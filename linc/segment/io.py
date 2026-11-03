"""Reading images and labels, writing masks in the input's grid. NIfTI through nibabel; 2D PNG/TIFF through Pillow.

An image is one file or a list of files (one per channel, nnU-Net style). It comes back as float32 (C, *spatial):
  NIfTI 3D          (X, Y, Z) -> C = 1; a 4D NIfTI (X, Y, Z, C) with C <= 4 is read as channels last
  NIfTI 2D          (X, Y) or (X, Y, 1) -> one 2D channel
  PNG / TIFF        (H, W) grey -> C = 1; (H, W, 3|4) RGB(A) -> C = 3 (alpha dropped)
meta carries what is needed to write a mask back in the same grid (affine and header for NIfTI; the size for PNG) and the voxel size
in mm (NIfTI zooms; for PNG the task's spacing, else 1).
"""
from pathlib import Path

import nibabel as nib
import numpy as np

NIFTI = (".nii", ".nii.gz")


def _is_nifti(p):
    return str(p).endswith(NIFTI)


def _read_one(p, dim):
    p = str(p)
    if _is_nifti(p):
        im = nib.load(p)
        a = np.asanyarray(im.dataobj)
        z = [float(v) for v in im.header.get_zooms()]
        if dim == 2:
            a = a.reshape(a.shape[:2]) if a.ndim == 3 and a.shape[2] == 1 else a
            assert a.ndim in (2, 3), f"{p}: shape {a.shape} is not a 2D image"
            ch = a[None] if a.ndim == 2 else np.moveaxis(a, -1, 0)
            return ch, dict(kind="nifti", affine=im.affine, header=im.header, shape=a.shape[:2], spacing=z[:2])
        if a.ndim == 4 and a.shape[3] <= 4:
            ch = np.moveaxis(a, -1, 0)
        else:
            assert a.ndim == 3, f"{p}: shape {a.shape} is not a 3D image"
            ch = a[None]
        return ch, dict(kind="nifti", affine=im.affine, header=im.header, shape=a.shape[:3], spacing=z[:3])
    from PIL import Image
    with Image.open(p) as im:
        a = np.asarray(im)
    if a.ndim == 3:
        a = a[..., :3]
        ch = np.moveaxis(a, -1, 0)
    else:
        ch = a[None]
    return ch, dict(kind="image", shape=a.shape[:2], spacing=[1.0, 1.0], suffix=Path(p).suffix)


def read_image(paths, dim, spacing=None):
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    chans, meta = [], None
    for p in paths:
        c, m = _read_one(p, dim)
        chans.append(c.astype(np.float32))
        meta = meta or m
        assert tuple(m["shape"]) == tuple(meta["shape"]), f"{p}: channel shape {m['shape']} differs from {meta['shape']}"
    if spacing is not None and meta["kind"] == "image":
        meta["spacing"] = [float(s) for s in spacing]
    return np.concatenate(chans, 0), meta


def read_label(path, dim):
    a, m = _read_one(path, dim)
    assert a.shape[0] == 1, f"{path}: a label must have one channel"
    return np.rint(a[0]).astype(np.uint8), m


def target_shape(shape, spacing, target):
    return [max(1, int(round(s * sp / tg))) for s, sp, tg in zip(shape, spacing, target)]


def resample(arr, shape, mode):
    """arr (C, *S) -> (C, *shape); mode "linear" (images, probabilities), "area" (block means) or "nearest" (labels); torch, CPU"""
    import torch
    import torch.nn.functional as F
    if list(arr.shape[1:]) == list(shape):
        return arr
    x = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))[None]
    if mode == "linear":
        y = F.interpolate(x, size=list(shape), mode="bilinear" if x.ndim == 4 else "trilinear", align_corners=False)
    elif mode == "area":                                                  # downsampling: the mean of each block (4 x 4 for /4)
        y = F.interpolate(x, size=list(shape), mode="area")
    else:
        y = F.interpolate(x, size=list(shape), mode="nearest-exact")
    return y[0].numpy()


def resample_label(lab, shape):
    """lab (*S) uint8 -> (*shape) uint8: nearest-exact (the input voxel nearest each output voxel's center: unbiased, keeps thin
    structures such as the tubes), except when every axis is reduced by an integer factor and at least one factor is even: there two
    input voxels are equally near and nearest-exact took the upper one (x 2: the label half a fine voxel toward low indices), so each
    class's block mean (linear, center-aligned) >= 0.5 decides (nnU-Net's order-1 rule; its ties go to the class)."""
    if list(lab.shape) == list(shape):
        return lab
    k = [s / t for s, t in zip(lab.shape, shape)]
    if not (all(abs(f - round(f)) < 1e-9 and round(f) >= 1 for f in k) and any(round(f) % 2 == 0 for f in k)):
        return np.rint(resample(lab[None].astype(np.float32), shape, "nearest")[0]).astype(np.uint8)
    out = np.zeros(list(shape), np.uint8)
    for c in np.unique(lab):
        if c:
            out[resample((lab == c)[None].astype(np.float32), shape, "linear")[0] >= 0.5] = c
    return out


NORMALIZATIONS = ("zscore", "zscore_nonzero")


def zscore(img, mode="zscore"):
    """per image and channel. "zscore": over the whole image (nnU-Net's MR normalization; the background is signal).
    "zscore_nonzero": over the voxels where any channel is nonzero (an input whose background is already zero, e.g. a brain-only stack);
    those voxels stay 0 (nnU-Net's normalization with a nonzero mask)."""
    C = img.shape[0]
    shape = (-1,) + (1,) * (img.ndim - 1)
    if mode == "zscore":
        flat = img.reshape(C, -1)
        return ((img - flat.mean(1).reshape(shape)) / (flat.std(1).reshape(shape) + 1e-8)).astype(np.float32)
    assert mode == "zscore_nonzero", mode
    m = (img != 0).any(0)
    if not m.any():
        return np.zeros_like(img, dtype=np.float32)
    v = img[:, m]
    out = (img - v.mean(1).reshape(shape)) / (v.std(1).reshape(shape) + 1e-8)
    out[:, ~m] = 0
    return out.astype(np.float32)


def write_mask(mask, meta, out):
    out = str(out)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    mask = np.asarray(mask, np.uint8)
    if meta["kind"] == "nifti":
        hdr = meta["header"].copy(); hdr.set_data_dtype(np.uint8)
        shape = tuple(meta["header"].get_data_shape())
        data = mask.reshape(shape[:mask.ndim] + (1,) * (len(shape) - mask.ndim)) if len(shape) > mask.ndim and all(
            s == 1 for s in shape[mask.ndim:]) else mask
        img = nib.Nifti1Image(data, meta["affine"], hdr)
        img.set_qform(*meta["header"].get_qform(coded=True)); img.set_sform(*meta["header"].get_sform(coded=True))
        nib.save(img, out)
    else:
        from PIL import Image
        Image.fromarray(mask).save(out)
