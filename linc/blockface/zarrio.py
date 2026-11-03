"""OME-Zarr I/O for the blockface pipeline.

Convention: store data in canonical (c, z, y, x) order with z=rows, y=slices,
x=cols and POSITIVE scales, so the zarr opens in neuroglancer and zarr2nii
gives the correct orientation at ANY pyramid level. Rows & cols are reversed
(lossless) at write time for an upright, correctly-lateralised volume; the
slice axis (y) is never downsampled. This module is the single source of truth
for how blockface/mask zarrs are written.

Each store keeps a shallow standard pyramid: level L is level 0 downsampled in-plane by 2**L
(consistent with the OME convention). We keep NLEVELS levels: L0 = full resolution (what you
edit / process), L1 = /2, L2 = /4 (the light QC-nifti / browse level). Export L0 for full-res
editing, L2 for QC. Deeper levels are dropped (regenerable); the slice axis is never downsampled.

Ported unchanged from the analysis project's code/zarrio.py.
"""
import math
import os
from glob import glob

import numpy as np
import zarr
from PIL import Image

CHUNK = 1024
SHARD = 8 * CHUNK
MASK_SCALE = 50           # mask PNGs store label*50 so labels 0..5 are visible (0,50,..,250)
NLEVELS = 3               # pyramid depth: level L is /2**L in-plane (L0 edit, L2=/4 QC/browse)
STACK_PAD = 200           # px/side safety margin on the initial raw stack (room for the jump warp)
CROP_ALIGN = 256          # reconstructed L0 rows/cols padded up to a multiple of this (=2**8: 8 clean
                          # in-plane halvings) with minimal background. NB changing it re-shapes the
                          # blockface grid -> re-crops 03/05 and invalidates the histology TIRL; bump to
                          # 512 (so dim/2**8 is even) only if you re-run stage 4 onward.
CROP_MARGIN_MM = 4.0      # tolerance kept around the tissue bbox on every side BEFORE the CROP_ALIGN pad:
                          # if the nnUNet mask under-segments, the real tissue just outside it still lands
                          # inside the crop (visible in the background-intact bg twin, editable into the
                          # mask) so nothing real is cropped away. Physical (mm) -> px via the subject vx.
Image.MAX_IMAGE_PIXELS = None


def ceildiv(x, y):
    return int(math.ceil(x / y))


def load_image(fname):
    """Read a photo/mask PNG or JPG as a numpy array, shape (H, W[, C])."""
    return np.array(Image.open(fname))


def load_mask(fname):
    """Read a *50-scaled mask PNG, returning integer labels (undo the scaling)."""
    return (np.asarray(Image.open(fname)).astype(np.int16) // MASK_SCALE).astype(np.uint8)


def save_mask(label, fname):
    """Write integer labels as a *50-scaled PNG (visible; reversible with load_mask)."""
    Image.fromarray((np.asarray(label).astype(np.uint8) * MASK_SCALE)).save(fname)


def normalize_mask_dir(folder):
    """Re-save every nnUNet label PNG in `folder` as *MASK_SCALE-scaled (visible: 0,50,100,...).
    Idempotent (unscales first if already scaled). Project-wide mask visibility standard,
    for both Dataset500 (background) and Dataset704 (segmentation) outputs."""
    files = sorted(glob(os.path.join(folder, "*.png")))
    for f in files:
        arr = np.asarray(Image.open(f)).astype(np.int16)
        label = arr // MASK_SCALE if arr.max() > 5 else arr
        save_mask(label, f)
    return len(files)


def new_group(path):
    store = zarr.storage.LocalStore(path)
    return zarr.group(store=store, zarr_format=3, overwrite=True)


def create_level(omz, level, shape):
    """Create one pyramid level with the standard chunk/shard/compressor setup.
    shape is (c, rows, slices, cols)."""
    return omz.create_array(
        f"{level}",
        shape=shape,
        chunks=(1, CHUNK, 1, CHUNK),
        shards=(1, SHARD, 8, SHARD),
        compressors=zarr.codecs.BloscCodec(cname="lz4", clevel=5, shuffle="shuffle"),
        config="F",
        dtype="uint8",
        fill_value=0,
    )


def _downsample_inplane(img, how):
    """Halve the in-plane axes of img (c, rows, cols).
    how: 'stride' (subsample), 'mean', or 'max' over 2x2 blocks."""
    if how == "stride":
        return img[:, ::2, ::2]
    c, h, w = img.shape
    if h % 2 or w % 2:                       # pad odd edges by replication
        img = np.pad(img, ((0, 0), (0, h % 2), (0, w % 2)), mode="edge")
    c, H, W = img.shape
    blocks = img.reshape(c, H // 2, 2, W // 2, 2)
    if how == "mean":
        return np.rint(blocks.mean(axis=(2, 4))).astype(img.dtype)
    if how == "max":
        return blocks.max(axis=(2, 4))
    raise ValueError(f"unknown reduce method {how!r}")


def add_pyramid(omz, reduce="mean", nlevels=NLEVELS):
    """Build levels 1..nlevels-1 by halving the in-plane axes (rows=1, cols=3) with `reduce`
    in {'stride','mean','max'}; the slice axis (2) is preserved, level 0 left untouched.
    nlevels defaults to NLEVELS (3); the canonical L1 blockface pyramid passes nlevels=4 (L1..L4)."""
    c, _, nz, _ = omz["0"].shape
    for level in range(1, nlevels):
        prev = omz[f"{level-1}"]
        arr = create_level(
            omz, level, [c, ceildiv(prev.shape[1], 2), nz, ceildiv(prev.shape[3], 2)]
        )
        for i in range(nz):
            arr[:, :, i, :] = _downsample_inplane(prev[:, :, i, :], reduce)
        print(f"level {level}: {reduce} /2")


def rebuild_pyramid(omz, reduce="mean"):
    """Regenerate levels 1..NLEVELS-1 after level 0 was modified in place."""
    for level in range(1, NLEVELS):
        if f"{level}" in omz:
            del omz[f"{level}"]
    add_pyramid(omz, reduce)


def base_translation(path):
    """The level-0 [c, z, y, x] translation of an OME store (world = scale * (voxel + translation)).
    Read by a derived stack (seg/histo) so it can inherit the blockface's crop offset and stay
    co-registered with it."""
    return read_group(path).attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["translation"]


def _slice_axis(vz, y_trans, y_anterior0):
    """(scale, translation) for the SLICE axis, in the OME convention world = scale * (voxel + translation), um.

    Blockface stacks run front to back, so the slice scale is NEGATIVE and slice 0 sits at the store's anterior
    end: world_y(j) = y_anterior0 - vz * j. Given in mm, returned in um. `y_anterior0=None` keeps the old
    positive mapping (non-blockface stores, and the pre-2026-09-19 record)."""
    if y_anterior0 is None:
        return float(vz), float(y_trans)
    sy = -float(vz)
    return sy, float(y_anterior0) * 1000.0 / sy


def write_metadata(omz, vx, vz, reduce="mean", y_trans=0.5, z_off=0.0, x_off=0.0, nlevels=NLEVELS,
                   y_anterior0=None):
    """OME multiscale metadata: axes [c, z, y, x]. Level L is /2**L in-plane
    (scale 2**L * vx on z=rows and x=cols); y=slices is vz (constant, never downsampled).

    THE SLICE AXIS IS NEGATIVE (2026-09-19). The photos are cut and ingested FRONT to BACK, so the slice index
    advances POSTERIORLY, while +y means ANTERIOR in NIfTI. Writing a positive slice scale therefore declared the
    A-P axis backwards on every blockface world in the project, and forced a reflection into every BF->MRI
    transform (13/13 negative determinants - the "mirrored digitisation" of the chirality work). The direction is
    encoded HERE, in the store's own coordinateTransformations (scale y < 0, translation set so the volume keeps
    its place), NOT in a reader-side convention: any tool that implements `world = scale * (voxel + translation)`
    - ours, zarr2nii, niizarr - then produces an anatomically correct NIfTI by default.
    `y_anterior0` is the world y (mm) of THIS store's slice 0, i.e. its most anterior slice; pass it so a cropped
    or sub-sampled store lands on its parent. Omit it (None) only for a non-blockface store with no A-P meaning.
    nlevels defaults to NLEVELS (3); the canonical L1 blockface pyramid passes nlevels=4. Here `vx`
    is the LEVEL-0 in-plane size (for the L1 store, vx_base = 2**RECON_LEVEL * subject_vx).

    Translations (world = scale * (voxel + translation)) keep a store co-registered with its source:
    - `y_trans` (slice-axis voxels) shifts the slice origin so a sub-sampled stack (e.g. the
      histology-matched blockface) sits at the world-z of its source slices;
    - `z_off`/`x_off` (row/col origin offset, in LEVEL-0 voxels) place a CROPPED store at the same
      world x/z as its uncropped source, so all exported niftis overlay. In-plane axes downsample by
      2**L, so the offset scales by 1/2**L per level while the 0.5 voxel-centre stays constant (the
      slice axis is never downsampled, so y_trans is level-independent)."""
    sy, ty = _slice_axis(vz, y_trans, y_anterior0)
    datasets = [
        {
            "path": str(n),
            "coordinateTransformations": [
                {"type": "translation",
                 "translation": [0.0, 0.5 + float(z_off) / 2**n, ty, 0.5 + float(x_off) / 2**n]},
                {"type": "scale",
                 "scale": [1.0, float(2**n) * vx, sy, float(2**n) * vx]},
            ],
        }
        for n in range(nlevels)
    ]
    omz.attrs["multiscales"] = [{
        "axes": [
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
        "datasets": datasets,
        "description": f"{NLEVELS}-level pyramid, 2x in-plane downsampling ({reduce}); slice axis kept",
        "version": "0.4",
    }]


def write_metadata_level(omz, vx, vz, level, reduce="mean", y_trans=0.5, z_off=0.0, x_off=0.0,
                         y_anterior0=None):
    """OME metadata for a store whose ONLY array '0' holds data at pyramid `level` (in-plane /2**level,
    slice axis full). Same world frame as write_metadata's datasets[level], so e.g. a store built directly
    at L2 co-registers exactly with another store's --level 2 export. Used by the background-intact bg twin
    (built straight at L2 to avoid ever materialising a full-res copy)."""
    f = float(2 ** level)
    sy, ty = _slice_axis(vz, y_trans, y_anterior0)
    omz.attrs["multiscales"] = [{
        "axes": [
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
        "datasets": [{
            "path": "0",
            "coordinateTransformations": [
                {"type": "translation",
                 "translation": [0.0, 0.5 + float(z_off) / f, ty, 0.5 + float(x_off) / f]},
                {"type": "scale", "scale": [1.0, f * vx, sy, f * vx]},
            ],
        }],
        "description": f"single array at pyramid level {level} (in-plane /{int(f)}; slice axis kept)",
        "version": "0.4",
    }]


def read_group(path):
    return zarr.open_group(path, mode="r")


def y_anterior0_from(src_path, first_src_slice=0.0, level=2):
    """World y (mm) that a DERIVED store's slice 0 must sit at, given the source store it is aligned to and the
    source slice its own slice 0 corresponds to. Pass the result as `write_metadata(y_anterior0=...)` so the
    derived store inherits the source's A-P direction (see _slice_axis). Returns None for a pre-2026-09-19
    source, which keeps the old positive mapping rather than guessing."""
    try:
        _, A = read_level(src_path, level)
    except Exception:
        return None
    if A[1, 1] >= 0:
        return None
    return float(A[1, 3] + A[1, 1] * float(first_src_slice))


def read_level(path, level):
    """Materialise pyramid `level` of an OME store as (array, nibabel affine) IN-PROCESS (no niizarr / no
    temp nifti), so recon/curate/segment/register (tirl env) can derive any level from the canonical pyramid.
    Reproduces zarr2nii's OME->affine: world = scale * (voxel + translation), micrometre->mm, and the axis
    permute (c,rows,slices,cols)->(cols,slices,rows[,c]) == the (W, nz, H[, c]) order the pipeline uses.
    The stored data already has rows/cols reversed for an upright volume, so the affine is positive-diagonal."""
    g = read_group(path)
    a = np.asarray(g[str(level)])                                   # (c, rows, slices, cols) uint8
    ct = g.attrs["multiscales"][0]["datasets"][level]["coordinateTransformations"]
    tr, sc = ct[0]["translation"], ct[1]["scale"]                  # [c, z=rows, y=slices, x=cols]
    arr = np.transpose(a, (3, 2, 1, 0)) if a.shape[0] > 1 else np.transpose(a[0], (2, 1, 0))  # ->(cols,slices,rows[,c])
    A = np.eye(4)
    A[0, 0], A[1, 1], A[2, 2] = sc[3] / 1000.0, sc[2] / 1000.0, sc[1] / 1000.0        # cols, slices, rows (mm)
    A[0, 3], A[1, 3], A[2, 3] = sc[3] * tr[3] / 1000.0, sc[2] * tr[2] / 1000.0, sc[1] * tr[1] / 1000.0
    return arr, A            # the slice axis is negative for blockface stores: the direction lives in the store


def write_stack(omz_dir, get_slice, nz, base_rows, base_cols, channels,
                vx, vz, reduce, pad=STACK_PAD, y_trans=0.5, z_off=0.0, x_off=0.0, y_anterior0=None):
    """Build a shallow-pyramid OME-Zarr from per-slice native-orientation arrays.
    get_slice(s) returns (H, W) or (H, W, C). Each slice is reversed (rows & cols)
    and stored canonically as (c, rows, slice, cols).

    `pad` px per side is a safety margin so later in-plane shifts (e.g. camera-jump
    registration) don't push tissue out of frame - a workaround for the initial stack
    from raw photos. Stacks derived from an already-cropped grid (e.g. the seg masks,
    which share the reconstructed blockface grid) pass pad=0 plus the blockface's
    z_off/x_off so they stay co-registered with it in world space (see write_metadata)."""
    omz = new_group(omz_dir)
    a0 = create_level(omz, 0, [channels, base_rows + 2 * pad, nz, base_cols + 2 * pad])
    end = -pad if pad else None
    for s in range(nz):
        img = get_slice(s)
        if img.ndim == 2:
            img = img[:, :, None]
        img = img[::-1, ::-1, :]
        a0[:, pad:end, s, pad:end] = img.transpose(2, 0, 1)
    print(f"level 0: {nz} slices x {channels}ch")
    add_pyramid(omz, reduce)
    write_metadata(omz, vx, vz, reduce, y_trans, z_off, x_off,
                   y_anterior0=(vz * (nz - 1 + y_trans) / 1000.0) if y_anterior0 is None else y_anterior0)
    return omz
