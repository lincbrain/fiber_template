"""A prepared brain folder (layout: code/blockface/synthesis/__init__.py) as arrays, and the paired training patches.

Grids: the input lives on the 0.5 mm box C, the output on F, C's center-aligned 2x refinement: output voxel 2k sits on input voxel
k, output voxel 2k+1 half way to k+1. An input patch starting at s (n^3) pairs with the output patch starting at 2s ((2n)^3).
The join plane: every output voxel k has a plane coordinate y_dep(k) = y0 + g . k (in output voxels along the normal of the
photographed sections, increasing rostrally); the cut is the plane y_dep = z_f.
Protocols: "full" trains on all photographed tissue rostral of the cut (the networks that made the completed volumes); "lso"
(leave slab out) also withholds the slab of dstar_planes planes rostral of the cut (the networks of the withheld-slab evaluation).
"""
import json

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from . import CFG


def recorded_path(p):
    """a source path recorded in a prepared brain's seam.json (on the analysis machine), re-rooted on this machine"""
    from shared import paths
    return paths.rerooted(p)


def y_dep_volume(rec, shape):
    """the plane coordinate y_dep of every output voxel (seam.json: y0 + g . k)"""
    y = rec["seam"]["y_dep"]; g = np.asarray(y["g"], np.float32)
    ii = [np.arange(n, dtype=np.float32) for n in shape]
    return (np.float32(y["y0"]) + ii[0][:, None, None] * g[0] + ii[1][None, :, None] * g[1] + ii[2][None, None, :] * g[2]).astype(np.float32)


def plane_masks(rec, ydep, protocol):
    """(trainable, withheld slab) on the continuous plane coordinate"""
    zf, n = rec["seam"]["z_f"], rec["seam"]["dstar_planes"]
    slab = (ydep >= zf - 0.5) & (ydep < zf + n - 0.5)
    if protocol == "full":
        return ydep >= zf - 0.5, np.zeros_like(slab)
    if protocol == "lso":
        return ydep >= zf + n - 0.5, slab
    raise ValueError(f"protocol {protocol!r}: full or lso")


def load_subject(D, protocol, photographed_only=CFG["training"]["photographed_only"]):
    """one prepared brain: input x (1, C...), target y on [0, 1] (the photographs normalized by the 1st/99th percentiles of the
    retained targets), loss mask m, and counts (info)"""
    rec = json.load(open(f"{D}/seam.json"))
    ld = lambda p: np.asanyarray(nib.load(f"{D}/{p}").dataobj)  # noqa: E731
    x = ld("coarse/in.nii.gz").astype(np.float32)[None]
    bf = ld("fine/bf.nii.gz").astype(np.float32); valid = ld("fine/valid.nii.gz") > 0
    ydep = y_dep_volume(rec, bf.shape)
    keep, slab = plane_masks(rec, ydep, protocol); del ydep
    m = valid & keep
    v0, v1 = int(valid.sum()), int(m.sum())
    photo = None
    if photographed_only is not None:
        pf = ld("fine/photo_fraction.nii.gz"); brain = ld("fine/brain.nii.gz") > 0
        low = brain & (pf < photographed_only)
        m2 = m & ~low
        photo = dict(threshold=photographed_only, removed=v1 - int(m2.sum()), removed_pct=round(100.0 * (v1 - int(m2.sum())) / max(v1, 1), 3))
        m = m2
    q1, q99 = (float(v) for v in np.percentile(bf[m], [1, 99]))             # python floats: float32 arithmetic under NumPy 1 and 2
    y = np.clip((bf - q1) / max(q99 - q1, 1e-6), 0, 1).astype(np.float32)
    info = dict(valid=v0, after_keep=v1, after_all=int(m.sum()), slab_vox=int(slab.sum()), photographed_only=photo,
                target_norm=[float(q1), float(q99)], coarse_shape=list(x.shape[1:]), fine_shape=list(y.shape))
    return dict(S=rec.get("subject"), x=x, y=y, m=m, info=info)


class Patches(Dataset):
    """random paired patches (input n^3, output (2n)^3) centered on loss voxels, pooled over brains"""
    def __init__(self, subjects, n=CFG["window"], length=CFG["training"]["patches_per_epoch"],
                 stride=CFG["training"]["patch_stride"], seed=0):
        self.subj = subjects; self.n = n; self.length = length
        self.rng = np.random.default_rng(seed)
        self.coords = [np.argwhere(sd["m"])[::stride] // 2 for sd in subjects]      # input-grid centers

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        n = self.n; h = n // 2
        si = int(self.rng.integers(0, len(self.subj))); sd = self.subj[si]; co = self.coords[si]
        c = co[int(self.rng.integers(0, len(co)))]
        Nc = np.array(sd["x"].shape[1:])
        s = np.clip(c - h, 0, np.maximum(0, Nc - n)).astype(int)
        x = _crop(sd["x"], (0,) + tuple(s), (1, n, n, n))
        y = _crop(sd["y"][None], (0,) + tuple(2 * s), (1, 2 * n, 2 * n, 2 * n))
        m = _crop(sd["m"][None].astype(np.float32), (0,) + tuple(2 * s), (1, 2 * n, 2 * n, 2 * n))
        return {"input": torch.from_numpy(x), "target": torch.from_numpy(y), "mask": torch.from_numpy(m)}


def _crop(a, lo, n):
    out = np.zeros(n, dtype=a.dtype)
    sl_src = tuple(slice(l, min(l + k, s)) for l, k, s in zip(lo, n, a.shape))  # noqa: E741
    sl_dst = tuple(slice(0, s.stop - s.start) for s in sl_src)
    out[sl_dst] = a[sl_src]
    return np.ascontiguousarray(out)
