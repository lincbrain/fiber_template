"""Display helpers for the figures of the completed volumes (Supplementary Sections S1, "The blockface pipeline", and S5.3,
"Accuracy of the synthesis on withheld slabs": code/figures/figS_synthesis_*.py). Display only.

Flat          views of a prepared brain's volumes on its section-aligned ("flat") deploy grid. The prepared fine grid (<ID>/fine,
              0.25 mm) is a box of the unrotated deploy grid; the flat grid is that grid rotated by R so that its axis 1 runs along the
              stack: every flat plane j is one photographed section (the plane coordinate y_dep of seam.json), and the junction plane is
              one plane. seam.json records the flat grid (affine, shape, R); a flat voxel maps to the fine grid as
                flat index -> world (flat affine) -> unrotated world (rotation R about the flat grid's center) -> fine index.
brain_colors  one luminance stretch per brain (1st-99.5th percentile of its photographed grey); photographed voxels drawn with their
              own chroma (RGB / its mean) x that luminance; predicted voxels with the brain's own luminance -> chroma lookup, fitted on
              its photographed voxels (median chroma in 48 luminance quantile bins, smoothed over 5 bins). The network predicts grey.
on_black      an RGB image composited onto black with a soft brain edge.
"""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


def _center(A, shape):
    return A[:3, :3] @ (0.5 * (np.array(shape[:3], float) - 1)) + A[:3, 3]


def _vox_world(idx, A):
    return idx @ A[:3, :3].T + A[:3, 3]


def _world_vox(X, A):
    Ai = np.linalg.inv(A)
    return X @ Ai[:3, :3].T + Ai[:3, 3]


class Flat:
    def __init__(self, brain_dir):
        D = Path(brain_dir)
        rec = json.load(open(D / "seam.json"))
        fl = rec["seam"]["flat"]
        self.S, self.rec = D.name, rec
        self.A = np.asarray(fl["affine"], float); self.shape = tuple(fl["shape"]); self.R = np.asarray(fl["R"], float)
        assert fl["ap_axis"] == 1
        self.Afi = nib.load(D / "fine" / "bf.nii.gz").affine
        self.z_f, self.n = rec["seam"]["z_f"], rec["seam"]["dstar_planes"]

    def coords(self, sl):
        """fine-grid voxel coordinates (3, ...) of the flat voxels selected by `sl` (a slice or an int per axis)"""
        axes = [np.arange(self.shape[d])[s] if isinstance(s, slice) else np.array([s]) for d, s in enumerate(sl)]
        idx = np.stack(np.meshgrid(*axes, indexing="ij"), -1)
        X = _vox_world(idx.reshape(-1, 3).astype(np.float64), self.A)
        c = _center(self.A, self.shape)
        v = _world_vox(c + (X - c) @ self.R, self.Afi)
        out = v.T.reshape(3, *idx.shape[:-1])
        keep = [d for d, s in enumerate(sl) if isinstance(s, slice)]
        return out.reshape(3, *[idx.shape[d] for d in keep])

    def sample(self, vol, sl, order=1, cval=0.0):
        c = self.coords(sl)
        if vol.ndim == 4:                                                 # RGB volumes: last axis = channel
            return np.stack([ndi.map_coordinates(vol[..., i], c, order=order, mode="constant", cval=cval)
                             for i in range(vol.shape[-1])], -1)
        return ndi.map_coordinates(vol, c, order=order, mode="constant", cval=cval)


def brain_colors(g_real, rgb, kept, n_bins=48):
    """g_real: photographed grey; rgb: photographs (..., 3); kept: photographed voxels to fit on.
    Returns (disp, photo, pred): luminance stretch, photographed coloring (g, rgb), predicted coloring (g)."""
    m_ = rgb[kept].mean(-1, keepdims=True)
    ok = m_[:, 0] > 1e-3
    chrom_v = rgb[kept][ok] / m_[ok]
    gv = g_real[kept][ok]
    edges = np.quantile(gv, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges, gv, side="right") - 1, 0, n_bins - 1)
    lut_g = np.array([np.median(gv[idx == k]) for k in range(n_bins)])
    lut_c = np.array([np.median(chrom_v[idx == k], axis=0) for k in range(n_bins)])
    lut_c = ndi.uniform_filter1d(lut_c, 5, axis=0, mode="nearest")
    lo, hi = np.percentile(g_real[kept], [1, 99.5])

    def disp(v):
        return np.clip((v - lo) / (hi - lo), 0, 1)

    def photo(g, r):
        mu = r.mean(-1, keepdims=True)
        ch = np.where(mu > 1e-3, r / np.maximum(mu, 1e-3), 1.0)
        return np.clip(ch * disp(g)[..., None], 0, 1)

    def pred(g):
        ch = np.stack([np.interp(g, lut_g, lut_c[:, k]) for k in range(3)], -1)
        return np.clip(ch * disp(g)[..., None], 0, 1)
    return disp, photo, pred


def on_black(img3, m, sigma=0.6):
    """composite an RGB image onto black with a soft brain edge"""
    a_ = np.clip(ndi.gaussian_filter(m.astype(np.float32), sigma), 0, 1)[..., None]
    a_ = np.where(m[..., None], np.maximum(a_, 0.6), a_ * (a_ > 0.25))
    return np.clip(a_ * img3, 0, 1)
