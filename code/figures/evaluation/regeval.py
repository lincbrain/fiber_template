#!/usr/bin/env python3
"""How well a blockface -> MRI registration aligns the inside of the brain, not only its outline.

    python -m figures.evaluation.regeval --mri <fixed.nii.gz> --bf <moved.nii.gz> [--warp <warp.nii.gz>] [--tag NAME]

You need
  the fixed MRI image and the moved blockface image on one grid; the warp, for its plausibility measures.
What it does
  It detects the interior edges of both images and measures:
  edge_mean     the mean distance (mm) from each interior edge voxel to the nearest interior edge of the other image, in the worse
                direction (the paper's "interior edge distance"). The median is quantized by the voxel lattice (a distance
                transform can only return sqrt(i^2+j^2+k^2) x voxel size), and the lattice differs between resolutions, so medians
                cannot be compared across working grids; the mean can.
  edge_frac1mm  the fraction of edge voxels more than 1 mm from their counterpart (independent of scale).
  edge_bf2mri   the median distance from each blockface interior edge voxel to the nearest MRI interior edge.
  edge_mri2bf   the same in reverse (the forward direction alone can be gamed: an over-flexible warp can squeeze blockface edges
                onto whatever MRI edge is nearest while leaving MRI structures unmatched).
  edge_sym      the larger of the two medians.
  nmi_interior  the normalized mutual information between the two images inside the eroded brain.
  jac_std, folds_pct, jac_out_0p5_2   the warp's plausibility: the standard deviation of its Jacobian determinant, the percentage
                of voxels where it is negative (folds), and where it is outside [0.5, 2].
You get
  the measures as JSON on stdout.

Ported unchanged from the analysis project's featurereg/regeval.py.
"""
import argparse
import json

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi


def norm(a, m):
    v = a[m]
    if v.size == 0:
        return np.zeros_like(a)
    lo, hi = np.percentile(v, (1, 99))
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def interior_edges(a, fg, pct=90, erode=6):
    """Ridge voxels strictly inside the brain. The tissue/background boundary is the strongest
    gradient in a blockface image by far, so it must be eroded away or the map is just the
    envelope — which envelope Dice already covers."""
    inner = ndi.binary_erosion(fg, np.ones((3, 3, 3)), iterations=erode)
    if inner.sum() < 500:
        return np.zeros(a.shape, bool), inner
    g = ndi.gaussian_filter(a, 1.6)
    m = np.sqrt(sum(x ** 2 for x in np.gradient(g)))
    return inner & (m > np.percentile(m[inner], pct)), inner


def nmi(x, y, bins=64):
    h, _, _ = np.histogram2d(x, y, bins=bins)
    p = h / h.sum()
    px, py = p.sum(1), p.sum(0)
    nz = p > 0
    hxy = -(p[nz] * np.log(p[nz])).sum()
    hx = -(px[px > 0] * np.log(px[px > 0])).sum()
    hy = -(py[py > 0] * np.log(py[py > 0])).sum()
    return float((hx + hy) / hxy)


# The saved field's x/y components are LPS-signed relative to the NIfTI RAS grid (determined
# empirically for provenance.apply_warp: LPS-pull reproduces bf_nonlin at corr 0.99, the
# alternatives at 0.28-0.49). It matters here too: WITHOUT the sign the Jacobian of MR252's
# production warp reads 0.77% folded, WITH it reads 0.000% — and the tool's own metrics.json says
# 0.0, so the signed version is the correct one. Getting this wrong invents folding that is not there.
LPS_SIGN = np.array([-1.0, -1.0, 1.0], np.float32)


def jac_stats(warp_path, vox):
    """Jacobian determinant of the displacement field: the plausibility guard."""
    try:
        W = np.squeeze(np.asanyarray(nib.load(warp_path).dataobj)).astype(np.float32)
    except Exception:
        return {}
    if W.ndim != 4 or W.shape[-1] != 3:
        return {}
    W = W * LPS_SIGN
    J = np.empty(W.shape[:3] + (3, 3), np.float32)
    for i in range(3):
        for j in range(3):
            J[..., i, j] = np.gradient(W[..., i], vox[j], axis=j)
        J[..., i, i] += 1.0
    d = np.linalg.det(J)
    sub = d[::2, ::2, ::2]
    return {"jac_std": float(sub.std()), "folds_pct": float(100.0 * (sub < 0).mean()),
            "jac_out_0p5_2": float(100.0 * ((sub < 0.5) | (sub > 2.0)).mean())}


def evaluate(mri_p, bf_p, warp_p=None, tag=""):
    mi = nib.load(mri_p); bi = nib.load(bf_p)
    M0 = np.asanyarray(mi.dataobj).astype(np.float32)
    B0 = np.asanyarray(bi.dataobj).astype(np.float32)
    vox = np.abs(np.diag(mi.affine)[:3])
    fg = B0 > 0.02 * B0.max()
    M, B = norm(M0, fg), norm(B0, fg)
    eb, inner = interior_edges(B, fg)
    em, _ = interior_edges(M, fg)
    out = {"tag": tag, "n_bf_edges": int(eb.sum()), "n_mri_edges": int(em.sum())}
    if eb.sum() > 100 and em.sum() > 100:
        d1 = ndi.distance_transform_edt(~em, sampling=vox)[eb]
        d2 = ndi.distance_transform_edt(~eb, sampling=vox)[em]
        out.update(edge_bf2mri=float(np.median(d1)), edge_mri2bf=float(np.median(d2)),
                   edge_sym=float(max(np.median(d1), np.median(d2))),
                   edge_mean=float(max(d1.mean(), d2.mean())),
                   edge_frac1mm=float(max((d1 > 1.0).mean(), (d2 > 1.0).mean())),
                   edge_bf2mri_p90=float(np.percentile(d1, 90)))
    if inner.sum() > 1000:
        out["nmi_interior"] = nmi(B[inner], M[inner])
    # envelope Dice, for continuity with the prior benchmark
    mb = M0 > 0.02 * M0.max()
    out["envelope_dice"] = float(2 * (fg & mb).sum() / max(fg.sum() + mb.sum(), 1))
    if warp_p:
        out.update(jac_stats(warp_p, vox))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mri", required=True); ap.add_argument("--bf", required=True)
    ap.add_argument("--warp", default=None); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    print(json.dumps(evaluate(a.mri, a.bf, a.warp, a.tag), indent=1))
