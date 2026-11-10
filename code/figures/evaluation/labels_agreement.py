"""Agreement of LGN/STN labels carried into the MRI template (paper: Results, "Labels in the template"; Table 7).

Shared by code/figures/paper_numbers/labels_overlap_current.py and lgnstn_geometry_current.py.

surface: mask voxels with a background 6-neighbor.
overlap_metrics: Dice, ASSD (mean symmetric surface distance, mm), centroid distance (mm), volume ratio and voxel counts of two masks.
one_voxel_ref: the Dice a structure scores against ITSELF moved by exactly one voxel of its grid (0.25 mm on the template) along the grid
  axis closest to its thinnest principal axis (PCA of its voxel world coordinates; integer shift, no interpolation; mean of + and -).
  It tells a reader what a one-voxel error costs that structure (a thin STN ~0.7, a thick LGN ~0.87), next to the measured Dice.
bf_ap_spacing: a brain's blockface anterior-posterior (slice) spacing = axis 1 of its blockface-native label grid.
AA_RULE: the anti-aliasing applied when the labels were carried onto the template (recorded as text in the overlap record).
"""
import glob
import os

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

# the analysis' own rule texts, verbatim (they are part of the records' data)
AA_RULE = "per source axis: Gaussian FWHM = max(source spacing, target voxel), then ONE linear resample, > 0.5 (orchestrator 2026-09-28)"
ONE_VOXEL_RULE = ("Dice of the label vs itself shifted exactly one voxel (0.25 mm) along the grid axis closest to its thinnest principal "
                  "axis (PCA of voxel world coords; integer shift, no interpolation; mean of +/-)")


def surface(m):
    return m & ~binary_erosion(m)


def overlap_metrics(a, b, vox, aff):
    """{dice, assd_mm, centroid_mm, vol_ratio, vox_a, vox_b} of boolean masks a, b on one grid (vox = voxel sizes, aff = affine)."""
    if not a.any() or not b.any():
        return dict(dice=None)
    inter = (a & b).sum()
    sa, sb = surface(a), surface(b)
    da = distance_transform_edt(~sb, sampling=vox)[sa]
    db = distance_transform_edt(~sa, sampling=vox)[sb]
    ca = nib.affines.apply_affine(aff, np.argwhere(a).mean(0)); cb = nib.affines.apply_affine(aff, np.argwhere(b).mean(0))
    return dict(dice=round(float(2 * inter / (a.sum() + b.sum())), 4),
                assd_mm=round(float((da.sum() + db.sum()) / (len(da) + len(db))), 3),
                centroid_mm=round(float(np.linalg.norm(ca - cb)), 3),
                vol_ratio=round(float(a.sum() / b.sum()), 3), vox_a=int(a.sum()), vox_b=int(b.sum()))


def one_voxel_ref(mask, aff):
    m = np.asarray(mask) > 0
    if m.sum() < 4:
        return None
    ijk = np.argwhere(m)
    M = aff[:3, :3]
    e = np.linalg.eigh(np.cov(((ijk - ijk.mean(0)) @ M.T).T))[1][:, 0]    # thinnest principal axis (world, unit)
    ax = int(np.argmax(np.abs(np.linalg.solve(M, e)) * nib.affines.voxel_sizes(aff)))   # grid axis closest to it
    d = []
    for sgn in (1, -1):                                                    # exact one-voxel shift (no interpolation), + and -
        q = ijk.copy(); q[:, ax] += sgn
        b = np.zeros(np.array(m.shape) + 2, bool); b[tuple((q + 1).T)] = True
        d.append(2 * (b[1:-1, 1:-1, 1:-1] & m).sum() / (b[1:-1, 1:-1, 1:-1].sum() + m.sum()))
    return round(float(np.mean(d)), 4)


def bf_ap_spacing(S, push_dir):
    """Blockface A-P spacing (mm) of brain S from its demons-interpolated blockface-native labels in push_dir/<S>/ (None if none)."""
    f = sorted(glob.glob(os.path.join(push_dir, S, f"{S}_*_bfnative_L2_demons.nii.gz")))
    return round(float(nib.load(f[0]).header.get_zooms()[1]), 4) if f else None
