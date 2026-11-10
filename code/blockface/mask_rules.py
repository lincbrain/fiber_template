"""The post-processing rules of the blockface masks, as plain numpy / scipy functions, so the pipeline and the tests share one copy.

You need
  a predicted mask (a boolean array).
What it does
    fill_photo_mask   step 1: a photograph's brain mask with its interior holes filled (photos.normalize_masks)
    largest_cc_fill   step 4: the largest 26-connected component of the 3D brain mask, interior holes filled (segment3d.post_mask)
    tube_rule         step 6: marker tubes kept left of the brain's x-centroid + a tolerance, inside the brain, components smaller
                      than min_vox dropped (segment3d.post_holes)
  These are the rules the paper's masks were made with (the analysis project's code/tasks.py normalize_masks and code/segment.py).
  Within linc-reconstruct each is applied once, here, to the network's raw mask; each linc-segment model's spec.json
  holds the same rule for use outside the pipeline (tests/test_blockface_rules.py checks that they agree).
You get
  the post-processed mask (the same shape).
"""
import numpy as np
from scipy import ndimage as ndi


def fill_photo_mask(m):
    """A 2D brain mask with its interior holes filled (scipy binary_fill_holes), uint8 0/1."""
    return ndi.binary_fill_holes(np.asarray(m) > 0).astype(np.uint8)


def largest_cc_fill(M):
    """Keep the largest 3D connected component (26-conn) + fill interior holes (drop wax/edge false
    positives). Returns (mask uint8, % foreground removed)."""
    M = np.asarray(M) > 0
    lab, n = ndi.label(M, structure=np.ones((3, 3, 3)))
    if n == 0:
        return M.astype(np.uint8), 0.0
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    big = int(np.argmax(sizes)); tot = int(M.sum())
    out = ndi.binary_fill_holes(lab == big)
    removed = 100.0 * (tot - int((lab == big).sum())) / max(tot, 1)
    return out.astype(np.uint8), removed


def tube_rule(P, brain, left_tol_vox, min_vox):
    """LEFT-hemisphere constraint (tubes are always anatomical-left; +X = Right, so drop predictions at/right of the brain X-centroid +
    left_tol_vox), clamp to the brain, drop specks < min_vox (each tube is its own component - NO keep-largest/fill).
    P, brain: the same grid. Returns (kept bool, voxels dropped on the right, voxels dropped as specks, number of tubes kept)."""
    P = np.asarray(P) > 0; brain = np.asarray(brain) > 0
    assert P.shape == brain.shape, (P.shape, brain.shape)
    rdrop = 0
    if brain.any():
        midx = ndi.center_of_mass(brain)[0] + left_tol_vox                         # brain midline + fissure tol
        right = P & (np.arange(P.shape[0])[:, None, None] >= midx)
        rdrop = int(right.sum()); P = P & ~right
    P = P & brain
    lab, n = ndi.label(P, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    keep = np.isin(lab, np.where(sizes >= min_vox)[0])
    return keep, rdrop, int((P & ~keep).sum()), int((sizes >= min_vox).sum())
