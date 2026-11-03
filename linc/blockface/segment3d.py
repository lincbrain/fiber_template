"""Steps 4 and 6: the 3D brain mask and the marker tubes, predicted so a brain is masked automatically.

You need
  for the mask, the volume with its background kept (nii/02_blockface_bg_L4: prepare.py); for the tubes, the masked color volume
  (steps/<S>_03_color.ome.zarr: reconstruct.py); the networks (configs/blockface.yaml `segmentation`: bf_mask3d and bf_holes,
  models/segmentation.md) run in between by external.py.
What it does
    prep_mask, post_mask     the uncropped L4 volume at 0.40 mm sections, RGB -> the network -> the largest 3D connected
                             component, holes filled
    prep_holes, post_holes   the tight-cropped color volume at 0.40 mm sections -> the network -> tubes in the left hemisphere
                             only, inside the brain, specks dropped
  The two rules are mask_rules.largest_cc_fill and tube_rule, applied once here whichever engine predicted.
You get
  curated/<S>_bg (the mask network's input), curated/<S>_mask and curated/<S>_holes on the curated grid. They are a starting point:
  a hand correction saved as <S>_mask_edited or <S>_holes_edited wins in the reconstruction, and is never overwritten here.

Ported unchanged from the analysis project's code/segment.py; parameters from configs/blockface.yaml `volume_segmentation`.
"""
import os
import shutil

import numpy as np
import nibabel as nib

from linc.blockface import curate, mask_rules, zarrio
from linc.blockface.subject import params

VS = params()["volume_segmentation"]


def _zblock(vol, f, how):
    """z-downsample native->0.40mm along the slice axis (axis 1) by factor f, last block padded by repeat.
    how='mean' (image) | 'max' (label). Matches the training-data recipe (build_holes3) exactly."""
    if f == 1:
        return vol
    x = np.moveaxis(vol, 1, 0)                                   # slice axis first
    pad = (-x.shape[0]) % f
    if pad:
        x = np.concatenate([x, np.repeat(x[-1:], pad, axis=0)], axis=0)
    x = x.reshape(x.shape[0] // f, f, *x.shape[1:])
    x = x.mean(1) if how == "mean" else x.max(1)
    return np.moveaxis(x, 0, 1)


def _z_affine(A, f):
    """native-z affine -> 0.40mm-z affine: scale the slice-axis (axis 1) column by f, block-centre origin."""
    A = A.copy().astype(float)
    if f > 1:
        A[:3, 3] += A[:3, 1] * ((f - 1) / 2.0)
        A[:3, 1] *= f
    return A


def _write_channels(img_rgb, aff, cfg):
    """Write the RGB volume as nnUNet channels <S>_0000/1/2.nii.gz into a fresh seg3d/in/ (+ ensure out/)."""
    IN = os.path.join(cfg.seg3d_dir, "in")
    if os.path.isdir(IN):
        shutil.rmtree(IN)
    os.makedirs(IN, exist_ok=True)
    os.makedirs(os.path.join(cfg.seg3d_dir, "out"), exist_ok=True)
    for c in range(3):
        nib.save(nib.Nifti1Image(np.ascontiguousarray(img_rgb[..., c]).astype(np.uint8), aff),
                 os.path.join(IN, f"{cfg.subject}_{c:04d}.nii.gz"))


_largest_cc_fill = mask_rules.largest_cc_fill                       # the rule of step 4 (linc/blockface/mask_rules.py)


def _f(cfg):
    # never below 1: a subject whose native z is already coarser than the model's 0.40 mm target (MF204 at
    # vz=810 um rounds to 0 -> ZeroDivisionError in _zblock) is simply fed at its native spacing, as every
    # vz >= 0.6 mm subject already was.
    return cfg.curation_f()                                       # pinned per brain (configs/subjects)


# ---------- brain mask (Dataset506) ----------
def prep_mask(cfg):
    """Build the mask model input: the uncropped L4 bg, z-downsampled native->0.40mm (block-mean), split
    to RGB channels. Rebuilds curated/<S>_bg.nii.gz from nii/02_blockface_bg_L4 when present (reproducible),
    else reuses an existing curated_bg (e.g. an already-migrated subject whose _work was reclaimed)."""
    f = _f(cfg)
    bgl4 = cfg.snap("02_blockface_bg_L4")
    if os.path.exists(bgl4):
        im = nib.load(bgl4); A = np.asanyarray(im.dataobj)                    # (W,nz,H,3) native-z, uncropped
        assert A.ndim == 4 and A.shape[-1] == 3, (bgl4, A.shape)
        img = np.clip(np.rint(_zblock(A.astype(np.float32), f, "mean")), 0, 255).astype(np.uint8)
        aff = _z_affine(im.affine, f)
        os.makedirs(cfg.curated_dir, exist_ok=True)
        nib.save(nib.Nifti1Image(np.ascontiguousarray(img), aff), cfg.curated_bg)
        print(f"  {cfg.subject}: built curated_bg {img.shape} (0.40mm-z) from 02_blockface_bg_L4 (f={f})")
    elif os.path.exists(cfg.curated_bg):
        im = nib.load(cfg.curated_bg); img = np.asanyarray(im.dataobj); aff = im.affine
        assert img.ndim == 4 and img.shape[-1] == 3, (cfg.curated_bg, img.shape)
        print(f"  {cfg.subject}: reuse existing curated_bg {img.shape}")
    else:
        raise SystemExit(f"no bg for the mask model: need {bgl4} (run `linc-reconstruct prepare`) or {cfg.curated_bg}")
    _write_channels(img, aff, cfg)


def post_mask(cfg):
    """Post-process the Dataset506 prediction (largest 3D CC + fill) and write it as the AUTO brain mask on
    the curated grid. Does NOT overwrite a hand <S>_mask_edited.nii.gz (that wins in reconstruction)."""
    pred = os.path.join(cfg.seg3d_dir, "out", f"{cfg.subject}.nii.gz")
    ref = nib.load(cfg.curated_bg)                                            # grid + affine authority
    out, removed = _largest_cc_fill(np.asanyarray(nib.load(pred).dataobj))
    assert out.shape == ref.shape[:3], (out.shape, ref.shape)
    nib.save(nib.Nifti1Image(out.astype(np.uint8), ref.affine), cfg.mask_model)
    note = " (NB: hand _mask_edited present -> it WINS in reconstruction)" if os.path.exists(cfg.mask_edited) else ""
    print(f"  {cfg.subject}: brain mask -> {os.path.basename(cfg.mask_model)} fg={int(out.sum())} "
          f"(largest-CC dropped {removed:.2f}%){note}")


# ---------- punch-holes (Dataset508) ----------
def prep_holes(cfg):
    """Build the holes model input: the masked colour-corrected brain (steps/<S>_03_color.ome.zarr, its L4
    level = tight-crop native-z) z-downsampled native->0.40mm (block-mean), split to RGB channels. The model
    stays L4/0.40mm-z regardless of the image reconstruction level. Requires `reconstruct` to have run."""
    f = _f(cfg)
    if not os.path.isdir(cfg.color_store):
        raise SystemExit(f"no {cfg.color_store}: run `linc-reconstruct volume --subject {cfg.subject}` first")
    *_, recon_level = curate._info(cfg)                                      # fail early if <S>.info missing
    arr, aff = zarrio.read_level(cfg.color_store, 4 - recon_level)           # L4 in-plane, native-z (W4,nz,H4,3)
    assert arr.ndim == 4 and arr.shape[-1] == 3, (cfg.color_store, arr.shape)
    img = np.clip(np.rint(_zblock(arr.astype(np.float32), f, "mean")), 0, 255).astype(np.uint8)
    _write_channels(img, _z_affine(aff, f), cfg)
    print(f"  {cfg.subject}: holes model input {img.shape} (0.40mm-z, tight-crop L4) from 03_color pyramid (f={f})")


def post_holes(cfg):
    """Post-process the Dataset508 prediction: LEFT-hemisphere constraint (holes are always anatomical-left;
    +X = Right, so drop predictions at/right of the brain X-centroid + tol), clamp to brain, drop specks
    < HOLES_MINVOX (each tube is its own component - NO keep-largest/fill). Padded back to the uncropped
    curated grid and written as the AUTO holes. A hand <S>_holes_edited.nii.gz wins in reconstruction."""
    pred = os.path.join(cfg.seg3d_dir, "out", f"{cfg.subject}.nii.gz")
    P = np.asanyarray(nib.load(pred).dataobj) > 0                             # tight-crop, 0.40mm-z
    _, _, i0, i1, j0, j1, _ = curate._info(cfg)
    # MUST be curated_mask (edited-wins), not mask_model: <S>.info's i0/i1/j0/j1 were produced by
    # reconstruct's crop, which used curated_mask. Preferring mask_model here made `brain` disagree
    # with the crop that produced P whenever a hand _mask_edited existed (7/17 subjects).
    mp = cfg.curated_mask
    mim = nib.load(mp); Mfull = np.asanyarray(mim.dataobj) > 0                # uncropped curated grid
    assert (i1 - i0, j1 - j0) == (P.shape[0], P.shape[2]), (P.shape, (i1 - i0, j1 - j0))
    brain = Mfull[i0:i1, :, j0:j1]                                           # brain on the pred (tight) grid
    zc = min(P.shape[1], brain.shape[1])                                     # reconcile any z off-by-one
    P, brain = P[:, :zc], brain[:, :zc]
    keep, rdrop, specks, ntubes = mask_rules.tube_rule(P, brain, VS['tubes_left_tolerance_vox'], VS['tubes_min_voxels'])
    full = np.zeros(Mfull.shape, np.uint8)                                   # pad tight -> uncropped grid
    full[i0:i1, :zc, j0:j1] = keep.astype(np.uint8)
    nib.save(nib.Nifti1Image(full, mim.affine), cfg.holes_model)
    note = " (NB: hand _holes_edited present -> it WINS)" if os.path.exists(cfg.holes_edited) else ""
    print(f"  {cfg.subject}: {ntubes} tubes, {int(full.sum())} vox -> "
          f"{os.path.basename(cfg.holes_model)} (dropped {rdrop} right-side, {specks} specks){note}")
