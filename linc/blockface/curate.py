"""Hand corrections of the curated masks, folded back onto their grid, and the quick tube re-fill (Methods, "Blockface
reconstruction": the masks were corrected by hand where needed).

You need
  a hand-edited final <S>_mask or <S>_holes (L4, native sections, tight crop), and the brain's <S>.info (the grid: factor, kept
  sections, crop).
What it does
  fold_edits(cfg, "mask"|"holes")  folds the edit back onto the curated grid (L4, 0.40 mm sections, uncropped)
  clamp_holes                      keeps every tube voxel inside the brain mask
  reinpaint                        re-fills the tubes on steps/<S>_03_color after a tubes-only edit (no full reconstruction)
You get
  curated/<S>_mask_edited or <S>_holes_edited (the hand corrections, which win over the models' outputs); after reinpaint, the
  reconstruction <S>_blockface.ome.zarr, <S>_holes and <S>.info rewritten.

Ported unchanged from the analysis project's code/curate.py (the one-off migration helpers are not part of the release).
"""
import math
import os
import shutil

import numpy as np
import nibabel as nib


def _info(cfg):
    """Parse the <S>.info grid contract: (factor f, kept-z, in-plane crop i0:i1_j0:j1 in L4 voxels,
    recon_level). recon_level (the canonical image pyramid's level 0; L4 crop -> L1 pixels via US=2**(4-lvl))
    defaults to 4 for legacy pre-L1 .info files."""
    d = dict(l.split("=", 1) for l in open(cfg.info_file).read().split("\n") if "=" in l)
    f, keep = int(d["factor"]), int(d["z_out"])
    i0, i1 = [int(x) for x in d["crop"].split("_")[0][1:].split(":")]
    j0, j1 = [int(x) for x in d["crop"].split("_")[1][1:].split(":")]
    return f, keep, i0, i1, j0, j1, int(d.get("recon_level", 4))


def fold_edits(cfg, kind):
    """Fold a hand-edited FINAL mask/holes (L4 / native-z / tight-crop) back onto the curated grid (L4 /
    0.40mm-z / uncropped) and write it as the _EDITED override (<S>_mask_edited / <S>_holes_edited), which
    wins over the model output - so a correction never clobbers the auto <S>_mask / <S>_holes. In-plane the
    crop maps 1:1; only a z-collapse (native -> 0.40mm block; MAX = foreground if any sub-slice) + an un-crop
    are needed. ONLY the in-crop region changes. Masks collapse orig-aware (block-MIN where it was brain so a
    removal wins, block-MAX where background so an addition wins); holes are additions-only -> block-MAX. The
    _edited file is seeded from itself if present, else from the model output; the previous _edited is .bak'd."""
    f, keep, i0, i1, j0, j1, _ = _info(cfg)
    src = cfg.final("mask") if kind == "mask" else cfg.final("holes")
    dst = cfg.mask_edited if kind == "mask" else cfg.holes_edited
    seed = dst if os.path.exists(dst) else (cfg.mask_model if kind == "mask" else cfg.holes_model)
    reproc = (np.asanyarray(nib.load(src).dataobj) > 0)                # (Wc, keep, Hc), native-z, cropped
    cim = nib.load(seed); cur = (np.asanyarray(cim.dataobj) > 0).astype(np.uint8)   # (Wr, Zc, Hr), 0.40mm-z
    Wc, Nk, Hc = reproc.shape
    assert Nk == keep and Wc == i1 - i0 and Hc == j1 - j0, (reproc.shape, keep, i1 - i0, j1 - j0)
    assert cur.shape[0] >= i1 and cur.shape[2] >= j1, (cur.shape, i1, j1)
    new = cur.copy()
    for K in range(math.ceil(keep / f)):
        zlo, zhi = K * f, min((K + 1) * f, keep)
        blk = reproc[:, zlo:zhi, :]
        if kind == "mask":
            ref = cur[i0:i1, K, j0:j1] > 0
            new[i0:i1, K, j0:j1] = np.where(ref, blk.min(axis=1), blk.max(axis=1)).astype(np.uint8)
        else:
            new[i0:i1, K, j0:j1] = blk.max(axis=1).astype(np.uint8)
    changed = int((new != cur).sum())
    if os.path.exists(dst):
        shutil.copy2(dst, dst + ".bak")
    nib.save(nib.Nifti1Image(np.ascontiguousarray(new), cim.affine), dst)
    print(f"  {cfg.subject} {kind}: {changed} vox changed, seeded from {os.path.basename(seed)} "
          f"(fg {int(cur.sum())}->{int(new.sum())}) -> {os.path.basename(dst)} (the _edited override)")


def clamp_holes(cfg, grid="final"):
    """Force every hole voxel to lie WITHIN the (binarized) brain mask, so no hole sits outside the brain.
    grid='final' clamps cfg.final('holes') to cfg.final('mask'); grid='curated' clamps the curated store.
    Binarizes the mask (label-2 interpolated slices count as brain). Overwrites only if something changed."""
    hp, mp = (cfg.curated_holes, cfg.curated_mask) if grid == "curated" else (cfg.final("holes"), cfg.final("mask"))
    him = nib.load(hp); H = np.asanyarray(him.dataobj) > 0
    M = np.asanyarray(nib.load(mp).dataobj) > 0
    assert H.shape == M.shape, (H.shape, M.shape)
    outside = int((H & ~M).sum())
    if outside:
        nib.save(nib.Nifti1Image((H & M).astype(np.uint8), him.affine), hp)
    print(f"  {cfg.subject} clamp_holes[{grid}]: fg {int(H.sum())}, outside-brain {outside} "
          f"-> {'MASKED (overwritten)' if outside else 'clean'}")


def reinpaint(cfg):
    """Fast holes-only re-fill after a curated-holes edit: re-inpaint the EXISTING colour-corrected L1 image
    (steps/<S>_03_color.ome.zarr, pre-inpaint) with the current curated holes (upsampled to L1 & clamped to
    brain), instead of a full reconstruction. Rewrites the canonical blockface pyramid, the L4 <S>_holes
    final and the .info hole count. Mask/shading/colour are untouched (their inputs are unchanged)."""
    from linc.blockface import reconstruct as recon, zarrio
    f, keep, i0, i1, j0, j1, recon_level = _info(cfg)
    US = 2 ** (4 - recon_level); s = cfg.subject
    color, A1 = zarrio.read_level(cfg.color_store, 0)                                 # L1 pre-inpaint (W1,nz,H1,3)
    mim = nib.load(cfg.final("mask")); A4 = mim.affine; brain = np.asanyarray(mim.dataobj) > 0   # L4 tight
    nz = min(brain.shape[1], color.shape[1])
    hcur = np.asanyarray(nib.load(cfg.curated_holes).dataobj) > 0                     # curated L4 (uncropped)
    hole_native = np.repeat(hcur[i0:i1, :, j0:j1], f, axis=1)[:, :nz, :] & brain[:, :nz, :]   # L4 tight, native-z
    h1 = recon._upsample_inplane(hole_native, US)                                     # -> L1 tight
    filled, nh = recon._inpaint_holes(color[:, :nz], h1)
    recon._write_pyramid(cfg.final_zarr("blockface"), filled, A1, cfg.reduce, 4 - recon_level + 1)
    recon._export_view_nii(cfg)                                                       # refresh the viewable QC nifti
    recon._save_nii(cfg.final("holes"), hole_native.astype(np.uint8), A4)
    lines = [l for l in open(cfg.info_file).read().split("\n") if l and not l.startswith("hole_voxels")]
    open(cfg.info_file, "w").write("\n".join(lines) + f"\nhole_voxels={nh}\n")
    print(f"  {s} reinpaint: {nh} hole voxels (L4 fg {int(hole_native.sum())}) -> blockface pyramid + holes; info updated")
