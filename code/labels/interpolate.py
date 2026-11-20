"""Demons interpolation between registered histology sections on the blockface grid: continuous RGB histology and continuous labels,
so labels carried into the template are not sheets one section apart.

Paper: Methods, "Histology registration and label propagation". Parameters: settings/labels.yaml (interpolate).

Grid: the target grid of the push (level 1 of the reconstructed store, native slice spacing).
Sections: every registered piece's RGB (<registration>/<piece>/moving4_nonlinear_rgb.png) 2 x 2 tissue-mean pooled and placed at its
matched slice; the pieces of one slice fill their own footprints.
Pairs: consecutive section slices (a, b) are interpolated only if (i) (b - a) x slice spacing <= gap_mm_max, (ii) their tissue footprints
agree (Dice >= footprint_dice_min: a one-hemisphere slice is never blended into a whole section). Everything skipped is recorded with its
reason.
Demons: multi-resolution symmetric forces (the reconstruction's schedule scaled by `upsample`) on the luminance of each section normalized
to its tissue p99.5 (capped at scale_max_over_median x the brain's median: `section_scales`), masked by the dilated union of the two
footprints; fields both ways.
Slice z (t = (z - a) / (b - a)): RGB = (1 - t) A warped by t f_AB + t B warped by (1 - t) f_BA. Labels: signed distance maps (negative
inside) of each label on a and b, warped the same way, blended with the same weights, thresholded at 0; a label present on only one of
the two sections tapers towards the other (its SDF there = +far).
Outputs: histo_rgb_L2.nii.gz (sections + interpolated), provenance_L2.nii.gz (1 real section, 2 interpolated), interp.json; per label
<name>_bfnative_L2_demons.nii.gz and one <ID>_labels_demons_prov.nii.gz per output folder (1 real section, 2 interpolated).

Ported from the analysis project's labels/histo_interp.py and the demons helpers of its code/recon.py (unchanged).
"""
import json
import os
import re
from glob import glob
from pathlib import Path

import numpy as np

from labels.labels import config, match_table


def pool2(rgb):
    """2 x 2 mean over tissue pixels (outside-the-section pixels are 0 and do not darken the rim) of an (H, W, 3) image."""
    h, w = rgb.shape[0] // 2 * 2, rgb.shape[1] // 2 * 2
    b = rgb[:h, :w].reshape(h // 2, 2, w // 2, 2, 3)
    t = (b.max(-1) > 0).sum((1, 3))
    return np.where(t[..., None] > 0, b.sum((1, 3)) / np.maximum(t, 1)[..., None], 0).astype(np.float32)


def sdf(m, far):
    from scipy.ndimage import distance_transform_edt
    if not m.any():
        return np.full(m.shape, far, np.float32)
    return (distance_transform_edt(~m) - distance_transform_edt(m)).astype(np.float32)


# --- demons (the reconstruction's helpers) ---

def _lum(rgb, w):
    return rgb.astype(np.float32) @ np.asarray(w, np.float32)


def _smooth_resample(image, shrink, sigma):
    import SimpleITK as sitk
    sm = sitk.SmoothingRecursiveGaussian(image, sigma) if sigma > 0 else image
    osz, osp = image.GetSize(), image.GetSpacing()
    nsz = [int(s / float(shrink) + 0.5) for s in osz]
    nsp = [((osz[i] - 1) * osp[i]) / (nsz[i] - 1) if nsz[i] > 1 else osp[i] for i in range(image.GetDimension())]
    return sitk.Resample(sm, nsz, sitk.Transform(), sitk.sitkLinear, image.GetOrigin(), nsp,
                         image.GetDirection(), 0.0, image.GetPixelID())


def ms_field(fixed, moving, shrinks, sigmas, iters, dev):
    """Multi-resolution symmetric-forces demons; returns the displacement field (moving -> fixed)."""
    import SimpleITK as sitk
    reg = sitk.FastSymmetricForcesDemonsRegistrationFilter()
    reg.SetNumberOfIterations(iters); reg.SetStandardDeviations(dev)
    fpyr, mpyr = [fixed], [moving]
    for s, sg in reversed(list(zip(shrinks, sigmas))):
        fpyr.append(_smooth_resample(fpyr[0], s, sg)); mpyr.append(_smooth_resample(mpyr[0], s, sg))
    field = sitk.Image(fpyr[-1].GetSize(), sitk.sitkVectorFloat64, fpyr[-1].GetDimension())
    field.CopyInformation(fpyr[-1])
    for fx, mv in reversed(list(zip(fpyr, mpyr))):
        field = sitk.Resample(field, fx)
        field = reg.Execute(fx, mv, field)
    return field


def warp3(rgb, df, fr):
    """Warp a 3-channel image by `fr` x the displacement field (linear, 0 outside), channel by channel."""
    import SimpleITK as sitk
    a = sitk.GetArrayFromImage(df) * float(fr); s = sitk.GetImageFromArray(a, isVector=True); s.CopyInformation(df)
    tx = sitk.DisplacementFieldTransform(sitk.Cast(s, sitk.sitkVectorFloat64))
    o = np.empty_like(rgb, dtype=np.float32)
    for c in range(3):
        im = sitk.GetImageFromArray(rgb[..., c].astype(np.float32))
        o[..., c] = sitk.GetArrayFromImage(sitk.Resample(im, im, tx, sitk.sitkLinear, 0.0))
    return o


# --- the interpolation ---

def section_scales(sec, idxs, limit=None):
    """Each section's intensity scale for the demons input: the 99.5th percentile of its tissue's RGB values (at least 1.0; 1.0 for a
    section with no tissue). A section whose scale is more than `limit` times the median scale of the brain's sections takes that
    median instead (one-sided: a scale below the limit is kept as it is). A saturated fold or stain spot can push one section's
    percentile far above the others, its luminance then collapses after the division, and the demons fields of its two pairs fail
    (MR252's sections 24 and 39: 5.7 and 4.3 times the median). None: no limit (the rule before 17 Nov 2026)."""
    hi = {i: max(np.percentile(sec[i][sec[i].max(-1) > 0], 99.5), 1.0) if (sec[i].max(-1) > 0).any() else 1.0 for i in idxs}
    if limit:                                                   # all of the run's sections, outliers included, in one pass; strictly >
        med = np.median(np.array([hi[i] for i in idxs], dtype=np.float64))
        # the median takes the type of the section's own scale, so the division keeps the section's float32: under NumPy 2,
        # np.percentile of float32 data is float32 and a float64 divisor would make the result float64, where the analysis (NumPy 1,
        # whose percentile is float64 but whose float32 / float64-scalar division stays float32) divided in float32
        hi = {i: (type(hi[i])(med) if hi[i] > limit * med else hi[i]) for i in idxs}
    return hi


def interpolate(subject, *, registrations, match_table_path, first_photo_number, grid, label_files, interp_dir, out_for=None,
                cfg=None, log=print):
    """Interpolate the registered sections and `label_files` (sheets on `grid`) between sections.
    out_for(label_path) -> output folder of its dense version (default: beside the input). Returns the interp.json record."""
    import nibabel as nib
    import SimpleITK as sitk
    from PIL import Image
    from scipy.ndimage import binary_dilation
    Image.MAX_IMAGE_PIXELS = None
    C = (cfg or config())["interpolate"]
    GAP_MM, DICE_MIN, US, FAR, WL = float(C["gap_mm_max"]), float(C["footprint_dice_min"]), int(C["upsample"]), float(C["far"]), C["luminance"]
    D = C["demons"]
    im = nib.load(str(grid))
    X, Y, Z = im.shape[:3]
    dz_mm = float(np.linalg.norm(im.affine[:3, 1]))                          # slice spacing (axis 1)
    look = match_table(match_table_path)
    # 1. sections on the grid (x = col, z = row), RGB float + footprint
    sec = {}
    for d in sorted(glob(os.path.join(str(registrations), "*", "moving4_nonlinear_rgb.png"))):
        piece = os.path.basename(os.path.dirname(d))
        idx = look(int(re.search(r"_s(\d+)", piece).group(1)))[int(piece[-4:])] - first_photo_number
        if not 0 <= idx < Y:
            continue
        rgb = np.asarray(Image.open(d)).astype(np.float32)[::-1, ::-1]        # store orientation
        p = pool2(rgb)[:Z, :X].transpose(1, 0, 2)                               # (X', Z', 3), X' <= X, Z' <= Z
        full = np.zeros((X, Z, 3), np.float32)
        full[: p.shape[0], : p.shape[1]] = p
        cur = sec.setdefault(idx, np.zeros((X, Z, 3), np.float32))
        fp = full.max(-1) > 0
        cur[fp] = full[fp]                                                      # pieces of one slice fill their own footprints
    idxs = sorted(sec)
    # 2. labels (sheets) on the same grid
    lab_files = [str(p) for p in label_files]
    labs = {p: np.asarray(nib.load(p).dataobj) > 0 for p in lab_files}
    for p, L in labs.items():
        assert L.shape[:3] == (X, Y, Z), (p, L.shape)
    out_rgb = np.zeros((X, Y, Z, 3), np.uint8)
    prov = np.zeros((X, Y, Z), np.uint8)
    dense = {p: L.copy() for p, L in labs.items()}
    hi = section_scales(sec, idxs, C.get("scale_max_over_median"))
    for i in idxs:
        out_rgb[:, i] = np.clip(sec[i] / hi[i] * 255, 0, 255).astype(np.uint8)
        prov[:, i][sec[i].max(-1) > 0] = 1
    shrink, sigma, dil = tuple(x * US for x in D["shrink"]), tuple(x * US for x in D["sigma"]), D["dilate"] * US // 4
    pairs = []
    for a, b in zip(idxs[:-1], idxs[1:]):
        rec = dict(a=int(a), b=int(b), gap_mm=round((b - a) * dz_mm, 3))
        fa, fb = sec[a].max(-1) > 0, sec[b].max(-1) > 0
        dice = float(2 * (fa & fb).sum() / max(fa.sum() + fb.sum(), 1))
        rec["footprint_dice"] = round(dice, 3)
        if b - a < 2:
            rec["status"] = "adjacent"; pairs.append(rec); continue
        if rec["gap_mm"] > GAP_MM:
            rec["status"] = f"skipped: gap > {GAP_MM:g} mm"; pairs.append(rec); continue
        if dice < DICE_MIN:
            rec["status"] = "skipped: footprints differ (partial section)"; pairs.append(rec); continue
        A, B = sec[a] / hi[a] * 255, sec[b] / hi[b] * 255
        reg = binary_dilation(fa | fb, iterations=max(dil, 1))
        Ag = sitk.GetImageFromArray((_lum(np.clip(A, 0, 255), WL) * reg).astype(np.float32))
        Bg = sitk.GetImageFromArray((_lum(np.clip(B, 0, 255), WL) * reg).astype(np.float32))
        fBA = ms_field(Ag, Bg, shrink, sigma, D["iters"], D["dev"])
        fAB = ms_field(Bg, Ag, shrink, sigma, D["iters"], D["dev"])
        la = [p for p in labs if labs[p][:, a].any() or labs[p][:, b].any()]
        Sa = np.stack([sdf(labs[p][:, a], FAR) for p in la], -1) if la else None
        Sb = np.stack([sdf(labs[p][:, b], FAR) for p in la], -1) if la else None
        for z in range(a + 1, b):
            t = (z - a) / (b - a)
            wa, wb = warp3(np.clip(A, 0, 255), fAB, t), warp3(np.clip(B, 0, 255), fBA, 1 - t)
            out_rgb[:, z] = np.clip(np.rint((1 - t) * wa + t * wb), 0, 255).astype(np.uint8)
            prov[:, z][out_rgb[:, z].max(-1) > 0] = 2
            if la:
                for k0 in range(0, len(la), 3):                                # warp3 warps 3 channels at a time
                    ch = list(range(k0, min(k0 + 3, len(la))))
                    pad = lambda X_: np.concatenate([X_[..., ch], np.full(X_.shape[:2] + (3 - len(ch),), FAR, np.float32)], -1)
                    s = (1 - t) * warp3(pad(Sa), fAB, t) + t * warp3(pad(Sb), fBA, 1 - t)
                    for j, k in enumerate(ch):
                        dense[la[k]][:, z] |= s[..., j] < 0
        rec["status"] = "interpolated"; rec["slices"] = int(b - a - 1); rec["labels"] = [os.path.basename(p) for p in la]
        pairs.append(rec)
        log(f"{subject} {a}->{b} ({rec['gap_mm']} mm, footprint Dice {dice:.2f}): {b - a - 1} slices, {len(la)} labels")
    od = Path(interp_dir)
    od.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(out_rgb, im.affine), str(od / "histo_rgb_L2.nii.gz"))
    nib.save(nib.Nifti1Image(prov, im.affine), str(od / "provenance_L2.nii.gz"))
    out_for = out_for or (lambda p: os.path.dirname(p))
    lab_prov, outputs = {}, []
    for p, L in dense.items():
        q = os.path.join(str(out_for(p)), os.path.basename(p).replace("_bfnative_L2.nii.gz", "_bfnative_L2_demons.nii.gz"))
        os.makedirs(os.path.dirname(q), exist_ok=True)
        nib.save(nib.Nifti1Image(L.astype(np.uint8), im.affine), q)
        outputs.append(q)
        pv = np.where(labs[p], 1, np.where(L, 2, 0)).astype(np.uint8)
        dp = os.path.join(os.path.dirname(q), f"{subject}_labels_demons_prov.nii.gz")
        lab_prov[dp] = np.maximum(lab_prov.get(dp, 0), pv)
    for dp, v in lab_prov.items():
        nib.save(nib.Nifti1Image(v, im.affine), dp)
    n_int = sum(r.get("slices", 0) for r in pairs if r["status"] == "interpolated")
    skip_gap = f"skipped: gap > {GAP_MM:g} mm"
    summ = dict(subject=subject, sections=len(idxs), pairs=len(pairs), interpolated_pairs=sum(r["status"] == "interpolated" for r in pairs),
                interpolated_slices=n_int, skipped={k: sum(r["status"] == k for r in pairs) for k in
                                                     (skip_gap, "skipped: footprints differ (partial section)")},
                rules=dict(gap_mm_max=GAP_MM, footprint_dice_min=DICE_MIN, demons=dict(shrink=shrink, sigma=sigma, iters=D["iters"],
                                                                                        dev=D["dev"])))
    out = dict(summary=summ, pairs=pairs)
    (od / "interp.json").write_text(json.dumps(out, indent=1))
    log(str(summ))
    return out, outputs


# --- the interpolation of record since 3 Nov 2026: groups, per-hemisphere pairing, end extension ---

GROUPS = {"Bundles": ["Fiber_light_bundle", "Fiber_moderate_bundle", "Fiber_dense_bundle", "Bundle", "Fiber_bundles", "Unnamed_outline"],
          "Terminals": ["Terminal_light", "Terminal_moderate", "Terminal_dense"],
          "Injection_site": ["Injection_site"], "Fibers": ["Fibers"]}


def regions_on_grid(regions_path, img, scale=None):
    """left (1) / right (2) / 0 of the atlas regions volume on the label grid (nearest), the regions' world scaled by `scale`
    (x, y, z factors about the origin: the brain's calibrated / made-at spacing, histology.pieces.regions_scale; None = the
    regions are on the grid's frame). Unlabeled voxels take the nearest labeled value; punch holes (4) count as left."""
    import nibabel as nib
    from scipy import ndimage as ndi
    ri = nib.load(str(regions_path)); Rg = np.asanyarray(ri.dataobj).astype(np.uint8)
    A = (np.diag(list(scale) + [1.0]) @ ri.affine) if scale is not None else ri.affine
    LR = np.where((Rg == 1) | (Rg == 4), 1, np.where(Rg == 2, 2, 0)).astype(np.uint8)
    _, ind = ndi.distance_transform_edt(LR == 0, return_indices=True)
    LRn = LR[ind[0], ind[1], ind[2]]
    sh = img.shape[:3]
    out = np.zeros(sh, np.uint8)
    M = np.linalg.inv(A) @ img.affine
    ii, kk = np.meshgrid(np.arange(sh[0]), np.arange(sh[2]), indexing="ij")
    for j in range(sh[1]):
        v = np.stack([ii, np.full(ii.shape, j), kk, np.ones(ii.shape)], -1) @ M.T
        q = np.clip(np.rint(v[..., :3]).astype(int), 0, np.array(LR.shape) - 1)
        out[:, j, :] = LRn[q[..., 0], q[..., 1], q[..., 2]]
    return out


def field_stats(f, mask):
    """Median and p90 of the displacement magnitude (px) of a SimpleITK displacement field over a mask."""
    import SimpleITK as sitk
    a = sitk.GetArrayFromImage(f)                                             # (H, W, 2)
    mag = np.sqrt((a ** 2).sum(-1))[mask]
    return (round(float(np.median(mag)), 3), round(float(np.percentile(mag, 90)), 3)) if mag.size else (None, None)


def interpolate_v2(subject, *, registrations, match_table_path, first_photo_number, grid, label_files, regions, out_dir,
                   regions_scale=None, cfg=None, log=print):
    """The interpolation of record since 3 Nov 2026 (the analysis project's labels/histo_interp_v2.py, unchanged in rule): the same
    grid, sections, demons engine and pair rules as `interpolate`, with three changes:
      1. GROUPS, not classes: the bundle classes are interpolated as ONE label, the terminal classes as one, Injection_site and Fibers
         on their own; the class grade is kept as a partition of each group (every interior voxel takes the class whose blended signed
         distance is smallest), so the per-class files tile the group with no gaps;
      2. PER-HEMISPHERE PAIRING: a pair of section slices is interpolated per hemisphere (the atlas regions `regions`, left / right):
         the consecutive slices that hold that hemisphere's tissue (>= hemisphere_min_share of the slice's footprint), with the gap
         and footprint-Dice rules inside the hemisphere; the demons field of a pair is computed on the luminance inside the valid
         hemispheres' footprints, and labels of a hemisphere that does not pair are not blended;
      3. END EXTENSION: a group present on one end only shrinks to a point at the midpoint of the gap (its signed distance on the
         absent side = its equivalent radius) instead of vanishing one slice away.
    `label_files`: the sheets per class (<ID>_<class>_bfnative_L2.nii.gz; classes outside GROUPS are ignored). Writes to `out_dir`:
    <ID>_<group>_bfnative_L2.nii.gz (the union sheets), <ID>_<group>_bfnative_L2_demons.nii.gz, <ID>_<class>_bfnative_L2_demons.nii.gz
    (the partition), <ID>_labels_demons_prov.nii.gz (1 real section, 2 interpolated), interp_v2.json (the summary, the pairs with their
    demons displacement statistics, the per-hemisphere pair rows). No RGB volume is written (the RGB interpolation of record is
    `interpolate`'s). Returns (the record, the output paths). Validated 3 Nov 2026 on MR252 against the products of record: the 11
    volumes voxel-identical, interp_v2.json's pairs and summary identical (34 min under load)."""
    import time
    import nibabel as nib
    import SimpleITK as sitk
    from PIL import Image
    from scipy.ndimage import binary_dilation
    Image.MAX_IMAGE_PIXELS = None
    t0 = time.time()
    C = (cfg or config())["interpolate"]
    GAP_MM, DICE_MIN, US, FAR, WL = float(C["gap_mm_max"]), float(C["footprint_dice_min"]), int(C["upsample"]), float(C["far"]), C["luminance"]
    HEMI_MIN = float(C.get("hemisphere_min_share", 0.05))
    D = C["demons"]
    im = nib.load(str(grid))
    X, Y, Z = im.shape[:3]
    dz_mm = float(np.linalg.norm(im.affine[:3, 1]))                          # slice spacing (axis 1)
    look = match_table(match_table_path)
    od = Path(out_dir); od.mkdir(parents=True, exist_ok=True)
    # 1. sections on the grid (x = col, z = row), RGB float + footprint (as the method of record)
    sec = {}
    for d in sorted(glob(os.path.join(str(registrations), "*", "moving4_nonlinear_rgb.png"))):
        piece = os.path.basename(os.path.dirname(d))
        idx = look(int(re.search(r"_s(\d+)", piece).group(1)))[int(piece[-4:])] - first_photo_number
        if not 0 <= idx < Y:
            continue
        rgb = np.asarray(Image.open(d)).astype(np.float32)[::-1, ::-1]
        p = pool2(rgb)[:Z, :X].transpose(1, 0, 2)
        full = np.zeros((X, Z, 3), np.float32)
        full[: p.shape[0], : p.shape[1]] = p
        cur = sec.setdefault(idx, np.zeros((X, Z, 3), np.float32))
        fp = full.max(-1) > 0
        cur[fp] = full[fp]
    idxs = sorted(sec)
    # 2. the sheets per class -> group sheets (unions) + the classes
    cls_sheet = {}
    by_class = {}
    for p in label_files:
        name = os.path.basename(str(p))
        if name.startswith(subject + "_") and name.endswith("_bfnative_L2.nii.gz"):
            by_class[name[len(subject) + 1:-len("_bfnative_L2.nii.gz")]] = str(p)
    for g, cls in GROUPS.items():
        for cl in cls:
            if cl in by_class:
                cls_sheet[cl] = np.asarray(nib.load(by_class[cl]).dataobj) > 0
                assert cls_sheet[cl].shape[:3] == (X, Y, Z), (by_class[cl], cls_sheet[cl].shape)
    if not idxs or not cls_sheet:
        raise SystemExit(f"{subject}: nothing to interpolate, nothing written: {len(idxs)} registered sections in {registrations}, "
                         f"{len(cls_sheet)} tracer classes among the {len(label_files)} sheets given.")
    grp_sheet = {g: np.any([cls_sheet[c] for c in cls if c in cls_sheet], 0) for g, cls in GROUPS.items() if any(c in cls_sheet for c in cls)}
    outputs = []
    for g, L in grp_sheet.items():
        q = od / f"{subject}_{g}_bfnative_L2.nii.gz"; nib.save(nib.Nifti1Image(L.astype(np.uint8), im.affine), str(q)); outputs.append(str(q))
    # 3. hemispheres on the grid; per-hemisphere section lists
    LR = regions_on_grid(regions, im, regions_scale)
    hi = section_scales(sec, idxs, C.get("scale_max_over_median"))
    foot = {i: sec[i].max(-1) > 0 for i in idxs}
    hemi_slices = {}
    for h, hv in (("left", 1), ("right", 2)):
        hemi_slices[h] = [i for i in idxs if (foot[i] & (LR[:, i] == hv)).sum() > HEMI_MIN * max(foot[i].sum(), 1)]
    # 4. pairs: a pair (a, b) is valid in hemisphere h when a and b are consecutive in h's list and pass the rules inside h
    valid = {}
    hemi_rows = {"left": [], "right": []}
    skip_gap = f"skipped: gap > {GAP_MM:g} mm"
    for h, hv in (("left", 1), ("right", 2)):
        sl = hemi_slices[h]
        for a, b in zip(sl[:-1], sl[1:]):
            fa, fb = foot[a] & (LR[:, a] == hv), foot[b] & (LR[:, b] == hv)
            s = fa.sum() + fb.sum()
            r = dict(a=int(a), b=int(b), gap_mm=round((b - a) * dz_mm, 3), dice=round(float(2 * (fa & fb).sum() / s) if s else 0.0, 3))
            if b - a < 2:
                r["status"] = "adjacent"
            elif r["gap_mm"] > GAP_MM:
                r["status"] = skip_gap
            elif r["dice"] < DICE_MIN:
                r["status"] = "skipped: footprints differ (partial section)"
            else:
                r["status"] = "interpolated"; valid.setdefault((a, b), []).append(hv)
            hemi_rows[h].append(r)
    shrink, sigma, dil = tuple(x * US for x in D["shrink"]), tuple(x * US for x in D["sigma"]), D["dilate"] * US // 4
    dense_g = {g: L.copy() for g, L in grp_sheet.items()}
    dense_c = {c: L.copy() for c, L in cls_sheet.items()}
    pairs = []
    for (a, b), hvs in sorted(valid.items()):
        hm_a = np.isin(LR[:, a], hvs); hm_b = np.isin(LR[:, b], hvs)
        fa, fb = foot[a] & hm_a, foot[b] & hm_b
        A, B = sec[a] / hi[a] * 255, sec[b] / hi[b] * 255
        reg = binary_dilation(fa | fb, iterations=max(dil, 1))
        Ag = sitk.GetImageFromArray((_lum(np.clip(A, 0, 255), WL) * reg).astype(np.float32))
        Bg = sitk.GetImageFromArray((_lum(np.clip(B, 0, 255), WL) * reg).astype(np.float32))
        fBA = ms_field(Ag, Bg, shrink, sigma, D["iters"], D["dev"])
        fAB = ms_field(Bg, Ag, shrink, sigma, D["iters"], D["dev"])
        s = fa.sum() + fb.sum()
        rec = dict(a=int(a), b=int(b), gap_mm=round((b - a) * dz_mm, 3), hemispheres=[{1: "left", 2: "right"}[v] for v in hvs],
                   footprint_dice=round(float(2 * (fa & fb).sum() / s) if s else 0.0, 3), slices=int(b - a - 1))
        rec["field_AB_px_median_p90"] = field_stats(fAB, fa | fb); rec["field_BA_px_median_p90"] = field_stats(fBA, fa | fb)
        # labels of the valid hemispheres only
        ga = {g: (grp_sheet[g][:, a] & hm_a) for g in grp_sheet}; gb = {g: (grp_sheet[g][:, b] & hm_b) for g in grp_sheet}
        gl = [g for g in grp_sheet if ga[g].any() or gb[g].any()]
        rec["groups"] = gl
        if not gl:
            pairs.append(rec); continue
        # group signed distances; a group on one end only shrinks to a point at the midpoint (absent side = + equivalent radius)
        Sa, Sb = {}, {}
        for g in gl:
            if ga[g].any() and gb[g].any():
                Sa[g], Sb[g] = sdf(ga[g], FAR), sdf(gb[g], FAR)
            elif ga[g].any():
                Sa[g] = sdf(ga[g], FAR); Sb[g] = np.full(ga[g].shape, float(np.sqrt(ga[g].sum() / np.pi)), np.float32)
            else:
                Sb[g] = sdf(gb[g], FAR); Sa[g] = np.full(gb[g].shape, float(np.sqrt(gb[g].sum() / np.pi)), np.float32)
        # class signed distances (absent side = +far; used only to partition the group's interior)
        Ca, Cb = {}, {}
        for g in gl:
            for c in GROUPS[g]:
                if c in cls_sheet:
                    ca, cb = cls_sheet[c][:, a] & hm_a, cls_sheet[c][:, b] & hm_b
                    if ca.any() or cb.any():
                        Ca[c], Cb[c] = sdf(ca, FAR), sdf(cb, FAR)
        keys_g = list(Sa); keys_c = list(Ca)
        SA = np.stack([Sa[k] for k in keys_g] + [Ca[k] for k in keys_c], -1)
        SB = np.stack([Sb[k] for k in keys_g] + [Cb[k] for k in keys_c], -1)
        for z in range(a + 1, b):
            t = (z - a) / (b - a)
            blend = np.zeros(SA.shape, np.float32)
            for k0 in range(0, SA.shape[-1], 3):                                 # warp3 warps 3 channels at a time
                ch = list(range(k0, min(k0 + 3, SA.shape[-1])))
                pad = lambda X_: np.concatenate([X_[..., ch], np.full(X_.shape[:2] + (3 - len(ch),), FAR, np.float32)], -1)  # noqa: E731
                sblend = (1 - t) * warp3(pad(SA), fAB, t) + t * warp3(pad(SB), fBA, 1 - t)
                blend[..., ch] = sblend[..., : len(ch)]
            hz = np.isin(LR[:, z], hvs)
            for j, g in enumerate(keys_g):
                inside = (blend[..., j] < 0) & hz
                dense_g[g][:, z] |= inside
                cls_of = [k for k in keys_c if k in GROUPS[g]]
                if cls_of and inside.any():
                    cb_ = np.stack([blend[..., len(keys_g) + keys_c.index(k)] for k in cls_of], -1)
                    arg = cb_.argmin(-1)
                    for ci, k in enumerate(cls_of):
                        dense_c[k][:, z] |= inside & (arg == ci)
        pairs.append(rec)
        log(f"{subject} {a}->{b} ({rec['gap_mm']} mm, {'+'.join(rec['hemispheres'])}, Dice {rec['footprint_dice']:.2f}): {b - a - 1} slices, "
            f"groups {gl}, field median {rec['field_AB_px_median_p90'][0]} px")
    for g, L in dense_g.items():
        q = od / f"{subject}_{g}_bfnative_L2_demons.nii.gz"; nib.save(nib.Nifti1Image(L.astype(np.uint8), im.affine), str(q)); outputs.append(str(q))
    for c, L in dense_c.items():
        q = od / f"{subject}_{c}_bfnative_L2_demons.nii.gz"; nib.save(nib.Nifti1Image(L.astype(np.uint8), im.affine), str(q)); outputs.append(str(q))
    lab_prov = np.zeros((X, Y, Z), np.uint8)
    for g, L in dense_g.items():
        lab_prov = np.maximum(lab_prov, np.where(grp_sheet[g], 1, np.where(L, 2, 0)).astype(np.uint8))
    q = od / f"{subject}_labels_demons_prov.nii.gz"; nib.save(nib.Nifti1Image(lab_prov, im.affine), str(q)); outputs.append(str(q))
    summ = dict(subject=subject, sections=len(idxs), hemisphere_sections={h: len(v) for h, v in hemi_slices.items()},
                pairs_interpolated=len(pairs), pairs_with_groups=sum(bool(p["groups"]) for p in pairs),
                slices_interpolated=int(sum(p["slices"] for p in pairs)),
                per_hemisphere={h: {k: sum(r["status"] == k for r in rows) for k in ("interpolated", "adjacent", skip_gap,
                                                                                    "skipped: footprints differ (partial section)")} for h, rows in hemi_rows.items()},
                groups=list(grp_sheet), classes=list(cls_sheet),
                rules=dict(gap_mm_max=GAP_MM, footprint_dice_min=DICE_MIN, hemisphere_min_share=HEMI_MIN,
                           end_extension="absent side = + equivalent radius (vanishes at mid-gap)",
                           partition="class of the smallest blended signed distance", demons=dict(shrink=shrink, sigma=sigma, iters=D["iters"], dev=D["dev"])),
                sources=dict(sheets=sorted(by_class.values()), rgb=str(registrations), regions=str(regions)), wall_s=round(time.time() - t0, 1))
    out = dict(summary=summ, pairs=pairs, hemisphere_pairs=hemi_rows)
    (od / "interp_v2.json").write_text(json.dumps(out, indent=1))
    log(str(summ))
    return out, outputs
