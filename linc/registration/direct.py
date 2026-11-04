"""A brain's blockface registered directly to a template, without its own MRI: to a blockface template (the blockface-based
alignment, `to_blockface_template`) or to the MRI template's b0 image (the cross-modal alignment, `direct_placement`: how the 8
brains without MRI were registered; Methods, "Blockface to MRI registration and synthesis of missing sections").

You need
  the brain's reconstruction and mask (<ID>_blockface.ome.zarr, <ID>_mask), the template image and its mask, FSL and the anatomix
  environment (featurereg.py); parameters: configs/registration.yaml `direct`. For the stand-in stage: the MRI template's 0.5 mm
  frame and the bridge from it to the template's stereotaxic frame; for a brain with its own b0, also its registration to that b0.
What it does
  1. a rigid registration (6 degrees of freedom) of the binary masks at 0.5 mm, searching all rotations (+-180 deg): masks have no
     contrast to mislead it, and a rigid transform cannot stretch a brain whose caudal part is missing;
  2. 7 degrees of freedom (rigid + isotropic scale, correlation ratio) on the masked grayscale, against a band of the template as
     long as the brain, anchored whole, anterior or caudal (the measured length says which end is missing), scored on the whole
     template (NCC x containment; the NCC's magnitude for the b0 target, whose contrast is inverted);
  3. cross-modal alignment only: 9 degrees of freedom from the best 7 (a scale along each axis), kept only if every scale lies in
     [0.6, 1.25];
  4. the template cut to the part the brain covers (1 mm inside both A-P faces), and the brainstem cut flat on both inputs; the
     feature-based deformable registration (featurereg.py) on the template's 0.25 mm grid, padded with zeros;
  5. the products by one resample from the native pyramid through the linear transform and the warp (resample.py).
  6. cross-modal alignment only, the stand-in stage (CPU, no registration; it runs only after a finished alignment), what
     `linc-complete prepare` reads: `stand_in` for a brain without MRI, `stand_in_own_b0` for a brain with its own b0.
You get
  the grayscale, RGB and mask on the template grid; for the cross-modal alignment, also its registry entry bf-native -> <template
  space> (linc/xfm.py) and, in the stand-in folder, <ID>_mri_full (the stand-in b0) and <ID>_pseudo_tplmask_0p5 and, for a brain
  without MRI, <ID>_mri, <ID>_bf_nonlin, <ID>_bf_nonlin_rgb, <ID>_bfmask, <ID>_bf2mri_dof9.mat, <ID>_pseudo_brain_n1_0p5.

Ported from the analysis project: bftemplate/direct/register_direct.py (the blockface-based alignment) and
featurereg/d2_s2_register.py (the cross-modal alignment), the final settings only, unchanged numerically.
"""
import hashlib
import json
import os
import shutil
import time

import numpy as np
import nibabel as nib
import nibabel.processing as nip

from linc import xfm
from linc.blockface import zarrio
from linc.registration import featurereg, fsl, xfmtools as X
from linc.registration.resample import W_LUM, antialias, bf_products

FLAT_CUT_MAX_PCT = 25.0                    # fail closed: a flat cut that removes more of either input is a frame error


def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ grids and cuts
def refine_grid(ref_img, factor):
    """Exact refinement of a grid: voxel/factor, same box (edges), so a centred x-flip stays exact."""
    A = ref_img.affine.copy()
    v = np.diag(A)[:3]
    A2 = A.copy(); A2[:3, :3] = A[:3, :3] / factor
    A2[:3, 3] = A[:3, 3] - v / 2 + v / (2 * factor)
    shape = tuple(int(s * factor) for s in ref_img.shape[:3])
    return shape, A2


def flat_cut_bridge(shape, affine, z_mm, bridge_json):
    """The flat brainstem cut on a grid in the run-113 frame: keep a voxel iff (Rb p)_z >= z_mm, p its centre, Rb the bridge rigid
    run-113 -> canonical frame (the canonical template brain mask's lowest axial slice is z = -2.0 mm)."""
    Rb = np.asarray(json.load(open(bridge_json))["rigid_world_run113_to_canonical"], float)
    if Rb.shape != (4, 4) or not np.allclose(Rb[:3, :3] @ Rb[:3, :3].T, np.eye(3), atol=1e-6) or np.linalg.det(Rb[:3, :3]) <= 0:
        raise RuntimeError(f"{bridge_json}: rigid_world_run113_to_canonical is not a proper rigid")
    c = (Rb @ affine)[2]
    i, j, k = (np.arange(n, dtype=np.float64) for n in shape[:3])
    return (c[0] * i[:, None, None] + c[1] * j[None, :, None] + c[2] * k[None, None, :] + c[3]) >= z_mm - 1e-6


def flat_cut_canonical(shape, affine, z_mm):
    """The flat brainstem cut on a grid in the canonical frame (the MRI template's own): z_world >= z_mm."""
    c = affine[2]
    i, j, k = (np.arange(n, dtype=np.float64) for n in shape[:3])
    return (c[0] * i[:, None, None] + c[1] * j[None, :, None] + c[2] * k[None, None, :] + c[3]) >= z_mm - 1e-6


def coarsen_2x(img_path, mask_path, out_img, out_mask, mask_image=True):
    """A 0.25 mm template image (x its mask) and its mask coarsened EXACTLY 2x (block mean; same box) to 0.5 mm."""
    b, m = nib.load(str(img_path)), nib.load(str(mask_path))
    assert b.shape[:3] == m.shape[:3] and np.allclose(b.affine, m.affine, atol=1e-4), "template image and mask grids differ"
    v = np.asanyarray(b.dataobj).astype(np.float32) * ((np.asanyarray(m.dataobj) > 0.5) if mask_image else 1)
    mm = (np.asanyarray(m.dataobj) > 0.5).astype(np.float32)
    sh = [s // 2 for s in v.shape[:3]]
    blk = lambda a: a[:2 * sh[0], :2 * sh[1], :2 * sh[2]].reshape(sh[0], 2, sh[1], 2, sh[2], 2).mean(axis=(1, 3, 5))  # noqa: E731
    A = b.affine.copy(); A[:3, :3] = b.affine[:3, :3] * 2; A[:3, 3] = b.affine[:3, 3] + b.affine[:3, :3] @ np.full(3, 0.5)
    nib.save(nib.Nifti1Image(blk(v), A), out_img); nib.save(nib.Nifti1Image((blk(mm) > 0.5).astype(np.uint8), A), out_mask)
    return out_img, out_mask


# ------------------------------------------------------------------------------------------------ the linear stages
def _ap_mm(mask, affine):
    """A-P extent of a binary mask in world mm."""
    i = np.argwhere(mask)
    if not i.size:
        return 0.0
    y = (affine[1, :3] @ i.T) + affine[1, 3]
    return float(y.max() - y.min())


def _placed_len(d, mask05, mat, template):
    """How long the subject is along A-P once the rigid has put it in template space (mm)."""
    p = f"{d}/_placed_rigid6.nii.gz"
    fsl.run("flirt", "-in", mask05, "-ref", template, "-applyxfm", "-init", mat, "-interp", "nearestneighbour", "-out", p)
    im = nib.load(p)
    return _ap_mm(np.asanyarray(im.dataobj) > 0, im.affine)


def _anchor_init(d, anchor, r6, mvimg, tpl, band, placed):
    """Init for a band fit: slide the rigid placement bodily until the subject's end meets the band's pole (FLIRT optimizes locally;
    the travel is handed to it, not searched for)."""
    pm, pA = placed
    idx = np.argwhere(pm); pw = (pA[:3, :3] @ idx.T).T + pA[:3, 3]
    bi = np.argwhere(band); bw = (tpl.affine[:3, :3] @ bi.T).T + tpl.affine[:3, 3]
    dy = (bw[:, 1].max() - pw[:, 1].max()) if anchor == "anterior" else (bw[:, 1].min() - pw[:, 1].min())
    t = bw.mean(0) - pw.mean(0); t[1] = dy                      # centre in L-R and D-V, anchor in A-P
    T = np.eye(4); T[:3, 3] = t
    W = T @ fsl.fsl_to_world(np.loadtxt(r6), mvimg, tpl)
    out = f"{d}/_init_{anchor}.mat"
    np.savetxt(out, fsl.world_to_fsl(W, mvimg, tpl), fmt="%.10f")
    return out, t


def _band_ref(d, tpl, cov, ax, ant_hi, anchor, length_mm, guard=3.0):
    """The template, and its coverage, restricted to an A-P band `length_mm` long anchored at one pole ('whole': untouched)."""
    grey = np.asanyarray(tpl.dataobj).astype(np.float32)
    band = np.ones(cov.shape[ax], bool)
    if anchor != "whole":
        idx = np.flatnonzero(cov.any(axis=tuple(k for k in range(3) if k != ax)))
        n = int(round((length_mm + guard) / abs(tpl.affine[1, ax])))
        hi_end = (anchor == "anterior") == bool(ant_hi)
        band[:] = False
        if hi_end:
            band[max(int(idx.max()) - n + 1, 0):int(idx.max()) + 1] = True
        else:
            band[int(idx.min()):min(int(idx.min()) + n, cov.shape[ax])] = True
    b = cov & band.reshape([-1 if i == ax else 1 for i in range(3)])
    ref, refm = f"{d}/_band_{anchor}.nii.gz", f"{d}/_band_{anchor}_mask.nii.gz"
    nib.save(nib.Nifti1Image(grey * b, tpl.affine), ref)
    nib.save(nib.Nifti1Image(b.astype(np.uint8), tpl.affine), refm)
    return ref, refm


def _score(template, placed, cov, placed_mask):
    """NCC where the placed brain meets the template, weighted by how much of it landed inside it."""
    T = np.asanyarray(nib.load(template).dataobj).astype(np.float32)
    P = np.asanyarray(nib.load(placed).dataobj).astype(np.float32)
    pm = np.asanyarray(nib.load(placed_mask).dataobj) > 0
    m = pm & cov
    cont = float(m.sum() / max(pm.sum(), 1))
    if m.sum() < 1000:
        return dict(ncc=-1.0, containment=cont, score=-1.0)
    x, y = T[m], P[m]
    ncc = float(((x - x.mean()) * (y - y.mean())).mean() / (x.std() * y.std() + 1e-9))
    return dict(ncc=ncc, containment=cont, score=ncc * cont)


def linear(d, S, mv, mm, template, cov, tmask, cross_modal, dof9_scale_range=None):
    """Steps 1-3 on the 0.5 mm working copies (mv, mm) against the 0.5 mm template (and its coverage `cov` / mask file `tmask`).
    Returns (M = blockface world -> template world, the FSL matrix of the placement used, info)."""
    tpl = nib.load(template)
    info = {}
    r6 = f"{d}/{S}_rigid6_masks.mat"
    fsl.run("flirt", "-in", mm, "-ref", tmask, "-dof", "6", "-cost", "normcorr", "-searchrx", -180, 180, "-searchry", -180, 180,
            "-searchrz", -180, 180, "-coarsesearch", 30, "-finesearch", 10, "-omat", r6)
    ax = int(np.argmax(np.abs(tpl.affine[:3, :3][1]))); ant_hi = tpl.affine[1, ax] > 0
    L = _placed_len(d, mm, r6, template)
    pimg = nib.load(f"{d}/_placed_rigid6.nii.gz"); placed = (np.asanyarray(pimg.dataobj) > 0, pimg.affine)
    info["ap_length_mm"], info["template_ap_mm"] = L, _ap_mm(cov, tpl.affine)
    best = None
    for anchor in ("whole", "anterior", "caudal"):
        ref, refm = _band_ref(d, tpl, cov, ax, ant_hi, anchor, L)
        cand, r6c = f"{d}/_dof7_{anchor}.mat", r6
        if anchor != "whole":                                  # re-seat the block inside the band before dof-7
            band = np.asanyarray(nib.load(refm).dataobj) > 0
            init, t = _anchor_init(d, anchor, r6, nib.load(mv), tpl, band, placed)
            r6c = f"{d}/_rigid6_{anchor}.mat"
            fsl.run("flirt", "-in", mm, "-ref", refm, "-dof", "6", "-cost", "normcorr", "-init", init,
                    "-searchrx", -20, 20, "-searchry", -20, 20, "-searchrz", -20, 20, "-omat", r6c)
            info.setdefault("anchor_shift_mm", {})[anchor] = [round(float(v), 1) for v in t]
        fsl.run("flirt", "-in", mv, "-ref", ref, "-dof", "7", "-cost", "corratio", "-init", r6c, "-nosearch",
                "-omat", cand, "-out", f"{d}/_bf_dof7_{anchor}.nii.gz")
        fsl.run("flirt", "-in", mm, "-ref", template, "-applyxfm", "-init", cand, "-interp", "nearestneighbour",
                "-out", f"{d}/_mask_dof7_{anchor}.nii.gz")
        sc = _score(template, f"{d}/_bf_dof7_{anchor}.nii.gz", cov, f"{d}/_mask_dof7_{anchor}.nii.gz")
        if cross_modal:
            sc = dict(sc, score=abs(sc["ncc"]) * sc["containment"])        # inter-modal: the NCC's magnitude counts
        info.setdefault("anchor_scores", {})[anchor] = sc
        print(f"  {S} anchor {anchor:8s} NCC {sc['ncc']:+.3f} x containment {sc['containment']:.3f} -> {sc['score']:+.3f}", flush=True)
        if best is None or sc["score"] > best[1]["score"]:
            best = (anchor, sc, cand, ref)
    info["anchor"] = best[0]
    M7 = fsl.fsl_to_world(np.loadtxt(best[2]), nib.load(mv), tpl)
    used, M = best[2], M7
    info["linear_dof7"] = fsl.decompose(M7, allow_180=True)
    if dof9_scale_range:                                       # dof-9 from the best dof-7, same band; kept if every scale is in range
        m9 = f"{d}/{S}_dof9.mat"
        fsl.run("flirt", "-in", mv, "-ref", best[3], "-dof", "9", "-cost", "corratio", "-init", best[2], "-nosearch", "-omat", m9)
        M9 = fsl.fsl_to_world(np.loadtxt(m9), nib.load(mv), tpl)
        d9 = fsl.decompose(M9, allow_180=True)
        ok9 = all(dof9_scale_range[0] <= s <= dof9_scale_range[1] for s in d9["scales"])
        info.update(linear_dof9=d9, linear_used="dof9" if ok9 else "dof7 (dof-9 scales out of range: flagged)")
        if ok9:
            used, M = m9, M9
    else:
        info["linear_used"] = "dof7"
    return M, used, info


def _working_copies(d, S, iso_grey, final_mask):
    """The 0.5 mm masked grayscale and mask of the blockface (the mask resampled onto the image grid first)."""
    mv, mm = f"{d}/{S}_bf_0.5.nii.gz", f"{d}/{S}_mask_0.5.nii.gz"
    g = nib.load(str(iso_grey))
    mo = nip.resample_from_to(nib.load(str(final_mask)), g, order=0)
    return g, mo, mv, mm


def _padded_crop(fmk, mmsk, shp, pad_rule):
    ijk = np.argwhere(fmk | mmsk)
    m_mm, mult = float(pad_rule.split(":")[0]), int(pad_rule.split(":")[1])
    mg = int(np.ceil(m_mm / 0.25 - 1e-9))
    lo = ijk.min(0) - mg; hi = ijk.max(0) + mg + 1
    n = hi - lo; nt = (np.ceil(n / mult) * mult).astype(int); lo = lo - (nt - n) // 2; hi = lo + nt
    pl = np.maximum(-lo, 0); ph = np.maximum(hi - np.array(shp), 0)

    def crop(v):                                              # zero-padded crop (identical to v[lo:hi] when inside the box)
        if pl.any() or ph.any():
            v = np.pad(v, [(int(pl[k]), int(ph[k])) for k in range(3)])
        return np.ascontiguousarray(v[tuple(slice(int(lo[k] + pl[k]), int(hi[k] + pl[k])) for k in range(3))])
    return crop, lo, hi


def _cut_keep(pm, ax, cut_margin_mm):
    """The template's A-P extent kept for the deformable: the placed brain's extent, `cut_margin_mm` past its faces (negative: inside)."""
    idx = np.flatnonzero(pm.any(axis=tuple(k for k in range(3) if k != ax)))
    keep = np.zeros(pm.shape[ax], bool); guard = -int(round(cut_margin_mm / 0.5))
    if idx.size:
        keep[max(min(int(idx.min()) + guard, pm.shape[ax] - 1), 0):min(max(int(idx.max()) - guard, 0), pm.shape[ax] - 1) + 1] = True
    return keep


def coarsen_template(grey_025, coverage_025, frame_05, out_grey, out_cov):
    """A blockface template's 0.25 mm grayscale and coverage coarsened EXACTLY 2x onto its 0.5 mm frame (the grid it is the 2x
    refinement of): the grayscale anti-aliased (Gaussian, sigma = 0.5 sqrt(3) voxels) then 2x2x2 block means; the coverage block means.
    (The route test's targets, bftemplate/cv/route_test_1c.py `coarse_template`.)"""
    ref = nib.load(str(frame_05))
    for src, sig, out in ((grey_025, 0.5 * np.sqrt(3), out_grey), (coverage_025, 0.0, out_cov)):
        im = nib.load(str(src)); v = np.asanyarray(im.dataobj).astype(np.float32)
        assert v.shape == tuple(2 * s for s in ref.shape[:3]), f"{src}: not the 2x refinement of the frame"
        c0 = im.affine[:3, :3] @ np.full(3, 0.5) + im.affine[:3, 3]           # centre of the first 2x2x2 block
        assert np.allclose(c0, ref.affine[:3, 3], atol=1e-4) and np.allclose(np.diag(im.affine)[:3] * 2, np.diag(ref.affine)[:3]), f"{src}: grid"
        if sig:
            from scipy import ndimage as ndi
            v = ndi.gaussian_filter(v, sig)
        c = v.reshape(ref.shape[0], 2, ref.shape[1], 2, ref.shape[2], 2).mean((1, 3, 5))
        nib.save(nib.Nifti1Image(c.astype(np.float32), ref.affine), str(out))
    return str(out_grey), str(out_cov)


# ------------------------------------------------------------------------------------------------ route A: onto a blockface template
def to_blockface_template(S, bf, template, template_coverage, outdir, cfg, bridge_json, device="cuda:0", xfm_root=None, overwrite=False):
    """Route A: the blockface onto a blockface template given at 0.5 mm (`template`, `template_coverage`; the route test coarsened its
    0.25 mm targets exactly 2x). bf = dict(iso_grey, final_mask, final_zarr). cfg = configs/registration.yaml `direct.to_blockface_template`.
    Products in outdir: <S>_bf_in_template, <S>_bf_rgb_in_template, <S>_mask_in_template (0.25 mm), <S>_mask_in_template_cut (with the
    template's flat brainstem cut), <S>_warp, <S>_dof7_world.txt; the transform entry bf-native -> bf-template under xfm_root (default the registry, $LINC_DATA/sub-<ID>/xfm; overwrite=True replaces an
    existing entry, which linc/xfm.py backs up)."""
    d = str(outdir); os.makedirs(d, exist_ok=True)
    tpl = nib.load(str(template)); cov = np.asanyarray(nib.load(str(template_coverage)).dataobj) > 0.5
    tmask = f"{d}/_template_mask.nii.gz"; nib.save(nib.Nifti1Image(cov.astype(np.uint8), tpl.affine), tmask)
    src, msk = str(bf["iso_grey"]), str(bf["final_mask"])
    g, mo, mv, mm = _working_copies(d, S, src, msk)
    gi = nib.Nifti1Image(antialias(np.asanyarray(g.dataobj).astype(np.float32), np.abs(np.diag(g.affine)[:3]), [0.5] * 3), g.affine)
    nib.save(nip.resample_to_output(gi, [0.5] * 3, order=1), mv)
    nib.save(nip.resample_from_to(mo, nib.load(mv), order=0), mm)
    info = dict(subject=S, template=str(template), config=cfg)
    M, used, lin = linear(d, S, mv, mm, str(template), cov, tmask, cross_modal=False)
    info.update(lin)
    np.savetxt(f"{d}/{S}_dof7_world.txt", M)
    fsl.run("flirt", "-in", mm, "-ref", str(template), "-applyxfm", "-init", used, "-interp", "nearestneighbour", "-out", f"{d}/{S}_mask_dof7.nii.gz")
    ax = int(np.argmax(np.abs(tpl.affine[:3, :3][1])))
    keep = _cut_keep(np.asanyarray(nib.load(f"{d}/{S}_mask_dof7.nii.gz").dataobj) > 0, ax, float(cfg["cut_margin_mm"]))
    shp, A25 = refine_grid(tpl, 2)                            # the deformable on the 0.25 mm refinement of the template box
    up = X.apply_once(np.asanyarray(tpl.dataobj).astype(np.float32), tpl.affine, shp, A25, X.Map(), order=3)
    upm = X.apply_once(cov.astype(np.float32), tpl.affine, shp, A25, X.Map(), order=1) > 0.5
    k25 = np.repeat(keep, 2).reshape([-1 if i == ax else 1 for i in range(3)])
    lin_map = X.Map(L=np.linalg.inv(M))
    mvol = X.apply_once(src, None, shp, A25, lin_map, order=1)
    mmsk = X.apply_once(np.asanyarray(mo.dataobj).astype(np.float32), mo.affine, shp, A25, lin_map, order=1) > 0.5
    mvol = mvol * mmsk
    fmk = upm & k25
    kz = flat_cut_bridge(shp, A25, float(cfg["flat_cut_mm"]), bridge_json)
    rf = 100 * float((fmk & ~kz).sum()) / max(int(fmk.sum()), 1); rm = 100 * float((mmsk & ~kz).sum()) / max(int(mmsk.sum()), 1)
    if max(rf, rm) > FLAT_CUT_MAX_PCT:
        raise RuntimeError(f"{S}: the flat cut removes {rf:.1f} % (fixed) / {rm:.1f} % (moving) > {FLAT_CUT_MAX_PCT} %: frame error?")
    fmk = fmk & kz; mmsk = mmsk & kz; mvol = mvol * mmsk
    info["flat_cut"] = dict(z_mm=float(cfg["flat_cut_mm"]), fixed_removed_pct=round(rf, 3), moving_removed_pct=round(rm, 3))
    crop, lo, hi = _padded_crop(fmk, mmsk, shp, cfg["pad_rule"])
    Ac = A25.copy(); Ac[:3, 3] = A25[:3, :3] @ lo + A25[:3, 3]
    paths = {k: f"{d}/_{k}_0.25.nii.gz" for k in ("fixed", "fixed_mask", "moving", "moving_mask")}
    nib.save(nib.Nifti1Image(crop(up * fmk), Ac), paths["fixed"]); nib.save(nib.Nifti1Image(crop(fmk).astype(np.uint8), Ac), paths["fixed_mask"])
    nib.save(nib.Nifti1Image(crop(mvol), Ac), paths["moving"]); nib.save(nib.Nifti1Image(crop(mmsk).astype(np.uint8), Ac), paths["moving_mask"])
    eng = f"{d}/_engine"; shutil.rmtree(eng, ignore_errors=True)
    t1 = time.time()
    m = featurereg.register(paths["fixed"], paths["moving"], paths["fixed_mask"], paths["moving_mask"], eng, affine=None,
                            resolution=float(cfg["resolution_mm"]), device=device, **cfg["featurereg"])
    info["deformable_s"] = round(time.time() - t1, 1)
    shutil.copyfile(m["warp"], f"{d}/{S}_warp.nii.gz")
    dfield, da = X.read_fireants_field(f"{d}/{S}_warp.nii.gz")
    grey, rgb, mk = bf_products(bf["final_zarr"], msk, shp, A25, X.Map(L=np.linalg.inv(M), d=dfield, d_affine=da), level=int(cfg["product_level"]))
    nib.save(nib.Nifti1Image(grey, A25), f"{d}/{S}_bf_in_template.nii.gz")
    nib.save(nib.Nifti1Image(rgb, A25), f"{d}/{S}_bf_rgb_in_template.nii.gz")
    nib.save(nib.Nifti1Image(mk.astype(np.uint8), A25), f"{d}/{S}_mask_in_template.nii.gz")
    nib.save(nib.Nifti1Image((mk & kz).astype(np.uint8), A25), f"{d}/{S}_mask_in_template_cut.nii.gz")   # at the template's flat brainstem cut: what the outline check compares
    shutil.rmtree(eng, ignore_errors=True)
    np.savetxt(f"{d}/{S}_linear_pull_ras.txt", np.linalg.inv(M))
    e = xfm.register("bf-native", "bf-template",                        # pull order: template point -> warp -> M^-1 -> blockface
                     [("warp", "itk_lps_disp", f"{d}/{S}_warp.nii.gz"), ("affine", "ras_pull", f"{d}/{S}_linear_pull_ras.txt")],
                     subject=S, dst_grid=str(template), src_grid=src,
                     engine="blockface-based alignment: rigid (masks) + anchored dof-7 (correlation ratio) + linc/registration/featurereg.py",
                     config=dict(cfg, linear_used=info.get("linear_used")), root=str(xfm_root) if xfm_root else None, overwrite=overwrite)
    info["entry"] = str(getattr(e, "dir", e))
    json.dump(info, open(f"{d}/{S}_direct.json", "w"), indent=1, default=float)
    return info


# ------------------------------------------------------------------------------------------------ direct placement: onto the MRI template's b0
def direct_placement(S, bf, template_b0, template_mask, outdir, cfg, device="cuda:0", template_space="mri-template", xfm_root=None,
                     overwrite=False, stand_in_dir=None, frame=None, bridge_json=None, own_entry=None, own_b0=None):
    """Direct placement: the blockface onto the MRI template's b0 (0.25 mm; masked by its brain mask). cfg = configs/registration.yaml
    `direct.direct_placement`. Products in outdir on the template's grid: <S>_bf_in_template, <S>_bf_rgb_in_template,
    <S>_mask_in_template, <S>_warp, <S>_linear_world.txt; the transform entry bf-native -> template_space under xfm_root (default the
    registry, $LINC_DATA/sub-<ID>/xfm; overwrite=True replaces an existing entry, which linc/xfm.py backs up). stand_in_dir: then
    `stand_in` into that folder, from this alignment (frame, bridge_json: its two more inputs), or `stand_in_own_b0` for a brain with
    its own b0 (own_entry: its bf-native -> b0-native entry; own_b0: that b0, on the template's frame)."""
    d = str(outdir); os.makedirs(d, exist_ok=True)
    t05, c05 = coarsen_2x(template_b0, template_mask, f"{d}/_template_b0_0.5.nii.gz", f"{d}/_template_cov_0.5.nii.gz")
    tpl = nib.load(t05); cov = np.asanyarray(nib.load(c05).dataobj) > 0.5
    tmask = f"{d}/_template_mask.nii.gz"; nib.save(nib.Nifti1Image(cov.astype(np.uint8), tpl.affine), tmask)
    src, msk = str(bf["iso_grey"]), str(bf["final_mask"])
    info = dict(subject=S, template_b0=str(template_b0), template_b0_md5=_md5(template_b0), template_mask=str(template_mask),
                template_mask_md5=_md5(template_mask), bf_grey=src, bf_grey_md5=_md5(src), bf_mask=msk, bf_mask_md5=_md5(msk), config=cfg)
    g, mo, mv, mm = _working_copies(d, S, src, msk)
    gv = np.asanyarray(g.dataobj).astype(np.float32) * (np.asanyarray(mo.dataobj) > 0)      # the masked grey first
    gi = nib.Nifti1Image(antialias(gv, np.abs(np.diag(g.affine)[:3]), [0.5] * 3), g.affine)
    nib.save(nip.resample_to_output(gi, [0.5] * 3, order=1), mv)
    nib.save(nip.resample_from_to(mo, nib.load(mv), order=0), mm)
    M, used, lin = linear(d, S, mv, mm, t05, cov, tmask, cross_modal=True, dof9_scale_range=tuple(cfg["dof9_scale_range"]))
    info.update(lin)
    np.savetxt(f"{d}/{S}_linear_world.txt", M)
    fsl.run("flirt", "-in", mm, "-ref", t05, "-applyxfm", "-init", used, "-interp", "nearestneighbour", "-out", f"{d}/{S}_mask_linear.nii.gz")
    ax = int(np.argmax(np.abs(tpl.affine[:3, :3][1])))
    keep = _cut_keep(np.asanyarray(nib.load(f"{d}/{S}_mask_linear.nii.gz").dataobj) > 0, ax, float(cfg["cut_margin_mm"]))
    t25 = nib.load(str(template_b0)); shp, A25 = t25.shape[:3], t25.affine
    upm = np.asanyarray(nib.load(str(template_mask)).dataobj) > 0.5
    up = np.asanyarray(t25.dataobj).astype(np.float32) * upm             # the template b0 masked
    keep25 = np.repeat(keep, 2)
    if keep25.size < shp[ax]:                                            # an odd template dimension: the last 0.25 mm voxel has no block
        keep25 = np.pad(keep25, (0, shp[ax] - keep25.size), mode="edge")
    k25 = keep25[:shp[ax]].reshape([-1 if i == ax else 1 for i in range(3)])
    lin_map = X.Map(L=np.linalg.inv(M))
    mvol = X.apply_once(src, None, shp, A25, lin_map, order=1)
    mmsk = X.apply_once(np.asanyarray(mo.dataobj).astype(np.float32), mo.affine, shp, A25, lin_map, order=1) > 0.5
    mvol = mvol * mmsk                                                   # the blockface masked
    fmk = upm & k25
    kz = flat_cut_canonical(shp, A25, float(cfg["flat_cut_mm"]))
    rf = 100 * float((fmk & ~kz).sum()) / max(int(fmk.sum()), 1); rm = 100 * float((mmsk & ~kz).sum()) / max(int(mmsk.sum()), 1)
    if max(rf, rm) > FLAT_CUT_MAX_PCT:
        raise RuntimeError(f"{S}: the flat cut removes {rf:.1f} % / {rm:.1f} %: frame error?")
    fmk = fmk & kz; mmsk = mmsk & kz; mvol = mvol * mmsk
    info["flat_cut"] = dict(z_mm=float(cfg["flat_cut_mm"]), fixed_removed_pct=round(rf, 3), moving_removed_pct=round(rm, 3))
    crop, lo, hi = _padded_crop(fmk, mmsk, shp, cfg["pad_rule"])
    Ac = A25.copy(); Ac[:3, 3] = A25[:3, :3] @ lo + A25[:3, 3]
    info["reg_grid"] = dict(shape=[int(t) for t in (hi - lo)], mvox=float(np.prod(hi - lo) / 1e6), pad_rule=cfg["pad_rule"])
    paths = {k: f"{d}/_{k}_0.25.nii.gz" for k in ("fixed", "fixed_mask", "moving", "moving_mask")}
    nib.save(nib.Nifti1Image(crop(up * fmk), Ac), paths["fixed"]); nib.save(nib.Nifti1Image(crop(fmk).astype(np.uint8), Ac), paths["fixed_mask"])
    nib.save(nib.Nifti1Image(crop(mvol), Ac), paths["moving"]); nib.save(nib.Nifti1Image(crop(mmsk).astype(np.uint8), Ac), paths["moving_mask"])
    eng = f"{d}/_engine"; shutil.rmtree(eng, ignore_errors=True)
    t1 = time.time()
    m = featurereg.register(paths["fixed"], paths["moving"], paths["fixed_mask"], paths["moving_mask"], eng, affine=None,
                            resolution=float(cfg["resolution_mm"]), device=device, **cfg["featurereg"])
    info["deformable_s"] = round(time.time() - t1, 1)
    shutil.copyfile(m["warp"], f"{d}/{S}_warp.nii.gz")
    dfield, da = X.read_fireants_field(f"{d}/{S}_warp.nii.gz")
    grey, rgb, mk = bf_products(bf["final_zarr"], msk, shp, A25, X.Map(L=np.linalg.inv(M), d=dfield, d_affine=da), level=int(cfg["product_level"]))
    nib.save(nib.Nifti1Image(grey, A25), f"{d}/{S}_bf_in_template.nii.gz")
    nib.save(nib.Nifti1Image(rgb, A25), f"{d}/{S}_bf_rgb_in_template.nii.gz")
    nib.save(nib.Nifti1Image(mk.astype(np.uint8), A25), f"{d}/{S}_mask_in_template.nii.gz")
    shutil.rmtree(eng, ignore_errors=True)
    np.savetxt(f"{d}/{S}_linear_pull_ras.txt", np.linalg.inv(M))
    e = xfm.register("bf-native", template_space,                       # pull order: template point -> warp -> M^-1 -> blockface
                     [("warp", "itk_lps_disp", f"{d}/{S}_warp.nii.gz"), ("affine", "ras_pull", f"{d}/{S}_linear_pull_ras.txt")],
                     subject=S, dst_grid=str(template_b0), src_grid=src,
                     engine="direct placement: full-sphere rigid (masks) + anchored dof-7 + dof-9 (correlation ratio) + linc/registration/featurereg.py",
                     config=dict(cfg, linear_used=info["linear_used"]), root=str(xfm_root) if xfm_root else None, overwrite=overwrite)
    info["entry"] = str(getattr(e, "dir", e))
    json.dump(info, open(f"{d}/{S}_direct_placement.json", "w"), indent=1, default=float)
    if stand_in_dir is not None:                                         # only after a finished alignment: any failure above raises
        info["stand_in"] = (stand_in_own_b0(S, own_entry, own_b0, info["entry"], template_b0, template_mask, stand_in_dir) if own_entry else
                            stand_in(S, bf, M, info["entry"], template_b0, template_mask, frame, bridge_json, stand_in_dir, float(cfg["flat_cut_mm"])))
    return info


# ------------------------------------------------------------------------------------------------ the stand-in frame
def _rigid_of(M, c):
    """The rotation of the 9-dof linear (polar decomposition) and the translation that keeps the world point c where M puts it."""
    U, sv, Vt = np.linalg.svd(M[:3, :3]); R = U @ Vt
    if np.linalg.det(R) < 0:
        raise RuntimeError("the 9-dof linear has a reflection: refused")
    Rr = np.eye(4); Rr[:3, :3] = R; Rr[:3, 3] = (M[:3, :3] @ c + M[:3, 3]) - R @ c
    return Rr, np.round(sv, 4).tolist()


def _box_025(A05, masks05, pad=4.5, mult=8):
    """The padded 0.25 mm box around the union of the 0.5 mm masks, on the frame's centre-aligned 2x lattice (fine voxel 2k on frame
    voxel k), dimensions a multiple of `mult`."""
    assert np.allclose(A05[:3, :3], np.diag(np.diag(A05)[:3])) and (np.diag(A05)[:3] > 0).all(), "the frame is not axis-aligned / positive"
    u = np.zeros(masks05[0].shape, bool)
    for m in masks05:
        u |= m
    idx = np.argwhere(u); lo, hi = idx.min(0), idx.max(0)
    p = int(np.ceil(pad / 0.25))
    a0, a1 = 2 * lo - 1 - p, 2 * hi + 1 + p + 1
    L = (np.ceil((a1 - a0) / mult) * mult).astype(int); a0 = a0 - (L - (a1 - a0)) // 2
    A = A05.copy(); A[:3, :3] = A05[:3, :3] / 2; A[:3, 3] = A05[:3, 3] + a0 * np.diag(A05)[:3] / 2
    return tuple(int(x) for x in L), A


def stand_in(S, bf, M, entry, template_b0, template_mask, frame, bridge_json, out, flat_cut_mm):
    """What `linc-complete prepare` reads for a brain without MRI, in `out`, from the finished alignment. CPU, no registration.
    The blockface goes into the MRI template's 0.5 mm frame rigidly: the rotation of the 9-dof linear M (its scales are not applied, so
    the brain keeps its own size), the translation that keeps the brain's centroid where M puts it, then the bridge. The template's b0
    and brain mask come into the brain through the alignment's full map, inverted: the stand-in b0 has the brain's shape where it
    was photographed and the template's anatomy, at the brain's scale, beyond. The brain mask (rule N1) = the brain's own mask where it
    was photographed, plus the template mask, cut flat at flat_cut_mm, caudal of the photographed end."""
    d = str(out); os.makedirs(d, exist_ok=True)
    Rb = np.array(json.load(open(bridge_json))["rigid_world_run113_to_canonical"])
    mimg = nib.load(str(bf["final_mask"])); mvol = np.asanyarray(mimg.dataobj) > 0
    c = np.argwhere(mvol).mean(0) @ mimg.affine[:3, :3].T + mimg.affine[:3, 3]
    Rr, sv = _rigid_of(M, c)
    T = np.linalg.inv(Rb) @ Rr                                           # blockface native world -> frame world
    Ti, Rri = np.linalg.inv(T), np.linalg.inv(Rr)
    inv = xfm.InverseEntry(xfm.Entry(entry))

    def q_to_template(Q):                                                # frame world -> template world (pull)
        q = Q @ Rb[:3, :3].T + Rb[:3, 3]
        return inv.pull(q @ Rri[:3, :3].T + Rri[:3, 3])
    to_native = lambda Q: Q @ Ti[:3, :3].T + Ti[:3, 3]                   # noqa: E731  frame world -> blockface native world
    b0i, tmi = nib.load(str(template_b0)), nib.load(str(template_mask))
    b0v = np.asanyarray(b0i.dataobj).astype(np.float32); tmv = (np.asanyarray(tmi.dataobj) > 0).astype(np.float32)
    r = nib.load(str(frame)); A05, shp05 = r.affine, r.shape[:3]
    pb05 = X.apply_once(b0v, b0i.affine, shp05, A05, q_to_template, order=1)
    pt05 = X.apply_once(tmv, tmi.affine, shp05, A05, q_to_template, order=1) > 0.5
    kz = (np.arange(tmv.shape[2]) * tmi.affine[2, 2] + tmi.affine[2, 3]) >= flat_cut_mm      # the mask cut flat in template space
    assert abs(tmi.affine[2, 0]) + abs(tmi.affine[2, 1]) < 1e-9, "the template grid is not axis-aligned in z"
    pd05 = X.apply_once(tmv * kz[None, None, :], tmi.affine, shp05, A05, q_to_template, order=1) > 0.5
    bm05 = X.apply_once(mvol.astype(np.float32), mimg.affine, shp05, A05, to_native, order=1) > 0.5
    for v, nm in ((pt05, "template mask"), (bm05, "blockface mask")):
        if not v.any():
            raise RuntimeError(f"{S}: the {nm} is empty on the frame")
    shp25, A25 = _box_025(A05, [pt05, bm05])
    pb25 = X.apply_once(b0v, b0i.affine, shp25, A25, q_to_template, order=1)
    rgb, Az = zarrio.read_level(str(bf["final_zarr"]), 1)
    rgb25 = np.zeros(tuple(shp25) + (3,), np.float32)
    for k in range(3):
        rgb25[..., k] = X.apply_once(antialias(rgb[..., k].astype(np.float32), np.abs(np.diag(Az)[:3]), [0.25] * 3), Az, shp25, A25, to_native, order=1)
    del rgb
    bm25 = X.apply_once(mvol.astype(np.float32), mimg.affine, shp25, A25, to_native, order=1) > 0.5
    grey = (rgb25 @ W_LUM) * bm25
    save = lambda v, A, name, dt_=np.float32: nib.save(nib.Nifti1Image(np.ascontiguousarray(v.astype(dt_)), A), f"{d}/{S}_{name}.nii.gz")  # noqa: E731
    save(pb05, A05, "mri_full"); save(pb25, A25, "mri")
    save(grey, A25, "bf_nonlin"); save(np.clip(rgb25 * bm25[..., None], 0, 255), A25, "bf_nonlin_rgb", np.uint8)
    save(bm25, A25, "bfmask", np.uint8); save(pt05, A05, "pseudo_tplmask_0p5", np.uint8); save(pd05, A05, "pseudo_tplmask_d11_0p5", np.uint8)
    # the rigid T as a FLIRT matrix: moving = the masked isotropic grey, reference = the masked 0.5 mm stand-in b0
    gi = nib.load(str(bf["iso_grey"]))
    mg = np.asanyarray(nip.resample_from_to(nib.Nifti1Image(mvol.astype(np.uint8), mimg.affine), gi, order=0).dataobj) > 0
    mov, ref = f"{d}/{S}_iso_grey_masked.nii.gz", f"{d}/{S}_pseudo_b0_masked_0p5.nii.gz"
    nib.save(nib.Nifti1Image(np.asanyarray(gi.dataobj) * mg, gi.affine, gi.header), mov); save(pb05 * pt05, A05, "pseudo_b0_masked_0p5")
    np.savetxt(f"{d}/{S}_bf2mri_dof9.mat", fsl.world_to_fsl(T, nib.load(mov), nib.load(ref)))
    # the brain mask, rule N1. The photographed end: the plane normal to the blockface's section axis through the caudal-most 0.1 % of
    # the photographed voxels; caudal = the side where the template-only tissue lies
    fimg = nib.load(f"{d}/{S}_mri_full.nii.gz"); tim = nib.load(f"{d}/{S}_pseudo_tplmask_d11_0p5.nii.gz")
    tpl = np.asanyarray(tim.dataobj) > 0
    bim = nib.load(f"{d}/{S}_bfmask.nii.gz")
    bfi = nib.Nifti1Image((np.asanyarray(bim.dataobj) > 0).astype(np.float32), bim.affine)
    bfm = np.asanyarray(nip.resample_from_to(bfi, (fimg.shape, fimg.affine), order=1).dataobj) >= 0.5
    n = T[:3, :3] @ (gi.affine[:3, 1] / np.linalg.norm(gi.affine[:3, 1])); n = n / np.linalg.norm(n)
    world = lambda m: (fimg.affine @ np.c_[np.argwhere(m), np.ones(int(m.sum()))].T)[:3].T  # noqa: E731
    pb = world(bfm) @ n
    only = tpl & ~bfm
    po = world(only) @ n if only.any() else np.array([0.0])
    caudal_sign = -1.0 if po.mean() < pb.mean() else 1.0
    end = np.percentile(pb, 0.1) if caudal_sign < 0 else np.percentile(pb, 99.9)
    allv = np.indices(fimg.shape).reshape(3, -1).T
    pw = (fimg.affine @ np.c_[allv, np.ones(len(allv))].T)[:3].T @ n
    caudal = ((pw - end) * caudal_sign > 0).reshape(fimg.shape)
    brain = bfm | (tpl & caudal)
    hdr = tim.header.copy(); hdr.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(brain.astype(np.uint8), fimg.affine, hdr), f"{d}/{S}_pseudo_brain_n1_0p5.nii.gz")
    vx = float(np.prod(np.abs(np.diag(fimg.affine)[:3])))
    rec = dict(subject=S, what="the stand-in b0 of a brain without MRI (the MRI template's b0 aligned to the brain; not an MRI)",
               T_world_bf_to_frame=T.tolist(), linear_singular_values=sv, entry=str(entry), box_025=dict(shape=list(shp25), affine=A25.tolist()),
               photographed_end=dict(normal_world=[round(float(v), 6) for v in n], offset_mm=round(float(end), 4), caudal_sign=caudal_sign),
               ml=dict(blockface=round(bfm.sum() * vx / 1000, 3), template_caudal_added=round((tpl & caudal & ~bfm).sum() * vx / 1000, 3),
                       brain=round(brain.sum() * vx / 1000, 3)),
               inputs={str(p): _md5(p) for p in (template_b0, template_mask, frame, bridge_json, bf["final_mask"], bf["iso_grey"])})
    json.dump(rec, open(f"{d}/{S}_stand_in.json", "w"), indent=1)
    return rec


def stand_in_own_b0(S, own_entry, own_b0, entry, template_b0, template_mask, out):
    """A brain with its own b0: the stand-in b0 on that b0's grid, so the two differ only in content (a network trained on the stand-in
    b0 of the brains with MRI sees what a brain without MRI gets). A point q of the grid -> b0-native -> blockface native (the brain's
    registration to its b0) -> template (this alignment, inverted) -> the template's b0 and brain mask. CPU, no registration."""
    d = str(out); os.makedirs(d, exist_ok=True)
    r1 = xfm.Entry(own_entry)
    c0 = r1.meta["components"][0]
    assert c0["kind"] == "affine" and c0["convention"] == "ras_pull" and "placed" in c0["file"], f"{S}: unexpected first component {c0}"
    Mbi = np.linalg.inv(np.loadtxt(f"{r1.dir}/{c0['file']}"))
    inv = xfm.InverseEntry(xfm.Entry(entry))
    real = nib.load(str(own_b0)); A, shp = real.affine, real.shape[:3]
    to_tpl = lambda Q: inv.pull(r1.pull(Q @ Mbi[:3, :3].T + Mbi[:3, 3]))  # noqa: E731
    b0i, tmi = nib.load(str(template_b0)), nib.load(str(template_mask))
    pb = X.apply_once(np.asanyarray(b0i.dataobj).astype(np.float32), b0i.affine, shp, A, to_tpl, order=1)
    pt = X.apply_once((np.asanyarray(tmi.dataobj) > 0).astype(np.float32), tmi.affine, shp, A, to_tpl, order=1) > 0.5
    nib.save(nib.Nifti1Image(pb.astype(np.float32), A), f"{d}/{S}_mri_full.nii.gz")
    nib.save(nib.Nifti1Image(pt.astype(np.uint8), A), f"{d}/{S}_pseudo_tplmask_0p5.nii.gz")
    rec = dict(subject=S, what="the stand-in b0 of a brain with MRI, on its own b0's grid (the MRI template's b0 aligned to the brain; not its MRI)",
               own_entry=str(own_entry), entry=str(entry), inputs={str(p): _md5(p) for p in (template_b0, template_mask, own_b0)})
    json.dump(rec, open(f"{d}/{S}_stand_in.json", "w"), indent=1)
    return rec
