"""Registration of a brain's blockface volume to its own b0 image, the first step of the MRI-based alignment (Methods, "Blockface to
MRI registration and synthesis of missing sections").

You need
  the brain's reconstruction and mask, its b0 and b0 brain mask (every input brain-masked), FSL, the anatomix environment and a
  GPU; parameters: settings/registration.yaml `to_own_mri`; per brain, settings/brains/<ID>.yaml (the starting pose of record, the
  native end crop).
What it does
  0. the b0 registered rigidly onto the MRI template's 0.5 mm grid (`place_b0`: FLIRT, 6 degrees of freedom, correlation ratio,
     +-90 deg search); the blockface resampled isotropically (`iso_pair`: 0.15 mm grayscale luminance and mask);
  1. the starting pose: a rotation and a translation only (the paper's brains: their pose of record; a new brain: `initial_pose`,
     FLIRT with 7 degrees of freedom and a search over all rotations);
  2. FLIRT on the masked inputs at 0.5 mm, correlation ratio, no search: 7 degrees of freedom (3 iterations), then 9 (rotation,
     translation and a scale along each axis; up to 10 iterations). The cost is restricted to the photographed slab (0.5 mm inside
     both end faces of the photographed stack, derived again from the current transform at each iteration; iterated until the axis
     scales change by < 2e-3 and the translation by < 0.05 mm), because without it a stack missing its caudal block is stretched
     to fill the b0;
  3. the fixed image: the b0 x its brain mask inside that slab;
  4. the working grid at 0.25 mm: the b0's 0.5 mm lattice with >= 4.5 mm of zeros on every side, dimensions rounded up to a
     multiple of 8; the masked blockface resampled onto it through the 9-dof transform;
  5. the feature-based deformable registration (featurereg.py: warp sigma 1.25, gradient sigma 1.5, Adam step 1.0, the loss
     including the background, the features not multiplied by the masks);
  6. the products by one resample from the native pyramid through the 9-dof transform and the warp, divided by the resampled
     photographed support, the brain's native end crop applied first (resample.py).
You get
  the grayscale, RGB and mask on the working grid, and the registry entry bf-native -> b0-native (code/shared/xfm.py; resampling order
  b0-native -> the b0 on the template grid -> (warp) -> blockface).

Ported from the analysis project: featurereg/r1_redo_v3.py, featurereg/r1_rewarp.py, code/register_mri.py,
featurereg/r1_dof9_test.py and code/masknorm.py, the final settings only, unchanged numerically.
"""
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import nibabel.processing as nip
from scipy import ndimage as ndi

from shared import xfm
from blockface import zarrio
from blockface.registration import featurereg, fsl, xfmtools as X
from blockface.registration.resample import W_LUM, bf_products


# ------------------------------------------------------------------------------------------------ inputs
def place_b0(b0_native, mask_native, mri_ref, outdir, subject, search_deg=90):
    """The b0 (and its brain mask) placed rigidly on the MRI template's 0.5 mm grid (FLIRT 6 dof, correlation ratio, +-search_deg on
    every axis) -> <S>_mri.nii.gz, <S>_mri_mask.nii.gz, <S>_mri2template.mat in outdir."""
    os.makedirs(outdir, exist_ok=True)
    mat, out_img, out_mask = (f"{outdir}/{subject}_mri2template.mat", f"{outdir}/{subject}_mri.nii.gz", f"{outdir}/{subject}_mri_mask.nii.gz")
    s = str(search_deg)
    fsl.run("flirt", "-in", b0_native, "-ref", mri_ref, "-dof", "6", "-cost", "corratio", "-searchrx", f"-{s}", s, "-searchry", f"-{s}", s,
            "-searchrz", f"-{s}", s, "-omat", mat, "-out", out_img)
    fsl.run("flirt", "-in", mask_native, "-ref", mri_ref, "-applyxfm", "-init", mat, "-interp", "nearestneighbour", "-out", out_mask)
    return dict(placed_b0=out_img, placed_mask=out_mask, mri2template_mat=mat)


def iso_pair(final_zarr, final_mask, out_grey, out_mask, iso=0.15, level=1):
    """The blockface's isotropic pair: the luminance of pyramid level `level` resampled to `iso` mm (linear) and the L4 mask
    (nearest)."""
    rgb, A = zarrio.read_level(final_zarr, level)
    grey = (rgb.astype(np.float32) @ W_LUM) if rgb.ndim == 4 else rgb.astype(np.float32)
    mimg = nib.load(str(final_mask))
    g_iso = nip.resample_to_output(nib.Nifti1Image(grey.astype(np.float32), A), [iso] * 3, order=1)
    m_iso = nip.resample_to_output(nib.Nifti1Image((np.asanyarray(mimg.dataobj) > 0).astype(np.uint8), mimg.affine), [iso] * 3, order=0)
    nib.save(g_iso, out_grey); nib.save(m_iso, out_mask)
    return out_grey, out_mask


def initial_pose(iso_grey, iso_mask, placed_b0, b0_mask, workdir):
    """A new brain's start pose: FLIRT 7 dof (correlation ratio, full-sphere search) of the masked blockface onto the masked b0;
    only its rotation and where it puts the blockface's centre are kept (the scale comes from the blockface spacing)."""
    os.makedirs(workdir, exist_ok=True)
    mov, ref = _masked_inputs(iso_grey, iso_mask, placed_b0, b0_mask, workdir, "pose")
    mat = f"{workdir}/pose_dof7.mat"
    fsl.run("flirt", "-in", mov, "-ref", ref, "-dof", "7", "-cost", "corratio", "-searchrx", "-180", "180", "-searchry", "-180", "180",
            "-searchrz", "-180", "180", "-omat", mat)
    T = fsl.fsl_to_world(np.loadtxt(mat), nib.load(mov), nib.load(ref))
    return pose_only(T, iso_mask)


def pose_only(T, iso_mask):
    """The nearest rotation of T's linear part, with the translation that keeps where T puts the blockface mask's centroid."""
    U, _, Vt = np.linalg.svd(T[:3, :3]); Rp = U @ Vt
    mm = nib.load(str(iso_mask)); xc = (np.argwhere(np.asanyarray(mm.dataobj) > 0).mean(0) @ mm.affine[:3, :3].T + mm.affine[:3, 3])
    T0 = np.eye(4); T0[:3, :3] = Rp; T0[:3, 3] = (T[:3, :3] @ xc + T[:3, 3]) - Rp @ xc
    return T0


def _masked_inputs(iso_grey, iso_mask, fixed_b0, b0_mask, outdir, prefix):
    g = nib.load(str(iso_grey)); m = nib.load(str(iso_mask))
    mg = np.asanyarray(nip.resample_from_to(nib.Nifti1Image((np.asanyarray(m.dataobj) > 0).astype(np.uint8), m.affine), g, order=0).dataobj) > 0
    mov = f"{outdir}/{prefix}_iso_grey_masked.nii.gz"; nib.save(nib.Nifti1Image(np.asanyarray(g.dataobj) * mg, g.affine, g.header), mov)
    b = nib.load(str(fixed_b0)); bm = nib.load(str(b0_mask))
    assert b.shape[:3] == bm.shape[:3] and np.allclose(b.affine, bm.affine), "the b0 and its mask must share a grid"
    ref = f"{outdir}/{prefix}_b0_masked.nii.gz"
    nib.save(nib.Nifti1Image(np.asanyarray(b.dataobj) * (np.asanyarray(bm.dataobj) > 0.5), b.affine, b.header), ref)
    return mov, ref


# ------------------------------------------------------------------------------------------------ the photographed slab
def _end_faces(T, iso_mask):
    bfm = nib.load(str(iso_mask)); M = np.asanyarray(bfm.dataobj) > 0
    ys = np.where(M.any(axis=(0, 2)))[0]; ic, kc = M.shape[0] / 2.0, M.shape[2] / 2.0
    p_hi = (T @ (bfm.affine @ np.array([ic, ys.max(), kc, 1.0])))[:3]
    p_lo = (T @ (bfm.affine @ np.array([ic, ys.min(), kc, 1.0])))[:3]
    n = np.linalg.inv(T[:3, :3]).T @ bfm.affine[:3, 1]; n /= np.linalg.norm(n)
    if np.dot(n, p_hi - p_lo) < 0:
        n = -n
    return p_hi, p_lo, n


def slab_weight(T, iso_mask, A, shp, inset=0.5):
    """1 on the grid (A, shp) between the blockface's photographed end faces mapped through T (blockface world -> MRI world), `inset`
    mm inside both; 0 elsewhere. Returns (weight, slab width mm)."""
    p_hi, p_lo, n = _end_faces(T, iso_mask)
    width = float(np.dot(p_hi - p_lo, n))
    ijk = np.stack(np.meshgrid(*[np.arange(k) for k in shp], indexing="ij"), -1).reshape(-1, 3)
    t = ((ijk @ A[:3, :3].T + A[:3, 3]) - p_lo) @ n
    return ((t >= inset) & (t <= width - inset)).reshape(shp).astype(np.float32), width


def slab_keep(T, iso_mask, A, shp):
    """The slab between the two end faces (no inset), on the grid (A, shp). Returns (keep, width mm)."""
    p_hi, p_lo, n = _end_faces(T, iso_mask)
    ijk = np.stack(np.meshgrid(*[np.arange(k) for k in shp], indexing="ij"), -1).reshape(-1, 3)
    w = ijk @ A[:3, :3].T + A[:3, 3]
    keep = (((w - p_hi) @ n) <= 0) & (((w - p_lo) @ n) >= 0)
    return keep.reshape(shp), float(np.dot(p_hi - p_lo, n))


def slab_fit(mov, ref, iso_mask, T_init, dof, iters, prefix, inset=0.5, tol_scale=2e-3, tol_mm=0.05):
    """FLIRT correlation ratio, no search, -refweight = the photographed slab under the current transform, iterated until the axis
    scales move < tol_scale and the translation < tol_mm. Returns (T, history)."""
    mi, ri = nib.load(mov), nib.load(ref)
    T, hist = T_init, []
    for it in range(iters):
        w, width = slab_weight(T, iso_mask, ri.affine, ri.shape[:3], inset)
        rw = f"{prefix}_rw{it}.nii.gz"; nib.save(nib.Nifti1Image(w, ri.affine), rw)
        ini = f"{prefix}_init{it}.mat"; np.savetxt(ini, fsl.world_to_fsl(T, mi, ri))
        cur = f"{prefix}_it{it}.mat"
        fsl.run("flirt", "-in", mov, "-ref", ref, "-init", ini, "-dof", dof, "-cost", "corratio", "-refweight", rw, "-nosearch", "-omat", cur)
        Tn = fsl.fsl_to_world(np.loadtxt(cur), mi, ri)
        dsc = float(np.abs(np.linalg.norm(Tn[:3, :3], axis=0) - np.linalg.norm(T[:3, :3], axis=0)).max())
        dtr = float(np.linalg.norm(Tn[:3, 3] - T[:3, 3]))
        hist.append(dict(it=it, scales=np.linalg.norm(Tn[:3, :3], axis=0).round(5).tolist(), d_scale=round(dsc, 5), d_trans_mm=round(dtr, 4),
                         slab_width_mm=round(width, 2)))
        T = Tn; shutil.copy2(cur, f"{prefix}_last.mat")
        if it and dsc < tol_scale and dtr < tol_mm:
            break
    return T, hist


# ------------------------------------------------------------------------------------------------ the linear stage
def linear(subject, bf, mri, start_pose, outdir, cfg):
    """Steps 2-3: the slab-weighted 7 -> 9 dof fits from `start_pose` and the masked fixed image. bf = dict(iso_grey, iso_mask);
    mri = dict(placed_b0 (the FLIRT reference grid), fixed_b0 (on that grid), b0_mask (the mask of record, on that grid)). Returns paths
    and the fit histories."""
    S, O = subject, outdir; os.makedirs(O, exist_ok=True)
    mov, ref = _masked_inputs(bf["iso_grey"], bf["iso_mask"], mri["fixed_b0"], mri["b0_mask"], O, S)
    mv_img = nib.load(mov)
    vlin = float(cfg["linear_resolution_mm"])
    sig = 0.5 * np.sqrt(max((vlin / abs(float(mv_img.affine[0, 0]))) ** 2 - 1, 0))      # anti-aliased, then at the linear resolution
    sm = ndi.gaussian_filter(np.asanyarray(mv_img.dataobj).astype(np.float32), sig)
    mov05 = f"{O}/{S}_iso_grey_masked_{vlin:g}.nii.gz"
    nib.save(nip.resample_to_output(nib.Nifti1Image(sm, mv_img.affine), [vlin] * 3, order=1), mov05)
    inset = float(cfg["slab_inset_mm"])
    T7, hist7 = slab_fit(mov05, ref, bf["iso_mask"], np.asarray(start_pose, float), 7, int(cfg["dof7_iterations"]), f"{O}/{S}_dof7", inset)
    T9, hist9 = slab_fit(mov05, ref, bf["iso_mask"], T7, 9, int(cfg["dof9_iterations"]), f"{O}/{S}_dof9", inset)
    m7, m9 = f"{O}/{S}_bf2mri_dof7.mat", f"{O}/{S}_bf2mri_dof9.mat"
    np.savetxt(m7, fsl.world_to_fsl(T7, mv_img, nib.load(ref)))
    np.savetxt(m9, fsl.world_to_fsl(T9, mv_img, nib.load(ref)))
    m9m = f"{O}/{S}_bf2mri_dof9_maskgrid.mat"; np.savetxt(m9m, fsl.world_to_fsl(T9, nib.load(bf["iso_mask"]), nib.load(ref)))
    bfm = f"{O}/{S}_bfmask_in_mri.nii.gz"                                # the blockface extent in MRI space through the 9-dof matrix
    fsl.run("flirt", "-in", bf["iso_mask"], "-ref", mri["placed_b0"], "-applyxfm", "-init", m9m, "-interp", "nearestneighbour", "-out", bfm)
    b = nib.load(str(mri["fixed_b0"])); bm = nib.load(str(mri["b0_mask"]))
    T9g = fsl.fsl_to_world(np.loadtxt(m9), nib.load(bf["iso_grey"]), nib.load(mri["placed_b0"]))
    keep, width = slab_keep(T9g, bf["iso_mask"], b.affine, b.shape[:3])
    fixed = f"{O}/{S}_mri_fixed.nii.gz"                                   # the masked fixed inside the 9-dof photographed slab
    nib.save(nib.Nifti1Image((np.asanyarray(b.dataobj) * ((np.asanyarray(bm.dataobj) > 0.5) & keep)).astype(np.float32), b.affine, b.header), fixed)
    return dict(mats=dict(dof7=m7, dof9=m9, dof9_maskgrid=m9m), bfmask_in_mri=bfm, fixed=fixed, slab_width_mm=round(width, 2),
                slab_fit_dof7=hist7, slab_fit_dof9=hist9, T9_world=T9g.tolist())


# ------------------------------------------------------------------------------------------------ the working grid
def working_grid(subject, bf, mri, mat, fixed, bfm_in_mri, workdir, iso=0.25, pad_rule="4.5:8", mri_ref_iso=0.5):
    """Step 4: the fixed image on the padded fine grid and the masked blockface (+ mask) placed onto it through `mat`. The grid is
    the b0's lattice at `iso` with >= M mm of zeros on every side around (b0 brain | blockface extent), never clipped, dims rounded UP
    to a multiple of K (pad_rule "M:K"), the extra voxels split evenly. Returns the paths."""
    os.makedirs(workdir, exist_ok=True)
    mc = nib.load(str(fixed)); mcd = np.asanyarray(mc.dataobj)
    bmr = nib.load(str(mri["frame_mask"]))                                # the b0 x its brain mask of record (same grid)
    assert bmr.shape[:3] == mc.shape[:3] and np.allclose(bmr.affine, mc.affine), "the brain mask must share the b0's grid"
    mcd = mcd * (np.asanyarray(bmr.dataobj) > 0.5)
    gimg = nib.load(str(bf["iso_grey"])); mimg = nib.load(str(bf["iso_mask"]))
    T = fsl.fsl_to_world(np.loadtxt(mat), gimg, nib.load(str(mri["placed_b0"])))   # blockface world -> MRI world
    bfm = np.asanyarray(nib.load(str(bfm_in_mri)).dataobj) > 0
    m_mm, mult = float(pad_rule.split(":")[0]), int(pad_rule.split(":")[1])
    mv = np.full(3, int(np.ceil(m_mm / mri_ref_iso - 1e-9)))
    pad = [(int(mv[k]), int(mv[k])) for k in range(3)]
    mcd = np.pad(mcd, pad); bfm = np.pad(bfm, pad)
    A0 = mc.affine.copy(); A0[:3, 3] -= A0[:3, :3] @ mv.astype(float)
    idx = np.argwhere((mcd > 0) | bfm); lo = idx.min(0) - mv; hi = idx.max(0) + mv
    r = int(round(mri_ref_iso / iso)); n_ext = (hi - lo) * r + 1                 # the extent-based fine size (2N-1 at 0.25 mm)
    fine_dims = (np.ceil(n_ext / mult) * mult).astype(int)
    sl = tuple(slice(int(lo[a]), int(hi[a]) + 1) for a in range(3))
    A = A0.copy(); A[:3, 3] += A[:3, :3] @ lo.astype(float)
    At = A.copy(); At[:3, :3] = A[:3, :3] * (iso / mri_ref_iso)               # the b0 lattice at `iso`, extra voxels split evenly
    At[:3, 3] -= At[:3, :3] @ ((fine_dims - n_ext) // 2).astype(float)
    fine_fixed = nip.resample_from_to(nib.Nifti1Image(np.ascontiguousarray(mcd[sl]), A, mc.header),
                                      (tuple(int(x) for x in fine_dims), At), order=1, cval=0.0)
    fmap = (fine_fixed.shape[:3], fine_fixed.affine)
    P = dict(fine_mri=f"{workdir}/mri_fine.nii.gz", fine_fmask=f"{workdir}/mri_fine_mask.nii.gz", bf_rigid=f"{workdir}/bf_rigid.nii.gz",
             bfm_fine=f"{workdir}/bfmask_fine.nii.gz")
    nib.save(fine_fixed, P["fine_mri"])
    nib.save(nib.Nifti1Image((np.asanyarray(fine_fixed.dataobj) > 0).astype(np.uint8), fine_fixed.affine), P["fine_fmask"])
    mg = np.asanyarray(nip.resample_from_to(                              # the iso grey bleeds ~1 voxel past its mask: masked first
        nib.Nifti1Image((np.asanyarray(mimg.dataobj) > 0).astype(np.uint8), mimg.affine), gimg, order=0).dataobj) > 0
    gdat = np.asanyarray(gimg.dataobj) * mg
    bfr = nip.resample_from_to(nib.Nifti1Image(gdat, T @ gimg.affine, gimg.header), fmap, order=1)
    bmr_ = nip.resample_from_to(nib.Nifti1Image((np.asanyarray(mimg.dataobj) > 0).astype(np.uint8), T @ mimg.affine), fmap, order=0)
    bfr = nib.Nifti1Image(np.asanyarray(bfr.dataobj) * (np.asanyarray(bmr_.dataobj) > 0), bfr.affine, bfr.header)   # masked after placement too
    nib.save(bfr, P["bf_rigid"]); nib.save(bmr_, P["bfm_fine"])
    P["nvox"] = float(np.prod(fine_fixed.shape[:3]) / 1e6)
    return P


# ------------------------------------------------------------------------------------------------ the whole chain
def register(subject, bf, mri, start_pose, outdir, cfg, device="cuda:0", xfm_root=None, overwrite=False, crop_mm=0.0, vz_mm=None):
    """Steps 2-6 for one brain. bf = dict(iso_grey, iso_mask, final_zarr, final_mask); mri = dict(placed_b0, fixed_b0, b0_mask,
    frame_mask, b0_native, mri2template_mat); cfg = settings/registration.yaml `to_own_mri`. Writes into outdir: the linear stage
    (linear/), <S>_mri.nii.gz (the fixed image on the working grid), <S>_bf_rigid.nii.gz, <S>_warp.nii.gz, <S>_bf_nonlin.nii.gz,
    <S>_bf_nonlin_rgb.nii.gz, <S>_bf_nonlin_mask.nii.gz, <S>_metrics.json; and the transform entry bf-native -> b0-native under
    xfm_root (default the registry, $LINC_DATA/sub-<ID>/xfm; overwrite=True replaces an existing entry, which code/shared/xfm.py backs up).
    crop_mm, vz_mm: the brain's native end crop and section step (mm), read by the product step (code/blockface/registration/resample.py)."""
    S = subject; outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    lin = linear(S, bf, mri, start_pose, str(outdir / "linear"), cfg)
    m9 = lin["mats"]["dof9"]
    eng = outdir / "_engine"; shutil.rmtree(eng, ignore_errors=True)       # (featurereg checks that both inputs are masked, fail-closed)
    res = float(cfg["resolution_mm"])
    P = working_grid(S, bf, mri, m9, lin["fixed"], lin["bfmask_in_mri"], str(eng), iso=res, pad_rule=cfg["pad_rule"])
    t0 = time.time()
    m = featurereg.register(P["fine_mri"], P["bf_rigid"], P["fine_fmask"], P["bfm_fine"], str(eng), affine=None, resolution=res,
                            device=device, **cfg["featurereg"])
    reg_s = time.time() - t0
    fine = nib.load(P["fine_mri"])
    shutil.copyfile(P["fine_mri"], outdir / f"{S}_mri.nii.gz")
    shutil.copyfile(P["bf_rigid"], outdir / f"{S}_bf_rigid.nii.gz")
    shutil.copyfile(m["warp"], outdir / f"{S}_warp.nii.gz")
    T = fsl.fsl_to_world(np.loadtxt(m9), nib.load(bf["iso_grey"]), nib.load(mri["placed_b0"]))
    d, da = X.read_fireants_field(str(outdir / f"{S}_warp.nii.gz"))
    fmap = X.Map(L=np.linalg.inv(T), d=d, d_affine=da)                    # fine world -> blockface world
    grey, rgb, mk = bf_products(bf["final_zarr"], bf["final_mask"], fine.shape, fine.affine, fmap, level=int(cfg["product_level"]),
                                norm=bool(cfg["product_normalize"]), crop_mm=float(crop_mm or 0.0), vz_mm=vz_mm)
    nib.save(nib.Nifti1Image(grey, fine.affine), outdir / f"{S}_bf_nonlin.nii.gz")
    nib.save(nib.Nifti1Image(rgb, fine.affine), outdir / f"{S}_bf_nonlin_rgb.nii.gz")
    nib.save(nib.Nifti1Image(mk.astype(np.uint8), fine.affine), outdir / f"{S}_bf_nonlin_mask.nii.gz")
    meta = {k: v for k, v in m.items() if k not in ("moved", "warp")}
    meta.update(config=cfg, grid_vox_M=round(P["nvox"], 1), register_wall_s=round(reg_s, 1), linear=lin,
                products="one resample from the native pyramid through 9-dof + warp, anti-aliased, support-normalized",
                native_end_crop_mm=float(crop_mm or 0.0))
    (outdir / f"{S}_metrics.json").write_text(json.dumps(meta, indent=2, default=float))
    shutil.rmtree(eng, ignore_errors=True)
    # the transform entry (pull order: b0-native world -> placed b0 world -> (warp) -> blockface world)
    Wn = fsl.fsl_to_world(np.loadtxt(mri["mri2template_mat"]), nib.load(mri["b0_native"]), nib.load(mri["placed_b0"]))
    tmp = outdir / "_xfm_tmp"; tmp.mkdir(exist_ok=True)
    np.savetxt(tmp / "b0native_to_placed_pull.txt", Wn); np.savetxt(tmp / "placed_to_bf_pull.txt", np.linalg.inv(T))
    e = xfm.register("bf-native", "b0-native",
                     [("affine", "ras_pull", str(tmp / "b0native_to_placed_pull.txt")), ("warp", "itk_lps_disp", str(outdir / f"{S}_warp.nii.gz")),
                      ("affine", "ras_pull", str(tmp / "placed_to_bf_pull.txt"))],
                     subject=S, dst_grid=mri["b0_native"], src_grid=bf["iso_grey"],
                     engine="FLIRT 7->9 dof correlation ratio, slab-weighted, masked inputs + code/blockface/registration/featurereg.py",
                     config=dict(cfg, b0_mask=str(mri["b0_mask"])), root=str(xfm_root) if xfm_root else None, overwrite=overwrite)
    shutil.rmtree(tmp, ignore_errors=True)
    return dict(subject=S, outdir=str(outdir), entry=str(getattr(e, "dir", e)), metrics=meta)
