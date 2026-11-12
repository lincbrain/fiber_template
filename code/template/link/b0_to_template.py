#!/usr/bin/env python3
"""The MRI route: a photographed brain's own b0 image registered to the MRI template's b0 channel (paper: Methods, "Link to the MRI
channels and placement of photographed brains", sec:m-link). Writes the registry entry b0-native -> mri-template (code/shared/xfm.py).

Recipe (settings/link.yaml `b0_to_template`; every input brain-masked):
  1. canonical placement P0 = R_bridge @ W_flirt: the b0's FLIRT 6-dof placement in the run-113 frame (<ID>_mri2template.mat with its
     reference image <ID>_mri.nii.gz; `blockface.registration.to_own_mri.place_b0` makes them) and the run-113 -> canonical bridge rigid;
  2. input QC: when the b0's residual low-frequency field (p95/p5) exceeds 1.5 it is re-debiased with ANTs N4 inside its brain mask;
     the N4 b0 feeds the deformable only;
  3. the linear start ('xag'): a 7-dof similarity of the P0-placed native b0 to the template b0 (NCC, Powell, x2 downsampled; a coarse
     rotation-grid retry when a rotation exceeds 0.25 rad), then a 12-dof affine of the similarity-placed b0; the affine is rejected
     (similarity only) when the singular-value ratio of (A S) exceeds 1.5. Pst = A S P0 (or S P0);
  4. ONE resample (b0 cubic, mask linear > 0.5) onto a 0.35 mm grid centered on (the Pst-placed b0 brain | the template brain) with
     8 mm of zeros on every side, sizes x16; the template (x its mask) linearly onto the same grid;
  5. D11 brainstem cut of the moving input: voxels that the similarity places below the template brain's lowest axial slice are zeroed
     (registration input only; the products use the uncropped mask);
  6. the feature-based engine (code/blockface/registration/featurereg.py, deformable stages only, 0.35 mm, warp smoothing 1.25 voxels);
  7. registry entry, components in PULL order: (warp, itk_lps_disp, <ID>_warp.nii.gz) on the working grid, then
     (affine, ras_pull, <ID>_start_pull.txt) = Pst^-1; dst grid = the template b0, src grid = the native b0. The moved b0 and the
     uncropped moved mask on the template grid and the outline check of the moved mask (code/template/link/qc.py) are written beside it.

    python -m template.link.b0_to_template --subject <ID> --b0 <b0 native> --mask <mask native> --template-b0 B0 --template-mask M \
        [--registry DIR] [--out DIR] [--device cuda:0] [--overwrite]
Inputs default to the data layout (docs/tutorials/00_start_here.md): $LINC_DATA/sub-<ID>/mri/<ID>_mri_b0_native.nii.gz, <ID>_mri_mask_native.nii.gz
(the brain's mask of record), <ID>_mri2template.mat, <ID>_mri.nii.gz; the bridge = settings/link.yaml inputs.bridge. The registry
entry goes to $LINC_DATA/sub-<ID>/xfm/b0-native__to__mri-template (or <--registry>/b0-native__to__mri-template); the working files
and products to --out (default $LINC_WORK/b0_to_template/<ID>).

Ported from the analysis project's mri/code/r5_canon_test.py (`run_canon`, arm canon with --lin xag, the D22 production runs of
_nightlogs/L2_r5prod_queue.sh with R5_RECORD_MASKS=1), its helpers in bftemplate/cv/b0_to_template.py (bridge, N4 rule) and the registry
write of mri/code/r5_production_register.py: the final path only, unchanged numerically. Not ported: the regqc scoring (global QC store,
montage, linear-part gate) and the edge diagnostics of the test harness; the other linear modes (fa, xa, sim).
"""
import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

from shared import paths, xfm
from blockface.registration import featurereg, fsl, xfmtools as X
from . import qc
from template.link.common import bridge, config

SRC, DST = "b0-native", "mri-template"
def cfg():
    return config()["b0_to_template"]


def engine_text(c=None):
    """The registry entry's engine field, from the configuration (settings/link.yaml b0_to_template)."""
    c = c or cfg()
    return (f"featurereg (anatomix + MIND-SSC), guarded start, {float(c['grid_res_mm']):g} mm grid, "
            f"{float(c['grid_margin_mm']):g} mm centered margin, brainstem crop")


# ------------------------------------------------------------------------------------------------ input QC (N4)
def field_ratio(V, m, sigma=8):
    """p95 / p5 of the `sigma`-voxel-smoothed masked intensity: the residual low-frequency field (anatomy at that scale ~1.05-1.2)."""
    sm = ndi.gaussian_filter(np.where(m, V, 0), sigma) / np.maximum(ndi.gaussian_filter(m.astype(np.float32), sigma), 1e-3)
    f = sm[m & (V > 0)]
    return float(np.percentile(f, 95) / np.percentile(f, 5))


def n4_b0(S, b0_path, mask_path, mk, d, c):
    """ANTs N4 inside the brain mask; returns the corrected intensities (same grid) and the record."""
    from . import ants
    exe = ants.executable("N4BiasFieldCorrection")
    ants.check(exe=exe)
    n = c["n4"]
    out, fld = f"{d}/{S}_b0_n4.nii.gz", f"{d}/{S}_b0_n4field.nii.gz"
    subprocess.run([exe, "-d", "3", "-i", str(b0_path), "-x", str(mask_path), "-s", str(n["shrink"]), "-c", n["convergence"],
                    "-b", f"[{n['spline_mm']}]", "-o", f"[{out},{fld}]"], check=True, capture_output=True)
    sg = c["n4_field_sigma_vox"]
    raw = np.asanyarray(nib.load(str(b0_path)).dataobj).astype(np.float32)
    cor = np.asanyarray(nib.load(out).dataobj).astype(np.float32)
    F = np.asanyarray(nib.load(fld).dataobj).astype(np.float32)[mk]
    info = dict(spline_mm=n["spline_mm"], residual_field_before=round(field_ratio(raw, mk, sg), 3),
                residual_field_after=round(field_ratio(cor, mk, sg), 3),
                n4_field_p95_over_p5=round(float(np.percentile(F, 95) / np.percentile(F, 5)), 3), corrected=out)
    return cor, info


# ------------------------------------------------------------------------------------------------ the linear start
def placement(b0_img, placement_mat, placed_b0, bridge_json=None):
    """P0 = R_bridge @ W_flirt: native b0 world -> canonical (MRI template) world."""
    Wn = fsl.fsl_to_world(np.loadtxt(str(placement_mat)), b0_img, nib.load(str(placed_b0)))
    return bridge(bridge_json) @ Wn


def linear_start(b0_native, b0_aff, mk, T, tpl_aff, Tm, P0, c):
    """Similarity S (with its rotation-grid retry) and the guarded affine. Returns (Pst, S, info)."""
    f = int(c["linear_factor"])
    L0 = X.Map(L=np.linalg.inv(P0))
    moved0 = X.apply_once(b0_native, b0_aff, T.shape, tpl_aff, L0, order=1)
    m0 = X.apply_once(mk.astype(np.float32), b0_aff, T.shape, tpl_aff, L0, order=1) > 0.5
    Sm, si = X.linear_register(T, tpl_aff, Tm, moved0, tpl_aff, m0, "similarity", factor=f)
    if max(abs(v) for v in si["params"][:3]) > float(c["similarity_retry_rad"]):
        g = np.array(c["similarity_retry_grid_rad"], float)
        Sm2, si2 = X.linear_register(T, tpl_aff, Tm, moved0, tpl_aff, m0, "similarity", factor=f, grid_search={0: g, 1: g, 2: g})
        if si2["ncc"] > si["ncc"]:
            Sm, si = Sm2, dict(si2, retry_from=si)
        else:
            si = dict(si, retry_rejected=si2)
    info = dict(similarity=si, lin_mode=c["linear_mode"], affine_fit=None, guard=None)
    if c["linear_mode"] != "xag":
        raise ValueError("only the linear mode of record ('xag') is ported")
    L1 = X.Map(L=np.linalg.inv(Sm @ P0))
    mv1 = X.apply_once(b0_native, b0_aff, T.shape, tpl_aff, L1, order=1)
    mm1 = X.apply_once(mk.astype(np.float32), b0_aff, T.shape, tpl_aff, L1, order=1) > 0.5
    Am, ai = X.linear_register(T, tpl_aff, Tm, mv1, tpl_aff, mm1, "affine", factor=f)
    Pst = Am @ Sm @ P0
    lim = float(c["affine_anisotropy_max"])
    sv = np.linalg.svd((Am @ Sm)[:3, :3], compute_uv=False)
    guard = dict(anisotropy=round(float(sv.max() / sv.min()), 4), limit=lim)
    guard["fallback_to_similarity"] = guard["anisotropy"] > lim
    if guard["fallback_to_similarity"]:
        Pst = Sm @ P0
    info.update(affine_fit=ai, guard=guard)
    return Pst, Sm, info


# ------------------------------------------------------------------------------------------------ the working grid + engine inputs
def engine_inputs(b0, b0_aff, mk, T, tpl_aff, Tm, P0, Sm, Pst, margin, c, d):
    """Steps 4-5: the centered working grid, the start-placed moving (D11-cut) and the template on it; saved under d."""
    mw = np.argwhere(mk) @ b0_aff[:3, :3].T + b0_aff[:3, 3]; mw = mw @ Pst[:3, :3].T + Pst[:3, 3]
    tw = np.argwhere(Tm) @ tpl_aff[:3, :3].T + tpl_aff[:3, 3]
    lo = np.minimum(mw.min(0), tw.min(0)) - margin; hi = np.maximum(mw.max(0), tw.max(0)) + margin
    gshape, gaff = qc.grid_from(lo, hi, float(c["grid_res_mm"]), int(c["grid_multiple"]))
    lin = X.Map(L=np.linalg.inv(Pst))
    mv = X.apply_once(b0, b0_aff, gshape, gaff, lin, order=3)
    mvm_full = X.apply_once(mk.astype(np.float32), b0_aff, gshape, gaff, lin, order=1) > 0.5
    tz = (np.where(Tm.any((0, 1)))[0].min()) * tpl_aff[2, 2] + tpl_aff[2, 3]          # the template brain's lowest axial slice (world z)
    if c["brainstem_cut"]:
        I = np.indices(gshape).reshape(3, -1).T; Wg = I @ gaff[:3, :3].T + gaff[:3, 3]
        Cz = Sm @ P0 @ np.linalg.inv(Pst)                                     # working frame -> template frame (similarity), for the cut only
        zt = (Wg @ Cz[:3, :3].T + Cz[:3, 3])[:, 2].reshape(gshape)
        keep = zt >= tz - 1e-6
    else:
        keep = np.ones(gshape, bool)
    mvm = mvm_full & keep
    Tg = X.apply_once(T, tpl_aff, gshape, gaff, X.Map(L=np.eye(4)), order=1)
    Tmg = X.apply_once(Tm.astype(np.float32), tpl_aff, gshape, gaff, X.Map(L=np.eye(4)), order=1) > 0.5
    P = {n: f"{d}/_{n}.nii.gz" for n in ("fixed", "fixed_mask", "moving_rig", "moving_rig_mask")}
    nib.save(nib.Nifti1Image((Tg * Tmg).astype(np.float32), gaff), P["fixed"])
    nib.save(nib.Nifti1Image(Tmg.astype(np.uint8), gaff), P["fixed_mask"])
    nib.save(nib.Nifti1Image((mv * mvm).astype(np.float32), gaff), P["moving_rig"])
    nib.save(nib.Nifti1Image(mvm.astype(np.uint8), gaff), P["moving_rig_mask"])
    res = float(c["grid_res_mm"])
    info = dict(margin_mm=margin, grid=list(gshape), grid_affine=gaff.tolist(), res=res, cut_z_mm=float(tz),
                brainstem_cropped_mm3=round(float((mvm_full & ~keep).sum() * res ** 3), 1))
    return P, info


# ------------------------------------------------------------------------------------------------ the whole route
def run(subject, b0_path, mask_path, template_b0, template_mask, placement_mat, placed_b0, out, *, registry=None, bridge_json=None,
        device="cuda:0", overwrite=False):
    S, c = subject, cfg()
    entry = xfm.xfm_dir(SRC, DST, S, registry)                       # fail before the GPU run, not after it
    if (entry / "meta.json").exists() and not overwrite:
        raise SystemExit(f"{entry} exists: pass --overwrite to replace it (the old entry is backed up by code/shared/xfm.py)")
    d = Path(out); d.mkdir(parents=True, exist_ok=True); d = str(d)
    b0n = nib.load(str(b0_path)); mkn = nib.load(str(mask_path))
    tpl = nib.load(str(template_b0)); T = np.asanyarray(tpl.dataobj).astype(np.float32)
    Tm = np.asanyarray(nib.load(str(template_mask)).dataobj) > 0.5
    P0 = placement(b0n, placement_mat, placed_b0, bridge_json)
    mk = np.asanyarray(mkn.dataobj) > 0
    b0 = np.asanyarray(b0n.dataobj).astype(np.float32); b0_native = b0
    fr = field_ratio(b0, mk, c["n4_field_sigma_vox"])
    n4info = dict(applied=False, residual_field_before=round(fr, 3), threshold=float(c["n4_field_max"]))
    if fr > float(c["n4_field_max"]):
        b0, n4info = n4_b0(S, b0_path, mask_path, mk, d, c)
        n4info.update(applied=True, threshold=float(c["n4_field_max"]), used_for="deformable (the linear fits use the native b0)")
    Pst, Sm, lin = linear_start(b0_native, b0n.affine, mk, T, tpl.affine, Tm, P0, c)
    P, ginfo = engine_inputs(b0, b0n.affine, mk, T, tpl.affine, Tm, P0, Sm, Pst, float(c["grid_margin_mm"]), c, d)
    eng = f"{d}/_engine"; shutil.rmtree(eng, ignore_errors=True); t0 = time.time()
    m = featurereg.register(P["fixed"], P["moving_rig"], P["fixed_mask"], P["moving_rig_mask"], eng, affine=None, device=device,
                            **c["engine"])                              # stops first, with one line, if the GPU is short of memory
    wall = time.time() - t0
    warp, start = f"{d}/{S}_warp.nii.gz", f"{d}/{S}_start_pull.txt"
    shutil.copyfile(m["warp"], warp); np.savetxt(start, np.linalg.inv(Pst))
    # products on the template grid by one pull: the b0 (cubic) and the UNCROPPED mask (linear > 0.5); the outline check of the mask
    dd, da = X.read_fireants_field(warp)
    full = X.Map(L=np.linalg.inv(Pst), d=dd, d_affine=da)
    outv = X.apply_once(b0, b0n.affine, tpl.shape, tpl.affine, full, order=3)
    outm = X.apply_once(mk.astype(np.float32), b0n.affine, tpl.shape, tpl.affine, full, order=1) > 0.5
    nib.save(nib.Nifti1Image(outv * outm, tpl.affine), f"{d}/{S}_b0_in_template.nii.gz")
    nib.save(nib.Nifti1Image(outm.astype(np.uint8), tpl.affine), f"{d}/{S}_mask_in_template.nii.gz")
    ol = qc.outline_regions(outm, Tm, tpl.affine); st, why = qc.outline_gate(ol)
    shutil.rmtree(f"{d}/_engine", ignore_errors=True)
    for k in ("moving_rig", "moving_rig_mask"):
        os.remove(P[k])
    # the registry entry (PULL order: the warp on the working grid, then the start)
    e = xfm.register(SRC, DST, [("warp", "itk_lps_disp", warp), ("affine", "ras_pull", start)], subject=S, dst_grid=template_b0,
                     src_grid=b0_path, engine=engine_text(c),
                     config=dict(recipe="settings/link.yaml b0_to_template", guard=lin["guard"], margin_mm=ginfo["margin_mm"], res=ginfo["res"], mask=str(mask_path)),
                     root=registry, overwrite=overwrite)
    xfm.set_qc(e.dir, "pass" if st in ("PASS", "WARN") else "fail", outline=st, outline_reasons=why,
               source="template.link.b0_to_template (outline check of the uncropped moved mask)")
    rec = dict(subject=S, entry=str(e.dir), outline_status=st, outline_reasons=why, outline={k: ol[k] for k in qc.REGIONS},
               wall_s=round(wall, 1), mask=str(mask_path), n4=n4info, P0=P0.tolist(), start_pull=np.linalg.inv(Pst).tolist(),
               engine_metrics={k: v for k, v in m.items() if k not in ("moved", "warp", "argv")}, **ginfo,
               similarity_for_cut=lin["similarity"], lin_mode=lin["lin_mode"], affine_fit=lin["affine_fit"], guard=lin["guard"])
    Path(f"{d}/{S}_b0_to_template.json").write_text(json.dumps(rec, indent=1, default=float))
    print(f"{S}: outline {st} {why} | guard {lin['guard']} | margin {ginfo['margin_mm']} grid {ginfo['grid']} | engine {wall:.0f} s "
          f"-> {e.dir}", flush=True)
    return rec


def main(argv=None, prog=None):
    ap = argparse.ArgumentParser(prog=prog, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True, help="LINC identifier, e.g. MR252")
    ap.add_argument("--b0", default=None, help="native b0 (default $LINC_DATA/sub-<ID>/mri/<ID>_mri_b0_native.nii.gz)")
    ap.add_argument("--mask", default=None, help="its brain mask of record (default .../<ID>_mri_mask_native.nii.gz)")
    ap.add_argument("--template-b0", required=True, help="the MRI template's b0 channel (linc_mri_b0.nii.gz)")
    ap.add_argument("--template-mask", required=True, help="the MRI template's brain mask (linc_mri_brainmask.nii.gz)")
    ap.add_argument("--placement-mat", default=None, help="FLIRT 6-dof placement of the b0 in the run-113 frame (default .../<ID>_mri2template.mat)")
    ap.add_argument("--placed-b0", default=None, help="its FLIRT reference output (default .../<ID>_mri.nii.gz)")
    ap.add_argument("--place-b0", default=None, metavar="FRAME_0P5",
                    help="make the placement instead (blockface.registration.to_own_mri.place_b0 onto this run-113 0.5 mm frame) into OUT/placement")
    ap.add_argument("--bridge", default=None, help="run-113 -> canonical bridge JSON (default settings/link.yaml inputs.bridge)")
    ap.add_argument("--registry", default=None, help="parent folder of the entry (default $LINC_DATA/sub-<ID>/xfm)")
    ap.add_argument("--out", default=None, help="working files and products (default $LINC_WORK/b0_to_template/<ID>)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing entry (backed up by code/shared/xfm.py)")
    a = ap.parse_args(argv)
    S = a.subject
    md = paths.subject_dir(S) / "mri" if not (a.b0 and a.mask and ((a.placement_mat and a.placed_b0) or a.place_b0)) else None
    b0 = a.b0 or str(md / f"{S}_mri_b0_native.nii.gz")
    mask = a.mask or str(md / f"{S}_mri_mask_native.nii.gz")
    out = Path(a.out or (paths.work_root() / "b0_to_template" / S))
    if a.place_b0:
        from blockface.registration import to_own_mri
        r = to_own_mri.place_b0(b0, mask, a.place_b0, str(out / "placement"), S)
        mat, placed = r["mri2template_mat"], r["placed_b0"]
    else:
        mat = a.placement_mat or str(md / f"{S}_mri2template.mat")
        placed = a.placed_b0 or str(md / f"{S}_mri.nii.gz")
    run(S, b0, mask, a.template_b0, a.template_mask, mat, placed, out, registry=a.registry, bridge_json=a.bridge, device=a.device,
        overwrite=a.overwrite)


if __name__ == "__main__":
    main()
