#!/usr/bin/env python3
"""The link: the blockface (dual-channel) template's b0 channel registered to the MRI template's b0 channel (paper: Methods,
"Registration of the blockface template into the MRI template"), and its export as pull maps in both directions.

Port of the analysis project's mri/code/s5_b0link.py --arm sym (the run of record bftemplate/cv/s5_direct_d2bs8/sym) and
mri/code/s5_export_maps.py (bftemplate/cv/template2template/d2bs8_s5sym). MOVING = the blockface template's b0 (its other channels are
carried with the same map), FIXED = the MRI template's b0; one pull map carries the blockface template onto the MRI template grid.

Recipe (configs/link.yaml `register`):
  1. masked inputs; start pose P0 = the frame bridge (run-113 frame -> canonical frame, rigid);
  2. 7-dof similarity, then a 12-dof affine (NCC, Powell, images downsampled x2), the affine rejected when its anisotropy > 1.5
     (similarity only): Pst = A S P0 (moving world -> fixed world);
  3. ONE resample of both images onto a working grid centered on both brains with 8 mm of zeros, 0.35 mm, sizes x16 (moving: cubic,
     mask linear > 0.5; fixed: linear); the moving input is zeroed below the fixed brain's lowest axial slice (registration input only);
  4. the symmetric engine (linc/link/symreg.py, A = fixed, B = placed moving, S = identity, 0.35 mm, warp smoothing 1.25 voxels);
  5. pull = a2b on the working grid, then Pst^-1; outputs on the fixed grid and the scores (outline check, b0 NCC at 0.5 mm FWHM,
     envelope Dice, interior edge distance, Jacobian, working-grid edge checks).

    python -m linc.link.register run     --out DIR [--device cuda:0]    (= prepare + solve + finish)
    python -m linc.link.register prepare --out DIR                      (CPU: steps 1-3, writes the engine inputs)
    python -m linc.link.register solve   --out DIR [--device cuda:0]    (GPU: step 4, DIR/sym/)
    python -m linc.link.register finish  --out DIR                      (CPU: step 5, DIR/s5_link.json)
    python -m linc.link.register export  LINK_DIR OUT_DIR               (CPU: the two full pull maps + round trips)
Inputs default to configs/link.yaml `inputs` (relative to $LINC_PRODUCTS); --moving/--fixed/... override them.

Products (as the run of record): link_linear_world.txt (Pst), link_linear_vs_bridge.txt (Pst P0^-1), link_warp.nii.gz (the pull
displacement a2b(x) - x on the working grid, ITK/FireANTs x/y LPS-signed), moving_b0_on_fixed, moving_mask_on_fixed (uncropped mask),
carried_<name>_on_fixed, s5_link.json, sym/ (the engine run).
Not ported: the greedy arm of s5_b0link.py (the one-way feature-based engine, a cross-check read by linc/link/estimate.py compare) and
the analysis project's QC-store bookkeeping (regqc rows, PNGs); the record's numbers are computed by linc/link/qc.py.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import nibabel as nib

from ..registration import xfmtools as X
from . import qc
from .common import apply_affine, bridge, chunked, config, inputs, product


def aniso(L):
    s = np.linalg.svd(L[:3, :3], compute_uv=False); return float(s.max() / s.min())


def _load_inputs(moving, moving_mask, fixed, fixed_mask):
    mi, mmi = nib.load(str(moving)), nib.load(str(moving_mask)); fi, fmi = nib.load(str(fixed)), nib.load(str(fixed_mask))
    Mv = np.asanyarray(mi.dataobj).astype(np.float32); Mm = np.asanyarray(mmi.dataobj) > 0.5
    Fv = np.asanyarray(fi.dataobj).astype(np.float32); Fm = np.asanyarray(fmi.dataobj) > 0.5
    return mi, fi, Mv * Mm, Mm, Fv * Fm, Fm                                     # masked inputs (standing rule)


def _start(init, bridge_path=None):
    if init == "bridge":
        return bridge(bridge_path)
    return np.eye(4) if init == "identity" else np.loadtxt(init)


def prepare(a):
    """Linear start, working grid, brainstem cut; writes the engine inputs _A, _Amask, _B, _Bmask, _S.txt and prepare.json."""
    c = config()["register"]
    d = Path(a.out); d.mkdir(parents=True, exist_ok=True)
    mi, fi, Mv, Mm, Fv, Fm = _load_inputs(a.moving, a.moving_mask, a.fixed, a.fixed_mask)
    MA = mi.affine.copy()
    P0 = _start(a.init, a.bridge)
    m0 = X.apply_once(Mv, MA, fi.shape, fi.affine, X.Map(L=np.linalg.inv(P0)), order=1)
    mm0 = X.apply_once(Mm.astype(np.float32), MA, fi.shape, fi.affine, X.Map(L=np.linalg.inv(P0)), order=1) > 0.5
    Sm, si = X.linear_register(Fv, fi.affine, Fm, m0, fi.affine, mm0, "similarity", factor=c["linear_factor"])
    m1 = X.apply_once(Mv, MA, fi.shape, fi.affine, X.Map(L=np.linalg.inv(Sm @ P0)), order=1)
    mm1 = X.apply_once(Mm.astype(np.float32), MA, fi.shape, fi.affine, X.Map(L=np.linalg.inv(Sm @ P0)), order=1) > 0.5
    Am, ai = X.linear_register(Fv, fi.affine, Fm, m1, fi.affine, mm1, "affine", factor=c["linear_factor"])
    an = aniso(Am @ Sm); use_aff = an <= c["affine_anisotropy_max"]
    Pst = (Am @ Sm @ P0) if use_aff else (Sm @ P0)
    # working grid: centered on both brains + margin, res, sizes x multiple
    mw = np.argwhere(Mm) @ MA[:3, :3].T + MA[:3, 3]; mw = mw @ Pst[:3, :3].T + Pst[:3, 3]
    fw = np.argwhere(Fm) @ fi.affine[:3, :3].T + fi.affine[:3, 3]
    gshape, gaff = qc.grid_from(np.minimum(mw.min(0), fw.min(0)) - c["grid_margin_mm"], np.maximum(mw.max(0), fw.max(0)) + c["grid_margin_mm"],
                                c["grid_res_mm"], c["grid_multiple"])
    lin = X.Map(L=np.linalg.inv(Pst))
    mv = X.apply_once(Mv, MA, gshape, gaff, lin, order=3); mvm_full = X.apply_once(Mm.astype(np.float32), MA, gshape, gaff, lin, order=1) > 0.5
    fg = X.apply_once(Fv, fi.affine, gshape, gaff, X.Map(), order=1); fgm = X.apply_once(Fm.astype(np.float32), fi.affine, gshape, gaff, X.Map(), order=1) > 0.5
    kz = np.arange(gshape[2]) * gaff[2, 2] + gaff[2, 3]; tz = kz[np.where(fgm.any((0, 1)))[0]].min()
    keep = np.broadcast_to(kz[None, None, :] >= tz - 1e-6, gshape) if c["brainstem_cut"] else np.ones(gshape, bool)
    mvm = mvm_full & keep                                                    # brainstem cut, moving input only
    e_lin = qc.edge(mvm_full, gaff)
    nib.save(nib.Nifti1Image((fg * fgm).astype(np.float32), gaff), str(d / "_A.nii.gz")); nib.save(nib.Nifti1Image(fgm.astype(np.uint8), gaff), str(d / "_Amask.nii.gz"))
    nib.save(nib.Nifti1Image((mv * mvm).astype(np.float32), gaff), str(d / "_B.nii.gz")); nib.save(nib.Nifti1Image(mvm.astype(np.uint8), gaff), str(d / "_Bmask.nii.gz"))
    np.savetxt(d / "_S.txt", np.eye(4))                                      # B is already linearly placed on A's grid
    st = dict(P0=P0.tolist(), Pst=Pst.tolist(), grid=list(gshape), grid_affine=gaff.tolist(), similarity=si, affine_fit=ai,
              affine_anisotropy=round(an, 4), affine_used=bool(use_aff), res=c["grid_res_mm"], margin_mm=c["grid_margin_mm"],
              brainstem_cut_z=round(float(tz), 2), cropped_mm3=round(float((mvm_full & ~keep).sum() * c["grid_res_mm"] ** 3), 1),
              edge_after_linear_start=e_lin, init=a.init)
    (d / "prepare.json").write_text(json.dumps(st, indent=1, default=float))
    print(f"prepare: similarity NCC {si['ncc']:.4f}, affine NCC {ai['ncc']:.4f}, anisotropy {an:.4f} (used {use_aff}); grid {gshape}", flush=True)
    return st


def solve(a):
    """The symmetric engine on the prepared inputs (GPU)."""
    from . import symreg
    e = config()["register"]["engine"]
    d = Path(a.out); t0 = time.time()
    symreg.main([str(d / "_A.nii.gz"), str(d / "_Amask.nii.gz"), str(d / "_B.nii.gz"), str(d / "_Bmask.nii.gz"), str(d / "_S.txt"), str(d / "sym"),
                 "--res", str(e["res"]), "--smooth-warp-sigma", str(e["smooth_warp_sigma"]), "--channel-chunk", str(e["channel_chunk"]),
                 "--max-gpu-frac", str(e["max_gpu_frac"]), "--device", a.device])
    (d / "solve.json").write_text(json.dumps(dict(wall_s=round(time.time() - t0, 1))))


def finish(a, keep_inputs=False):
    """Pull map, outputs on the fixed grid, scores -> s5_link.json."""
    from . import symreg
    c = config()["register"]; n = c["point_chunk"]
    d = Path(a.out); st = json.loads((d / "prepare.json").read_text()); t0 = time.time()
    mi, fi, Mv, Mm, Fv, Fm = _load_inputs(a.moving, a.moving_mask, a.fixed, a.fixed_mask)
    MA = mi.affine.copy()
    P0, Pst = np.array(st["P0"]), np.array(st["Pst"]); gshape, gaff = tuple(st["grid"]), np.array(st["grid_affine"])
    a2b, _ = symreg.maps(a.run or d / "sym")
    Pi = np.linalg.inv(Pst)
    pull = lambda x: apply_affine(Pi, chunked(a2b, x, n))                    # noqa: E731  fixed world -> moving world
    Xg = X.grid_world(gshape, gaff); Z = chunked(a2b, Xg, n) - Xg
    Z[:, :2] *= -1; nib.save(nib.Nifti1Image(Z.reshape(tuple(gshape) + (1, 3)).astype(np.float32), gaff), str(d / "link_warp.nii.gz"))
    del Xg, Z
    Jg = qc.detj(d / "link_warp.nii.gz")[0]
    if not keep_inputs:
        for f in ("_A", "_Amask", "_B", "_Bmask"):
            (d / f"{f}.nii.gz").unlink(missing_ok=True)
    solve_wall = json.loads((d / "solve.json").read_text())["wall_s"] if (d / "solve.json").exists() else 0.0
    wall = solve_wall + time.time() - t0
    np.savetxt(d / "link_linear_world.txt", Pst); np.savetxt(d / "link_linear_vs_bridge.txt", Pst @ np.linalg.inv(P0))
    out = X.apply_once(Mv, MA, fi.shape, fi.affine, pull, order=3)
    outm = X.apply_once(Mm.astype(np.float32), MA, fi.shape, fi.affine, pull, order=1) > 0.5             # UNCROPPED moving mask
    nib.save(nib.Nifti1Image((out * outm).astype(np.float32), fi.affine), str(d / "moving_b0_on_fixed.nii.gz"))
    nib.save(nib.Nifti1Image(outm.astype(np.uint8), fi.affine), str(d / "moving_mask_on_fixed.nii.gz"))
    for name, p in a.carry.items():                                          # extra channels of the moving template, same map
        ci = nib.load(str(p))
        nib.save(nib.Nifti1Image(X.apply_once(np.asanyarray(ci.dataobj).astype(np.float32), ci.affine, fi.shape, fi.affine, pull, order=3).astype(np.float32),
                                 fi.affine), str(d / f"carried_{name}_on_fixed.nii.gz"))
    outm_g = X.apply_once(Mm.astype(np.float32), MA, gshape, gaff, pull, order=1) > 0.5
    e_reg = qc.edge(outm_g, gaff, Jg if Jg.shape == tuple(gshape) else None)
    ol = qc.outline_regions(outm, Fm, fi.affine); status, why = qc.outline_gate(ol)
    row = qc.global_qc(a.fixed, d / "moving_b0_on_fixed.nii.gz", fixed_mask=a.fixed_mask, moved_mask=d / "moving_mask_on_fixed.nii.gz",
                       warp=d / "link_warp.nii.gz", ncc_fwhm_mm=c["qc"]["ncc_fwhm_mm"])
    md5 = lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest()           # noqa: E731
    rec = dict(arm="sym", moving=str(a.moving), fixed=str(a.fixed), dry_frame=False, init=st["init"], similarity=st["similarity"],
               affine_fit=st["affine_fit"], affine_anisotropy=st["affine_anisotropy"], affine_used=st["affine_used"], grid=st["grid"],
               res=st["res"], margin_mm=st["margin_mm"], brainstem_cut_z=st["brainstem_cut_z"], cropped_mm3=st["cropped_mm3"],
               wall_s=round(wall, 1), outline_status=status, outline_reasons=why, outline={k: ol[k] for k in qc.REGIONS},
               ncc=row["ncc"], envelope_dice=row["envelope_dice"], edge_mean=(row.get("regeval") or {}).get("edge_mean"),
               jacobian={k: (row.get("jacobian") or {}).get(k) for k in ("sd_log_detJ", "jac_min", "jac_max", "jac_out_0p5_2_pct", "folds_n_in_mask")},
               edge_after_linear_start=st["edge_after_linear_start"], edge_after_registration=e_reg,
               mask_md5={str(p): md5(p) for p in (a.moving_mask, a.fixed_mask)})
    (d / "s5_link.json").write_text(json.dumps(rec, indent=1, default=float))
    print(f"link: outline {status} {why} | NCC {row['ncc']:.3f} Dice {row['envelope_dice']:.3f} edge_mean {rec['edge_mean']} | "
          f"edge {e_reg} | {wall:.0f} s", flush=True)
    return rec


def export(link, bf_grid, bf_mask, out, mri_grid, mri_mask, every=None):
    """The link as two full pull maps (xfmtools convention src_world = L (x + d(x)), L = identity) plus round trips (export.json):
      <out>/bftemplate__to__mri-template/full_*   pull MRI-template point -> BF-template point (displacement on the MRI template grid)
      <out>/mri-template__to__bftemplate/full_*   pull BF-template point -> MRI-template point (displacement on the BF template grid)
    MRI y -> BF: Pst^-1 (a2b(y));  BF x -> MRI: b2a(Pst x)."""
    from . import symreg
    c = config()["register"]; n = c["point_chunk"]; every = every or c["export_roundtrip_every"]
    link = str(link); out = str(out)
    Pst = np.loadtxt(f"{link}/link_linear_world.txt"); Pi = np.linalg.inv(Pst)
    if not os.path.isdir(f"{link}/sym"):
        raise SystemExit(f"{link}/sym not found: only the symmetric engine's links are exported here")
    a2b, b2a = symreg.maps(f"{link}/sym"); engine = "symmetric (symreg_upstream)"
    mri2bf = lambda y: apply_affine(Pi, np.asarray(a2b(y)))                 # noqa: E731
    bf2mri = lambda x: np.asarray(b2a(apply_affine(Pst, x)))                # noqa: E731
    maps = {}
    for name, fn, grid in (("bftemplate__to__mri-template", mri2bf, mri_grid), ("mri-template__to__bftemplate", bf2mri, bf_grid)):
        g = nib.load(str(grid)); G = X.grid_world(g.shape[:3], g.affine); D = (chunked(fn, G, n) - G).reshape(tuple(g.shape[:3]) + (3,))
        X.save_map(f"{out}/{name}", "full", X.Map(L=np.eye(4), d=D.astype(np.float32), d_affine=g.affine),
                   meta=dict(source_link=link, engine=engine, direction=("pull: MRI-template point -> BF-template point" if name.startswith("bftemplate")
                                                                         else "pull: BF-template point -> MRI-template point"), grid=str(grid)))
        del G, D
        maps[name] = X.load_map(f"{out}/{name}", "full")
    rt = {}
    for nm, mk, f1, f2 in (("mri_to_bf_to_mri_in_mri_mask", mri_mask, maps["bftemplate__to__mri-template"], maps["mri-template__to__bftemplate"]),
                           ("bf_to_mri_to_bf_in_bf_mask", bf_mask, maps["mri-template__to__bftemplate"], maps["bftemplate__to__mri-template"])):
        m = nib.load(str(mk)); P = (np.argwhere(np.asanyarray(m.dataobj) > 0.5)[::every]) @ m.affine[:3, :3].T + m.affine[:3, 3]
        e = np.linalg.norm(chunked(f2, chunked(f1, P, n), n) - P, axis=1)
        rt[nm] = dict(p50=round(float(np.median(e)), 4), p99=round(float(np.percentile(e, 99)), 4), max=round(float(e.max()), 4), n=int(len(P)))
    json.dump(dict(source_link=link, engine=engine, bf_grid=str(bf_grid), bf_mask=str(bf_mask), round_trip_mm=rt, script="linc/link/register.py export"),
              open(f"{out}/export.json", "w"), indent=1)
    print(engine, rt, flush=True)
    return rt


def main(argv=None):
    I = inputs()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "prepare", "solve", "finish"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--moving", default=None); p.add_argument("--moving-mask", default=None)
        p.add_argument("--fixed", default=None); p.add_argument("--fixed-mask", default=None)
        p.add_argument("--init", default=config()["register"]["init"], help="bridge | identity | <4x4 txt, moving world -> fixed world>")
        p.add_argument("--bridge", default=None, help="bridge JSON (default: configs/link.yaml inputs.bridge)")
        p.add_argument("--carry", nargs="*", default=None, help="name=path ... (default: inputs.bf_carry)")
        p.add_argument("--run", default=None, help="finish: the engine run folder (default: OUT/sym)")
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--keep-inputs", action="store_true", help="finish: keep the engine inputs _A/_B (default: removed, as the record)")
    p = sub.add_parser("export")
    p.add_argument("link"); p.add_argument("out")
    p.add_argument("--bf-grid", default=None); p.add_argument("--bf-mask", default=None)
    p.add_argument("--mri-grid", default=None); p.add_argument("--mri-mask", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "export":
        return export(a.link, a.bf_grid or product(I["bf_b0"]), a.bf_mask or product(I["bf_mask"]), a.out,
                      a.mri_grid or product(I["mri_b0"]), a.mri_mask or product(I["mri_mask"]))
    a.moving = a.moving or product(I["bf_b0"]); a.moving_mask = a.moving_mask or product(I["bf_mask"])
    a.fixed = a.fixed or product(I["mri_b0"]); a.fixed_mask = a.fixed_mask or product(I["mri_mask"])
    a.carry = dict(c.split("=", 1) for c in a.carry) if a.carry is not None else {k: product(v) for k, v in I["bf_carry"].items()}
    if a.cmd in ("run", "prepare"):
        prepare(a)
    if a.cmd in ("run", "solve"):
        solve(a)
    if a.cmd in ("run", "finish"):
        finish(a, keep_inputs=a.keep_inputs)


if __name__ == "__main__":
    main()
