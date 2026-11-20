#!/usr/bin/env python3
"""The construction estimate of the link, and its comparison with the link (paper: Methods, "Registration of the blockface template
to the MRI template": "a second mapping from the blockface template to the MRI template").

Port of the analysis project's mri/code/s5_compose.py (final path: the member list of the dual-channel template, --members-json, with
the inverse guard) and mri/code/s5_compare.py; runs of record bftemplate/cv/s5_compose_d2bs8_r05_guard (all brain members),
s5_tpl_d2bs8_r05_guard (the MRI template's own member, --only TPL) and bftemplate/cv/s5_compare_d2bs8.json.

compose. For every member i of the dual-channel template, a point y of the MRI template is pulled to the dual-channel template through
    y --(mirror members: M_c, the canonical x-mirror)--> the member's own registration into the MRI template, pulled:
        brains with MRI ('r5'):   b0-native -> mri-template (registry), then b0-native -> run-113 placement (registry, used forward)
        brains without MRI ('s2'): their direct placement mri-template -> bf-native, then the rigid T (blockface -> run-113)
        the MRI template's own b0 member ('tpl'): the frame bridge, inverted
      --(mirror members: M_r, the run-113 x-mirror)--> the member's input world --> inverse of the member's construction warp
    = x_i(y), a point of the dual-channel template.
The inverse of each construction warp is computed pointwise (fixed-point iteration, 40 iterations, tol 1e-3); GUARD (0.1 mm): a point is
dropped for that member when its round trip |warp(x) - q| > 0.1 mm or its solution x lies outside the warp's grid (the step-8 box, where
the warp is identity and the iteration converges to a spurious solution). Estimate = the per-point mean of x_i(y) over the members that
kept the point (displacement mean); spread = RMS distance of the x_i from the mean. Scores: the dual-channel template's mask carried onto
the MRI grid vs the MRI template mask (outline check), the estimate's symmetry (link(M_c y) vs M_r link(y)), unmirrored vs all, groups.

compare. Per MRI-template brain voxel (every 8th), |x_estimate(y) - x_link(y)| for each direct link (greedy, sym) and each extra
estimate (tpl), with the estimate's spread for scale.

    python -m template.link.estimate compose --out DIR [--only TPL] [-j 8]        (CPU; inputs from settings/link.yaml)
    python -m template.link.estimate compare ESTIMATE_DIR LINK_ROOT OUT.json [--extra tpl=DIR]
Needs $LINC_DATA (the brains' registrations, code/shared/xfm.py registry) and $LINC_PRODUCTS (settings/link.yaml inputs).

Differences from the analysis script: only the final path is ported (no --run/--step build-folder mode); members are accumulated in
member order (the analysis script, with -j > 1, added them in completion order; float64 sums: on the record, member order changes
2 of 2M spread voxels by 6e-8 mm and the summary percentiles at 1e-14 mm; --order-from reproduces a record's order bit-exactly);
`points=` evaluates a subset of the estimate's grid (validation).
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import nibabel as nib

from shared import xfm
from blockface.registration import xfmtools as X
from . import qc
from template.link.common import bridge, config, inputs, product

MIRX = np.array([-1, 1, 1])


def target_grid(res, mri_mask):
    """The MRI template grid resampled to `res` (same field of view, voxel centers shifted to keep the box)."""
    m = nib.load(str(mri_mask)); A = m.affine; f = abs(A[0, 0]) / res
    shp = tuple(int(round(s * f)) for s in m.shape)
    B = A.copy(); B[:3, :3] = A[:3, :3] / f; B[:3, 3] = A[:3, 3] - np.diag(A)[:3] / 2 + np.diag(B)[:3] / 2
    return shp, B


def inv_pull(warp, src_grid, q, guard_mm, iters, tol, chunk):
    """Pointwise inverse of a member's construction warp (fsl_rel) at points q; with guard_mm > 0, points whose round trip exceeds it or
    whose solution lies outside the field's grid are NaN."""
    W = xfm.Warp(warp, "fsl_rel", src_grid=src_grid); inv = W.inverse(iters=iters, tol=tol)
    x = np.concatenate([np.asarray(inv.pull(q[k:k + chunk])) for k in range(0, len(q), chunk)])
    if guard_mm > 0:
        r = np.concatenate([np.linalg.norm(np.asarray(W.pull(x[k:k + chunk])) - q[k:k + chunk], axis=1) for k in range(0, len(q), chunk)])
        v = x @ W.V[:3, :3].T + W.V[:3, 3]; sh = np.array(W.d[0].shape)
        x[(r > guard_mm) | ~np.all((v >= 0) & (v <= sh - 1), axis=1)] = np.nan
    return x


def has_r5(S):
    try:
        return bool(xfm.find_chain("b0-native", "mri-template", subject=S))
    except LookupError:
        return False


def member_points(m, Y, est=None, bridge_path=None):
    """x_i(Y) for one member spec (see members())."""
    est = est or config()["estimate"]
    g = dict(guard_mm=est["guard_mm"], iters=est["inverse_iters"], tol=est["inverse_tol_mm"], chunk=est["point_chunk"])
    Yc = Y * MIRX if m["mirror"] else Y                                      # M_c
    if m["kind"] == "tpl":                                                   # the MRI template itself, placed in the run-113 frame
        Ri = np.linalg.inv(bridge(bridge_path)); q = Yc @ Ri[:3, :3].T + Ri[:3, 3]
        return inv_pull(m["warp"], m["src_grid"], q, **g)
    if m["kind"] == "r5":
        b = xfm.Chain.between("b0-native", "mri-template", subject=m["subject"]).pull(Yc)
        p = xfm.Chain.between("mri-run113", "b0-native", subject=m["subject"]).pull(b)   # b0-native -> run-113 world
    else:                                                                    # s2: mri-template -> bf-native (pull), then the rigid T
        b = xfm.Entry(m["s2_entry"]).pull(Yc)
        T = np.asarray(m["T"]); p = b @ T[:3, :3].T + T[:3, 3]
    q = p * MIRX if m["mirror"] else p                                       # M_r: into the mirrored frame
    return inv_pull(m["warp"], m["src_grid"], q, **g)


def members(path=None):
    """Member specs from the dual-channel template's member list (d2_member_warps.json): sid, subject, mirror, warp, src_grid,
    kind ('r5': brain with MRI; 's2': brain without MRI; 'tpl': the MRI template's own b0 member), s2_entry, T."""
    I = inputs()
    path = Path(path) if path else product(I["member_warps"])
    J = json.loads(path.read_text())
    rows = J["members"] if isinstance(J, dict) and "members" in J else (J["data"] if isinstance(J, dict) and "data" in J else J)
    base = path.parent
    out = []
    for r in rows:
        if r.get("kind") == "template" and r["member"] != "TPL":            # external templates are not members of the composition
            continue
        mir = bool(r["mirror"]); S = r.get("subject") or (r["member"][:-1] if mir else r["member"])
        w = r["file"] if os.path.isabs(r["file"]) else str(base / r["file"])
        if S == "TPL":
            out.append(dict(sid="TPL", subject="TPL", mirror=False, warp=w, src_grid=r["src_grid"], kind="tpl"))
            continue
        src = r.get("src_grid") or str(product(I["member_src_grid"], S=S, flip="_flip" if mir else ""))
        assert r.get("convention", "fsl_rel") == "fsl_rel", r
        if has_r5(S):
            out.append(dict(sid=r["member"], subject=S, mirror=mir, warp=w, src_grid=src, kind="r5"))
        else:
            T = json.loads(product(I["s2_linear"], S=S).read_text())["T_world_bf_to_run113"]
            out.append(dict(sid=r["member"], subject=S, mirror=mir, warp=w, src_grid=src, kind="s2", T=T,
                            s2_entry=str(product(I["s2_entry"], S=S))))
    return out


def compose(out, template_mask, *, members_json=None, mri_mask=None, res=None, groups=None, only=None, exclude=None, jobs=1,
            points=None, accumulate_order=None, write=True):
    """The construction estimate (see the module docstring). points: optional flat indices into the estimate grid (subset evaluation:
    returns the per-point arrays and writes nothing). accumulate_order: optional member order of the float sums (validation)."""
    est = config()["estimate"]
    res = est["res_mm"] if res is None else res
    groups = est["groups"] if groups is None else groups
    exclude = est["exclude"] if (exclude is None and only is None) else (exclude or [])
    mri_mask = mri_mask or product(inputs()["mri_mask"])
    specs = members(members_json)
    if only:
        specs = [m for m in specs if m["sid"] in only]
    if exclude:
        specs = [m for m in specs if m["sid"] not in exclude]
    shp, G = target_grid(res, mri_mask); Yall = X.grid_world(shp, G)
    Y = Yall if points is None else Yall[points]
    t0 = time.time()
    names = [m["sid"] for m in specs]
    gset = {g: set(v) for g, v in groups.items()}
    acc = {k: [np.zeros((len(Y), 3)), np.zeros(len(Y))] for k in ["all", "unmirrored"] + [f"g:{g}" for g in gset] + [f"r:{g}" for g in gset]}
    sq = np.zeros(len(Y)); n_bad = {}

    def add(name, X_):
        X_ = X_.astype(np.float64); ok = np.isfinite(X_).all(1); X_ = np.where(ok[:, None], X_, 0.0)
        keys = ["all"] + (["unmirrored"] if not name.endswith("f") else [])
        keys += [f"g:{g}" if name.rstrip("f") in m else f"r:{g}" for g, m in gset.items()]
        for k in keys:
            acc[k][0] += X_; acc[k][1] += ok
        sq[:] += (X_ ** 2).sum(1)
        n_bad[name] = int((~ok).sum())

    results = {}
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(jobs) as ex:
            futs = {ex.submit(member_points, m, Y, est): m["sid"] for m in specs}
            for f in as_completed(futs):
                results[futs[f]] = f.result(); print(f"  {futs[f]}: {time.time() - t0:.0f} s", flush=True)
    else:
        for m in specs:
            results[m["sid"]] = member_points(m, Y, est); print(f"  {m['sid']}: {time.time() - t0:.0f} s", flush=True)
    for name in (accumulate_order or names):                                 # deterministic accumulation order
        add(name, results.pop(name))
    cnt = acc["all"][1]; mean = acc["all"][0] / np.maximum(cnt, 1)[:, None]
    mean[cnt == 0] = Y[cnt == 0]                                             # no converged member: identity (flagged by n_valid = 0)
    spread = np.sqrt(np.maximum(sq / np.maximum(cnt, 1) - (mean ** 2).sum(1), 0))
    spread[cnt == 0] = 0.0
    mean_un = acc["unmirrored"][0] / np.maximum(acc["unmirrored"][1], 1)[:, None]
    if points is not None:
        return dict(Y=Y, mean=mean, spread=spread, count=cnt, mean_unmirrored=mean_un, dropped=n_bad, grid=shp, grid_affine=G)
    os.makedirs(out, exist_ok=True)
    vol = lambda v: v.reshape(shp + v.shape[1:])                             # noqa: E731
    D = vol(mean - Y); nib.save(nib.Nifti1Image(D.astype(np.float32), G), f"{out}/link_ras_disp.nii.gz")      # x_src = y + d(y)
    nib.save(nib.Nifti1Image(vol(spread).astype(np.float32), G), f"{out}/link_spread_mm.nii.gz")
    tm = nib.load(str(template_mask))
    carried = X.apply_once(np.asanyarray(tm.dataobj).astype(np.float32), tm.affine, shp, G, X.Map(d=D, d_affine=G), order=1) > 0.5
    carried &= vol(cnt > 0) if est["guard_mm"] > 0 else True
    nib.save(nib.Nifti1Image(carried.astype(np.uint8), G), f"{out}/template_mask_on_mri_grid.nii.gz")
    mmi = nib.load(str(mri_mask))
    mm = X.apply_once(np.asanyarray(mmi.dataobj).astype(np.float32), mmi.affine, shp, G, X.Map(), order=1) > 0.5
    ol = qc.outline_regions(carried, mm, G); st, why = qc.outline_gate(ol)
    Lm = vol(mean); sym = np.linalg.norm(np.flip(Lm, 0) * MIRX - Lm, axis=-1)
    inb = mm.reshape(-1)
    du = np.linalg.norm(mean_un - mean, axis=1)[inb]
    res_ = dict(script="code/template/link/estimate.py compose", run=str(members_json or product(inputs()["member_warps"])), step=None, iteration=None,
                members=names, member_kinds={m["sid"]: m["kind"] for m in specs}, res_mm=res, grid=list(shp),
                averaging="displacement mean of member-composed points (see docstring)",
                spread_mm_in_mri_mask=dict(p50=float(np.percentile(spread[inb], 50)), p95=float(np.percentile(spread[inb], 95)), max=float(spread[inb].max())),
                unmirrored_vs_all_mm_in_mask=dict(p50=float(np.percentile(du, 50)), p95=float(np.percentile(du, 95))),
                link_symmetry_mm_in_mask=dict(p50=float(np.percentile(sym.reshape(-1)[inb], 50)), p95=float(np.percentile(sym.reshape(-1)[inb], 95))),
                outline_status=st, outline_reasons=why, outline={k: ol[k] for k in qc.REGIONS},
                groups={}, wall_s=round(time.time() - t0, 1))
    for g in gset:
        ng, nr = acc[f"g:{g}"][1], acc[f"r:{g}"][1]
        both = (ng > 0) & (nr > 0)
        if both.any():
            dg = np.linalg.norm(acc[f"g:{g}"][0] / np.maximum(ng, 1)[:, None] - acc[f"r:{g}"][0] / np.maximum(nr, 1)[:, None], axis=1)[inb & both]
            res_["groups"][g] = dict(n=int(ng.max()), n_rest=int(nr.max()), vs_rest_mm=dict(p50=float(np.percentile(dg, 50)), p95=float(np.percentile(dg, 95))))
    res_["inverse_guard"] = dict(S5_INV_TOL_mm=est["guard_mm"], dropped_points_per_member={k: v for k, v in n_bad.items() if v},
                                 mri_mask_points_without_any_member=int((cnt[inb] == 0).sum()))
    json.dump(res_, open(f"{out}/s5_compose.json", "w"), indent=1)
    print(json.dumps({k: res_[k] for k in ("outline_status", "outline_reasons", "spread_mm_in_mri_mask", "unmirrored_vs_all_mm_in_mask",
                                           "link_symmetry_mm_in_mask", "wall_s")}, indent=1))
    return res_


def compare_points(mri_mask=None, every=None):
    """Every `every`-th MRI-template brain voxel (world RAS), in array order."""
    mri_mask = mri_mask or product(inputs()["mri_mask"]); every = every or config()["estimate"]["compare_every"]
    m = nib.load(str(mri_mask)); M = np.asanyarray(m.dataobj) > 0.5
    return X.grid_world(m.shape, m.affine)[M.reshape(-1)][::every]


def compare(estimate_dir, link_root, out, extra=None, arms=None, mri_mask=None):
    """|x_estimate(y) - x_link(y)| per direct link (and per extra estimate) at every 8th MRI-template brain voxel."""
    est = config()["estimate"]
    arms = est["compare_direct"] if arms is None else arms
    extra = {k: str(Path(estimate_dir).parent / v) for k, v in est["compare_extra"].items()} if extra is None else extra
    Y = compare_points(mri_mask)
    D = nib.load(f"{estimate_dir}/link_ras_disp.nii.gz"); xc = Y + X.sample(np.asanyarray(D.dataobj).astype(np.float32), D.affine, Y, order=1)
    sp = nib.load(f"{estimate_dir}/link_spread_mm.nii.gz"); spread = X.sample(np.asanyarray(sp.dataobj).astype(np.float32), sp.affine, Y, order=1)
    res = {"n_points": int(len(Y)), "composition_spread_mm": dict(p50=float(np.percentile(spread, 50)), p95=float(np.percentile(spread, 95)))}
    for k, dpath in extra.items():
        Dx = nib.load(f"{dpath}/link_ras_disp.nii.gz")
        xe = Y + X.sample(np.asanyarray(Dx.dataobj).astype(np.float32), Dx.affine, Y, order=1); e = np.linalg.norm(xc - xe, axis=1)
        res[f"composition_vs_{k}_mm"] = dict(p50=float(np.percentile(e, 50)), p95=float(np.percentile(e, 95)), max=float(e.max()),
                                             frac_within_member_spread=float((e <= spread).mean()))
    for arm in arms:
        d = f"{link_root}/{arm}"
        Pst = np.loadtxt(f"{d}/link_linear_world.txt"); dd, da = X.read_fireants_field(f"{d}/link_warp.nii.gz")
        xd = X.Map(L=np.linalg.inv(Pst), d=dd, d_affine=da)(Y)
        e = np.linalg.norm(xc - xd, axis=1)
        res[f"composition_vs_direct_{arm}_mm"] = dict(p50=float(np.percentile(e, 50)), p95=float(np.percentile(e, 95)), max=float(e.max()),
                                                       frac_within_member_spread=float((e <= spread).mean()))
        res[f"direct_{arm}_summary"] = {k: json.load(open(f"{d}/s5_link.json"))[k] for k in ("outline_status", "ncc", "envelope_dice", "edge_mean")}
    if out:
        json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("compose")
    p.add_argument("--out", required=True)
    p.add_argument("--members-json", default=None, help="default: settings/link.yaml inputs.member_warps")
    p.add_argument("--template-mask", default=None, help="default: inputs.bf_mask")
    p.add_argument("--only", default=None, help="comma list of member sids (e.g. TPL: the MRI template member's own link)")
    p.add_argument("--exclude", default=None, help="comma list of member sids left out (default: settings/link.yaml estimate.exclude)")
    p.add_argument("-j", "--jobs", type=int, default=1, help="members in parallel (processes)")
    p.add_argument("--order-from", default=None, help="an s5_compose.json whose inverse_guard.dropped_points_per_member key order gives "
                   "the accumulation order (bit-exact reproduction of a record made with -j > 1)")
    p = sub.add_parser("compare")
    p.add_argument("estimate"); p.add_argument("link_root"); p.add_argument("out")
    p.add_argument("--extra", nargs="*", default=None, help="name=estimate_dir ... (default: settings/link.yaml estimate.compare_extra)")
    a = ap.parse_args(argv)
    if a.cmd == "compose":
        order = list(json.loads(Path(a.order_from).read_text())["inverse_guard"]["dropped_points_per_member"]) if a.order_from else None
        compose(a.out, a.template_mask or product(inputs()["bf_mask"]), members_json=a.members_json,
                only=a.only.split(",") if a.only else None, exclude=a.exclude.split(",") if a.exclude else None, jobs=a.jobs,
                accumulate_order=order)
    else:
        compare(a.estimate, a.link_root, a.out, extra=dict(kv.split("=", 1) for kv in a.extra) if a.extra is not None else None)


if __name__ == "__main__":
    main()
