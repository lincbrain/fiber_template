"""The blockface template's construction, except MMORF itself (reproduce/templates/blockface.md; Methods, "Blockface template"):
the members, their frames, the TPL member, the step-8 weights, the averages the driver's own code computes, and the two run trees.

The build had two stages: plan A, 7 nonlinear steps at 0.5 mm, then step 8 at 0.25 mm. Its 43 inputs are the 21 brains, each with its
exact mirror, plus TPL, the MRI template's b0 entered once as a b0-only member (its warp is the link from the MRI template to the
blockface template). Each input has two channels, the blockface luminance (BF) and the b0 (B0), each with a cost mask that is also its
averaging weight (the patched driver, LINC_MASKWEIGHTED=1):
  members      (cmd_members, 0.5 mm; cmd_s8_inputs, 0.25 mm) every brain's completed volume (linc/completion) placed in the run-113
               frame and resampled once (anti-aliased), with its photographed fraction, color, brain mask and b0 (one rule:
               clip(b0 / mhi, 0, 1) x brain); cost masks: BF 1 for a brain with MRI, and for a brain without MRI 1 on its kept native
               sections with a cosine ramp from 1 mm before its caudal cut face (slab_cost); B0 1 everywhere. Mirrors are exact (the
               grids are symmetric about x = 0).
  frames       the prep frame (frame1c grown 8 mm), the build frame (the union of the members + 4 mm, x-symmetric, x4 voxels), its exact
               2x lattice for step 8, and the step-8 box (plan A's template support + 4 mm).
  TPL          (cmd_tpl_canon, cmd_tpl_costmask, cmd_tpl_affine) the MRI template's b0 in the run-113 frame through the bridge; its b0 cost
               0 below the brainstem plane (+ 1 mm, cosine over 2 mm); its affine a 12-dof FLIRT to plan A's affine b0 template,
               symmetrized.
  step-8 rule  (cmd_step8_weights) the 8 brains without MRI: b0 cost = their blockface cost wherever not fully photographed.
  run trees    (cmd_assemble_plan_a, cmd_assemble_step8) the driver's data tree, configs (linc/construction/blockface_mmorf.py) and
               standardspace; plan A's affine stage is the copied design-2 stage + TPL (cmd_copy_affine, cmd_affine_templates); step 8
               starts from plan A's affines (cmd_s8_mats) and step-7 state (cmd_s8_step7, cmd_step7_templates).
The averages call the patched driver's own functions: set LINC_TEMPLATE_DRIVER to the template_construction_v1 folder of an
mm-template-construction checkout at f4907f4 edited with linc/construction/patches/mm-template-construction_v1_blockface.edits.json
(apply_edits.py).

Ported unchanged (function bodies) from the analysis project: bftemplate/prep_t1.py, bftemplate/d2/{prep_d2, make_frame_d2, prep_d2s8,
build_d2s8_run}.py, bftemplate/d2b/{prep_d2b, prep_d2b_tplfix, prep_d2bs8_weights, prep_d2b_affavg, prep_d2bs8_avg, build_d2b_run}.py.
Paths recorded inside the products are re-rooted (linc.paths.rerooted); the template's own files come from configs/products_blockface.yaml.
"""
import datetime as dt
import glob
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

from linc import paths, provenance, xfm
from linc.construction import blockface_mmorf as M
from linc.construction.blockface_finish import _prod, b0_source, md5, unflatten_map
from linc.registration import fsl
from linc.registration import xfmtools as X

MODS = [("BF", "t1_grey", "t1_costmask"), ("B0", "d2_b0", "d2_b0_costmask")]           # the driver's two scalar channels
MODS_K = [("01", "t1_grey", "t1_costmask"), ("02", "d2_b0", "d2_b0_costmask")]           # scalar_mod_01 = BF, scalar_mod_02 = B0
NATIVE = {"BF": 0.25, "B0": 0.5}                    # the channels' own resolution (mm): the step-8 smoothing floors
MRI13 = "MF275,MF278,MF282,MR243,MR249,MR250,MR251,MR252,MR253,MR254,MR256,MR257,MR261"
NOMRI8 = "MF191,MF196,MF204,MR244,MR247,MR248,MR263,MR273"
DRIVER_PATCHED = "mm-template-construction f4907f4 + linc/construction/patches/mm-template-construction_v1_blockface.edits.json"
STEP8_RULE = dict(question="At step 8 only, should the 8 brains' stand-in b0 count less in never-photographed tissue, the same way their "
                           "synthetic blockface already does? Steps 1-7 keep full weight.",
                  answer="Yes, step 8 only", who="the author",
                  scope="step 8 only; the 8 brains without MRI and their mirrors; the 13 MRI brains and TPL unchanged")
PLANE, K, BAND = -2.0, 1.0, 2.0     # TPL's b0 cost: the brainstem plane (canonical z, mm), then 0 for K mm and a cosine over BAND mm
_FRAME = [None]


def set_frame(path):
    """The 0.5 mm frame the members' grids refine (the analysis' T1_GRID)."""
    _FRAME[0] = str(path)


def frame():
    return _FRAME[0] or os.environ.get("T1_GRID") or _prod("run113_frame")


def ab(p):
    return p if os.path.isabs(p) else os.path.abspath(p)


def _canon_grid(canon):
    g = json.load(open(f"{canon}/qc.json"))["grid"]
    return tuple(g["shape"]), np.array(g["affine"], float)


def _driver_dir():
    d = os.environ.get("LINC_TEMPLATE_DRIVER")
    if not d or not os.path.exists(os.path.join(d, "run_template_construction.py")):
        raise RuntimeError("set LINC_TEMPLATE_DRIVER to the patched template_construction_v1 folder (" + DRIVER_PATCHED + ")")
    return d


def _driver():
    """The patched driver module, with the mask-weighted averages switched on (LINC_MASKWEIGHTED=1)."""
    os.environ["LINC_MASKWEIGHTED"] = "1"
    sys.path.insert(0, _driver_dir())
    RT = importlib.import_module("run_template_construction")
    assert RT.MASKW, "LINC_MASKWEIGHTED must be on"
    return RT


# ---------------------------------------------------------------------------------------------------- grid
def grid(factor=1):
    ref = nib.load(frame())
    v = np.diag(ref.affine)[:3]
    A = ref.affine.copy(); A[:3, :3] = ref.affine[:3, :3] / factor
    A[:3, 3] = ref.affine[:3, 3] - v / 2 + v / (2 * factor)
    shp = tuple(int(s * factor) for s in ref.shape[:3])
    xc = A[0, 3] + A[0, 0] * np.arange(shp[0])
    assert np.allclose(xc, -xc[::-1], atol=1e-6), "grid not x-symmetric"
    return shp, A


def member(S, root, cache, shp, A):
    src = f"{root}/{S}"
    ci = nib.load(f"{src}/{S}_bf_completed.nii.gz")
    dep = nib.load(f"{cache}/{S}/deploy/brain.nii.gz")
    assert np.allclose(ci.affine, dep.affine) and ci.shape[:3] == dep.shape[:3], f"{S}: product and cache grids differ"
    flat, _ = unflatten_map(S, cache, ci.affine, ci.shape)
    iso_src = abs(ci.affine[0, 0]); vox = abs(A[0, 0])
    sig = 0.5 * np.sqrt(max((vox / iso_src) ** 2 - 1, 0))
    aa = lambda v: ndi.gaussian_filter(v, sig) if sig > 0 else v
    grey = X.apply_once(aa(np.asanyarray(ci.dataobj).astype(np.float32)), ci.affine, shp, A, flat, order=1)
    brain = X.apply_once(np.asanyarray(dep.dataobj).astype(np.float32), dep.affine, shp, A, flat, order=1) > 0.5
    prov = X.apply_once(aa(np.asanyarray(nib.load(f"{src}/{S}_provenance.nii.gz").dataobj).astype(np.float32)),
                        ci.affine, shp, A, flat, order=1)
    r = np.asanyarray(nib.load(f"{src}/{S}_bf_completed_rgb.nii.gz").dataobj).astype(np.float32)
    rgb = np.stack([X.apply_once(aa(r[..., k]), ci.affine, shp, A, flat, order=1) for k in range(3)], -1)
    ins = {f: md5(f"{src}/{S}_{f}.nii.gz") for f in ("bf_completed", "provenance", "bf_completed_rgb")}
    return np.clip(grey, 0, 1) * brain, brain, np.clip(prov, 0, 1) * brain, np.clip(rgb, 0, 1) * brain[..., None], ins


def save_pair(d, name, vol, A, dtype=np.float32):
    nib.save(nib.Nifti1Image(np.ascontiguousarray(vol.astype(dtype)), A), f"{d}/{name}.nii.gz")
    nib.save(nib.Nifti1Image(np.ascontiguousarray(np.flip(vol, 0).astype(dtype)), A), f"{d}/{name}_flip.nii.gz")


# ---------------------------------------------------------------------------------------------------- frame
def cut(v, sl):
    """v[sl] for slices that may reach outside v (zeros there)."""
    out = np.zeros(tuple(s.stop - s.start for s in sl) + v.shape[3:], v.dtype)
    src = tuple(slice(max(s.start, 0), min(s.stop, n)) for s, n in zip(sl, v.shape[:3]))
    dst = tuple(slice(a.start - s.start, a.stop - s.start) for a, s in zip(src, sl))
    out[dst] = v[src]
    return out


def make_prep_frame(out, grow_mm):
    f = nib.load(_prod("frame1c")); A = f.affine; vx = np.abs(np.diag(A)[:3]); g = int(np.ceil(grow_mm / vx.min()))
    sl = tuple(slice(-g, n + g) for n in f.shape[:3])
    A2 = A.copy(); A2[:3, 3] = A[:3, :3] @ np.array([s.start for s in sl]) + A[:3, 3]
    shp = tuple(s.stop - s.start for s in sl)
    xc = A2[0, 3] + A2[0, 0] * np.arange(shp[0]); assert np.allclose(xc, -xc[::-1], atol=1e-6)
    assert not os.path.exists(out), f"{out} exists"
    nib.save(nib.Nifti1Image(cut(np.asanyarray(f.dataobj), sl), A2, f.header), out)
    print(json.dumps(dict(prep_frame=out, from_frame=_prod("frame1c"), grow_vox=g, shape=list(shp), mvox=round(np.prod(shp) / 1e6, 3))))


def frame_box(canon, pad, mult, extra=()):
    qc = json.load(open(f"{canon}/qc.json"))
    shp = tuple(qc["grid"]["shape"]); A = np.array(qc["grid"]["affine"]); vx = np.abs(np.diag(A)[:3])
    u = np.zeros(shp, bool); n = 0
    for S in qc["subjects"]:
        for suf in ("", "_flip"):
            u |= np.asanyarray(nib.load(f"{canon}/{S}/t1_grey_mask{suf}.nii.gz").dataobj) > 0; n += 1
    for p in extra:
        e = nib.load(p); assert e.shape[:3] == shp and np.allclose(e.affine, A, atol=1e-6), f"{p}: not on the canon grid"
        u |= np.asanyarray(e.dataobj) > 0; n += 1
    idx = np.argwhere(u); lo, hi = idx.min(0), idx.max(0)
    p = np.ceil(pad / vx - 1e-9).astype(int)
    xc = A[0, 3] + A[0, 0] * np.arange(shp[0]); assert np.allclose(xc, -xc[::-1], atol=1e-6), "canon grid not x-symmetric"
    half = int(max(shp[0] // 2 - lo[0], hi[0] - (shp[0] // 2 - 1)) + p[0])     # voxels per side from the centre (shp[0] even)
    assert shp[0] % 2 == 0
    nx = 2 * half; nx = int(np.ceil(nx / mult) * mult); nx += nx % 2
    sl = [slice(shp[0] // 2 - nx // 2, shp[0] // 2 + nx // 2)]
    for ax in (1, 2):
        a0, a1 = lo[ax] - p[ax], hi[ax] + p[ax] + 1
        L = int(np.ceil((a1 - a0) / mult) * mult); a0 -= (L - (a1 - a0)) // 2
        sl.append(slice(int(a0), int(a0 + L)))
    A2 = A.copy(); A2[:3, 3] = A[:3, :3] @ np.array([s.start for s in sl]) + A[:3, 3]
    nshape = [s.stop - s.start for s in sl]
    xc2 = A2[0, 3] + A2[0, 0] * np.arange(nshape[0])
    margins = {ax: [round(float((lo[k] - sl[k].start) * vx[k]), 2), round(float((sl[k].stop - 1 - hi[k]) * vx[k]), 2)]
               for k, ax in enumerate("xyz")}
    outside = {ax: [int(max(-sl[k].start, 0)), int(max(sl[k].stop - shp[k], 0))] for k, ax in enumerate("xyz")}
    # the members touching the canon grid's own edge would have been clipped by prep: refuse
    edge = {ax: [int(lo[k]), int(shp[k] - 1 - hi[k])] for k, ax in enumerate("xyz")}
    return dict(canon_shape=list(shp), union_inputs=n, union_bbox_vox=[lo.tolist(), hi.tolist()], union_to_canon_edge_vox=edge,
                slices=[[s.start, s.stop] for s in sl], shape=nshape, affine=A2.tolist(), zero_padded_vox=outside,
                x_symmetric=bool(np.allclose(xc2, -xc2[::-1], atol=1e-6)), margins_mm=margins,
                min_margin_mm=min(min(v) for v in margins.values()), pad_mm=pad, mult=mult,
                mvox_05=round(float(np.prod(nshape)) / 1e6, 3), mvox_025=round(float(np.prod(nshape)) * 8 / 1e6, 3)), sl, A2


# ---------------------------------------------------------------------------------------------------- members
def b0_member(S, root, cache, shp, A, brain):
    ci = nib.load(f"{root}/{S}/{S}_bf_completed.nii.gz")
    flat, R = unflatten_map(S, cache, ci.affine, ci.shape)
    assert np.allclose(R, np.eye(3), atol=1e-9), f"{S}: d3's flattening is not the identity (seam.json normal)"
    v, Av, rule = b0_source(S, root)
    v = X.apply_once(v, Av, shp, A, flat, order=1)
    return np.clip(v, 0, 1) * brain, md5(f"{root}/{S}/{S}_b0_input_0p5.nii.gz"), rule


def slab_cost(S, shp, A, suffix, k_mm, band_mm):
    """bftemplate/b1/direct_members.py's MMORF cost mask on the blockface native mask array (axis 1 = native section, increasing
    caudally), pulled onto the grid through the member's RIGID placement T (blockface native world -> run-113 world)."""
    j = json.load(open(f"{paths.subject_dir(S)}/mri/s2lin{suffix}/{S}_s2lin.json"))
    T = np.array(j["T_world_bf_to_run113"]); Ti = np.linalg.inv(T)
    mimg = nib.load(paths.rerooted(j["bf_mask"])); marr = np.asanyarray(mimg.dataobj) > 0
    ks = np.where(marr.any(axis=(0, 2)))[0]; last_kept = int(ks.max())
    step = float(np.linalg.norm(mimg.affine[:3, 1]))
    jj = np.arange(marr.shape[1], dtype=np.float64)
    dist = (last_kept - jj) * step                                      # mm before the last kept section (>= 0 kept)
    t = np.clip((dist - k_mm) / band_mm, 0.0, 1.0)
    c = np.where(dist < 0, 0.0, np.where(dist < k_mm, 0.0, 0.5 - 0.5 * np.cos(np.pi * t)))
    slab = np.ascontiguousarray(np.broadcast_to(c[None, :, None].astype(np.float32), marr.shape))
    to_native = lambda Q: Q @ Ti[:3, :3].T + Ti[:3, 3]
    cost = np.clip(X.apply_once(slab, mimg.affine, shp, A, to_native, order=1), 0.0, 1.0).astype(np.float32)
    return cost, dict(s2lin_json=f"{paths.subject_dir(S)}/mri/s2lin{suffix}/{S}_s2lin.json", last_kept_section=last_kept,
                      section_step_mm=round(step, 4), k_mm=k_mm, band_mm=band_mm, bf_mask=j["bf_mask"])


def cmd_members(a):
    """The members of the 0.5 mm build (bftemplate/d2/prep_d2.py main): every brain and its mirror on the prep frame."""
    for k in ("out", "root", "cache"):
        setattr(a, k, os.path.abspath(getattr(a, k)))
    suf = f"_{a.s2lin_suffix}" if a.s2lin_suffix else ""
    shp, A = grid(1)
    os.makedirs(a.out, exist_ok=True)
    assert not [d for d in os.listdir(a.out) if os.path.isdir(f"{a.out}/{d}")], f"{a.out} not empty: use a new dir"
    todo = [(s, "mri") for s in a.mri_subjects.split(",") if s] + [(s, "nomri") for s in a.nomri_subjects.split(",") if s]
    qc = {}
    for S, kind in todo:
        g, m, p, rgb, ins = member(S, a.root, a.cache, shp, A)
        b0, b0md5, b0rule = b0_member(S, a.root, a.cache, shp, A, m)
        ins["b0_input_0p5"] = b0md5
        if kind == "mri":
            cost, cinfo = np.ones(shp, np.float32), None
        else:
            cost, cinfo = slab_cost(S, shp, A, suf, a.k_mm, a.band_mm)
            ins["s2lin_json"] = md5(cinfo["s2lin_json"])
        d = f"{a.out}/{S}"; os.makedirs(d, exist_ok=True)
        save_pair(d, "t1_grey", g, A); save_pair(d, "t1_grey_mask", m, A, np.uint8)
        save_pair(d, "t1_photofrac", p, A); save_pair(d, "t1_rgb", rgb, A)
        save_pair(d, "t1_costmask", cost, A)
        save_pair(d, "d2_b0", b0, A); save_pair(d, "d2_b0_costmask", np.ones(shp, np.float32), A)
        e = np.argwhere(m); edge = [int(min(e[:, k].min(), shp[k] - 1 - e[:, k].max())) for k in range(3)]
        qc[S] = dict(kind=kind, root=a.root, cache=a.cache, inputs_md5=ins, brain_vox=int(m.sum()),
                     photographed_frac=float((p >= 0.5).sum() / max(m.sum(), 1)), photo_fraction_mean=float(p[m].mean()),
                     self_mirror_corr=float(np.corrcoef(g[m], np.flip(g, 0)[m])[0, 1]),
                     cost_mean_in_brain=float(cost[m].mean()), cost_zero_vox=int((cost < 0.01).sum()),
                     b0_mean_in_brain=float(b0[m].mean()), b0_zero_in_brain=int((b0[m] <= 0).sum()),
                     brain_to_grid_edge_vox=edge, slab=cinfo, b0_rule=b0rule)
        print(f"  {S} ({kind}): brain {m.sum()} photographed {qc[S]['photographed_frac']:.3f} cost mean {qc[S]['cost_mean_in_brain']:.3f} "
              f"b0 mean {qc[S]['b0_mean_in_brain']:.3f} (zero in brain {qc[S]['b0_zero_in_brain']}) edge {edge}", flush=True)
        if min(edge) < 1:
            raise SystemExit(f"{S}: the brain touches the grid edge (clipped?): use a larger T1_GRID")
    json.dump(dict(grid=dict(shape=shp, affine=A.tolist()), subjects=qc, args=vars(a), t1_grid=frame(),
                   d2=dict(step="S4", cost_bf="1 (MRI) / 1c's direct-member slab via the rigid placement (no MRI)",
                           cost_b0="1 everywhere (D35 full weight)", b0="<S>_b0_input_0p5 (the completion's exact b0 input), put on one rule for all 21: b0 / mhi (lo = 0), x brain")),
              open(f"{a.out}/qc.json", "w"), indent=1)
    provenance.write_provenance(a.out, inputs=[frame()], extra=dict(cmd=" ".join(sys.argv)))


# ---------------------------------------------------------------------------------------------------- s8
# ------------------------------------------------------------------------------------------------------------------- frame
def cmd_frame(a):
    g = json.load(open(f"{ab(a.canon)}/qc.json"))["grid"]
    A = np.array(g["affine"], float); shp = tuple(g["shape"])
    xc = A[0, 3] + A[0, 0] * np.arange(shp[0]); assert np.allclose(xc, -xc[::-1], atol=1e-6), "canon frame not x-symmetric"
    os.makedirs(os.path.dirname(ab(a.out)), exist_ok=True)
    assert not os.path.exists(ab(a.out)), f"{a.out} exists"
    nib.save(nib.Nifti1Image(np.zeros(shp, np.uint8), A), ab(a.out))
    print("frame", ab(a.out), shp, A[:3, 3])


# ------------------------------------------------------------------------------------------------------------------ inputs
def one_member(job):
    S, kind, root, cache, suf, k_mm, band_mm, out, frame_path = job
    set_frame(frame_path)
    shp, A = grid(2)
    t0 = time.time()
    g, m, p, rgb, ins = member(S, root, cache, shp, A)
    b0, b0md5, b0rule = b0_member(S, root, cache, shp, A, m)
    ins["b0_input_0p5"] = b0md5
    if kind == "mri":
        cost, cinfo = np.ones(shp, np.float32), None
    else:
        cost, cinfo = slab_cost(S, shp, A, suf, k_mm, band_mm)
        ins["s2lin_json"] = md5(cinfo["s2lin_json"])
    d = f"{out}/{S}"; os.makedirs(d, exist_ok=True)
    save_pair(d, "t1_grey", g, A); save_pair(d, "t1_grey_mask", m, A, np.uint8)
    save_pair(d, "t1_photofrac", p, A); save_pair(d, "t1_rgb", rgb, A)
    save_pair(d, "t1_costmask", cost, A)
    save_pair(d, "d2_b0", b0, A); save_pair(d, "d2_b0_costmask", np.ones(shp, np.float32), A)
    e = np.argwhere(m); edge = [int(min(e[:, k].min(), shp[k] - 1 - e[:, k].max())) for k in range(3)]
    q = dict(kind=kind, root=root, cache=cache, inputs_md5=ins, brain_vox=int(m.sum()),
             photographed_frac=float((p >= 0.5).sum() / max(m.sum(), 1)), photo_fraction_mean=float(p[m].mean()),
             self_mirror_corr=float(np.corrcoef(g[m], np.flip(g, 0)[m])[0, 1]),
             cost_mean_in_brain=float(cost[m].mean()), cost_zero_vox=int((cost < 0.01).sum()),
             b0_mean_in_brain=float(b0[m].mean()), b0_zero_in_brain=int((b0[m] <= 0).sum()),
             brain_to_grid_edge_vox=edge, slab=cinfo, b0_rule=b0rule, seconds=round(time.time() - t0, 1))
    print(f"  {S} ({kind}): brain {m.sum()} cost mean {q['cost_mean_in_brain']:.3f} b0 mean {q['b0_mean_in_brain']:.3f} "
          f"edge {edge} {q['seconds']} s", flush=True)
    return S, q


def cmd_inputs(a):
    out = ab(a.out); os.makedirs(out, exist_ok=True)
    assert not [d for d in os.listdir(out) if os.path.isdir(f"{out}/{d}")], f"{out} not empty: use a new dir"
    s4 = json.load(open(f"{ab(a.canon)}/qc.json"))
    args = s4["args"]
    subs = s4["subjects"]
    suf = f"_{args['s2lin_suffix']}" if args["s2lin_suffix"] else ""
    if a.subjects:
        subs = {S: subs[S] for S in a.subjects.split(",")}
    jobs = [(S, q["kind"], paths.rerooted(q["root"]), paths.rerooted(q["cache"]), suf, args["k_mm"], args["band_mm"], out, ab(a.frame))
            for S, q in subs.items()]
    set_frame(ab(a.frame))
    shp, A = grid(2)
    t0 = time.time()
    with Pool(a.j) as pool:
        res = dict(pool.map(one_member, jobs, chunksize=1))
    bad = {S: q["brain_to_grid_edge_vox"] for S, q in res.items() if min(q["brain_to_grid_edge_vox"]) < 1}
    if bad:
        raise SystemExit(f"brains touching the grid edge: {bad}")
    json.dump(dict(grid=dict(shape=list(shp), affine=A.tolist()), subjects={S: res[S] for S in subs}, args=args,
                   t1_grid=frame(), s8=dict(step="the S4 inputs at 0.25 mm (canon frame's exact 2x lattice)",
                                            s4_canon=ab(a.canon), seconds=round(time.time() - t0, 1))),
              open(f"{out}/qc.json", "w"), indent=1)
    provenance.write_provenance(out, inputs=[frame()], extra=dict(cmd=" ".join(sys.argv)))
    print(f"inputs -> {out}: {len(res)} brains, grid {shp}, {time.time() - t0:.0f} s")


# --------------------------------------------------------------------------------------------------------------------- box
def cmd_box(a):
    c2 = json.load(open(f"{ab(a.canon2)}/qc.json"))["grid"]
    A2 = np.array(c2["affine"], float); s2 = tuple(c2["shape"])
    lo = np.full(3, np.inf); hi = np.full(3, -np.inf); sup = {}
    for t in a.template:
        im = nib.load(ab(t)); v = np.asanyarray(im.dataobj).astype(np.float32)
        m = v > a.thr * np.percentile(v[v > 0], 99)
        idx = np.argwhere(m); w0 = im.affine @ np.r_[idx.min(0), 1]; w1 = im.affine @ np.r_[idx.max(0), 1]
        lo = np.minimum(lo, np.minimum(w0[:3], w1[:3])); hi = np.maximum(hi, np.maximum(w0[:3], w1[:3]))
        sup[t] = dict(world_lo=w0[:3].tolist(), world_hi=w1[:3].tolist(), voxel_mm=float(abs(im.affine[0, 0])))
    vox = np.abs(np.diag(A2)[:3]); assert np.all(np.diag(A2)[:3] > 0), "canon2 axes must be positive"
    # half a coarse voxel more on each side (the support is measured at voxel centres of the 0.5 mm template), then PAD
    wlo = lo - a.pad - 0.25; whi = hi + a.pad + 0.25
    i0 = np.floor((wlo - A2[:3, 3]) / vox).astype(int); i1 = np.ceil((whi - A2[:3, 3]) / vox).astype(int)
    i0[0] = min(i0[0], s2[0] - 1 - i1[0]); i1[0] = s2[0] - 1 - i0[0]               # x-symmetric on canon2's symmetric lattice
    shp = tuple(int(x) for x in i1 - i0 + 1)
    A = A2.copy(); A[:3, 3] = A2[:3, 3] + i0 * vox
    xc = A[0, 3] + A[0, 0] * np.arange(shp[0]); assert np.allclose(xc, -xc[::-1], atol=1e-6), "box not x-symmetric"
    out = ab(a.out); assert not os.path.exists(out), f"{out} exists"
    nib.save(nib.Nifti1Image(np.zeros(shp, np.uint8), A), out)
    marg = dict(lo=(lo - A[:3, 3]).tolist(), hi=(A[:3, 3] + (np.array(shp) - 1) * vox - hi).tolist())
    rec = dict(box=out, shape=list(shp), affine=A.tolist(), mvox=round(float(np.prod(shp)) / 1e6, 3), canon2_offset_vox=i0.tolist(),
               support_world=dict(lo=lo.tolist(), hi=hi.tolist()), margins_mm=marg, pad_mm=a.pad, thr_of_p99=a.thr, templates=sup,
               rule="support(BF or B0 template brain > thr x p99) +- (pad + 0.25) mm, snapped outward to canon2's lattice, x-symmetric")
    json.dump(rec, open(out.replace(".nii.gz", ".json"), "w"), indent=1)
    print(json.dumps({k: rec[k] for k in ("shape", "mvox", "canon2_offset_vox", "margins_mm")}))


# -------------------------------------------------------------------------------------------------------------------- mats
def fsl_from_world(fwd, src_grid, dst_grid):
    """The inverse of xfm.read_fsl_affine's construction: a src RAS -> dst RAS map as a FLIRT .mat for (src_grid, dst_grid)."""
    ss, As, zs = xfm._grid(src_grid); sd, Ad, zd = xfm._grid(dst_grid)
    Fs, Fd = xfm.fsl_vox2mm(ss, As, zs), xfm.fsl_vox2mm(sd, Ad, zd)
    return Fd @ np.linalg.inv(Ad) @ fwd @ As @ np.linalg.inv(Fs)


def cmd_mats(a):
    run = ab(a.run); spec = json.load(open(f"{run}/run_spec.json")); box = ab(a.box)
    std05 = ab(a.standard05)
    rows = {}
    for sid in spec["subject_ids"]:
        S = sid[:-1] if sid.endswith("f") else sid; fl = "_flip" if sid.endswith("f") else ""
        src05 = f"{ab(a.canon)}/{S}/t1_grey{fl}.nii.gz"; src25 = f"{ab(a.canon2)}/{S}/t1_grey{fl}.nii.gz"
        mats = {}
        for k, _, _ in MODS_K:
            p = f"{ab(a.affine_dir)}/{sid}/scalar_mod_{k}_to_standardspace.mat"
            fwd = np.linalg.inv(xfm.read_fsl_affine(p, src05, std05))         # src RAS -> standardspace RAS (plan A)
            M = fsl_from_world(fwd, src25, box)
            d = f"{run}/out/affine_step2/{sid}"; os.makedirs(d, exist_ok=True)
            q = f"{d}/scalar_mod_{k}_to_standardspace.mat"
            np.savetxt(q, M, fmt="%.10f")
            chk = np.linalg.inv(xfm.read_fsl_affine(q, src25, box))
            err = float(np.abs(chk - fwd).max())
            assert err < 1e-6, f"{sid} mod {k}: world map changed by {err}"
            mats[k] = dict(source=p, out=q, world_max_abs_err=err)
        e01 = np.loadtxt(mats["01"]["out"]); e02 = np.loadtxt(mats["02"]["out"])
        rows[sid] = dict(mats, mods_identical=bool(np.allclose(e01, e02, atol=1e-9)))
    json.dump(dict(rows=rows, standard05=std05, box=box, canon=ab(a.canon), canon2=ab(a.canon2),
                   rule="FLIRT .mat (canon 0.5 -> standardspace 0.5) -> RAS world map (xfm.read_fsl_affine) -> FLIRT .mat (canon2 -> box)"),
              open(f"{run}/out/affine_step2/s8_mats.json", "w"), indent=1)
    print(f"mats: {len(rows)} inputs x {len(MODS_K)}; max world error {max(r[k]['world_max_abs_err'] for r in rows.values() for k, _, _ in MODS_K):.2e}; "
          f"mods identical for all: {all(r['mods_identical'] for r in rows.values())}")


# ------------------------------------------------------------------------------------------------------------------- step7
def pull_one(job):
    sid, it, canon, canon2, box, outd = job
    S = sid[:-1] if sid.endswith("f") else sid; fl = "_flip" if sid.endswith("f") else ""
    bi = nib.load(box); shp = bi.shape[:3]; A = bi.affine
    W = xfm.Warp(f"{it}/{sid}_scalar_mod_01_to_template_warp_resampled_unbiased_full.nii.gz", "fsl_rel",
                 src_grid=f"{canon}/{S}/t1_grey{fl}.nii.gz")
    W2 = xfm.Warp(f"{it}/{sid}_scalar_mod_02_to_template_warp_resampled_unbiased_full.nii.gz", "fsl_rel",
                  src_grid=f"{canon}/{S}/t1_grey{fl}.nii.gz")
    vols = {}
    for k, payload, cmask in MODS_K:
        for kind, name in (("brain", payload), ("mask", cmask)):
            im = nib.load(f"{canon2}/{S}/{name}{fl}.nii.gz")
            vols[(k, kind)] = (np.asanyarray(im.dataobj).astype(np.float32), im.affine)
    bias = {}
    for b in (0, 1):
        im = nib.load(f"{it}/{sid}_to_template_bias_{b}.nii.gz")
        bias[b] = (np.asanyarray(im.dataobj).astype(np.float32), im.affine)
    out = {key: np.zeros(shp, np.float32) for key in list(vols) + [("bias", 0), ("bias", 1)]}
    dmax = 0.0
    for z0 in range(0, shp[2], 8):
        z1 = min(shp[2], z0 + 8); sh = (shp[0], shp[1], z1 - z0)
        P = X.grid_world(shp, A, z0, z1)
        Q = W.pull(P)
        if z0 % 64 == 0:
            dmax = max(dmax, float(np.abs(W2.pull(P) - Q).max()))
        for key, (v, va) in vols.items():
            out[key][..., z0:z1] = X.sample(v, va, Q, order=1).reshape(sh)
        for b in (0, 1):
            out[("bias", b)][..., z0:z1] = X.sample(bias[b][0], bias[b][1], P, order=1).reshape(sh)
    assert dmax < 1e-4, f"{sid}: the two modalities' full warps differ by {dmax} mm"
    for k, _, _ in MODS_K:
        nib.save(nib.Nifti1Image(out[(k, "brain")], A), f"{outd}/{sid}_scalar_mod_{k}_brain_to_template.nii.gz")
        nib.save(nib.Nifti1Image(np.clip(out[(k, "mask")], 0, None), A), f"{outd}/{sid}_scalar_mod_{k}_brain_mask_to_template.nii.gz")
    for b in (0, 1):
        nib.save(nib.Nifti1Image(out[("bias", b)], A), f"{outd}/{sid}_to_template_bias_{b}.nii.gz")
    print(f"  {sid} pulled (mod 01 vs 02 warp max diff {dmax:.1e} mm)", flush=True)
    return sid, dmax


def cmd_step7(a):
    run = ab(a.run); spec = json.load(open(f"{run}/run_spec.json")); box = paths.rerooted(spec["s8"]["box"])
    outd = f"{run}/out/nln_step_07/iteration_03"; os.makedirs(outd, exist_ok=True)
    assert not [f for f in os.listdir(outd) if f.endswith(".nii.gz")], f"{outd} not empty"
    t0 = time.time()
    jobs = [(sid, ab(a.plan_a_it), ab(a.canon), ab(a.canon2), box, outd) for sid in spec["subject_ids"]]
    with Pool(a.j) as pool:
        res = dict(pool.map(pull_one, jobs, chunksize=1))
    json.dump(dict(plan_a_iteration=ab(a.plan_a_it), canon=ab(a.canon), canon2=ab(a.canon2), box=box, inputs=len(res),
                   warp_mod01_vs_mod02_max_mm=max(res.values()), seconds=round(time.time() - t0, 1),
                   rule="every input's 2 channels + 2 cost masks pulled from canon2 through plan A's step-7 iteration-3 unbiased full "
                        "warp (xfm.Warp fsl_rel, linear); the bias fields resampled linearly; averages by prep_d2s8_avg.py"),
              open(f"{outd}/s8_step7_pull.json", "w"), indent=1)
    print(f"step7 pulls -> {outd}: {len(res)} inputs, {time.time() - t0:.0f} s")


def cmd_boxcheck(a):
    bi = nib.load(ab(a.box)); A = bi.affine; shp = np.array(bi.shape[:3]); vox = np.abs(np.diag(A)[:3])
    blo = A[:3, 3] - vox / 2; bhi = A[:3, 3] + (shp - 1) * vox + vox / 2              # the box's outer faces (world)
    rec = {}
    for t in a.template:
        im = nib.load(ab(t)); v = np.asanyarray(im.dataobj).astype(np.float32)
        m = v > a.thr * np.percentile(v[v > 0], 99); idx = np.argwhere(m)
        h = np.abs(np.diag(im.affine)[:3]) / 2
        w0 = im.affine @ np.r_[idx.min(0), 1]; w1 = im.affine @ np.r_[idx.max(0), 1]
        lo = np.minimum(w0[:3], w1[:3]) - h; hi = np.maximum(w0[:3], w1[:3]) + h      # the support's outer faces
        rec[t] = dict(margin_lo_mm=(lo - blo).round(3).tolist(), margin_hi_mm=(bhi - hi).round(3).tolist())
    mn = min(min(r["margin_lo_mm"] + r["margin_hi_mm"]) for r in rec.values())
    print(json.dumps(dict(min_margin_mm=mn, need_mm=a.min_mm, templates=rec)))
    if mn < a.min_mm:
        raise SystemExit(f"the plan-A support comes within {mn:.2f} mm of the step-8 box (< {a.min_mm}): re-cut the box")


# ---------------------------------------------------------------------------------------------------- tpl
def cmd_tpl_canon(a):
    SRC = {a.vox: os.path.abspath(a.source_canon)}
    vox = a.vox; shp, A = _canon_grid(SRC[vox])
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=False)
    q = json.load(open(f"{SRC[vox]}/qc.json"))
    for S in q["subjects"]:
        os.symlink(f"{SRC[vox]}/{S}", f"{out}/{S}")
    P = np.array(json.load(open(_prod("bridge")))["rigid_world_run113_to_canonical"])
    assert np.allclose(P[0], [1, 0, 0, 0]) and np.allclose(P[:, 0], [1, 0, 0, 0]), "the bridge moves x: TPL would not stay symmetric"
    bi, mi = nib.load(_prod("tpl_b0")), nib.load(_prod("tpl_mask"))
    b0 = np.asanyarray(bi.dataobj).astype(np.float32); m0 = np.asanyarray(mi.dataobj) > 0
    mhi = float(np.percentile(b0[m0], 99.5))
    src_vox = abs(bi.affine[0, 0]); sig = 0.5 * np.sqrt(max((vox / src_vox) ** 2 - 1, 0))
    b0a = ndi.gaussian_filter(b0, sig) if sig > 0 else b0
    to_can = lambda Q: Q @ P[:3, :3].T + P[:3, 3]
    v = X.apply_once(b0a, bi.affine, shp, A, to_can, order=1)
    brain = X.apply_once(m0.astype(np.float32), mi.affine, shp, A, to_can, order=1) > 0.5
    b = np.clip(v / mhi, 0, 1) * brain
    d = f"{out}/TPL"; os.makedirs(d)
    sv = lambda n, x, dt=np.float32: nib.save(nib.Nifti1Image(np.ascontiguousarray(x.astype(dt)), A), f"{d}/{n}.nii.gz")
    z = np.zeros(shp, np.float32)
    sv("t1_grey", z); sv("t1_grey_mask", brain, np.uint8); sv("t1_photofrac", z); sv("t1_rgb", np.zeros(shp + (3,), np.float32))
    sv("t1_costmask", z); sv("d2_b0", b); sv("d2_b0_costmask", np.ones(shp, np.float32))
    e = np.argwhere(brain); edge = [int(min(e[:, k].min(), shp[k] - 1 - e[:, k].max())) for k in range(3)]
    mirror = float(np.abs(b - np.flip(b, 0)).max())
    q["subjects"]["TPL"] = dict(kind="template", root=None, cache=None, brain_vox=int(brain.sum()), brain_to_grid_edge_vox=edge,
                                b0_mean_in_brain=float(b[brain].mean()), b0_rule="clip(b0 / mhi, 0, 1) x brain (the one rule; lo = 0)",
                                mhi=mhi, mirror_max_abs_diff=mirror,
                                inputs_md5=dict(b0=md5(_prod("tpl_b0")), brainmask=md5(_prod("tpl_mask")), bridge=md5(_prod("bridge"))),
                                channels="B0 only: BF image 0 with BF cost 0; b0 cost 1 everywhere (D35); entered once (its own mirror)")
    q["d2b"] = dict(decision="design 2b: the 21 brains x 2 mirrors + TPL, the MRI template's b0 only",
                    source_canon=SRC[vox], tpl=f"{out}/TPL")
    json.dump(q, open(f"{out}/qc.json", "w"), indent=1)
    print(f"canon {vox} mm -> {out}: 21 symlinks + TPL (brain {int(brain.sum())} vox, edge {edge}, b0 mean {float(b[brain].mean()):.3f}, "
          f"mhi {mhi:.2f}, mirror max diff {mirror:.2e})")


def cmd_copy_affine(a):
    run = os.path.abspath(a.run)
    spec = json.load(open(f"{run}/run_spec.json")); ids = spec["subject_ids"]
    src = os.path.abspath(a.plan_a_affine)
    dst = f"{run}/out/affine_step2"; os.makedirs(dst, exist_ok=True)
    ident = os.path.join(fsl.fsldir(), "etc/flirtsch/ident.mat")
    canon = paths.rerooted(spec["b1"]["canon"]); std = nib.load(f"{run}/standardspace.nii.gz")
    n = 0
    for sid in ids:
        d = f"{dst}/{sid}"; os.makedirs(d, exist_ok=True)
        for k in ("01", "02"):
            if sid == "TPL":
                shutil.copy2(ident, f"{d}/scalar_mod_{k}_to_standardspace.mat")
                pay, cm = (("t1_grey", "t1_costmask") if k == "01" else ("d2_b0", "d2_b0_costmask"))
                for nm, f in ((f"scalar_mod_{k}_to_standardspace_brain", pay), (f"scalar_mod_{k}_to_standardspace_brain_mask", cm)):
                    im = nib.load(f"{canon}/TPL/{f}.nii.gz")
                    assert im.shape[:3] == std.shape[:3] and np.allclose(im.affine, std.affine), "TPL not on the standardspace grid"
                    shutil.copy2(f"{canon}/TPL/{f}.nii.gz", f"{d}/{nm}.nii.gz")
            else:
                for f in (f"scalar_mod_{k}_to_standardspace.mat", f"scalar_mod_{k}_to_standardspace_brain.nii.gz",
                          f"scalar_mod_{k}_to_standardspace_brain_mask.nii.gz"):
                    shutil.copy2(f"{src}/{sid}/{f}", f"{d}/{f}")
            n += 1
    json.dump(dict(source=src, inputs=len(ids), tpl="identity (FSL ident.mat): TPL already sits in the run-113 frame on the standardspace grid",
                   note="design 2's affine stage for the 42 (identical inputs); the affine templates are recomputed over all inputs by prep_d2b_affavg.py"),
              open(f"{dst}/d2b_affine.json", "w"), indent=1)
    print(f"affine_step2 -> {dst}: {len(ids)} inputs x 2 channels ({n} entries)")


# ---------------------------------------------------------------------------------------------------- tplfix
def zcan(shape, A):
    P = np.array(json.load(open(_prod("bridge")))["rigid_world_run113_to_canonical"])
    ijk = np.indices(shape[:3]).reshape(3, -1).T
    w = ijk @ A[:3, :3].T + A[:3, 3]
    return (w @ P[2, :3] + P[2, 3]).reshape(shape[:3])


def profile(z):
    d = z - PLANE                                        # mm above the plane
    t = np.clip((d - K) / BAND, 0.0, 1.0)
    return np.where(d < K, 0.0, 0.5 - 0.5 * np.cos(np.pi * t)).astype(np.float32)


def cmd_tpl_costmask(a):
    out = {}
    for d in (f"{os.path.abspath(c)}/TPL" for c in a.canons):
        f = f"{d}/d2_b0_costmask.nii.gz"; old = f"{d}/d2_b0_costmask_ones_v1.nii.gz"
        im = nib.load(f)
        if not os.path.exists(old):
            shutil.copy2(f, old)
        c = profile(zcan(im.shape, im.affine))
        tmp = f + ".tmp.nii.gz"; nib.save(nib.Nifti1Image(c, im.affine), tmp); os.replace(tmp, f)
        b = np.asanyarray(nib.load(f"{d}/d2_b0.nii.gz").dataobj) > 0
        out[d] = dict(cost_mean_in_tpl_brain=float(c[b].mean()), zero_vox=int((c == 0).sum()), mirror_max_abs_diff=float(np.abs(c - np.flip(c, 0)).max()))
    run_mask = f"{os.path.abspath(a.run)}/data/B0/TPL/B0/B0_brain_mask.nii.gz" if a.run else ""
    if run_mask and os.path.exists(run_mask):
        shutil.copy2(f"{os.path.abspath(a.canons[0])}/TPL/d2_b0_costmask.nii.gz", run_mask + ".new"); os.replace(run_mask + ".new", run_mask)
        out["run_data"] = run_mask
    rec = dict(rule=f"cost = 0 below canonical z = {PLANE} + {K} mm, cosine to 1 over {BAND} mm, 1 above (design 1c's cut-face rule)", files=out)
    json.dump(rec, open(os.path.abspath(a.record), "w"), indent=1)
    print(json.dumps(rec, indent=1))


def extents(v, A, keep):
    idx = np.argwhere((v > 0) & keep)
    lo, hi = idx.min(0), idx.max(0)
    return [round(float((hi[k] - lo[k] + 1) * abs(A[k, k])), 2) for k in range(3)]


def cmd_tpl_affine(a):
    run = os.path.abspath(a.run)
    spec = json.load(open(f"{run}/run_spec.json")); canon = paths.rerooted(spec["b1"]["canon"])
    std = f"{run}/standardspace.nii.gz"; si = nib.load(std)
    ref = os.path.abspath(a.ref_template)
    ri = nib.load(ref); assert ri.shape[:3] == si.shape[:3] and np.allclose(ri.affine, si.affine), "reference not on the standardspace grid"
    tb, tc = f"{canon}/TPL/d2_b0.nii.gz", f"{canon}/TPL/d2_b0_costmask.nii.gz"
    tmp = tempfile.mkdtemp(prefix="tplfix_", dir=run)
    keep_r = zcan(ri.shape, ri.affine) >= PLANE
    rv = np.asanyarray(ri.dataobj).astype(np.float32) * keep_r
    nib.save(nib.Nifti1Image(rv, ri.affine), f"{tmp}/ref_cut.nii.gz")
    ti = nib.load(tb); keep_t = zcan(ti.shape, ti.affine) >= PLANE
    nib.save(nib.Nifti1Image(np.asanyarray(ti.dataobj).astype(np.float32) * keep_t, ti.affine), f"{tmp}/tpl_cut.nii.gz")
    FSL = os.path.join(fsl.fsldir(), "bin"); env = dict(os.environ, FSLDIR=fsl.fsldir(), FSLOUTPUTTYPE="NIFTI_GZ")
    subprocess.run([f"{FSL}/flirt", "-in", f"{tmp}/tpl_cut.nii.gz", "-ref", f"{tmp}/ref_cut.nii.gz", "-omat", f"{tmp}/tpl12.mat",
                    "-dof", "12", "-inweight", tc], check=True, env=env)
    W = np.linalg.inv(xfm.read_fsl_affine(f"{tmp}/tpl12.mat", tb, std))          # TPL RAS -> standardspace RAS
    F = np.diag([-1.0, 1.0, 1.0, 1.0])
    Ws = 0.5 * (W + F @ W @ F)
    ss, As, zs = xfm._grid(tb); sd, Ad, zd = xfm._grid(std)
    Fs, Fd = xfm.fsl_vox2mm(ss, As, zs), xfm.fsl_vox2mm(sd, Ad, zd)
    M = Fd @ np.linalg.inv(Ad) @ Ws @ As @ np.linalg.inv(Fs)
    sv = np.linalg.svd(Ws[:3, :3], compute_uv=False)
    d = f"{run}/out/affine_step2/TPL"
    for k in ("01", "02"):
        m = f"{d}/scalar_mod_{k}_to_standardspace.mat"
        if not os.path.exists(m.replace(".mat", "_identity_v1.mat")):
            shutil.copy2(m, m.replace(".mat", "_identity_v1.mat"))
        np.savetxt(m, M, fmt="%.10f")
        chk = np.linalg.inv(xfm.read_fsl_affine(m, tb, std)); assert np.abs(chk - Ws).max() < 1e-6
        pay, cm = (("t1_grey", "t1_costmask") if k == "01" else ("d2_b0", "d2_b0_costmask"))
        for nm, f, interp in ((f"scalar_mod_{k}_to_standardspace_brain", pay, "spline"), (f"scalar_mod_{k}_to_standardspace_brain_mask", cm, "trilinear")):
            o = f"{d}/{nm}.nii.gz"
            if not os.path.exists(o.replace(".nii.gz", "_identity_v1.nii.gz")):
                shutil.copy2(o, o.replace(".nii.gz", "_identity_v1.nii.gz"))
            subprocess.run([f"{FSL}/flirt", "-in", f"{canon}/TPL/{f}.nii.gz", "-ref", std, "-applyxfm", "-init", m, "-interp", interp,
                            "-out", o], check=True, env=env)
    tv = np.asanyarray(nib.load(f"{d}/scalar_mod_02_to_standardspace_brain.nii.gz").dataobj).astype(np.float32)
    keep_s = zcan(si.shape, si.affine) >= PLANE
    e_tpl, e_ref = extents(tv, si.affine, keep_s), extents(rv, si.affine, keep_s)
    e_id = extents(np.asanyarray(nib.load(f"{d}/scalar_mod_02_to_standardspace_brain_identity_v1.nii.gz").dataobj).astype(np.float32), si.affine, keep_s)
    sym = float(np.abs(tv - np.flip(tv, 0)).max())
    rec = dict(flirt=dict(dof=12, cost="flirt default (corratio)", inweight=tc, input="TPL d2_b0 cut at the D11 plane", reference=f"{ref} cut at the D11 plane"),
               world_affine_raw=W.tolist(), world_affine_symmetrized=Ws.tolist(), raw_asymmetry=float(np.abs(W - F @ W @ F).max()),
               singular_values=sv.tolist(), det=float(np.linalg.det(Ws[:3, :3])),
               extents_mm_LR_AP_IS_above_plane=dict(tpl_identity=e_id, tpl_affine=e_tpl, affine_b0_template=e_ref),
               tpl_affine_mirror_max_abs_diff=sym, tmp=tmp)
    json.dump(rec, open(f"{run}/out/affine_step2/TPL/tpl_affine.json", "w"), indent=1)
    print(json.dumps({k: rec[k] for k in ("singular_values", "det", "raw_asymmetry", "extents_mm_LR_AP_IS_above_plane", "tpl_affine_mirror_max_abs_diff")}, indent=1))


# ---------------------------------------------------------------------------------------------------- weights
def cmd_step8_weights(a):
    SRC, OUT = os.path.abspath(a.canon2), os.path.abspath(a.out)
    q = json.load(open(f"{SRC}/qc.json"))
    nomri = sorted(S for S, r in q["subjects"].items() if r.get("kind") == "nomri")
    assert len(nomri) == 8, nomri
    os.makedirs(OUT, exist_ok=False)
    for e in sorted(os.listdir(SRC)):
        if e in nomri or e == "qc.json":
            continue
        os.symlink(os.path.realpath(f"{SRC}/{e}") if os.path.islink(f"{SRC}/{e}") else f"{SRC}/{e}", f"{OUT}/{e}")
    chk = {}
    for S in nomri:
        src = os.path.realpath(f"{SRC}/{S}"); d = f"{OUT}/{S}"; os.makedirs(d)
        for p in sorted(glob.glob(f"{src}/*.nii.gz")):
            n = os.path.basename(p)
            if n not in ("d2_b0_costmask.nii.gz", "d2_b0_costmask_flip.nii.gz"):
                os.symlink(p, f"{d}/{n}")
        for suf in ("", "_flip"):
            ti = nib.load(f"{src}/t1_costmask{suf}.nii.gz"); tc = np.asanyarray(ti.dataobj).astype(np.float32)
            pf = np.asanyarray(nib.load(f"{src}/t1_photofrac{suf}.nii.gz").dataobj).astype(np.float32)
            old = np.asanyarray(nib.load(f"{src}/d2_b0_costmask{suf}.nii.gz").dataobj).astype(np.float32)
            brain = np.asanyarray(nib.load(f"{src}/t1_grey_mask{suf}.nii.gz").dataobj) > 0
            photo = pf >= 1.0
            assert np.all(tc[photo] == 1.0), f"{S}{suf}: t1_costmask != 1 somewhere photographed"
            new = np.where(photo, 1.0, tc).astype(np.float32)
            assert np.array_equal(new, tc), f"{S}{suf}: the new b0 mask differs from the blockface weight"
            nib.save(nib.Nifti1Image(new, ti.affine), f"{d}/d2_b0_costmask{suf}.nii.gz")
            nphot = brain & (pf < 1.0)
            chk[f"{S}{'f' if suf else ''}"] = dict(
                brain_vox=int(brain.sum()), photographed_vox=int((brain & photo).sum()), not_fully_photographed_vox=int(nphot.sum()),
                equals_blockface_weight_where_pf_lt_1=bool(np.array_equal(new[nphot], tc[nphot])), one_where_photographed=bool(np.all(new[photo] == 1.0)),
                old_b0_cost_mean_in_brain=float(old[brain].mean()), new_b0_cost_mean_in_brain=float(new[brain].mean()),
                new_mean_where_pf_eq_0=float(new[brain & (pf <= 0)].mean()) if (brain & (pf <= 0)).any() else None,
                new_mean_where_0_lt_pf_lt_1=float(new[brain & (pf > 0) & (pf < 1)].mean()) if (brain & (pf > 0) & (pf < 1)).any() else None)
    q2 = dict(q); q2["s8w"] = dict(decision=STEP8_RULE, rule="d2_b0_costmask_s8 = where(t1_photofrac >= 1, 1, t1_costmask) for the 8 (both mirrors) = their blockface weight",
                                   source=SRC, members_changed=nomri, created=dt.datetime.now().astimezone().isoformat(timespec="seconds"))
    json.dump(q2, open(f"{OUT}/qc.json", "w"), indent=1)
    rec = dict(decision=STEP8_RULE, rule=q2["s8w"]["rule"], canon2=OUT, check=chk, all_pass=all(v["equals_blockface_weight_where_pf_lt_1"] and v["one_where_photographed"] for v in chk.values()),
               created=q2["s8w"]["created"])
    json.dump(rec, open(f"{OUT}/s8w_check.json", "w"), indent=1)
    for k, v in chk.items():
        print(f"{k:7s} brain {v['brain_vox']:8d}  not fully photographed {v['not_fully_photographed_vox']:8d}  b0 cost mean {v['old_b0_cost_mean_in_brain']:.3f} -> "
              f"{v['new_b0_cost_mean_in_brain']:.3f} (pf = 0: {v['new_mean_where_pf_eq_0']}, 0 < pf < 1: {v['new_mean_where_0_lt_pf_lt_1']:.3f})  "
              f"check {'PASS' if v['equals_blockface_weight_where_pf_lt_1'] and v['one_where_photographed'] else 'FAIL'}")
    print("all pass:", rec["all_pass"], "->", OUT)


# ---------------------------------------------------------------------------------------------------- avg
def cmd_affine_templates(a):
    """The affine templates over all inputs, by the patched driver's own averaging (bftemplate/d2b/prep_d2b_affavg.py)."""
    RT = _driver()
    run = os.path.abspath(a.run)
    assert RT.MASKW
    ids = json.load(open(f"{run}/run_spec.json"))["subject_ids"]
    d = f"{run}/out/affine_step2"; rec = {}
    for k in ("01", "02"):
        mod = f"scalar_mod_{k}"
        imgs = [f"{d}/{sid}/{mod}_to_standardspace_brain.nii.gz" for sid in ids]
        masks = [f"{d}/{sid}/{mod}_to_standardspace_brain_mask.nii.gz" for sid in ids]
        for p in imgs + masks:
            assert os.path.exists(p), p
        tb, tm, tw = (f"{d}/{mod}_affine_template_brain.nii.gz", f"{d}/{mod}_affine_template_brain_mask.nii.gz",
                      f"{d}/{mod}_affine_template_brain_mask_weighted.nii.gz")
        for p in (tb, tm, tw):
            assert not os.path.exists(p), f"{p} exists"
        RT.averageImagesWeighted(imgs, tb, masks, None, True)
        RT.averageImages(masks, tm, None, "mean", False)
        subprocess.run(f"fslmaths {tm} -thr 0.1 -bin -mul 7 -add 1 -inm 1 {tw}", shell=True, check=True)
        v = np.asanyarray(nib.load(tb).dataobj).astype(np.float64)
        rec[mod] = dict(brain=tb, mask=tm, weighted=tw, mirror_max_abs_diff=float(np.abs(v - np.flip(v, 0)).max()),
                        p99=float(np.percentile(v[v > 0], 99)))
    json.dump(dict(templates=rec, inputs=len(ids), driver=RT.__file__,
                   rule="the driver's own affT_average_template_in_standardspace functions over all inputs (design 2's 42 + TPL)"),
              open(f"{d}/d2b_affavg.json", "w"), indent=1)
    print(json.dumps(rec, indent=1))


def cmd_step7_templates(a):
    """The step-7 iteration-3 templates of the step-8 run, by the patched driver's own averaging (bftemplate/d2b/prep_d2bs8_avg.py)."""
    RT = _driver()
    run = os.path.abspath(a.run)
    assert RT.MASKW, "LINC_MASKWEIGHTED must be on"
    ids = json.load(open(f"{run}/run_spec.json"))["subject_ids"]
    it = f"{run}/out/nln_step_07/iteration_03"
    t0 = time.time(); rec = {}
    for mod_id, k in enumerate(("01", "02")):
        mod = f"scalar_mod_{k}"
        imgs = [f"{it}/{sid}_{mod}_brain_to_template.nii.gz" for sid in ids]
        masks = [f"{it}/{sid}_{mod}_brain_mask_to_template.nii.gz" for sid in ids]
        bias = [f"{it}/{sid}_to_template_bias_{mod_id}.nii.gz" for sid in ids]
        for p in imgs + masks + bias:
            assert os.path.exists(p), f"missing {p} (prep_d2s8.py step7 first)"
        tb, tm, tw = (f"{it}/nln_template_07_03_{mod}_brain.nii.gz", f"{it}/nln_template_07_03_{mod}_brain_mask.nii.gz",
                      f"{it}/nln_template_07_03_{mod}_brain_mask_weighted.nii.gz")
        for p in (tb, tm, tw):
            assert not os.path.exists(p), f"{p} exists"
        RT.averageImagesWeighted(imgs, tb, masks, bias, True)
        RT.averageImages(masks, tm, None, "mean", False)
        subprocess.run(f"fslmaths {tm} -thr 0.1 -bin -mul 7 -add 1 -inm 1 {tw}", shell=True, check=True)
        v = np.asanyarray(nib.load(tb).dataobj).astype(np.float64)
        rec[mod] = dict(brain=tb, mask=tm, weighted=tw, mirror_r=float(np.corrcoef(v.ravel(), np.flip(v, 0).ravel())[0, 1]),
                        nonzero_frac=float((v != 0).mean()), p99=float(np.percentile(v[v > 0], 99)))
    shutil.copy2(f"{it}/nln_template_07_03_scalar_mod_01_brain.nii.gz", f"{run}/standardspace.nii.gz")
    json.dump(dict(templates=rec, driver=RT.__file__, seconds=round(time.time() - t0, 1),
                   rule="the driver's own average_modalities functions on prep_d2s8.py step7's pulls; standardspace = the BF template"),
              open(f"{it}/s8_step7_avg.json", "w"), indent=1)
    print(json.dumps(rec, indent=1))


# ---------------------------------------------------------------------------------------------------- run trees
def x_symmetric(img):
    A = img.affine
    xc = A[0, 3] + A[0, 0] * np.arange(img.shape[0])
    return abs(A[0, 1]) + abs(A[0, 2]) < 1e-9 and np.allclose(xc, -xc[::-1], atol=1e-6)


def cmd_assemble_plan_a(a):
    """The plan-A MMORF run (bftemplate/d2b/build_d2b_run.py): 42 inputs + TPL, two scalar channels, cost masks, 7 steps at 0.5 mm."""
    B = M; CANON = Path(os.path.abspath(a.canon))
    M.set_ladder(a.steps)
    w = {"BF": a.bf_weight, "B0": a.b0_weight}
    for mod, payload, cmask in MODS:
        B.MODALITIES[mod] = {"brain": payload, "mask": cmask, "quant": False}
        B.NATIVE_MM[mod] = 0.5                       # BF registered at 0.5 mm; the in-vivo b0s are 0.5 mm (FWHM floor 0.562 mm)
    qc = json.load(open(CANON / "qc.json"))["subjects"]
    subs = [s for s in qc if qc[s]["kind"] != "template"] if a.subjects == "all" else a.subjects.split(",")
    subs = sorted(s for s in subs if qc[s]["kind"] == "mri") + sorted(s for s in subs if qc[s]["kind"] != "mri")
    assert "TPL" in qc and qc["TPL"]["kind"] == "template", "no TPL in the canon (prep_d2b.py canon)"
    assert qc[subs[0]]["kind"] == "mri", "the first input (the affine reference) must be a member with MRI"
    run = Path(a.root) / a.name
    if run.exists():
        sys.exit(f"{run} exists: use a new name")

    # every file first: refuse before creating anything
    files = {}
    for s in subs + ["TPL"]:
        for sid, suffix in ([(s, ""), (s + "f", "_flip")] if s != "TPL" else [("TPL", "")]):
            for mod, payload, cmask in MODS:
                brain, cm = CANON / s / f"{payload}{suffix}.nii.gz", CANON / s / f"{cmask}{suffix}.nii.gz"
                for src in (brain, cm):
                    if not src.exists():
                        raise SystemExit(f"{src} missing -- prepare the members first")
                files[(sid, mod)] = (brain, cm)
    ref = nib.load(str(files[(subs[0], "BF")][0]))
    assert x_symmetric(ref), "grid not x-symmetric"
    for (sid, mod), (brain, cm) in files.items():
        for p in (brain, cm):
            h = nib.load(str(p))
            assert h.shape[:3] == ref.shape[:3] and np.allclose(h.affine, ref.affine, atol=1e-6), f"{p}: grid differs from the reference"

    cfg = run / "config"; cfg.mkdir(parents=True)
    (run / "out").mkdir()
    ids = []
    for s in subs + ["TPL"]:
        for sid in ((s, s + "f") if s != "TPL" else ("TPL",)):
            for mod, _, _ in MODS:
                d = run / "data" / mod / sid / mod; d.mkdir(parents=True)
                brain, cm = files[(sid, mod)]
                M.link_or_copy(brain, d / f"{mod}_unbiased_brain.nii.gz")
                M.link_or_copy(cm, d / f"{mod}_brain_mask.nii.gz")
            ids.append(sid)
    (run / "subject_ids.csv").write_text("\n".join(ids) + "\n")

    # ---- standard space: the cost-weighted mean of modality 01 (BF) over a mirror-closed set, hence exactly symmetric ----
    num = den = None
    for sid in ids:
        g, c = files[(sid, "BF")]
        gv = np.asanyarray(nib.load(str(g)).dataobj).astype(np.float64)
        cv = np.clip(np.asanyarray(nib.load(str(c)).dataobj).astype(np.float64), 0, None)
        num = cv * gv if num is None else num + cv * gv
        den = cv if den is None else den + cv
    mean = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0).astype(np.float32)
    nib.save(nib.Nifti1Image(mean, ref.affine), str(run / "standardspace.nii.gz"))
    asym = float(np.abs(mean - np.flip(mean, 0)).max())
    assert asym < 1e-3 * max(float(np.abs(mean).max()), 1e-6), f"standardspace not symmetric (max diff {asym})"

    # ---- config: two scalar blocks per step, the moving cost mask switched on at every level for both --------------------
    mods = [(mod, w[mod]) for mod, _, _ in MODS]
    tree = B.DATA_TREE_HEAD
    for k, (mod, _, _) in enumerate(MODS, 1):
        tree += f"\n!{{scalar_mod_{k:02d}_dir}}\n    ->scalar_mod_{mod} (scalar_mod_{k:02d})\n"
    tree += "\n!{config_dir}\n"
    tree += "".join(f"    mmorf_params_step{i:02d}{{ext_ini}} (mmorf_params_{i:02d})\n" for i in range(1, B.N_STEPS + 1))
    tree += "    affine_zero_warp_for_deltaWarp{ext_nii} (zero_warp_affine)\n\n"
    up = Path(_driver_dir()) / "tempconfig"
    tree += (up / "data_structure.tree").read_text().split("!{template_dir}", 1)[1].join(["!{template_dir}", ""])
    (cfg / "data_structure.tree").write_text(tree)
    for mod, _, _ in MODS:
        (cfg / f"scalar_mod_{mod}.tree").write_text(
            "{sub_id} (scalar_sub_dir)\n"
            f"    {mod} (scalar_mod_dir)\n"
            f"        {mod}_unbiased_brain{{ext_nii}} (scalar_mod_brain)\n"
            f"        {mod}_brain_mask{{ext_nii}} (scalar_mod_brain_mask)\n")
    for i in range(1, B.N_STEPS + 1):
        ini = B.step_ini(i, a.lambda_scale, mods, (), fwhm_floor=True)
        n = i + 1
        old = f"use_mask_mov_scalar     = {' '.join(['0'] * n)}\n"
        assert ini.count(old) == len(MODS), f"step {i}: {ini.count(old)} use_mask_mov_scalar lines, expected {len(MODS)}"
        ini = ini.replace(old, f"use_mask_mov_scalar     = {' '.join(['1'] * n)}\n")
        assert ini.count(f"use_mask_ref_scalar     = {' '.join(['0'] * n)}\n") == len(MODS)
        (cfg / f"mmorf_params_step{i:02d}.ini").write_text(ini)
    nib.save(nib.Nifti1Image(np.zeros(ref.shape[:3] + (3,), np.float32), ref.affine), str(cfg / "affine_zero_warp_for_deltaWarp.nii.gz"))

    spec = {"name": a.name, "payload": {m: p for m, p, _ in MODS},
            "modalities": [{"name": m, "lambda_scalar": w[m], "quantitative": False} for m, _, _ in MODS],
            "tensor_modalities": [], "quant_scalar_mods": [],
            "lambda_scale": a.lambda_scale, "fwhm_floor": True, "native_mm": {m: B.NATIVE_MM[m] for m, _, _ in MODS},
            "fwhm_floor_mm": {m: B.fwhm_floor_mm(B.NATIVE_MM[m]) for m, _, _ in MODS},
            "n_steps": B.N_STEPS, "subject_ids": ids, "subjects": subs,
            "mirrors": {sid: sid.endswith("f") and sid[:-1] in subs for sid in ids},
            "d2b": {"decision": "design 2b: the 21 brains x 2 mirrors + TPL; every member's b0 at full weight in steps 1-7",
                    "tpl": {"member": "TPL", "canon": str(CANON / "TPL"), "channels": "B0 only (BF image 0, BF cost 0); entered once",
                            "link": "TPL's construction warp = the MRI template -> 2b template link"},
                    "affine": "not run: prep_d2b.py affine (design 2's 42 + TPL identity) + prep_d2b_affavg.py; driver -aff False -nln True",
                    "unbiasing": "(a) as is: the average warp is weighted by the BF cost (TPL at eps)"},
            "b1": {"stage": 2, "canon": str(CANON), "cost_mask": {m: c for m, _, c in MODS},
                   "use_mask_mov_scalar": 1, "use_mask_ref_scalar": 0,
                   "standardspace": "cost-weighted mean of modality 01 (BF)",
                   "driver": DRIVER_PATCHED, "driver_switch": "LINC_MASKWEIGHTED=1",
                   "members": dict({s: qc[s]["kind"] for s in subs}, TPL="template")},
            "d2": {"decisions": {"D34": "b0 + blockface, 21 brains x 2 mirrors, design 1c's recipe",
                                 "D35": "full weight for the b0 channel of the brains without MRI (steps 1-7)"}}}
    json.dump(spec, open(run / "run_spec.json", "w"), indent=1)
    print(f"run           : {run}")
    print(f"inputs        : {len(ids)} = {len(subs)} brains x 2 mirrors ({sum(qc[s]['kind'] == 'mri' for s in subs)} with MRI first) + TPL")
    print(f"modalities    : " + ", ".join(f"{m} ({p}, cost {c}, lambda {w[m]})" for m, p, c in MODS))
    print(f"standardspace : cost-weighted mean of BF; max|x - mirror(x)| = {asym:.3e}")
    print(f"steps         : {B.N_STEPS}; lambda scale {a.lambda_scale}; use_mask_mov_scalar = 1 at every level for both")


def cmd_assemble_step8(a):
    """The step-8 MMORF run (bftemplate/d2/build_d2s8_run.py): plan A's inputs on canon2 (0.25 mm), plan A's configs for steps 1-7,
    step 8 from the 8-step ladder; the driver resumes at 8:1."""
    B = M
    canon2, pa, box = Path(ab(a.canon2)), Path(ab(a.plan_a)), Path(ab(a.box))
    M.set_ladder(8)
    for mod, payload, cmask in MODS:
        B.MODALITIES[mod] = {"brain": payload, "mask": cmask, "quant": False}
        B.NATIVE_MM[mod] = NATIVE[mod]
    spa = json.load(open(pa / "run_spec.json"))
    ids = [l.strip() for l in open(pa / "subject_ids.csv") if l.strip()]
    assert ids == spa["subject_ids"], "plan A's subject_ids.csv and run_spec disagree"
    assert [m["name"] for m in spa["modalities"]] == [m for m, _, _ in MODS], spa["modalities"]
    assert spa["lambda_scale"] == 10.0 and spa["n_steps"] == 7, (spa["lambda_scale"], spa["n_steps"])
    w = {m["name"]: m["lambda_scalar"] for m in spa["modalities"]}
    subs = spa["subjects"]
    q2 = json.load(open(canon2 / "qc.json"))
    assert set(subs) <= set(q2["subjects"]), f"canon2 lacks {set(subs) - set(q2['subjects'])}"
    run = Path(ab(a.root)) / a.name
    if run.exists():
        sys.exit(f"{run} exists: use a new name")

    files = {}
    for sid in ids:
        S = sid[:-1] if sid.endswith("f") else sid; suffix = "_flip" if sid.endswith("f") else ""
        for mod, payload, cmask in MODS:
            brain, cm = canon2 / S / f"{payload}{suffix}.nii.gz", canon2 / S / f"{cmask}{suffix}.nii.gz"
            for src in (brain, cm):
                if not src.exists():
                    raise SystemExit(f"{src} missing -- prepare the 0.25 mm members first")
            files[(sid, mod)] = (brain, cm)
    ref = nib.load(str(files[(ids[0], "BF")][0]))
    for (sid, mod), (brain, cm) in files.items():
        for p in (brain, cm):
            h = nib.load(str(p))
            assert h.shape[:3] == ref.shape[:3] and np.allclose(h.affine, ref.affine, atol=1e-6), f"{p}: grid differs"
    bimg = nib.load(str(box)); bA = bimg.affine
    xc = bA[0, 3] + bA[0, 0] * np.arange(bimg.shape[0]); assert np.allclose(xc, -xc[::-1], atol=1e-6), "box not x-symmetric"
    off = (bA[:3, 3] - ref.affine[:3, 3]) / np.diag(ref.affine)[:3]
    assert np.allclose(np.diag(bA)[:3], np.diag(ref.affine)[:3]) and np.allclose(off, np.round(off), atol=1e-4), "box not on canon2's lattice"

    cfg = run / "config"; cfg.mkdir(parents=True)
    (run / "out").mkdir()
    for sid in ids:
        for mod, _, _ in MODS:
            d = run / "data" / mod / sid / mod; d.mkdir(parents=True)
            brain, cm = files[(sid, mod)]
            M.link_or_copy(brain, d / f"{mod}_unbiased_brain.nii.gz")
            M.link_or_copy(cm, d / f"{mod}_brain_mask.nii.gz")
    (run / "subject_ids.csv").write_text("\n".join(ids) + "\n")

    # ---- config: plan A's steps 1-7 verbatim (skipped by --resume_from 8:1), step 8 from the 8-step ladder -------------------------
    tree = B.DATA_TREE_HEAD
    for k, (mod, _, _) in enumerate(MODS, 1):
        tree += f"\n!{{scalar_mod_{k:02d}_dir}}\n    ->scalar_mod_{mod} (scalar_mod_{k:02d})\n"
    tree += "\n!{config_dir}\n"
    tree += "".join(f"    mmorf_params_step{i:02d}{{ext_ini}} (mmorf_params_{i:02d})\n" for i in range(1, 9))
    tree += "    affine_zero_warp_for_deltaWarp{ext_nii} (zero_warp_affine)\n\n"
    up = Path(_driver_dir()) / "tempconfig"
    tree += (up / "data_structure.tree").read_text().split("!{template_dir}", 1)[1].join(["!{template_dir}", ""])
    (cfg / "data_structure.tree").write_text(tree)
    for mod, _, _ in MODS:
        shutil.copy2(pa / "config" / f"scalar_mod_{mod}.tree", cfg / f"scalar_mod_{mod}.tree")
    for i in range(1, 8):
        shutil.copy2(pa / "config" / f"mmorf_params_step{i:02d}.ini", cfg / f"mmorf_params_step{i:02d}.ini")
    mods = [(mod, w[mod]) for mod, _, _ in MODS]
    ini = B.step_ini(8, spa["lambda_scale"], mods, (), fwhm_floor=True)
    old = f"use_mask_mov_scalar     = {' '.join(['0'] * 9)}\n"
    assert ini.count(old) == len(MODS), f"step 8: {ini.count(old)} use_mask_mov_scalar lines, expected {len(MODS)}"
    ini = ini.replace(old, f"use_mask_mov_scalar     = {' '.join(['1'] * 9)}\n")
    assert ini.count(f"use_mask_ref_scalar     = {' '.join(['0'] * 9)}\n") == len(MODS)
    # steps 1-7 of the 8-step ladder must be plan A's own (same knots, lambdas, iterations) except the BF FWHM floor
    p7 = (pa / "config" / "mmorf_params_step07.ini").read_text()
    head = lambda s: [l for l in s.splitlines() if l.split("=")[0].strip() in ("warp_res_init", "hires", "optimiser_max_it_lowres",
                                                                               "optimiser_max_it_hires")]
    assert head(p7) == head(ini), (head(p7), head(ini))
    (cfg / "mmorf_params_step08.ini").write_text(ini)
    nib.save(nib.Nifti1Image(np.zeros(bimg.shape[:3] + (3,), np.float32), bA), str(cfg / "affine_zero_warp_for_deltaWarp.nii.gz"))

    spec = {"name": a.name, "payload": {m: p for m, p, _ in MODS},
            "modalities": [{"name": m, "lambda_scalar": w[m], "quantitative": False} for m, _, _ in MODS],
            "tensor_modalities": [], "quant_scalar_mods": [],
            "lambda_scale": spa["lambda_scale"], "fwhm_floor": True, "native_mm": dict(NATIVE),
            "fwhm_floor_mm": {m: B.fwhm_floor_mm(NATIVE[m]) for m, _, _ in MODS},
            "n_steps": 8, "subject_ids": ids, "subjects": subs, "mirrors": spa["mirrors"],
            "b1": dict(spa["b1"], canon=str(canon2), stage=2),
            "d2": spa["d2"],
            "s8": {"decision": "step 8 at 0.25 mm knots after plan A's step 7, -nit 3 as the MRI rebuild",
                   "plan_a_run_spec": str(pa / "run_spec.json"), "box": str(box), "canon2": str(canon2),
                   "resume": "-aff False -nln True --resume_from 8:1 -nres 8 -nit 3",
                   "steps_1_7": "plan A's own configs (bftemplate/d2/s8/plan_a/config), skipped",
                   "step_8": "build_bf_run.py 8-step ladder: 9 levels, knots 16,16,16,8,4,2,1,0.5,0.25 mm; FWHM floor per channel bandwidth",
                   "state_from_plan_a": "affine_step2 (prep_d2s8.py mats) + nln_step_07/iteration_03 (prep_d2s8.py step7 + prep_d2s8_avg.py)"}}
    json.dump(spec, open(run / "run_spec.json", "w"), indent=1)
    print(f"run     : {run}")
    print(f"inputs  : {len(ids)} (plan A's order), canon2 {ref.shape[:3]}, box {bimg.shape[:3]} ({np.prod(bimg.shape[:3]) / 1e6:.1f} Mvox)")
    print("step 8  :\n" + "\n".join("  " + l for l in ini.splitlines()[:6]))
    print("fwhm    : " + ", ".join(f"{m} floor {B.fwhm_floor_mm(NATIVE[m]):.3f}" for m, _, _ in MODS))


def cmd_crop(a):
    """The build frame (bftemplate/d2/make_frame_d2.py --canon C --crop): the union of every member's brain mask (both mirrors) +
    pad mm on each side, x-symmetric about x = 0, dims a multiple of mult, cut from the prep canon as whole voxels."""
    rep, sl, A2 = frame_box(ab(a.canon), a.pad, a.mult, a.extra)
    print(json.dumps({k: v for k, v in rep.items() if k != "affine"}))
    if min(min(v) for v in rep["union_to_canon_edge_vox"].values()) < 1:
        print("REFUSED: a member touches the canon grid's edge (it may have been clipped by prep): prep on a larger frame"); sys.exit(2)
    if not rep["x_symmetric"] or rep["min_margin_mm"] < a.pad - 1e-6:
        print("REFUSED: not x-symmetric, or < pad mm somewhere"); sys.exit(2)
    assert a.out and not os.path.exists(a.out), "--out must be a new directory"
    qc = json.load(open(f"{ab(a.canon)}/qc.json"))
    os.makedirs(a.out)
    for S in qc["subjects"]:
        os.makedirs(f"{a.out}/{S}")
        for f in sorted(glob.glob(f"{ab(a.canon)}/{S}/*.nii.gz")):
            img = nib.load(f)
            assert img.shape[:3] == tuple(rep["canon_shape"]), f"{f}: not on the canon grid"
            nib.save(nib.Nifti1Image(cut(np.asanyarray(img.dataobj), sl), A2, img.header), f"{a.out}/{S}/{os.path.basename(f)}")
    qc["grid"] = dict(shape=rep["shape"], affine=A2.tolist()); qc["frame_d2"] = dict(rep, source_canon=os.path.abspath(a.canon))
    json.dump(qc, open(f"{a.out}/qc.json", "w"), indent=1)
    print("cut ->", a.out)


# ---------------------------------------------------------------------------------------------------- the patched driver's command
def driver_command(run, extra=()):
    """The driver's command line for a run (bftemplate/d2b/run_template_d2b.sh): the patched driver, the run's two scalar channels,
    configs and tree, the 6-dof standard-space alignment; run it with LINC_MASKWEIGHTED=1, FSLDIR set, MMORF 1.0.1 on PATH and, first
    on PATH, linc/construction/blockface_shim (TPL's MMORF call rewritten). Plan A: extra = -aff False -nln True -nres 7 -nit 3;
    step 8: -aff False -nln True --resume_from 8:1 -nres 8 -nit 3."""
    j = json.load(open(f"{run}/run_spec.json"))
    scal = [os.path.join(run, "data", m["name"]) + "/" for m in j["modalities"]]
    return ([os.path.join(_driver_dir(), "run_template_construction.py"), "--input_scalar", *scal, "--config", f"{run}/config/",
             "--tree", f"{run}/config/data_structure.tree", "--standardspace", f"{run}/standardspace.nii.gz",
             "--subids", f"{run}/subject_ids.csv", "--dof", "6", "-o", f"{run}/out/", "--cpuq", "cpu.q", "--gpuq", "gpu.q",
             "-aff", "True", "-nln", "True", *extra])


def cmd_tpl_standins(a):
    """TPL's stand-ins for its MMORF call (the step-8 chain): its blockface moving image = its own b0, its blockface mask = ones."""
    run = ab(a.run); canon2 = ab(a.canon2)
    d = f"{run}/data/BF/TPL/BF"
    shutil.copy(f"{canon2}/TPL/d2_b0.nii.gz", f"{d}/BF_mmorf_image.nii.gz")
    m = nib.load(f"{d}/BF_brain_mask.nii.gz")
    nib.save(nib.Nifti1Image(np.ones(m.shape[:3], np.float32), m.affine), f"{d}/BF_mmorf_mask.nii.gz")


def main(argv=None):
    import argparse
    from linc.construction import blockface_finish as F
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep-frame", help="frame1c grown by whole voxels: the grid the 0.5 mm members are prepared on")
    p.add_argument("--out", required=True); p.add_argument("--grow-mm", type=float, default=8.0)
    p = sp.add_parser("members", help="the 0.5 mm members (both mirrors) on --frame")
    p.add_argument("--frame", required=True); p.add_argument("--out", required=True)
    p.add_argument("--root", required=True, help="the completion's per-brain folders"); p.add_argument("--cache", required=True)
    p.add_argument("--mri-subjects", default=MRI13); p.add_argument("--nomri-subjects", default=NOMRI8)
    p.add_argument("--s2lin-suffix", default="prod"); p.add_argument("--k-mm", type=float, default=1.0); p.add_argument("--band-mm", type=float, default=2.0)
    p = sp.add_parser("crop", help="the build frame: the members' union + pad mm, x-symmetric, dims x mult; cut the canon")
    p.add_argument("--canon", required=True); p.add_argument("--pad", type=float, default=4.0); p.add_argument("--mult", type=int, default=4)
    p.add_argument("--extra", nargs="*", default=[]); p.add_argument("--out", required=True)
    p = sp.add_parser("s8-frame", help="the 0.5 mm canon's grid as a NIfTI (its exact 2x lattice holds the 0.25 mm members)")
    p.add_argument("--canon", required=True); p.add_argument("--out", required=True)
    p = sp.add_parser("s8-inputs", help="the 0.25 mm members"); p.add_argument("--canon", required=True); p.add_argument("--frame", required=True)
    p.add_argument("--out", required=True); p.add_argument("-j", type=int, default=6); p.add_argument("--subjects", default="")
    p = sp.add_parser("s8-box", help="the step-8 grid"); p.add_argument("--template", nargs="+", required=True); p.add_argument("--canon2", required=True)
    p.add_argument("--out", required=True); p.add_argument("--pad", type=float, default=4.0); p.add_argument("--thr", type=float, default=0.01)
    p = sp.add_parser("s8-boxcheck"); p.add_argument("--template", nargs="+", required=True); p.add_argument("--box", required=True)
    p.add_argument("--thr", type=float, default=0.01); p.add_argument("--min-mm", type=float, default=3.0)
    p = sp.add_parser("tpl-canon", help="TPL in a canon (0.5 or 0.25 mm) beside symlinks to the 21 members")
    p.add_argument("--vox", type=float, required=True, choices=[0.5, 0.25]); p.add_argument("--source-canon", required=True); p.add_argument("--out", required=True)
    p = sp.add_parser("tpl-costmask", help="TPL's b0 cost mask (the brainstem plane rule)"); p.add_argument("--canons", nargs="+", required=True)
    p.add_argument("--run", default=""); p.add_argument("--record", required=True)
    p = sp.add_parser("run-plan-a", help="the plan-A run tree"); p.add_argument("--canon", required=True); p.add_argument("--root", required=True)
    p.add_argument("--name", required=True); p.add_argument("--steps", type=int, default=7, choices=[7]); p.add_argument("--subjects", default="all")
    p.add_argument("--lambda-scale", type=float, default=10.0); p.add_argument("--bf-weight", type=float, default=1.0); p.add_argument("--b0-weight", type=float, default=1.0)
    p = sp.add_parser("copy-affine", help="plan A's affine stage: the design-2 affines of the 42 + TPL at the identity")
    p.add_argument("--plan-a-affine", required=True); p.add_argument("--run", required=True)
    p = sp.add_parser("tpl-affine", help="TPL's 12-dof affine to the affine b0 template, symmetrized"); p.add_argument("--run", required=True)
    p.add_argument("--ref-template", required=True)
    p = sp.add_parser("affine-templates", help="the affine templates over all inputs (the patched driver's averaging; MMORF env)")
    p.add_argument("--run", required=True)
    p = sp.add_parser("step8-weights", help="the step-8 rule: the 8 brains' b0 cost = their blockface cost where not fully photographed")
    p.add_argument("--canon2", required=True); p.add_argument("--out", required=True)
    p = sp.add_parser("run-step8", help="the step-8 run tree"); p.add_argument("--canon2", required=True); p.add_argument("--plan-a", required=True)
    p.add_argument("--box", required=True); p.add_argument("--root", required=True); p.add_argument("--name", required=True)
    p = sp.add_parser("tpl-standins", help="TPL's stand-in blockface image and mask for its MMORF call"); p.add_argument("--run", required=True)
    p.add_argument("--canon2", required=True)
    p = sp.add_parser("s8-mats", help="plan A's affines for canon2 -> box"); p.add_argument("--affine-dir", required=True); p.add_argument("--canon", required=True)
    p.add_argument("--canon2", required=True); p.add_argument("--box", required=True); p.add_argument("--run", required=True); p.add_argument("--standard05", required=True)
    p = sp.add_parser("s8-step7", help="the step-7 iteration-3 state at 0.25 mm"); p.add_argument("--plan-a-it", required=True)
    p.add_argument("--canon", required=True); p.add_argument("--canon2", required=True); p.add_argument("--run", required=True); p.add_argument("-j", type=int, default=8)
    p = sp.add_parser("step7-templates", help="the step-7 templates of the step-8 run (the patched driver's averaging; MMORF env)")
    p.add_argument("--run", required=True)
    p = sp.add_parser("driver-command", help="print the patched driver's command line for a run"); p.add_argument("--run", required=True)
    p.add_argument("extra", nargs=argparse.REMAINDER)
    p = sp.add_parser("finish", help="the template's blockface channel from the step-8 warps"); p.add_argument("--run", required=True)
    p.add_argument("--arm", required=True); p.add_argument("--canon", required=True); p.add_argument("--iteration", required=True)
    p.add_argument("--out", required=True); p.add_argument("--precision-json", required=True); p.add_argument("-j", type=int, default=6)
    p = sp.add_parser("finish-b0", help="the template's b0 channel from the step-8 warps"); p.add_argument("--run", required=True)
    p.add_argument("--arm", required=True); p.add_argument("--canon", required=True); p.add_argument("--iteration", required=True)
    p.add_argument("--out", required=True); p.add_argument("--weights-canon", default=""); p.add_argument("-j", type=int, default=6)
    p = sp.add_parser("embed", help="the box products onto the run-113 grid's 2x refinement"); p.add_argument("--run", required=True)
    p.add_argument("--src", required=True); p.add_argument("--dst", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "prep-frame":
        return make_prep_frame(ab(a.out), a.grow_mm)
    if a.cmd == "members":
        set_frame(ab(a.frame)); return cmd_members(a)
    if a.cmd == "crop":
        return cmd_crop(a)
    if a.cmd == "driver-command":
        print(" ".join(driver_command(ab(a.run), a.extra))); return
    if a.cmd == "finish":
        return F.finish_blockface(ab(a.run), a.arm, ab(a.canon), ab(a.iteration), ab(a.out), ab(a.precision_json), j=a.j)
    if a.cmd == "finish-b0":
        return F.finish_b0(ab(a.run), a.arm, ab(a.canon), ab(a.iteration), ab(a.out), ab(a.weights_canon) if a.weights_canon else "", j=a.j)
    if a.cmd == "embed":
        return F.embed(ab(a.run), ab(a.src), ab(a.dst))
    {"s8-frame": cmd_frame, "s8-inputs": cmd_inputs, "s8-box": cmd_box, "s8-boxcheck": cmd_boxcheck, "tpl-canon": cmd_tpl_canon,
     "tpl-costmask": cmd_tpl_costmask, "run-plan-a": cmd_assemble_plan_a, "copy-affine": cmd_copy_affine, "tpl-affine": cmd_tpl_affine,
     "affine-templates": cmd_affine_templates, "step8-weights": cmd_step8_weights, "run-step8": cmd_assemble_step8,
     "tpl-standins": cmd_tpl_standins, "s8-mats": cmd_mats, "s8-step7": cmd_step7, "step7-templates": cmd_step7_templates}[a.cmd](a)


if __name__ == "__main__":
    main()
