"""The blockface template's finish (Methods, "Blockface template"; reproduce/templates/blockface.md, stage E): the published template
is not MMORF's own average. Every input is pulled ONCE from its 0.25 mm completed volume through its final (step-8) MMORF warp,
on the step-8 box (the run-113 box's 0.25 mm lattice), and the members are averaged with precision weights:
    recon point x -> the input's MMORF warp (fsl_rel, on its canon2 0.25 mm grid) -> x-mirror if it is a flipped twin (exact: the grid
    is symmetric about x = 0) -> the seam flattening (from the completion cache's seam.json; the identity here) -> sample.
Per member, a REAL part and a SYNTHETIC part (precision rule; the author's ruling, w = 1/(tau^2 + sigma^2)):
    a_i = r_i [phi_i / tau2 + (1 - phi_i) / (tau2 + s2_interp)],   b_i = (1 - r_i) m_i / (tau2 + s2_syn(D_i))
    template = sum_i (a_i R_i + b_i S_i) / sum_i (a_i + b_i)
with r_i the member's seam ramp, phi_i its photographed fraction, D_i the depth past its photographed end, m_i = 1 for a brain with
MRI and 0 for one without (whose synthesis carries no weight); tau2, s2_interp and s2_syn(d) from the precision JSON. The
non-tissue shell inside each member's photographed extent gets weight 0 (shell rule, 1 mm). Every product is symmetrized,
0.5 (V + M(V)), and its exact symmetry asserted.
The b0 channel is pulled through the same warps: every member's brain at full weight, except the 8 brains without MRI, whose
weight is their step-8 b0 cost mask (the author's step-8 rule); TPL (the MRI template's b0) enters once, through the bridge.
`embed` pads the box products onto the run-113 grid's 2x refinement at an integer offset (no resampling).

Ported from the analysis project: bftemplate/finish_t1.py (the precision rule with the shell rule, the rule of record; the eps and
handover rules were superseded), bftemplate/d2b/finish_d2bs8w_b0.py, bftemplate/d2b/finish_d2bs8.py (the step-8 box and `embed`),
bftemplate/finish_hr.py, bftemplate/prep_t1.py, bftemplate/d2/prep_d2.py (the one b0 rule) and synthesis/code/flatten_seam.py
(`rotation_to`); function bodies unchanged. Paths recorded inside the products are re-rooted (linc.paths.rerooted); the others
come from configs/products_blockface.yaml.
"""
import argparse
import glob
import hashlib
import json
import os
import tempfile
import time

import numpy as np
import nibabel as nib

from linc import paths, xfm
from linc.registration import xfmtools as X

ACC_NEW = ("a", "aph", "bw", "gnum", "b", "r0n", "r1n", "r2n", "w0n", "w1n", "w2n", "wr", "gr", "wsyn", "gsyn",
           "rr0", "rr1", "rr2", "rs0", "rs1", "rs2", "wwr0", "wwr1", "wwr2", "wws0", "wws1", "wws2", "bsh")
SHELL_MM = 1.0
MIRROR = np.diag([-1.0, 1.0, 1.0, 1.0])


def _prod(name):
    return str(paths.products("products_blockface")[name])


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def run113_grid(factor):
    """The run-113 0.5 mm box refined `factor` times (bftemplate/finish_hr.py `grid`)."""
    ref = nib.load(_prod("run113_frame")); v = np.diag(ref.affine)[:3]
    A = ref.affine.copy(); A[:3, :3] = ref.affine[:3, :3] / factor
    A[:3, 3] = ref.affine[:3, 3] - v / 2 + v / (2 * factor)
    shp = tuple(int(s * factor) for s in ref.shape[:3])
    xc = A[0, 3] + A[0, 0] * np.arange(shp[0]); assert np.allclose(xc, -xc[::-1], atol=1e-6)
    return shp, A


def _subjects(canon):
    """canon/qc.json's members, their recorded completion folders re-rooted on this machine."""
    subj = json.load(open(f"{canon}/qc.json"))["subjects"]
    for q in subj.values():
        if isinstance(q, dict):
            for k in ("root", "cache"):
                if q.get(k):
                    q[k] = paths.rerooted(q[k])
    return subj


def rotation_to(n, target):
    """Smallest rotation carrying unit vector n onto unit vector target (Rodrigues)."""
    n = n / np.linalg.norm(n); target = target / np.linalg.norm(target)
    v = np.cross(n, target); c = float(np.dot(n, target))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + c))


def unflatten_map(S, cache, aff, shape):
    """Template/placed world -> seam-flattened world, from the rotation recorded in <cache>/seam.json."""
    n = np.array(json.load(open(f"{cache}/seam.json"))[S]["normal"], float)
    n /= np.linalg.norm(n)
    ax = int(np.argmax(np.abs(aff[:3, :3][1])))
    tgt = aff[:3, ax] / np.linalg.norm(aff[:3, ax])
    if np.dot(n, tgt) < 0:
        tgt = -tgt
    R = rotation_to(n, tgt)
    c = (aff @ np.r_[0.5 * (np.array(shape[:3]) - 1), 1])[:3]
    return X.Map(L=np.r_[np.c_[R, c - R @ c], [[0, 0, 0, 1]]]), R


def iteration_dir(run, which):
    its = [d for d in sorted(glob.glob(f"{run}/out/nln_step_*/iteration_*"))
           if glob.glob(f"{d}/*_to_template_warp_resampled_unbiased_full.nii.gz")]
    if which == "latest":
        return its[-1]
    return [d for d in its if d.endswith(which)][-1]


def s2_syn_fn(pj):
    """sigma^2_syn(d): the fitted monotone curve (d_mm, s2) with its extrapolation (flat beyond the last point unless the
    JSON gives a slope)."""
    c = pj["s2_syn"]
    d = np.asarray(c["d_mm"], np.float32); v = np.asarray(c["s2"], np.float32); slope = float(c.get("slope_beyond", 0.0))
    def f(x):
        x = np.maximum(np.asarray(x, np.float32), 0.0)
        return np.where(x <= d[-1], np.interp(x, d, v), v[-1] + slope * (x - d[-1])).astype(np.float32)
    return f


def shell_mask(S, q, affine, brain):
    """v2 SHELL rule (orchestrator 2026-09-27 02:2x, the user's principle: average real tissue, synthesis only where nothing real
    exists): a synthesised voxel of this member within SHELL_MM of its photographed tissue AND inside that column's photographed
    A-P extent (seam_depth >= 0: not in the caudal block past the column's last real voxel) is a REAL observation of "no tissue"
    (interhemispheric fissure, sulcal CSF, pial mismatch). Template weight 0, counted as uncovered. The caudal exclusion keeps the
    first mm past every cut face synthetic (without it the join would get a zero-weight gap line)."""
    from scipy import ndimage as ndi
    src = f"{q['root']}/{S}"
    rm = np.asanyarray(nib.load(f"{q['cache']}/{S}/deploy/real_mask.nii.gz").dataobj) > 0
    d = np.asanyarray(nib.load(f"{src}/{S}_seam_depth.nii.gz").dataobj).astype(np.float32)
    near = ndi.distance_transform_edt(~rm, sampling=np.abs(np.diag(affine)[:3])) <= SHELL_MM
    return (brain > 0.5) & ~rm & near & (d > -998.0) & (d >= 0.0)


def source_for(S, q):
    src = f"{q['root']}/{S}"
    ci = nib.load(f"{src}/{S}_bf_completed.nii.gz")
    flat, _ = unflatten_map(S, q["cache"], ci.affine, ci.shape)
    vols = dict(grey=np.asanyarray(ci.dataobj).astype(np.float32),
                prov=np.asanyarray(nib.load(f"{src}/{S}_provenance.nii.gz").dataobj).astype(np.float32),
                brain=np.asanyarray(nib.load(f"{q['cache']}/{S}/deploy/brain.nii.gz").dataobj).astype(np.float32))
    r = np.asanyarray(nib.load(f"{src}/{S}_bf_completed_rgb.nii.gz").dataobj).astype(np.float32)
    for k in range(3):
        vols[f"rgb{k}"] = r[..., k]
    ramp_nm = "seam_ramp_real" if os.path.exists(f"{src}/{S}_seam_ramp_real.nii.gz") else "seam_ramp"   # the reliability
    for key, nm in (("real", "bf_real"), ("synth", "bf_synth"), ("ramp", ramp_nm), ("depth", "seam_depth"),   # ramp, not the
                    ("phi", "photo_fraction")):                                                            # composite's fade
        f = f"{src}/{S}_{nm}.nii.gz"
        if os.path.exists(f):
            vols[key] = np.asanyarray(nib.load(f).dataobj).astype(np.float32)
    if q.get("shell_zero"):                               # v2: the non-tissue shell inside the photographed extent (shell_mask)
        vols["shell"] = shell_mask(S, q, ci.affine, vols["brain"]).astype(np.float32)
    if "depth" in vols:                                   # columns with no photographed voxel (-999): the member's deepest
        d = vols["depth"]; fin = d > -998.0               # synthesis, finite, so the trilinear sample stays well-behaved
        vols["depth"] = np.where(fin, d, float(d[fin].min()) if fin.any() else 0.0).astype(np.float32)
    aff = ci.affine

    def sample(pts):
        qq = flat(pts)
        return {k: X.sample(v, aff, qq, order=1) for k, v in vols.items()}
    return sample


def accumulate(job):
    """The precision rule: real and synthetic parts accumulated separately with continuous per-member weights (module
    docstring)."""
    ids, a, it, shp, A, subj, pre = job
    acc = {k: np.zeros(shp, np.float32) for k in ACC_NEW}
    pj = json.load(open(a.precision_json)) if a.rule == "precision" else None
    f_s2 = s2_syn_fn(pj) if pj else None
    for sid in ids:
        S = sid[:-1] if sid.endswith("f") else sid
        flip = sid.endswith("f")
        m_i = 1.0 if subj[S]["kind"] == "mri" else 0.0                      # scaffold synthesis: weight 0
        W = xfm.Warp(f"{it}/{sid}_scalar_mod_01_to_template_warp_resampled_unbiased_full.nii.gz", "fsl_rel",
                     src_grid=f"{a.canon}/{S}/t1_grey{'_flip' if flip else ''}.nii.gz")
        smp = source_for(S, subj[S])
        for z0 in range(0, shp[2], 8):
            z1 = min(shp[2], z0 + 8)
            g = X.grid_world(shp, A, z0, z1)
            if pre is not None:
                g = g @ pre[:3, :3].T + pre[:3, 3]
            pts = W.pull(g)
            if flip:
                pts = pts @ MIRROR[:3, :3].T
            v = smp(pts)
            assert "ramp" in v, f"{S}: no seam_ramp -- deploy the members with --blend colcos for rule {a.rule}"
            sh = (shp[0], shp[1], z1 - z0)
            R = lambda k: v[k].reshape(sh)
            b = (R("brain") > 0.5).astype(np.float32)
            if "shell" in v:                                                 # v2 shell rule: uncovered, weight 0
                shl = (R("shell") > 0.5).astype(np.float32) * b
                acc["bsh"][(Ellipsis, slice(z0, z1))] += shl
                b = b - shl
            r = np.clip(R("ramp"), 0, 1) * b; phi = np.clip(R("phi"), 0, 1)
            real, syn = R("real"), R("synth")
            if a.rule == "precision":
                t2, s2i = float(pj["tau2"]), float(pj["s2_interp"])
                past = np.maximum(-R("depth"), 0.0)                        # mm past the member's photographed end (the
                                                                             # sigma2_syn curve's held-out-slab depth)
                ai = r * (phi / t2 + (1.0 - phi) / (t2 + s2i))
                aph = r * phi / t2
                bi = (1.0 - r) * b * m_i / (t2 + f_s2(past))
            sl = (Ellipsis, slice(z0, z1))
            acc["b"][sl] += b
            acc["a"][sl] += ai; acc["aph"][sl] += aph; acc["bw"][sl] += bi
            acc["wr"][sl] += ai; acc["gr"][sl] += ai * real
            acc["wsyn"][sl] += bi; acc["gsyn"][sl] += bi * syn
            gain = subj[S]["wb_gain"]
            for k in range(3):
                rv = R(f"rgb{k}")
                acc[f"rr{k}"][sl] += ai * rv; acc[f"rs{k}"][sl] += bi * rv
                acc[f"wwr{k}"][sl] += ai * np.clip(rv * gain[k], 0, 1); acc[f"wws{k}"][sl] += bi * np.clip(rv * gain[k], 0, 1)
        print(f"  [{os.getpid()}] {sid} done", flush=True)
    fd, pth = tempfile.mkstemp(suffix=".npz", dir=a.tmp); os.close(fd)
    np.savez(pth, **acc)
    return pth


def combine(acc, a, n_inputs, A):
    """num/den of the precision rule; returns the product volumes. n_inputs counts the mirror twins (every member enters
    twice)."""
    out = {}
    lam = 1.0
    den = acc["wr"] + lam * acc["wsyn"]
    ok = den > 0; dd = np.where(ok, den, 1.0)
    out["grey"] = np.where(ok, (acc["gr"] + lam * acc["gsyn"]) / dd, 0.0)
    out["rgb"] = np.stack([np.where(ok, (acc[f"rr{k}"] + lam * acc[f"rs{k}"]) / dd, 0.0) for k in range(3)], -1)
    out["rgb_wb"] = np.stack([np.where(ok, (acc[f"wwr{k}"] + lam * acc[f"wws{k}"]) / dd, 0.0) for k in range(3)], -1)
    out["real_eff"] = np.where(ok, acc["wr"] / dd, 0.0)                       # effective real fraction sum(w_real) / sum(w)
    out["photo_share"] = np.where(ok, acc["aph"] / dd, 0.0)                  # photographed share (drives real_mask)
    out["photofrac"] = acc["aph"] / np.maximum(acc["b"], 1e-6) * (float(json.load(open(a.precision_json))["tau2"])
                                                                   if a.rule == "precision" else 1.0)
    out["coverage"] = acc["b"] * 0.5
    out["nmap"] = out["photofrac"] * acc["b"] * 0.5
    return out


def b0_scale(S, root):
    """The member's b0 normalisation (mlo, mhi) as its export recorded it (summary_d3_full_A_s0 for the 13, summary_d3p_full_A_s0
    for the 8; L4 04:52: the 8 have mlo = 0, the 13 their own 0.5th percentile, lo/hi 0.02-0.20)."""
    for f in ("summary_d3_full_A_s0.json", "summary_d3p_full_A_s0.json"):
        p = f"{root}/{f}"
        if os.path.exists(p):
            m = json.load(open(p)).get("members", {})
            if S in m and "mlo" in m[S] and "mhi" in m[S]:
                return float(m[S]["mlo"]), float(m[S]["mhi"]), f
    raise SystemExit(f"{S}: (mlo, mhi) not found in the export summaries under {root}")


def b0_source(S, root):
    """The member's b0 on its own grid, on ONE rule for all 21 (4 Oct 2026): x = clip(b0 / mhi, 0, 1)
    (lo = 0). The 8 (mlo = 0): their exported <S>_b0_input_0p5 IS that already (clip(pseudo_b0 / mhi) x N1 brain). The 13 (exported as
    (b0 - mlo) / (mhi - mlo), whose 0.5 % tail below mlo was clipped): the RAW b0 the network saw (canon_d3/<S>/seam.json sources.b0,
    md5-checked), divided by the same mhi, so the result is exact, not a remap. Returns (array, affine, rule); the caller masks."""
    lo, hi, src = b0_scale(S, root)
    if lo == 0.0:
        bi = nib.load(f"{root}/{S}/{S}_b0_input_0p5.nii.gz")
        return np.clip(np.asanyarray(bi.dataobj).astype(np.float32), 0, 1), bi.affine, dict(mlo=lo, mhi=hi, source=src, b0=f"{root}/{S}/{S}_b0_input_0p5.nii.gz",
                                                                                           rule="exported file (already clip(b0 / mhi))")
    sj = json.load(open(f"{_prod('completion_canon_d3')}/{S}/seam.json")); s = sj["sources"]
    if isinstance(s, str):
        import ast
        s = ast.literal_eval(s)
    raw = paths.rerooted(s["b0"])
    if md5(raw) != s["b0_md5"]:
        raise SystemExit(f"{S}: {raw} md5 differs from canon_d3's record {s['b0_md5']}: not the b0 the network saw")
    if abs(float(sj["mhi"]) - hi) > 1e-6 * max(abs(hi), 1):
        raise SystemExit(f"{S}: canon_d3 mhi {sj['mhi']} != the export summary's {hi}")
    ri = nib.load(raw)
    return np.clip(np.asanyarray(ri.dataobj).astype(np.float32) / hi, 0, 1), ri.affine, dict(mlo=lo, mhi=hi, source=src, b0=raw, b0_md5=s["b0_md5"],
                                                                                             rule="raw b0 / mhi (exact; no export clipping)")


def accumulate_b0(job):
    ids, run_it, canon, subj, shp, A, tmp, wcanon = job
    num = np.zeros(shp, np.float32); den = np.zeros(shp, np.float32)
    for sid in ids:
        if sid == "TPL":                                                   # design 2b: the MRI template's b0 through the bridge
            tq = subj["TPL"]
            W = xfm.Warp(f"{run_it}/TPL_scalar_mod_01_to_template_warp_resampled_unbiased_full.nii.gz", "fsl_rel",
                         src_grid=f"{canon}/TPL/t1_grey.nii.gz")     # TPL's input grid in THIS run's canon (0.5 mm canon or 0.25 mm canon2)
            P = np.array(json.load(open(_prod('bridge')))["rigid_world_run113_to_canonical"])
            bi, mi = nib.load(_prod('tpl_b0')), nib.load(_prod('tpl_mask'))
            bv = np.clip(np.asanyarray(bi.dataobj).astype(np.float32) / float(tq["mhi"]), 0, 1); mv = np.asanyarray(mi.dataobj).astype(np.float32)
            for z0 in range(0, shp[2], 8):
                z1 = min(shp[2], z0 + 8)
                pts = W.pull(X.grid_world(shp, A, z0, z1)); pc = pts @ P[:3, :3].T + P[:3, 3]
                sh = (shp[0], shp[1], z1 - z0)
                b = (X.sample(mv, mi.affine, pc, order=1) > 0.5).reshape(sh).astype(np.float32)
                v = X.sample(bv, bi.affine, pc, order=1).reshape(sh)
                num[..., z0:z1] += b * v; den[..., z0:z1] += b
            print(f"  [{os.getpid()}] TPL done (the MRI template's b0 through the bridge)", flush=True)
            continue
        S = sid[:-1] if sid.endswith("f") else sid; flip = sid.endswith("f"); q = subj[S]
        W = xfm.Warp(f"{run_it}/{sid}_scalar_mod_01_to_template_warp_resampled_unbiased_full.nii.gz", "fsl_rel",
                     src_grid=f"{canon}/{S}/t1_grey{'_flip' if flip else ''}.nii.gz")
        ci = nib.load(f"{q['root']}/{S}/{S}_bf_completed.nii.gz")
        flat, R = unflatten_map(S, q["cache"], ci.affine, ci.shape)
        assert np.allclose(R, np.eye(3), atol=1e-9), f"{S}: flattening is not the identity"
        bv, bA, _ = b0_source(S, q["root"])                                  # ONE rule for all 21, exactly as prep_d2 (b0 / mhi)
        di = nib.load(f"{q['cache']}/{S}/deploy/brain.nii.gz"); dv = np.asanyarray(di.dataobj).astype(np.float32)
        wv = None
        if wcanon and q.get("kind") == "nomri":                         # the step-8 stand-in weighting (see the header)
            wi = nib.load(f"{wcanon}/{S}/d2_b0_costmask.nii.gz"); wv = np.asanyarray(wi.dataobj).astype(np.float32)
        for z0 in range(0, shp[2], 8):
            z1 = min(shp[2], z0 + 8)
            pts = W.pull(X.grid_world(shp, A, z0, z1))
            if flip:
                pts = pts @ MIRROR[:3, :3].T
            pf = flat(pts)
            sh = (shp[0], shp[1], z1 - z0)
            b = (X.sample(dv, di.affine, pf, order=1) > 0.5).reshape(sh).astype(np.float32)
            if wv is not None:                                               # pts: the member's own (unflipped) canon world
                b = b * np.clip(X.sample(wv, wi.affine, pts, order=1).reshape(sh), 0.0, 1.0)
            v = X.sample(bv, bA, pf, order=1).reshape(sh)
            num[..., z0:z1] += b * v; den[..., z0:z1] += b
        print(f"  [{os.getpid()}] {sid} done", flush=True)
    fd, pth = tempfile.mkstemp(suffix=".npz", dir=tmp); os.close(fd)
    np.savez(pth, num=num, den=den)
    return pth


def box_of(run):
    b = nib.load(paths.rerooted(json.load(open(f"{run}/run_spec.json"))["s8"]["box"]))
    return tuple(int(x) for x in b.shape[:3]), b.affine.astype(np.float64)

def finish_blockface(run, arm, canon, it, out, precision_json, trim_mm=1.0, shell_zero=True, j=6, tmp=None):
    """The blockface channel on the step-8 box (bftemplate/finish_t1.py main, as bftemplate/d2b/finish_d2bs8.py ran it: grid = the
    box, iteration pinned, rule precision, --shell-zero, -j 6). Writes <out>/<arm>_{grey, grey_presym, rgb, rgb_wb, real_eff,
    photo_share, photofrac, coverage, nmap, brainmask, real_mask, synth_mask, shell_removed}.nii.gz and metrics.json."""
    from concurrent.futures import ProcessPoolExecutor
    a = argparse.Namespace(run=run, arm=arm, canon=canon, factor=2, iteration=it, j=j, out=out, tmp=tmp, frame="run113", vox=0.25,
                           rule="precision", precision_json=precision_json, trim_mm=trim_mm, c0=0.25, shell_zero=shell_zero)
    t0 = time.time()
    step = (it.rstrip("/").split("nln_step_")[1].split("/")[0], it.rstrip("/").split("iteration_")[1])
    subj = _subjects(a.canon)
    spec = json.load(open(f"{a.run}/run_spec.json"))
    ids = spec["subject_ids"]
    # white balance (as the pilot's rgb_wb): each member's mean photographed chromaticity (its completed RGB where
    # provenance >= 0.5) is scaled to the cohort mean; gains are per member, per channel, luminance-free
    chrom = {}
    for S, q in subj.items():
        src = f"{q['root']}/{S}"
        r = np.asanyarray(nib.load(f"{src}/{S}_bf_completed_rgb.nii.gz").dataobj).astype(np.float32)
        ph = np.asanyarray(nib.load(f"{src}/{S}_provenance.nii.gz").dataobj) >= 0.5
        v = r[ph]; c = v.mean(0) / max(float(v.mean()), 1e-6); chrom[S] = c
    cm = np.mean(list(chrom.values()), 0)
    for S in subj:
        subj[S]["wb_gain"] = (cm / chrom[S]).tolist()
        subj[S]["shell_zero"] = bool(getattr(a, "shell_zero", False))
    n_brains = len(ids) / 2.0
    shp, A = box_of(a.run); pre = None
    os.makedirs(out, exist_ok=True)
    a.tmp = a.tmp or tempfile.mkdtemp(prefix="finish_t1_", dir=out)
    print(f"{a.arm}: step {step[0]} it {step[1]}, {len(ids)} inputs, grid {shp} @ {abs(A[0,0]):.3f} mm", flush=True)
    chunks = [ids[i::a.j] for i in range(a.j)]
    with ProcessPoolExecutor(a.j) as ex:
        parts = list(ex.map(accumulate, [(c, a, it, shp, A, subj, pre) for c in chunks if c]))
    acc = None
    for pth in parts:
        z = np.load(pth)
        acc = {k: z[k].astype(np.float64) for k in z.files} if acc is None else {k: acc[k] + z[k] for k in acc}
        os.remove(pth)
    prods = combine(acc, a, len(ids), A)
    met = dict(arm=a.arm, frame=a.frame, step=step[0], iteration=step[1], iteration_dir=it, n_inputs=len(ids),
               n_brains=n_brains, grid_mm=float(abs(A[0, 0])), shape=list(shp), eps=1e-3,
               members={S: v["kind"] for S, v in subj.items()},
               wb_gains={S: [round(g, 4) for g in v["wb_gain"]] for S, v in subj.items()},
               rule=a.rule, precision_json=a.precision_json, trim_mm=a.trim_mm, c0=None,
               weights="num/den with per-member real part a_i and synthetic part b_i; see the module docstring (" + a.rule + ")",
               masks="brainmask = coverage >= n_brains/2; real = brainmask & photo_share >= 0.5; synth = brainmask & ~real")
    sym = {}
    for name, vol in prods.items():
        vol = vol.astype(np.float32)
        met[f"{name}_asym_before"] = float(np.abs(vol - np.flip(vol, 0)).max())
        if name == "grey":                                   # the pre-symmetrisation average, for its mirror r
            nib.save(nib.Nifti1Image(vol, A), f"{out}/{a.arm}_grey_presym.nii.gz")
        s = 0.5 * (vol + np.flip(vol, 0))
        assert float(np.abs(s - np.flip(s, 0)).max()) == 0.0, f"{name}: not exactly symmetric after symmetrisation"
        sym[name] = s
        nib.save(nib.Nifti1Image(s, A), f"{out}/{a.arm}_{name}.nii.gz")
    brain = sym["coverage"] >= n_brains / 2.0
    real = brain & (sym["photo_share"] >= 0.5)
    for name, m in (("brainmask", brain), ("real_mask", real), ("synth_mask", brain & ~real)):
        assert np.array_equal(m, np.flip(m, 0)), f"{name} not symmetric"
        nib.save(nib.Nifti1Image(m.astype(np.uint8), A), f"{out}/{a.arm}_{name}.nii.gz")
    vv = float(np.prod(np.abs(np.diag(A)[:3])))
    met.update(brain_cm3=float(brain.sum() * vv / 1000), real_cm3=float(real.sum() * vv / 1000),
               synth_cm3=float((brain & ~real).sum() * vv / 1000),
               synth_frac_of_brain=float((brain & ~real).sum() / max(brain.sum(), 1)),
               wall_s=round(time.time() - t0, 1))
    if a.shell_zero:                                         # template-level shell share (the rule's footprint)
        before = (acc["b"] + acc["bsh"]) * 0.5 >= n_brains / 2.0
        after = acc["b"] * 0.5 >= n_brains / 2.0
        touched = before & (acc["bsh"] > 0)
        met.update(shell_rule=dict(shell_mm=SHELL_MM, support_before_vox=int(before.sum()), support_after_vox=int(after.sum()),
                                   support_lost_frac=float(1 - after.sum() / max(before.sum(), 1)),
                                   touched_frac_of_support=float(touched.sum() / max(before.sum(), 1)),
                                   shell_weight_share=float(acc["bsh"][before].sum() / max((acc["b"] + acc["bsh"])[before].sum(), 1))))
        nib.save(nib.Nifti1Image((acc["bsh"] * 0.5).astype(np.float32), A), f"{out}/{a.arm}_shell_removed.nii.gz")
    json.dump(met, open(f"{out}/metrics.json", "w"), indent=1)
    try:
        os.rmdir(a.tmp)
    except OSError:
        pass
    return met


def finish_b0(run, arm, canon, it, out, weights_canon="", j=6, tmp=None):
    """The b0 channel on the step-8 box (bftemplate/d2b/finish_d2bs8w_b0.py main, as finish_d2bs8.py ran it: grid = the box,
    iteration pinned, --weights-canon canon2_s8w, -j 6). Writes <out>/<arm>_b0.nii.gz, <arm>_b0_coverage.nii.gz,
    <arm>_b0_metrics.json."""
    from multiprocessing import Pool
    t0 = time.time()
    subj = _subjects(canon)
    ids = json.load(open(f"{run}/run_spec.json"))["subject_ids"]
    shp, A = box_of(run)
    os.makedirs(out, exist_ok=True)
    tmp = tmp or tempfile.mkdtemp(prefix="d2b0_", dir=out)
    chunks = [ids[k::j] for k in range(j) if ids[k::j]]
    with Pool(len(chunks)) as pool:
        parts = pool.map(accumulate_b0, [(c, it, canon, subj, shp, A, tmp, weights_canon) for c in chunks])
    num = np.zeros(shp, np.float64); den = np.zeros(shp, np.float64)
    for p in parts:
        z = np.load(p); num += z["num"]; den += z["den"]; os.remove(p)
    tpl = np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)
    xc = A[0, 3] + A[0, 0] * np.arange(shp[0]); assert np.allclose(xc, -xc[::-1], atol=1e-6), "recon grid not x-symmetric"
    before = dict(mirror_r=float(np.corrcoef(tpl.ravel(), np.flip(tpl, 0).ravel())[0, 1]), max_abs_diff=float(np.abs(tpl - np.flip(tpl, 0)).max()))
    tpl = 0.5 * (tpl + np.flip(tpl, 0)); cov = 0.5 * (den + np.flip(den, 0))
    after = dict(mirror_r=float(np.corrcoef(tpl.ravel(), np.flip(tpl, 0).ravel())[0, 1]), max_abs_diff=float(np.abs(tpl - np.flip(tpl, 0)).max()))
    assert after["max_abs_diff"] == 0.0, f"b0 template not exactly symmetric after (A + M(A))/2: {after}"
    nib.save(nib.Nifti1Image(tpl.astype(np.float32), A), f"{out}/{arm}_b0.nii.gz")
    nib.save(nib.Nifti1Image(cov.astype(np.float32), A), f"{out}/{arm}_b0_coverage.nii.gz")
    met = dict(arm=arm, iteration_dir=it, n_inputs=len(ids), grid=dict(shape=list(shp), affine=A.tolist()),
               weights=("every member's brain at full weight (both mirrors)" if not weights_canon else
                        "step-8 stand-in weighting: the 8 without MRI x their <weights canon>/<S>/d2_b0_costmask; MRI brains and TPL full weight"),
               symmetry_before=before, symmetry_after=after, seconds=round(time.time() - t0, 1))
    json.dump(met, open(f"{out}/{arm}_b0_metrics.json", "w"), indent=1)
    try:
        os.rmdir(tmp)
    except OSError:
        pass
    return met


def embed(run, src, dst):
    """Zero-pad every box product of `src` onto the run-113 grid's 2x refinement at an integer offset (finish_d2bs8.py embed)."""
    (bs, bA), (fs, fA) = box_of(run), run113_grid(2)
    off = (bA[:3, 3] - fA[:3, 3]) / np.diag(fA)[:3]; o = np.rint(off).astype(int)
    assert np.allclose(bA[:3, :3], fA[:3, :3]) and np.allclose(off, o, atol=1e-4), "box not on the run-113 2x lattice"
    assert np.all(o >= 0) and np.all(o + np.array(bs) <= np.array(fs)), f"box not inside the run-113 2x grid: offset {o}, {bs} vs {fs}"
    os.makedirs(dst, exist_ok=False)
    rows = {}
    for f in sorted(os.listdir(src)):
        p = f"{src}/{f}"
        if not f.endswith(".nii.gz"):
            continue
        im = nib.load(p)
        if im.shape[:3] != bs or not np.allclose(im.affine, bA, atol=1e-6):
            rows[f] = "not on the box: skipped"; continue
        v = np.asanyarray(im.dataobj)
        out = np.zeros(tuple(fs) + v.shape[3:], v.dtype)
        out[o[0]:o[0] + bs[0], o[1]:o[1] + bs[1], o[2]:o[2] + bs[2]] = v
        nib.save(nib.Nifti1Image(out, fA, im.header), f"{dst}/{f}")
        rows[f] = "embedded"
    json.dump(dict(box_shape=list(bs), grid_shape=list(fs), offset_vox=o.tolist(), files=rows,
                   rule="zero-padded onto the run-113 2x grid at an integer offset (no resampling)"), open(f"{dst}/embed.json", "w"), indent=1)
    return o.tolist()
