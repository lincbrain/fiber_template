#!/usr/bin/env python3
"""Our MRI template as a registration target for ex vivo brains, against NMT v2 (symmetric, in vivo T1w) and CIVM-R (ex vivo b0), with
ONE engine and ONE setting for all three targets (paper: Methods, sec:m-target; Results, Table 3). Settings: configs/targets.yaml.

The engine is the paper's: linc/link/symreg.py (anatomix + MIND-SSC features, FireANTs fork, symmetric, halfway space) with the
settings of the template <-> atlas mappings (0.25 mm, warp smoothing 1.25 voxels), from an ANTs MI rigid + affine start
(linc/link/ants.py). This module is only the driver around it:
  prep      every target channel and every brain image multiplied by its brain mask (registration inputs are always masked). Template
            brains: T2 pulled to b0-native by the within-brain registration of record, FA from the tensor fit.
  place     upright P0 (rigid, native b0 world -> the canonical NMT-v2-stereotaxic frame of our template), as the paper's b0 -> template
            step places a b0: sectioned brains = the frame bridge @ their FLIRT 6-dof placement of record; template brains =
            canonical.json T_subject_to_canonical. Needed because ANTs starts from the centers of mass without a rotation search, and
            five b0 headers are 34-56 deg from upright.
  affine    the two ANTs registrations of the upright b0 (header P0 @ affine): target fixed (fwd) and brain fixed (bwd).
  choose    the start S' (upright brain world -> target world) = the best of fwd, bwd and their log-Euclidean mean by the Dice of the
            brain mask carried by the candidate with the target's mask (target grid, every 2nd voxel). The mean alone fails where the two
            directions disagree. The multi arm reuses its pair's b0-arm start.
  grid      the upright images resampled ONCE onto an axis-aligned 0.25 mm grid (mask box + 4 mm; images cubic, clipped at 0 (FA also at
            1) and masked; masks linear > 0.5): the engine sizes its halfway grid from the corners of each mask's voxel box, so a tilted
            voxel grid inflates it (out of GPU memory on 48 GB cards for the tilted brains).
  solve     the engine: A = target, B = upright brain on its grid, S' (GPU).
  registry  linc.xfm entries b0-native -> {atlas:NMT_v2.0_sym, atlas:CIVM-R, mri-template} = [affine Hinv, warp F, affine Hinv,
            affine P0^-1] (the engine's a2b, then upright -> native), checked against the engine's own map.
  measure   each brain's b0 (cubic) and mask (linear > 0.5) pulled onto each target's grid; per pair of different brains: Pearson r of
            the two b0 inside both masks minus a 1.5 mm rim, and the Dice of the two masks; both over the whole masks and inside the
            target's own brain mask (the targets' brains end at different levels of the brainstem). Per target and group: median and
            IQR; per pair whether ours is higher than nmt / civmr. Groups: the 13 sectioned brains (78 pairs, the headline), the 4
            template brains (6 pairs, in sample for ours). Reference beside ours: the sectioned brains through the paper's own b0 ->
            template registration (registry b0-native -> mri-template). -> $LINC_NUMBERS/r060_target_comparison.json
Work folder (default $LINC_WORK/r060): targets/, inputs/ (prep); place/, start/, b0/, multi/, xfm/ (the runs); warped/ (measure).
The targets' intensities and anything derived from them (targets/, the runs' mid_A* images, warped/) stay local: CIVM-R's license does
not allow redistribution.

    python -m linc.evaluation.targets prep | place | grid | choose | registry [--work DIR]
    python -m linc.evaluation.targets affine <ID> <tgt> [--work DIR]              (CPU, ANTs)
    python -m linc.evaluation.targets solve <arm> <tgt> <ID> [--chunk N] [--device cuda:0] [--work DIR]   (GPU)
    python -m linc.evaluation.targets measure [--runs DIR] [--jobs N] [--work DIR]    (CPU; --runs = the runs' folder, default --work)
Ported from the analysis project's mri/code/r060_targets.py (the measure, starts and registry entries regenerate identically).
"""
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy.linalg import expm, logm
from scipy.ndimage import map_coordinates

from .. import paths

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "targets.yaml"
ARMS = ("b0", "multi")
KEYS = ("b0_r", "mask_dice", "b0_r_in_target", "mask_dice_in_target")
R5 = "ours_r5_of_record"
CHANNELS = {"b0": {"nmt": "brain b0 <-> NMT v2 T1w", "civmr": "brain b0 <-> CIVM-R b0", "ours": "brain b0 <-> our b0"},
            "multi": {"nmt": "brain T2 <-> NMT v2 T1w", "civmr": "b0 <-> CIVM-R b0 + FA <-> CIVM-R FA",
                      "ours": "b0 <-> our b0 + T2 <-> our T2 + FA <-> our FA"}}
# the record's text fields, verbatim
NOTE_MULTI = "the multi arm is NOT like-for-like: each target got the channels it has (channels.multi)"
HEADLINE = "sectioned (13 brains, out of sample for every target), b0 arm"
KEY_TEXT = dict(b0_r="b0 Pearson r inside both warped masks minus a 1.5 mm rim", mask_dice="Dice of the two warped brain masks",
                b0_r_in_target="the same inside the target's brain mask", mask_dice_in_target="Dice of the two masks cut to the target's brain mask")
SPECIES = dict(note="ours is built from 4 M. fascicularis brains (MF283-286); NMT v2 and CIVM-R are M. mulatta; of the 13 sectioned "
                    "brains MF275/MF278/MF282 are M. fascicularis, the 10 MR brains M. mulatta", rhesus_pairs="MR-MR pairs, 45 of the 78")
LGNSTN = dict(record="lgnstn_targets_r060c", note="LGN/STN centroid and surface distances of MM1-MM3 through these registrations "
              "(the blockface-to-b0 registration of record + the entries in mri/tc/r060_up/xfm); cite that record directly")
ENGINE_TEXT = "target comparison (30 Oct 2026): symreg_upstream (anatomix + MIND-SSC, FireANTs fork), ANTs MI affine start, 0.25 mm, warp sigma 1.25"


def config():
    import yaml
    return yaml.safe_load(CONFIG.read_text())


def _fmt(p, **kw):
    from ..link.common import inputs, product
    return Path(str(p).format(ref=product(inputs()["atlas_reference"]), prod=paths.products_root(), **kw))


def _src(ID, key):
    C = config(); grp = "sectioned" if ID in C["brains"]["sectioned"] else "template"
    return paths.products_root() / C["inputs"][grp][key].format(ID=ID)


def brains():
    b = config()["brains"]; return b["sectioned"], b["template"]


def load(p):
    i = nib.load(str(p)); return np.asanyarray(i.dataobj), i


class Work:
    """The folder layout of one target-comparison run (the analysis project's mri/tc/r060 + mri/tc/r060_up)."""

    def __init__(self, work=None, runs=None):
        self.w = Path(work or paths.work_root() / "r060"); self.r = Path(runs) if runs else self.w

    def tgt_ch(self, t, c="img"):
        return self.w / "targets" / f"{t}_{c}.nii.gz"

    def tgt_mask(self, t):
        return self.w / "targets" / f"{t}_mask.nii.gz"

    def inp(self, ID, c="b0"):
        return self.w / "inputs" / f"{ID}_{c}.nii.gz"

    def upright(self, ID, c="b0"):
        return self.w / "place" / f"{ID}_{c}_upright.nii.gz"

    def upgrid(self, ID, c="b0"):
        return self.w / "place" / f"{ID}_{c}_upgrid.nii.gz"

    def p0(self, ID):
        return self.r / "place" / f"{ID}_P0.txt"

    def start(self, t, ID):
        return self.r / "start" / t / ID

    def run(self, arm, t, ID):
        return self.r / arm / t / ID / "run"

    def warped(self, arm, t, ID):
        return self.w / "warped" / arm / t / ID


# ------------------------------------------------------------------------------------------------ prep (CPU)
def prep(W):
    from .. import xfm as LX
    C = config(); SECTIONED, TEMPLATE = brains()
    (W.w / "targets").mkdir(parents=True, exist_ok=True); (W.w / "inputs").mkdir(parents=True, exist_ok=True)
    for t, T in C["targets"].items():
        M, mi = load(_fmt(T["msk"])); M = M.reshape(M.shape[:3]) > 0.5
        nib.save(nib.Nifti1Image(M.astype(np.uint8), mi.affine), str(W.tgt_mask(t)))
        for c, p in T["ch"].items():
            V, vi = load(_fmt(p)); V = V.reshape(V.shape[:3]).astype(np.float32)
            assert V.shape == M.shape and np.allclose(vi.affine, mi.affine, atol=1e-4), (t, c)
            nib.save(nib.Nifti1Image(np.where(M, V, 0).astype(np.float32), mi.affine), str(W.tgt_ch(t, c)))
    for ID in SECTIONED + TEMPLATE:
        B, bi = load(_src(ID, "b0")); M, mi = load(_src(ID, "mask")); M = M.reshape(M.shape[:3]) > 0.5
        assert B.shape[:3] == M.shape and np.allclose(bi.affine, mi.affine, atol=1e-4), ID
        nib.save(nib.Nifti1Image(M.astype(np.uint8), bi.affine), str(W.inp(ID, "mask")))
        nib.save(nib.Nifti1Image(np.where(M, B.reshape(M.shape), 0).astype(np.float32), bi.affine), str(W.inp(ID, "b0")))
        if ID in TEMPLATE:
            reg = _src(ID, "t2_reg")
            meta = json.loads((reg / "meta.json").read_text()); (kind, conv, f), = meta["pull_chain"]
            T2, ti = load(_src(ID, "t2"))
            P = np.indices(M.shape).reshape(3, -1).T @ bi.affine[:3, :3].T + bi.affine[:3, 3]
            Q = LX.Warp(reg / f, conv).pull(P); V = np.linalg.inv(ti.affine)
            t2 = map_coordinates(T2.astype(np.float32), (Q @ V[:3, :3].T + V[:3, 3]).T, order=3, cval=0).reshape(M.shape)
            nib.save(nib.Nifti1Image(np.where(M, np.clip(t2, 0, None), 0).astype(np.float32), bi.affine), str(W.inp(ID, "T2")))
            FA, fi = load(_src(ID, "fa")); assert FA.shape[:3] == M.shape and np.allclose(fi.affine, bi.affine, atol=1e-4), ID
            nib.save(nib.Nifti1Image(np.where(M, FA.reshape(M.shape), 0).astype(np.float32), bi.affine), str(W.inp(ID, "FA")))
        print(ID, "prepared", flush=True)


# ------------------------------------------------------------------------------------------------ upright placement P0 (CPU)
def place(W):
    from ..registration import fsl
    C = config(); SECTIONED, TEMPLATE = brains()
    (W.w / "place").mkdir(parents=True, exist_ok=True)
    Rb = np.array(json.loads((paths.products_root() / C["inputs"]["bridge"]).read_text())["rigid_world_run113_to_canonical"])
    for ID in SECTIONED + TEMPLATE:
        bi = nib.load(str(W.inp(ID)))
        if ID in SECTIONED:
            assert np.allclose(nib.load(str(_src(ID, "b0"))).affine, bi.affine, atol=1e-4), ID
            P0 = Rb @ fsl.fsl_to_world(np.loadtxt(_src(ID, "place_mat")), bi, nib.load(str(_src(ID, "place_ref"))))
            src = f"bridge @ fsl_to_world(data/sub-{ID}/mri/{ID}_mri2template.mat)"
        else:
            P0 = np.array(json.loads(_src(ID, "canonical").read_text())["T_subject_to_canonical"]); src = "canonical.json"
        R = P0[:3, :3]; assert np.allclose(R @ R.T, np.eye(3), atol=1e-3) and np.linalg.det(R) > 0, (ID, "P0 not a rotation")
        ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        np.savetxt(W.w / "place" / f"{ID}_P0.txt", P0)
        for c in ("b0", "mask") + (("T2", "FA") if ID in TEMPLATE else ()):
            i = nib.load(str(W.inp(ID, c))); nib.save(nib.Nifti1Image(np.asanyarray(i.dataobj), P0 @ i.affine), str(W.upright(ID, c)))
        json.dump(dict(P0=P0.tolist(), source=src, rotation_deg=round(ang, 1)), open(W.w / "place" / f"{ID}_P0.json", "w"), indent=1)
        print(ID, "P0 rotation %.1f deg" % ang, src, flush=True)


def grid(W):
    """every upright brain image resampled once onto an axis-aligned grid in the upright world (mask box + pad)"""
    g = config()["upgrid"]; res, pad = float(g["res_mm"]), float(g["pad_mm"]); SECTIONED, TEMPLATE = brains()
    for ID in SECTIONED + TEMPLATE:
        K, ki = load(W.upright(ID, "mask")); X = np.argwhere(K > 0) @ ki.affine[:3, :3].T + ki.affine[:3, 3]
        lo, hi = X.min(0) - pad, X.max(0) + pad; shape = tuple(int(np.ceil((h - l) / res)) + 1 for l, h in zip(lo, hi))
        G = np.diag([res, res, res, 1.0]); G[:3, 3] = lo
        P = np.indices(shape).reshape(3, -1).T @ G[:3, :3].T + G[:3, 3]

        def samp(img, order):
            V = np.linalg.inv(img.affine)
            return map_coordinates(np.asanyarray(img.dataobj).astype(np.float32), (P @ V[:3, :3].T + V[:3, 3]).T, order=order, cval=0).reshape(shape)
        k = samp(ki, 1) > 0.5
        nib.save(nib.Nifti1Image(k.astype(np.uint8), G), str(W.upgrid(ID, "mask")))
        for c in ("b0",) + (("T2", "FA") if ID in TEMPLATE else ()):
            v = np.clip(samp(nib.load(str(W.upright(ID, c))), 3), 0, 1 if c == "FA" else None)
            nib.save(nib.Nifti1Image(np.where(k, v, 0).astype(np.float32), G), str(W.upgrid(ID, c)))
        print(ID, "upright grid", shape, flush=True)


# ------------------------------------------------------------------------------------------------ affine start (CPU, ANTs)
def affine(W, ID, t):
    from ..link.ants import affine_init
    d = W.start(t, ID); (d / "fwd").mkdir(parents=True, exist_ok=True); (d / "bwd").mkdir(parents=True, exist_ok=True)
    A, Am, B, Bm = str(W.tgt_ch(t)), str(W.tgt_mask(t)), str(W.upright(ID)), str(W.upright(ID, "mask"))
    affine_init(A, B, Am, Bm, str(d / "fwd" / "affine_"))                    # moving upright brain -> fixed target
    affine_init(B, A, Bm, Am, str(d / "bwd" / "affine_"))                    # moving target -> fixed upright brain
    return choose(W, ID, t)


def choose(W, ID, t, out=None):
    """the start from the saved ANTs pair (W.start): the best of fwd / bwd / their log-Euclidean mean by carried-mask Dice in the target
    grid. Writes S_ras_upright_to_target.txt, S_ras_brain_to_target.txt (= S' @ P0) and prep.json into `out` (default: the work
    folder's start/<tgt>/<ID>; never into a --runs folder of record)."""
    from .. import xfm
    d = W.start(t, ID); o = Path(out) if out else W.w / "start" / t / ID; o.mkdir(parents=True, exist_ok=True); st = int(config()["start_dice_step"])
    S1 = np.linalg.inv(xfm.read_itk_affine(d / "fwd" / "affine_0GenericAffine.mat")); S2 = xfm.read_itk_affine(d / "bwd" / "affine_0GenericAffine.mat")
    cand = dict(fwd=S1, bwd=S2, mean=np.real(expm(0.5 * (np.real(logm(S1)) + np.real(logm(S2))))))
    T, ti = load(W.tgt_mask(t)); T = T[::st, ::st, ::st] > 0; A = ti.affine.copy(); A[:3, :3] *= st
    G = np.indices(T.shape).reshape(3, -1).T @ A[:3, :3].T + A[:3, 3]
    M, mi = load(W.upright(ID, "mask")); M = M.astype(np.float32); V = np.linalg.inv(mi.affine)
    dice = {}
    for k, C in cand.items():
        Ci = np.linalg.inv(C); Q = G @ Ci[:3, :3].T + Ci[:3, 3]
        m = map_coordinates(M, (Q @ V[:3, :3].T + V[:3, 3]).T, order=1).reshape(T.shape) > 0.5
        dice[k] = round(float(2 * (m & T).sum() / (m.sum() + T.sum())), 4)
    best = max(dice, key=dice.get); Su = cand[best]
    P0 = np.loadtxt(W.w / "place" / f"{ID}_P0.txt"); S = Su @ P0                # native b0 world -> upright -> target world
    X = np.argwhere(M > 0)[::50] @ mi.affine[:3, :3].T + mi.affine[:3, 3]        # upright brain points
    e = np.linalg.norm(X @ (S1 - S2)[:3, :3].T + (S1 - S2)[:3, 3], axis=1)
    sv = lambda M: np.linalg.svd(M[:3, :3], compute_uv=False).round(4).tolist()  # noqa: E731
    np.savetxt(o / "S_ras_upright_to_target.txt", Su); np.savetxt(o / "S_ras_brain_to_target.txt", S)
    json.dump(dict(chosen=best, mask_dice=dice, S1_scales=sv(S1), S2_scales=sv(S2), S_scales=sv(Su),
                   fwd_vs_bwd_mm=dict(median=round(float(np.median(e)), 3), max=round(float(e.max()), 3))), open(o / "prep.json", "w"), indent=1)
    print(ID, t, "chosen", best, dice, "fwd vs bwd median %.2f mm" % np.median(e), flush=True)
    return Su


# ------------------------------------------------------------------------------------------------ the GPU solve
def solve_args(W, arm, t, ID, chunk=None, device="cuda:0"):
    e = config()["engine"]; m = config()["multi"][t]
    if arm == "b0":
        A, B, pairs = W.tgt_ch(t), W.upgrid(ID), []
    else:
        A, B = W.tgt_ch(t), W.upgrid(ID, m["B"]); pairs = [(W.tgt_ch(t, a), W.upgrid(ID, b)) for a, b in m["pairs"]]
    c = [A, W.tgt_mask(t), B, W.upgrid(ID, "mask"), W.start(t, ID) / "S_ras_upright_to_target.txt", W.run(arm, t, ID),
         "--res", e["res"], "--device", device, "--smooth-warp-sigma", e["smooth_warp_sigma"],
         "--channel-chunk", chunk or e["channel_chunk"], "--max-gpu-frac", e["max_gpu_frac"]]
    for a, b in pairs:
        c += ["--pair", a, b]
    return [str(x) for x in c]


def solve(W, arm, t, ID, chunk=None, device="cuda:0"):
    from ..link import symreg
    symreg.main(solve_args(W, arm, t, ID, chunk, device))


# ------------------------------------------------------------------------------------------------ registry entries
def p0inv(W, ID):
    f = W.r / "place" / f"{ID}_P0inv.txt"
    return np.loadtxt(f) if f.exists() else np.linalg.inv(np.loadtxt(W.p0(ID)))


def native_pull(W, run, ID):
    """target point -> the brain's native b0 point: the solve's a2b (target -> upright brain), then P0^-1 (upright -> native)"""
    from ..link import symreg
    a2b, _ = symreg.maps(run); Pi = p0inv(W, ID)
    return lambda X: np.asarray(a2b(X)) @ Pi[:3, :3].T + Pi[:3, 3]


def registry(W, root=None):
    from .. import xfm as LX
    C = config(); SECTIONED, TEMPLATE = brains(); rng = np.random.default_rng(0)
    root = Path(root) if root else W.w / "xfm"
    for ID in SECTIONED + TEMPLATE:
        f = W.w / "place" / f"{ID}_P0inv.txt"
        if not f.exists():
            np.savetxt(f, np.linalg.inv(np.loadtxt(W.p0(ID))))
        for t, T in C["targets"].items():
            run = W.run("b0", t, ID)
            if not (run / "metrics.json").exists():
                print("missing", run); continue
            e = LX.register("b0-native", T["space"], [("affine", "ras_pull", run / "Hinv.txt"), ("warp", "ras_disp", run / "F_ras_disp.nii.gz"),
                                                      ("affine", "ras_pull", run / "Hinv.txt"), ("affine", "ras_pull", f)],
                            subject=ID, dst_grid=W.tgt_ch(t), src_grid=W.inp(ID), root=root / f"sub-{ID}", copy=False, overwrite=True,
                            engine=ENGINE_TEXT, config=dict(run=str(run), arm="b0", target=t))
            M, mi = load(W.tgt_mask(t)); P = np.argwhere(M > 0); P = P[rng.choice(len(P), 2000, replace=False)] @ mi.affine[:3, :3].T + mi.affine[:3, 3]
            err = float(np.abs(e.pull(P) - native_pull(W, run, ID)(P)).max())
            assert err < 1e-4, (ID, t, err)
            print(ID, t, "entry ok (max |entry - a2b| %.1e mm)" % err, flush=True)


# ------------------------------------------------------------------------------------------------ measures (CPU)
def warp_to(W, t, pull, ID):
    """the brain's masked b0 (cubic) and mask (linear > 0.5) on the target's grid, through a pull map target point -> brain point"""
    n = int(config()["measure"]["point_chunk"])
    M, mi = load(W.tgt_mask(t)); P = np.indices(M.shape).reshape(3, -1).T @ mi.affine[:3, :3].T + mi.affine[:3, 3]
    Q = np.concatenate([np.asarray(pull(P[k:k + n])) for k in range(0, len(P), n)])
    B, bi = load(W.inp(ID)); K, _ = load(W.inp(ID, "mask")); V = np.linalg.inv(bi.affine); C = (Q @ V[:3, :3].T + V[:3, 3]).T
    b = map_coordinates(B.astype(np.float32), C, order=3, cval=0).reshape(M.shape)
    k = map_coordinates(K.astype(np.float32), C, order=1, cval=0).reshape(M.shape) > 0.5
    return b, k, mi


def warped(W, arm, t, ID):
    """the brain's b0 and mask in target t (computed once into W.warped, then reused); t = ours_r5_of_record: through the paper's own
    b0 -> template registration (registry entry), on our grid"""
    from .. import xfm as LX
    if t == R5:
        src = _src(ID, "r5_entry") / "meta.json"; tt = "ours"
    else:
        src = W.run(arm, t, ID) / "metrics.json"; tt = t
    d = W.warped(arm, t, ID); fb, fk = d / f"{ID}_b0_in_{tt}.nii.gz", d / f"{ID}_mask_in_{tt}.nii.gz"
    if not (fb.exists() and fk.exists()):
        pull = LX.Entry(src.parent).pull if t == R5 else native_pull(W, W.run(arm, t, ID), ID)
        b, k, mi = warp_to(W, tt, pull, ID); d.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(b, mi.affine), str(fb)); nib.save(nib.Nifti1Image(k.astype(np.uint8), mi.affine), str(fk))
    return np.asanyarray(nib.load(str(fb)).dataobj), np.asanyarray(nib.load(str(fk)).dataobj) > 0, src


def _jobs(W):
    SECTIONED, TEMPLATE = brains(); T = list(config()["targets"])
    return ([("b0", t, ID) for t in T for ID in SECTIONED + TEMPLATE] + [("b0", R5, ID) for ID in SECTIONED]
            + [("multi", t, ID) for t in T for ID in TEMPLATE])


def _warp1(args):
    w, r, job = args; warped(Work(w, r), *job); return job


def warps(W, jobs=4):
    """every warped b0 / mask the measure needs, in parallel"""
    from concurrent.futures import ProcessPoolExecutor
    todo = [(str(W.w), str(W.r), j) for j in _jobs(W)]
    with ProcessPoolExecutor(int(jobs)) as ex:
        for job in ex.map(_warp1, todo):
            print(*job, "warped", flush=True)


def pair_measures(ids, Wd, kt):
    """per pair: over the whole warped masks and inside the target's own brain mask kt"""
    c = config()["measure"]; ne, nmin = int(c["rim_erosions"]), int(c["min_core_voxels"])
    rows = []
    for a, b in combinations(ids, 2):
        (ba, ka), (bb, kb) = Wd[a], Wd[b]
        r = {}
        for tag, (ma, mb) in (("", (ka, kb)), ("_in_target", (ka & kt, kb & kt))):
            core = ndi.binary_erosion(ma & mb, iterations=ne)
            cc = float(np.corrcoef(ba[core], bb[core])[0, 1]) if core.sum() > nmin else None
            r.update({f"b0_r{tag}": None if cc is None else round(cc, 4), f"mask_dice{tag}": round(float(2 * (ma & mb).sum() / (ma.sum() + mb.sum())), 4),
                      f"core_voxels{tag}": int(core.sum())})
        rows.append(dict(pair=[a, b], **r))
    return rows


def summary(rows, k):
    v = np.array([r[k] for r in rows if r[k] is not None])
    return dict(n=int(len(v)), median=round(float(np.median(v)), 4), q25=round(float(np.percentile(v, 25)), 4), q75=round(float(np.percentile(v, 75)), 4))


def measure(W, jobs=4, method=None):
    """-> the record's data block (and its inputs); warps first, in parallel"""
    warps(W, jobs)
    C = config(); SECTIONED, TEMPLATE = brains(); TARGETS = list(C["targets"])
    out = dict(method=method or __doc__, channels=CHANNELS, note_multi=NOTE_MULTI, headline=HEADLINE, keys=KEY_TEXT, species=SPECIES,
               target_extent_mm={}, arms={})
    ins = [W.tgt_ch(t) for t in TARGETS] + [W.tgt_mask(t) for t in TARGETS]
    for t in TARGETS:
        M, mi = load(W.tgt_mask(t)); X = np.argwhere(M > 0) @ mi.affine[:3, :3].T + mi.affine[:3, 3]
        out["target_extent_mm"][t] = dict(xyz=(X.max(0) - X.min(0)).round(1).tolist(), volume_ml=round(float((M > 0).sum() * abs(np.linalg.det(mi.affine[:3, :3]))) / 1000, 1))
    for arm, groups in (("b0", dict(sectioned=SECTIONED, template=TEMPLATE)), ("multi", dict(template=TEMPLATE))):
        out["arms"][arm] = {}
        for g, ids in groups.items():
            res, per = {}, {}
            srcs = list(TARGETS) + ([R5] if arm == "b0" and g == "sectioned" else [])
            for t in srcs:
                Wd = {}; kt = load(W.tgt_mask("ours" if t == R5 else t))[0] > 0
                for ID in ids:
                    b, k, src = warped(W, arm, t, ID); ins.append(src); Wd[ID] = (b, k)
                rows = pair_measures(ids, Wd, kt); per[t] = rows; del Wd
                res[t] = {k: summary(rows, k) for k in KEYS}
                if g == "sectioned":
                    rh = [r for r in rows if all(i.startswith("MR") for i in r["pair"])]
                    res[t]["rhesus_pairs"] = {k: summary(rh, k) for k in KEYS}
                print(arm, g, t, {k: (v["median"], v["q25"], v["q75"]) for k, v in res[t].items() if k in KEYS}, flush=True)
            for other in ("nmt", "civmr"):
                for k in KEYS:
                    better = [o[k] > x[k] for o, x in zip(per["ours"], per[other]) if o[k] is not None and x[k] is not None]
                    res.setdefault("ours_higher", {}).setdefault(f"vs_{other}", {})[k] = dict(pairs=len(better), ours_higher=int(sum(better)))
            out["arms"][arm][g] = dict(brains=ids, pairs=len(per["ours"]), summary=res, per_pair=per)
    qc = {}                                                            # the engine's own QC and the starts, per arm and target
    for arm, ids in (("b0", SECTIONED + TEMPLATE), ("multi", TEMPLATE)):
        for t in TARGETS:
            ms = [json.loads((W.run(arm, t, ID) / "metrics.json").read_text()) for ID in ids]
            md = [m["mask_dice_mid_after"] for m in ms]
            qc.setdefault(arm, {})[t] = dict(n=len(ms), mid_mask_dice_after_median=round(float(np.median(md)), 4), mid_mask_dice_after_min=round(float(min(md)), 4),
                                             folds_fraction_max=float(max(max(m["folds_fraction"].values()) for m in ms)))
    for t in TARGETS:
        ch = [json.loads((W.start(t, ID) / "prep.json").read_text()) for ID in SECTIONED + TEMPLATE]
        qc.setdefault("starts", {})[t] = dict(chosen={k: sum(c["chosen"] == k for c in ch) for k in ("fwd", "bwd", "mean")},
                                              mask_dice_min=round(float(min(c["mask_dice"][c["chosen"]] for c in ch)), 4))
        ins += [W.start(t, ID) / "prep.json" for ID in SECTIONED + TEMPLATE]
    out["registration_qc"] = qc
    out["lgnstn"] = LGNSTN
    return out, ins


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prep", "place", "grid", "affine", "choose", "solve", "registry", "measure"])
    ap.add_argument("args", nargs="*", help="affine: <ID> <tgt>; solve: <arm> <tgt> <ID>")
    ap.add_argument("--work", default=None, help="work folder (default $LINC_WORK/r060)")
    ap.add_argument("--runs", default=None, help="the runs' folder (place/, start/, b0/, multi/; default --work), e.g. the runs of record")
    ap.add_argument("--chunk", type=int, default=None); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    if a.runs and a.step in ("affine", "solve"):
        ap.error("affine and solve write into the runs' folder: give --work only (--runs is for reading runs of record)")
    W = Work(a.work, a.runs)
    if a.step == "affine":
        affine(W, *a.args)
    elif a.step == "choose":
        SECTIONED, TEMPLATE = brains()
        for ID in SECTIONED + TEMPLATE:
            for t in config()["targets"]:
                choose(W, ID, t)
    elif a.step == "solve":
        solve(W, *a.args, chunk=a.chunk, device=a.device)
    elif a.step == "measure":
        from .. import provenance
        out, ins = measure(W, a.jobs)
        provenance.write_numbers("r060_target_comparison", out, inputs=ins, md5=False,
                                 cmd=" ".join(["python -m linc.evaluation.targets"] + list(argv if argv is not None else sys.argv[1:])))
    else:
        dict(prep=prep, place=place, grid=grid, registry=registry)[a.step](W)


if __name__ == "__main__":
    main()
