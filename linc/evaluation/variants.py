"""Measures that compare the seven MRI template variants (paper: Results "MRI template", Table 1, Supplementary Tables).

Every variant ("arm") is built from the same inputs: the brains of configs/mri_construction.yaml `inputs` (the paper's four, MF1-MF4 =
MF283..MF286) and their exact left-right mirrors (suffix f). Each input's canonical images were pushed through the arm's composed full warp once (one interpolation) and cached as
<cache>/<arm>__<MOD>[_vr]__<id>.nii.gz, MOD in T2, FRAC1, FRAC2, DTI (tensors, vecreg-canonicalized, tag _vr), B0, MASK.

Two families of measures:
  template-referenced  each input against a reference built from inputs (T2 NMI, soma/neurite fraction MAE, tensor overlap OVL);
                       the reference is either the mean of all eight inputs (in sample) or of the six inputs of the other three
                       brains (leave self out)
  pairwise             between two inputs (b0 Pearson r, warped brain-mask Dice, angle between principal eigenvectors)

The primitive measures are ported unchanged from the analysis project's evaluators: `fsl_mi_nmi`, `mats`, `ovl`, `logm`, `expm` and
the float64 loader from the old-metrics evaluator; `coverage_core`, `tensor_fields` and the float32 loader from the cross-modality
evaluator. Each measure is computed with the loader its evaluator used.
"""
import itertools

import nibabel as nib
import numpy as np

from linc.construction.tools import recipe as _recipe

BRAINS = list(_recipe()["inputs"])                                                   # the independent brains (configs/mri_construction.yaml)
IDS = [s + m for s in BRAINS for m in ("", "f")]                                     # the inputs in build order: each brain, then its mirror
TENSOR_TAG = "_vr"                     # cached tensors are the vecreg-canonicalized ones (Methods, tensor reorientation)

NMI_BINS = 64                          # Parzen joint histogram bins (FSL-like NMI)
NMI_ROBUST = (0.5, 99.5)               # percentiles of the in-mask robust rescale to [0, 1]
EIG_FLOOR_REL = 1e-6                   # log-Euclidean mean: eigenvalue floor = 1e-6 x median MD of the first two inputs
CORE_COVERAGE = 0.999                  # deliverable core: template coverage mask >= 0.999 ...
CORE_ERODE = 6                         # ... eroded by 6 voxels
MASK_THRESHOLD = 0.5                   # warped brain masks are binarized at >= 0.5

TMEAS = ["T2_NMI", "T2_NMI_WM", "FRAC1_MAE", "FRAC1_MAE_WM", "FRAC2_MAE", "DTI_OVL", "DTI_OVL_WM"]
PMEAS = ["fiber_angle_deg", "B0_r", "mask_dice"]
HIGHER = {"T2_NMI": True, "T2_NMI_WM": True, "FRAC1_MAE": False, "FRAC1_MAE_WM": False, "FRAC2_MAE": False,
          "DTI_OVL": True, "DTI_OVL_WM": True, "fiber_angle_deg": False, "B0_r": True, "mask_dice": True}

PAIRS = list(itertools.combinations(range(len(IDS)), 2))                     # the 28 input pairs
brain = lambda sid: sid.rstrip("f")                                          # noqa: E731
DISTINCT = [(i, j) for i, j in PAIRS if brain(IDS[i]) != brain(IDS[j])]      # 24 pairs between distinct brains
MIRROR = [(i, j) for i, j in PAIRS if brain(IDS[i]) == brain(IDS[j])]        # 4 brain-vs-own-mirror pairs


def cached(cache, arm, mod, sid):
    """The cached warped volume of one input for one arm and modality."""
    return cache / f"{arm}__{mod}{TENSOR_TAG if mod == 'DTI' else ''}__{sid}.nii.gz"


# --------------------------------------------------------------------------- old-metrics primitives (float64)

def load(p):
    return np.asanyarray(nib.load(str(p)).dataobj).astype(np.float64)


def fsl_mi_nmi(a, b, mask, bins=NMI_BINS, robust=NMI_ROBUST, eps=1e-12):
    """FSL-like Parzen MI and NMI = (H(A)+H(B))/H(A,B): in-mask robust rescale to [0, 1], linear Parzen window."""
    m = mask.astype(bool)
    amin, amax = np.percentile(a[m], robust[0]), np.percentile(a[m], robust[1])
    bmin, bmax = np.percentile(b[m], robust[0]), np.percentile(b[m], robust[1])
    ax = np.clip((a[m] - amin) / (amax - amin + eps), 0, 1) * (bins - 1)
    bx = np.clip((b[m] - bmin) / (bmax - bmin + eps), 0, 1) * (bins - 1)
    i0, j0 = np.floor(ax).astype(int), np.floor(bx).astype(int)
    wa, wb = ax - i0, bx - j0
    i1, j1 = np.clip(i0 + 1, 0, bins - 1), np.clip(j0 + 1, 0, bins - 1)
    H = np.zeros((bins, bins))
    np.add.at(H, (i0, j0), (1 - wa) * (1 - wb)); np.add.at(H, (i0, j1), (1 - wa) * wb)
    np.add.at(H, (i1, j0), wa * (1 - wb));       np.add.at(H, (i1, j1), wa * wb)
    P = H / (H.sum() + eps); Pa, Pb = P.sum(1), P.sum(0)
    Ha, Hb, Hab = (-np.sum(x * np.log(x + eps)) for x in (Pa, Pb, P))
    return float(Ha + Hb - Hab), float((Ha + Hb) / (Hab + eps))


def mats(t):
    """(..., 6) FSL-order tensor -> (..., 3, 3) symmetric matrices."""
    M = np.empty(t.shape[:-1] + (3, 3))
    M[..., 0, 0], M[..., 0, 1], M[..., 0, 2] = t[..., 0], t[..., 1], t[..., 2]
    M[..., 1, 0], M[..., 1, 1], M[..., 1, 2] = t[..., 1], t[..., 3], t[..., 4]
    M[..., 2, 0], M[..., 2, 1], M[..., 2, 2] = t[..., 2], t[..., 4], t[..., 5]
    return M


def ovl(Ma, Mb):
    """Basser-Pajevic tensor overlap sum_i l_i m_i (v_i . w_i)^2 / sum_i l_i m_i, per voxel."""
    la, va = np.linalg.eigh(Ma); lb, vb = np.linalg.eigh(Mb)
    la, lb = np.clip(la, 0, None), np.clip(lb, 0, None)
    d2 = np.einsum("nki,nki->ni", va, vb) ** 2          # same-rank eigenvector pairs (columns i)
    num = (la * lb * d2).sum(1); den = (la * lb).sum(1)
    return np.clip(np.where(den > 0, num / np.maximum(den, 1e-30), 0), 0, 1)


def logm(M, floor):
    w, v = np.linalg.eigh(M)
    return np.einsum("nij,nj,nkj->nik", v, np.log(np.maximum(w, floor)), v)


def expm(M):
    w, v = np.linalg.eigh(M)
    return np.einsum("nij,nj,nkj->nik", v, np.exp(w), v)


# --------------------------------------------------------------------------- cross-modality primitives (float32)

def load32(p):
    return np.asanyarray(nib.load(str(p)).dataobj).astype(np.float32)


def final_iteration(run):
    """The last iteration of the last nonlinear step of a construction run (<run>/out/nln_step_*/iteration_*)."""
    steps = sorted((run / "out").glob("nln_step_*"))
    if not steps:
        return None
    its = sorted(steps[-1].glob("iteration_*"))
    return its[-1] if its else None


def coverage_core(it, thresh=CORE_COVERAGE, erode=CORE_ERODE):
    """A construction run's full-coverage core: its template coverage mask >= thresh (the last scalar-channel mask in sorted order;
    the tensor channel's if there is no scalar one), eroded `erode` voxels."""
    m = None
    for f in sorted(it.glob("nln_template_*_scalar_mod_01_brain_mask.nii.gz")):
        m = load32(f) >= thresh
    if m is None:
        for f in sorted(it.glob("nln_template_*_tensor_mod_01_brain_mask.nii.gz")):
            m = load32(f) >= thresh
    if m is None:
        raise RuntimeError(f"no template coverage mask in {it}")
    if erode:
        from scipy import ndimage
        m = ndimage.binary_erosion(m, iterations=erode)
    return m


def tensor_fields(vols, mask):
    """Principal eigenvector and FA per input, inside `mask`. vols are (X, Y, Z, 6) FSL-order tensors."""
    def eig(t):
        n = t.shape[:3]
        M = np.zeros(n + (3, 3), dtype=np.float32)
        M[..., 0, 0], M[..., 0, 1], M[..., 0, 2] = t[..., 0], t[..., 1], t[..., 2]
        M[..., 1, 0], M[..., 1, 1], M[..., 1, 2] = t[..., 1], t[..., 3], t[..., 4]
        M[..., 2, 0], M[..., 2, 1], M[..., 2, 2] = t[..., 2], t[..., 4], t[..., 5]
        w, v = np.linalg.eigh(M[mask])
        lam = np.clip(w, 0, None)
        md = lam.mean(1, keepdims=True)
        num = ((lam - md) ** 2).sum(1)
        den = (lam ** 2).sum(1)
        fa = np.sqrt(1.5 * num / np.maximum(den, 1e-20))
        return v[:, :, -1], fa            # principal eigenvector, FA

    V, FA = zip(*[eig(t) for t in vols])
    return list(V), list(FA)


# --------------------------------------------------------------------------- the variant measures

def template_measures(cache, arm, sup):
    """Per input, the template-referenced measures against an in-sample reference (mean of all eight inputs) and a leave-self-out
    reference (mean of the six inputs of the other three brains; a brain and its mirror share one). The reference is built as the
    old-metrics evaluator builds the template: T2 each input scaled to mean 1000 in the whole support, then plain mean; fractions plain
    mean; tensor log-Euclidean mean on the union of the whole and WM supports. sup = {"whole", "micro", "wm"} boolean volumes."""
    f = lambda mod, s: cached(cache, arm, mod, s)                                # noqa: E731
    whole, micro, wm = sup["whole"], sup["micro"], sup["wm"]
    t2 = [load(f("T2", s)) for s in IDS]
    t2 = [v * (1000.0 / v[whole].mean()) for v in t2]
    fr = {m: [load(f(m, s)) for s in IDS] for m in ("FRAC1", "FRAC2")}
    sel = whole | wm
    Ts = [mats(load(f("DTI", s))[sel]) for s in IDS]
    md = np.median([np.trace(M, axis1=1, axis2=2).mean() / 3 for M in Ts[:2]])
    L = [logm(M, EIG_FLOOR_REL * md) for M in Ts]
    idx_w, idx_m = whole[sel], wm[sel]
    refs = {"in_sample": lambda k: list(range(len(IDS))),
            "leave_self_out": lambda k: [j for j in range(len(IDS)) if brain(IDS[j]) != brain(IDS[k])]}
    out = {}
    for kind, members in refs.items():
        rows = []
        cache_ref = {}
        for k, s in enumerate(IDS):
            mem = tuple(members(k))
            if mem not in cache_ref:
                cache_ref.clear()
                cache_ref[mem] = (np.mean([t2[j] for j in mem], 0), {m: np.mean([fr[m][j] for j in mem], 0) for m in fr},
                                  expm(np.mean([L[j] for j in mem], 0)))
            T2t, frt, Tt = cache_ref[mem]
            r = {"subject": s}
            _, r["T2_NMI"] = fsl_mi_nmi(t2[k], T2t, whole)
            _, r["T2_NMI_WM"] = fsl_mi_nmi(t2[k], T2t, wm)
            for m in ("FRAC1", "FRAC2"):
                r[f"{m}_MAE"] = float(np.abs(fr[m][k] - frt[m])[micro].mean())
                r[f"{m}_MAE_WM"] = float(np.abs(fr[m][k] - frt[m])[wm & micro].mean())
            o = ovl(Ts[k], Tt)
            r["DTI_OVL"], r["DTI_OVL_WM"] = float(o[idx_w].mean()), float(o[idx_m].mean())
            rows.append(r)
        out[kind] = rows
    return out


def pairwise_measures(cache, arm, common, wm):
    """Per input pair: b0 Pearson r in the common support, Dice of the warped masks (>= 0.5, whole grid), and the mean angle between
    principal eigenvectors in `wm` (a boolean vector over the common-support voxels)."""
    f = lambda mod, s: cached(cache, arm, mod, s)                                # noqa: E731
    X = np.stack([load32(f("B0", s))[common] for s in IDS])
    r = {p: float(np.corrcoef(X[p[0]], X[p[1]])[0, 1]) for p in PAIRS}
    del X
    bm = [load32(f("MASK", s)) >= MASK_THRESHOLD for s in IDS]
    d = {(i, j): float(2 * (bm[i] & bm[j]).sum() / max(bm[i].sum() + bm[j].sum(), 1)) for i, j in PAIRS}
    del bm
    V = []
    for s in IDS:
        v, _ = tensor_fields([load32(f("DTI", s))], common)
        V.append(v[0])
    ang = {}
    for i, j in PAIRS:
        c = np.abs(np.einsum("ij,ij->i", V[i][wm], V[j][wm])).clip(0, 1)
        ang[(i, j)] = float(np.degrees(np.arccos(c)).mean())
    return {"fiber_angle_deg": ang, "B0_r": r, "mask_dice": d}


def summarize_pairs(per_pair):
    res = {}
    for m, pp in per_pair.items():
        per_brain = {b: float(np.mean([pp[p] for p in DISTINCT if b in (brain(IDS[p[0]]), brain(IDS[p[1]]))])) for b in BRAINS}
        brain_pairs = {f"{a}-{b}": float(np.mean([pp[p] for p in DISTINCT if {brain(IDS[p[0]]), brain(IDS[p[1]])} == {a, b}]))
                       for a, b in itertools.combinations(BRAINS, 2)}
        res[m] = {"all28_mean": float(np.mean([pp[p] for p in PAIRS])),
                  "distinct24_mean": float(np.mean([pp[p] for p in DISTINCT])),
                  "own_mirror_pairs": {brain(IDS[i]): pp[(i, j)] for i, j in MIRROR},
                  "own_mirror_mean": float(np.mean([pp[p] for p in MIRROR])),
                  "per_brain_distinct": per_brain, "per_unique_brain_pair": brain_pairs,
                  "per_pair": {f"{IDS[i]}~{IDS[j]}": pp[(i, j)] for i, j in PAIRS}}
    return res


def summarize_template(rows):
    res = {}
    for m in TMEAS:
        v = {r["subject"]: r[m] for r in rows}
        res[m] = {"mean": float(np.mean(list(v.values()))), "per_input": v,
                  "per_brain": {b: float((v[b] + v[b + "f"]) / 2) for b in BRAINS}}
    return res


def exact_bootstrap(d):
    """All len(d)^len(d) resamples with replacement, equally weighted; returns the resample means."""
    d = np.asarray(d, float)
    n = len(d)
    return np.array([d[list(ix)].mean() for ix in itertools.product(range(n), repeat=n)])


def paired(per_brain_a, per_brain_b, m):
    """Arm a minus arm b per brain, with the exact bootstrap over the four brains."""
    diff = [per_brain_a[b] - per_brain_b[b] for b in BRAINS]
    bs = exact_bootstrap(diff)
    better_a = [(x > 0) if HIGHER[m] else (x < 0) for x in diff]
    return {"diff_four_channel_minus_t2tensor_per_brain": dict(zip(BRAINS, diff)),
            "mean_diff": float(np.mean(diff)), "min_diff": float(np.min(diff)), "max_diff": float(np.max(diff)),
            "bootstrap_n_resamples": int(bs.size), "bootstrap_q025": float(np.quantile(bs, 0.025)),
            "bootstrap_q975": float(np.quantile(bs, 0.975)),
            "bootstrap_frac_resamples_favoring_four_channel": float(np.mean(bs > 0) if HIGHER[m] else np.mean(bs < 0)),
            "n_brains_favor_four_channel": int(sum(better_a)),
            "n_brains_favor_t2tensor": int(sum((x < 0) if HIGHER[m] else (x > 0) for x in diff)),
            "higher_is_better": HIGHER[m]}


def ranking(vals, m):
    return sorted(vals, key=lambda a: vals[a], reverse=HIGHER[m])
