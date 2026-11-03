"""sigma^2_pred(D): the completion's squared error by depth past the cut, which the blockface template's construction turns into
precision weights for predicted tissue (reproduce/numbers/precision_weights_b1_d2b_s8.py reads d3_A_sigma2_pred_flat).

Measured on the withheld slabs of the 12 evaluation brains, from the leave-slab-out network's joined predictions (deploy.join, saved
by the evaluation as <run>_<ID>_deployed.nii.gz on [0, 1]), mapped back to the photographs' grey with each brain's target
percentiles: per brain, at each depth D = 0.25 j mm (j = rint(cut - y_dep), planes past the simulated cut along the section normal),
the MSE and mean residual (prediction - photographs) over the scored slab voxels of that depth (>= 50 voxels). Two variants: `all`
and `edge3` (voxels >= 3 mm inside the photographed tissue of their anterior-posterior column; the headline).
Fit (`pooled_from`, `fit`): the per-depth MSE adjusted for brain (log MSE = a_brain + b_depth, alternating least squares),
isotonic non-decreasing over the contiguous depths reached by >= 3 brains (weights = brains per depth), linear beyond with the
non-negative weighted least-squares slope.
s2_interp: the photographs' own interpolation floor in the same units, the MSE between each photographed voxel of the first scored
depth and its photographed neighbor 0.25 mm along the grid's anterior-posterior axis, averaged over the brains.
"""
import json

import nibabel as nib
import numpy as np

from . import CFG, deploy

ISO = CFG["output_mm"]
MIN_SUBJ = 3                    # depths reached by fewer brains end the measured curve
EDGE_MM = 3.0                   # edge3: voxels at least this far inside their column's photographed tissue
MIN_VOX = 50                    # depth bins with fewer voxels per brain are dropped
AT_MM = (0.5, 1.0, 2.0, 4.0, 8.0)


def ap_axis(affine):
    """the array axis most aligned with world anterior-posterior (RAS +y), and whether anterior is at the high index"""
    ax = int(np.argmax(np.abs(affine[:3, :3][1])))
    return ax, bool(affine[1, ax] > 0)


def signed_distance(real_mask, affine, iso):
    """per anterior-posterior column: mm from the column's photographed end (> 0 inside the photographed tissue, < 0 beyond it)"""
    ax, _ = ap_axis(affine)
    ant_low = affine[1, ax] > 0
    rm = np.moveaxis(real_mask, ax, 0); shp = rm.shape; A = shp[0]
    flat = rm.reshape(A, -1).astype(bool); ap = np.arange(A, dtype=np.float32)[:, None]; BIG = A + 10.0
    has = flat.any(0)
    if ant_low:
        seam = np.where(flat, ap, BIG).min(0); s = ap - seam[None, :]
    else:
        seam = np.where(flat, ap, -BIG).max(0); s = seam[None, :] - ap
    s = np.where(has[None, :], s, np.nan)
    return np.moveaxis((s * iso).reshape(shp), 0, ax)


def depth_tables(ev, D_of, pred_of, members):
    """{variant: {brain: {j: (mse, bias, n)}}}, {brain: interpolation-floor record}, {brain: scored voxels}.
    ev = the evaluation record (target_norm per brain); D_of(S) = the prepared folder; pred_of(S) = the saved joined prediction."""
    tabs, lags, nvox = {"all": {}, "edge3": {}}, {}, {}
    for S in members:
        D = D_of(S)
        rg = deploy.regions(D, "lso")
        q1, q99 = ev["subjects"][S]["target_norm"]
        pred = q1 + np.asanyarray(nib.load(pred_of(S)).dataobj).astype(np.float32) * (q99 - q1)
        real = np.asanyarray(nib.load(f"{D}/fine/bf.nii.gz").dataobj).astype(np.float32)
        rm = np.asanyarray(nib.load(f"{D}/fine/real_mask.nii.gz").dataobj) > 0
        j = np.rint(rg["cut"] - rg["ydep"]).astype(np.int32)
        d_true = signed_distance(rm & rg["brain"], rg["affine"], ISO)
        masks = dict(all=rg["score"], edge3=rg["score"] & np.isfinite(d_true) & (d_true >= EDGE_MM))
        e = pred - real
        for v, m in masks.items():
            row = {}
            for jj in np.unique(j[m]):
                mk = m & (j == jj)
                if mk.sum() >= MIN_VOX:
                    row[int(jj)] = (float(np.mean(e[mk] ** 2)), float(np.mean(e[mk])), int(mk.sum()))
            if row:
                tabs[v][S] = row
        k0 = min(tabs["all"][S]) if S in tabs["all"] else None
        if k0 is not None:
            m0 = rg["score"] & (j == k0)
            nb = np.zeros_like(m0); nb[:, :-1, :] = m0[:, :-1, :] & rm[:, 1:, :]
            idx = np.argwhere(nb)
            if len(idx) >= MIN_VOX:
                g0 = real[tuple(idx.T)]; g1 = real[idx[:, 0], idx[:, 1] + 1, idx[:, 2]]
                lags[S] = dict(mse=float(np.mean((g1 - g0) ** 2)), n=int(len(idx)), d_mm=k0 * ISO)
        nvox[S] = int(rg["score"].sum())
        print(S, "bins", len(tabs["all"].get(S, {})), "edge3", len(tabs["edge3"].get(S, {})), flush=True)
    return tabs, lags, nvox


def adjusted(tab):
    """per-bin cohort-mean MSE had every brain reached the bin: log mse_sk = a_s + b_k on the unbalanced table (alternating least
    squares), reported as mean_s exp(a_s + b_k)"""
    subs = sorted(tab)
    ks = sorted({k for d in tab.values() for k in d})
    Y = np.full((len(subs), len(ks)), np.nan)
    for i, S in enumerate(subs):
        for j, k in enumerate(ks):
            if k in tab[S]:
                Y[i, j] = np.log(tab[S][k])
    a = np.zeros(len(subs))
    for _ in range(500):
        b = np.nanmean(Y - a[:, None], 0)
        a = np.nanmean(Y - b[None, :], 1)
    return {k: float(np.exp(b[j]) * np.mean(np.exp(a))) for j, k in enumerate(ks)}


def pava(y, w):
    """weighted isotonic (non-decreasing) regression, pool-adjacent-violators"""
    v, ww, c = [], [], []
    for yi, wi in zip(y, w):
        v.append(float(yi)); ww.append(float(wi)); c.append(1)
        while len(v) > 1 and v[-2] > v[-1]:
            t = ww[-2] + ww[-1]
            v[-2] = (v[-2] * ww[-2] + v[-1] * ww[-1]) / t; ww[-2] = t; c[-2] += c[-1]
            v.pop(); ww.pop(); c.pop()
    return np.repeat(v, c)


def fit(pool):
    need = min(MIN_SUBJ, pool[0]["n_subj"])
    P = []
    for p in pool:                                      # contiguous from the cut, >= MIN_SUBJ brains per bin
        if p["n_subj"] < need or (P and p["k"] != P[-1]["k"] + 1):
            break
        P.append(p)
    d = np.array([p["d_mid"] for p in P]); y = np.array([p["mse_adj"] for p in P])
    w = np.array([p["n_subj"] for p in P], float)
    iso = pava(y, w)
    sw = np.sqrt(w)
    ls = lambda s: float(np.linalg.lstsq(np.c_[d[s], np.ones(s.sum())] * sw[s, None], y[s] * sw[s], rcond=None)[0][0])  # noqa: E731
    slope, slope_tail = ls(np.ones(d.size, bool)), ls(d >= d[-1] - 2.0)
    b = max(slope, 0.0)

    def f(x):
        x = np.asarray(x, float)
        return np.where(x <= d[-1], np.interp(x, d, iso), iso[-1] + b * (x - d[-1]))
    return dict(
        curve="per-bin MSE adjusted for subject (mse_adj), isotonic non-decreasing, weights = subjects per bin",
        min_subjects=need, measured_bins_mm=[float(P[0]["d_lo"]), float(P[-1]["d_lo"] + ISO)],
        all_subjects_to_mm=float(max(p["d_lo"] + ISO for p in P if p["n_subj"] == pool[0]["n_subj"])),
        isotonic=[[float(a), float(v)] for a, v in zip(d, iso)],
        rule=(f"d <= {d[-1]:g} mm: linear interpolation of `isotonic` (constant below {d[0]:g}); "
              f"d > {d[-1]:g} mm (EXTRAPOLATED): {iso[-1]:.6g} + {b:.6g} * (d - {d[-1]:g})"),
        extrapolated_beyond_mm=float(d[-1]), anchor=float(iso[-1]), slope_per_mm=b,
        slope_wls_full_range=slope, slope_wls_last_2mm=slope_tail,
        at_mm={f"{x:g}": float(f(x)) for x in AT_MM},
        table=[[float(x), float(f(x)), bool(x > d[-1])] for x in np.arange(0.125, 16.0, 0.25)])


def pooled_from(tab):
    """pooling of a table {brain: {j: (mse, bias, n)}}; bin j = the voxels at D = 0.25 j, reported with d_mid = 0.25 j and
    d_lo = d_mid - 0.125"""
    adj = adjusted({S: {k: t[0] for k, t in d.items()} for S, d in tab.items()})
    out = []
    for k in sorted(adj):
        pts = np.array([d[k] for d in tab.values() if k in d])
        mse, bias, n = pts[:, 0], pts[:, 1], pts[:, 2]
        out.append(dict(k=k, d_lo=k * ISO - ISO / 2, d_mid=k * ISO, n_subj=len(pts), n_vox=int(n.sum()),
                        mse_mean=float(mse.mean()), mse_sd=float(mse.std(ddof=1)) if len(pts) > 1 else None,
                        mse_vox=float((mse * n).sum() / n.sum()), bias_mean=float(bias.mean()),
                        bias_vox=float((bias * n).sum() / n.sum()), mse_adj=adj[k]))
    return out


def tables_json(tabs, lags, nvox):
    return json.dumps(dict(tables={v: {S: {str(k): t for k, t in d.items()} for S, d in tabs[v].items()} for v in tabs},
                           lags=lags, nvox=nvox), indent=1)
