#!/usr/bin/env python3
"""The paper's supplementary figure "From the chart to the template, one section per case": does the chain keep the tracer labels
where the anatomist charted them? The histology, the blockface and the MRI template are shown in one space: the plane of the
blockface photograph each section was registered to.

You need
  the products of record under $LINC_PRODUCTS (settings/internal/products_figures.yaml): the pushed tracer sheets (push_sheets), the
  tracer labels in the template (tracer_template), the template's T2 and brain mask, the histology registration each brain's labels
  went through (settings/internal/products_blockface.yaml labels_histology_arm), the blockface volumes and the transforms of record
  (MM1-MM3: blockface -> b0 -> template; MF8: through the blockface template). settings/env/tirl.yml ($PY_TIRL), the environment the
  paper's figure was made in.

What it does
  One row per case, at its injection level (the section with the most injection-site pixels; MM3, without an annotated site, the
  section with the most label). The three panels of a row are one plane, the blockface photograph the section was registered to,
  at one scale and in one field of view (the registered section plus 1 mm). Every panel draws the same outline (cyan: the section's
  tissue after the full registration); only the image beneath changes:
    (i)   the section's darkfield image after the full registration (the registration's own output), with its chart transformed
          with it;
    (ii)  the blockface photograph, with the same chart;
    (iii) the MRI template's T2 on the photograph's plane, with the labels the template carries for this section (the template
          voxels whose pull through the chain of record lands in the section's sheet), at the template's 0.25 mm. Each pixel is
          mapped to the template by inverting the chain of record: Newton steps on its pull, from the nearest pulled template voxel;
          the residual is in the provenance.
  A section is cut obliquely to the template's coronal plane, so (iii) is not one coronal slice of the template; the template y the
  registered section spans is in the provenance. MF8's two frontal lobes were matched to adjacent photographs (0.75 mm apart): each
  pixel is drawn on the photograph of the nearest lobe. Anatomical convention (the subject's left on the left, dorsal up; no case
  mirrored; L and R once, beside the first row). Row labels, on a line above each row: the case, the section's number in its
  labeled series, the photograph(s); the row's 5 mm bar on the same line at its right (each row has its own scale).
  Space (the author, 12 Nov 2026): one gutter (figstyle GUTTER_MM) between the columns and between a row and the next row's label,
  MARGIN_MM from the outermost ink to the page's edges; a zoomed panel's bar in its standard place, at its title's level, top right
  (figstyle.scalebar_title), one per row (before, one under every panel); every text in sentence case.
  Classes: bundles (dense and moderate, and MM2's single bundle class), light bundles, terminal fields (all three density classes),
  injection site; single-fiber strokes are not drawn.

You get
  --out (default figures/s_tracer_fidelity.png) and its provenance.json (the rows: case, section, why, blockface slices, window,
  registration folder, the template y of the registered section, the inverse's residual).

    $PY_TIRL code/figures/s_tracer_fidelity.py [--out figures/s_tracer_fidelity.png]

Ported from the analysis project's paper/fig_tracer_fidelity_v2_r058.py (--set r058d --layout s3) and the helpers it imported from
paper/fig_tracer_fidelity_r058.py; drawn in the photograph's plane since 11 Nov 2026.
"""
import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "code"))
import figstyle as FS                                           # noqa: E402
from shared import paths, xfm                                     # noqa: E402

Image.MAX_IMAGE_PIXELS = None
CASES = ["MR243", "MR252", "MR256", "MF191"]
TV = 0.25
OUTLINE = "#00e5ff"
SB_MM = 5                                                # one bar per row, on its label's line, at the right
TITLE_GAP_PT = 4.0                                       # the column titles this far above the first row's label (pt): apart from it
GROUPS = [("bundles", ("Fiber_dense_bundle", "Fiber_moderate_bundle", "Bundle"), "#D55E00", 0.9),
          ("light bundles", ("Fiber_light_bundle",), "#E69F00", 0.5),
          ("terminal fields", ("Terminal_dense", "Terminal_moderate", "Terminal_light"), "#56B4E9", 0.5),
          ("injection site", ("Injection_site",), "#ffffff", 0.9)]


def text_mm(plt):
    """a function (text) -> (width, height) mm of a figstyle.MIN_PT text, measured at figstyle's DPI (at the screen's 100,
    hinting widens a 7 pt text by up to 8 %)"""
    fig = plt.figure(figsize=(7, 2), dpi=FS.DPI); r = fig.canvas.get_renderer()
    def T(s_):
        t = fig.text(0, 0, s_, fontsize=FS.MIN_PT); e = t.get_window_extent(r); t.remove()
        return e.width / fig.dpi * 25.4, e.height / fig.dpi * 25.4
    return T


def load(p):
    return np.asanyarray(nib.load(str(p)).dataobj)


def chain_pull(S, P):
    """Template world points -> the brain's blockface world points, through the chain of record."""
    if S == "MF191":
        import tracer
        tracer.roots()
        return tracer.mf8_chain()
    B = json.loads(Path(P["r1_record"]).read_text()); B = (B.get("data") or B)["brains"]
    assert B[S]["qc"] == "pass"
    return xfm.Chain([xfm.Entry(paths.rerooted(B[S]["registry"])), xfm.Entry(str(P["data"] / f"sub-{S}" / "xfm" / "b0-native__to__mri-template"))],
                     strict=True).pull


def carried(S, P):
    """Per blockface slice: {class: template ijk of the voxels that land in that section's sheet}."""
    files = sorted(glob(str(P["push_sheets"] / S / f"{S}_*_bfnative_L2.nii.gz")))
    classes = [Path(f).name[len(S) + 1:-len("_bfnative_L2.nii.gz")] for f in files]
    keep = [i for i, c in enumerate(classes) if any(c in g[1] for g in GROUPS)]
    im0 = nib.load(files[0]); A = im0.affine; shp = im0.shape[:3]; z = np.array(im0.header.get_zooms()[:3], float)
    half = int(round(TV / z[1] / 2)) if z[1] < TV else 0
    bits = np.zeros(shp, np.uint16)
    for k, i in enumerate(keep):
        bits |= (load(files[i]) > 0).astype(np.uint16) << k
    cls = [classes[i] for i in keep]
    sec = np.full(shp, -1, np.int32); wb = np.zeros(shp, np.uint16)
    for j in np.nonzero(bits.any((0, 2)))[0]:
        sl = bits[:, j, :]
        for d in range(-half, half + 1):
            jj = j + d
            if 0 <= jj < shp[1]:
                put = (sl > 0) & (sec[:, jj, :] < 0)
                sec[:, jj, :][put] = j; wb[:, jj, :][put] = sl[put]
    ref = nib.load(str(P["template_t2"])); At = ref.affine
    foot = np.zeros(ref.shape[:3], bool)
    for f in glob(str(P["tracer_template"] / S / f"{S}_*_mri-template_footprint.nii.gz")):
        foot |= load(f) > 0
    foot = ndi.binary_dilation(foot, iterations=3)
    tv = np.argwhere(foot); yw = nib.affines.apply_affine(At, tv)
    pull = chain_pull(S, P)
    pb = np.concatenate([np.asarray(pull(yw[k0:k0 + 200000])) for k0 in range(0, len(yw), 200000)])
    q = np.rint(nib.affines.apply_affine(np.linalg.inv(A), pb)).astype(int)
    ok = np.all((q >= 0) & (q < np.array(shp)), 1)
    qs = np.full(len(q), -1); qb = np.zeros(len(q), np.uint16)
    qs[ok] = sec[q[ok, 0], q[ok, 1], q[ok, 2]]; qb[ok] = wb[q[ok, 0], q[ok, 1], q[ok, 2]]
    out = {}
    for j in np.unique(qs[qs >= 0]):
        m = qs == j
        out[int(j)] = {c: tv[m & (((qb >> k) & 1) > 0)] for k, c in enumerate(cls)}
    return out, files, classes, A


def choose(car):
    """Median template y of each section's bundle voxels (sections with at least 30)."""
    yinfo = {}
    for j, d in car.items():
        b = np.concatenate([v for c, v in d.items() if c in GROUPS[0][1]] or [np.zeros((0, 3), int)])
        if len(b) >= 30:
            yinfo[j] = float(np.median(b[:, 1]))
    return yinfo


def to_template(pull, X, At, shape, step=4, iters=12, h=0.1, tol=1e-4):
    """Blockface world points X -> template world points T with pull(T) = X: Newton steps (forward-difference Jacobian, h mm) from
    the template voxel (every step-th) whose pull lands nearest; returns T and the residual |pull(T) - X| (mm)."""
    g = np.stack(np.meshgrid(*[np.arange(0, n, step) for n in shape], indexing="ij"), -1).reshape(-1, 3)
    gw = nib.affines.apply_affine(At, g)
    gp = np.concatenate([np.asarray(pull(gw[k0:k0 + 500000])) for k0 in range(0, len(gw), 500000)])
    ok = np.isfinite(gp).all(1)
    T = gw[ok][cKDTree(gp[ok]).query(X)[1]]
    E = np.eye(3) * h
    for _ in range(iters):
        F = np.asarray(pull(T)); R = X - F
        J = np.stack([(np.asarray(pull(T + E[k])) - F) / h for k in range(3)], axis=2)
        dT = np.linalg.solve(J, R[..., None])[..., 0]
        T = T + dT
        if np.abs(dT).max() < tol:
            break
    return T, np.linalg.norm(np.asarray(pull(T)) - X, axis=1)


def world_y(At, jy):
    return At[1, 3] + At[1, 1] * jy


def tissue_extent_mm(mask, px_mm):
    ij = np.argwhere(mask)
    lo, hi = ij.min(0) * px_mm, (ij.max(0) + 1) * px_mm
    return (lo + hi) / 2, (hi - lo)


def show(ax, img, px_mm, center, half, **kw):
    H, W = img.shape[:2]; hy, hx = half
    ax.imshow(img, extent=(0, W * px_mm, H * px_mm, 0), interpolation="none", **kw)
    ax.set_xlim(center[1] - hx, center[1] + hx); ax.set_ylim(center[0] + hy, center[0] - hy)
    ax.set_xticks([]); ax.set_yticks([]); [s_.set_visible(False) for s_ in ax.spines.values()]


def contour(ax, m, px_mm, color, lw, ls="-"):
    if m.any():
        H, W = m.shape
        ax.contour(np.linspace(px_mm / 2, W * px_mm - px_mm / 2, W), np.linspace(px_mm / 2, H * px_mm - px_mm / 2, H), m.astype(float), [0.5],
                   colors=[color], linewidths=lw, linestyles=ls)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figures/s_tracer_fidelity.png")
    a = ap.parse_args(argv)
    P = paths.products("products_figures")
    arms = paths.products("products_blockface")["labels_histology_arm"]
    FS.setup()
    import matplotlib.pyplot as plt
    ref = nib.load(str(P["template_t2"])); At = ref.affine; iAt = np.linalg.inv(At); T2 = load(P["template_t2"]).astype(float)
    brain = load(P["template_mask"]) > 0.5
    rows_out, cache, inputs = [], {}, [Path(P["template_t2"]), Path(P["template_mask"])]
    for S in CASES:
        car, files, classes, A = carried(S, P)
        pj = json.load(open(P["push_sheets"] / S / f"push_tracer_{S}.json"))
        inputs += files + [P["push_sheets"] / S / f"push_tracer_{S}.json"]
        sec_sl, sl_secs = {}, {}
        for p in pj["pieces"]:
            sec_sl.setdefault(p["section"], set()).add(p["bf_idx"]); sl_secs.setdefault(p["bf_idx"], set()).add(p["section"])
        carS = {}
        for sid, sls in sec_sl.items():
            if any(len(sl_secs[j]) > 1 for j in sls):
                continue                                                       # a slice shared with another section: ambiguous
            carS[sid] = {c: np.concatenate([car[j][c] for j in sls if j in car and c in car[j]] or [np.zeros((0, 3), int)])
                         for c in set().union(*[set(car.get(j, {})) for j in sls])}
        yinfo = choose(carS)
        for sid_, d_ in carS.items():                                          # sections with < 30 bundle voxels: the median y of all their labels
            if sid_ not in yinfo:
                allv = np.concatenate([v for v in d_.values() if len(v)] or [np.zeros((0, 3), int)])
                if len(allv):
                    yinfo[sid_] = float(np.median(allv[:, 1]))
        inj = {sid: sum(p["classes"].get("Injection_site", {}).get("px_L2", 0) for p in pj["pieces"] if p["section"] == sid) for sid in carS}
        tot = {sid: sum(sum(v["px_L2"] for v in p["classes"].values()) for p in pj["pieces"] if p["section"] == sid) for sid in carS}
        pick = max(carS, key=lambda s_: (inj[s_], tot[s_] if not any(inj.values()) else 0))
        has_site = any(v.get("px_L2", 0) > 0 for p_ in pj["pieces"] for c_, v in p_["classes"].items() if c_ == "Injection_site")
        why = "injection level" if inj[pick] else ("most labeled\n(site on a shared\nphotograph)" if has_site else "most labeled\n(no site annotated)")
        cache[S] = (carS, files, classes, A, pj, sec_sl)
        rows_out.append((S, pick, why))
    W, GAP, MG, pt = FS.FULL_W_IN * 25.4, FS.GUTTER_MM, FS.MARGIN_MM, 25.4 / 72
    titles = ["(i) The section (darkfield),\nregistered to its photograph", "(ii) The blockface photograph",
              "(iii) The MRI template (T2)\non the photograph's plane"]
    T = text_mm(plt); plt.close("all")
    t_h, ct_h = T("MM1, section 6")[1], max(T(t)[1] for t in titles)
    DPAD = 1.0                                                             # pt: the L and R beside the first row
    LEFT, RIGHT = MG + T("L")[0] + DPAD * pt, T("R")[0] + DPAD * pt + MG
    HEAD = t_h + 2.0 * pt                                                  # a row's label line, 2 pt above its panels
    TOP = MG + ct_h + TITLE_GAP_PT * pt + HEAD                              # the column titles above the first row's label
    pw = (W - LEFT - 2 * GAP - RIGHT) / 3
    disp = lambda v: np.flipud(np.swapaxes(v, 0, 1))  # noqa: E731      volume (x, z) -> display (rows dorsal up, columns x)
    rows, prepped = [], []
    for S, sid, why in rows_out:
        carS, files, classes, A, pj, sec_sl = cache[S]
        root = paths.products_root() / arms[S]                                 # the histology registration the labels went through
        bf_root = root / f"sub-{S}" / "blockface"
        pieces = [p for p in pj["pieces"] if p["section"] == sid]
        sls = sorted(sec_sl[sid])
        bfp = P["data"] / f"sub-{S}" / "blockface" / f"{S}_blockface_L2.nii.gz"
        bf = nib.load(str(bfp)); assert np.allclose(bf.affine, A), "blockface and tracer sheets on different grids"
        nx, nz = bf.shape[0], bf.shape[2]; px_b = float(abs(A[0, 0]))
        # the registered pieces: their tissue after the full registration (the L2 grid) and their registered darkfield (TIRL's canvas,
        # level 0 = half the L2 pixel)
        reg = np.zeros((nz, nx), bool); by_sl = {}; his0 = np.zeros((2 * nz, 2 * nx))
        for p in pieces:
            f = bf_root / "06_histo2bf" / "tirl" / p["piece"] / "moving4_nonlinear.png"; inputs.append(f)
            H_ = np.asarray(Image.open(f)); H_ = (H_.max(-1) if H_.ndim == 3 else H_)[::-1, ::-1]   # slide -> store orientation
            m0 = (H_ > 0) & (H_ < 254.5)
            h, w_ = min(H_.shape[0], 2 * nz), min(H_.shape[1], 2 * nx)
            c0 = np.zeros((2 * nz, 2 * nx), bool); c0[:h, :w_] = m0[:h, :w_]
            his0[c0] = H_[:h, :w_][m0[:h, :w_]]
            r1 = np.flipud(c0.reshape(nz, 2, nx, 2).max((1, 3)))              # level 1 = the L2 grid; store rows -> display (dorsal up)
            reg |= r1; by_sl[p["bf_idx"]] = by_sl.get(p["bf_idx"], False) | r1
        his = np.flipud(his0)
        sl_ = sorted(by_sl)                                                    # each pixel on the photograph of the nearest registered piece
        jmap = (np.full((nz, nx), sl_[0]) if len(sl_) == 1 else
                np.asarray(sl_)[np.argmin([ndi.distance_transform_edt(~by_sl[s]) for s in sl_], axis=0)])
        ph = np.zeros((nz, nx, 3)); lab = {g: np.zeros((nz, nx), bool) for g, *_ in GROUPS}
        for s in np.unique(jmap):
            on = jmap == s
            ph[on] = disp(np.asarray(bf.dataobj[:, s, :, :]))[on]
            for g, cls, col, lw in GROUPS:
                for c in cls:
                    if c in classes:
                        lab[g] |= on & disp(np.asarray(nib.load(files[classes.index(c)]).dataobj[:, s, :]) > 0)
        inputs.append(bfp)
        # one field of view for the row: the registered section plus 1 mm
        cb, eb = tissue_extent_mm(reg, px_b); half = (float(eb[0]) / 2 + 1.0, float(eb[1]) / 2 + 1.0)
        r0, r1_ = max(int((cb[0] - half[0]) / px_b), 0), min(int(np.ceil((cb[0] + half[0]) / px_b)) + 1, nz)
        k0, k1 = max(int((cb[1] - half[1]) / px_b), 0), min(int(np.ceil((cb[1] + half[1]) / px_b)) + 1, nx)
        dd, cc = np.mgrid[r0:r1_, k0:k1]
        Xb = nib.affines.apply_affine(A, np.stack([cc.ravel(), jmap[dd, cc].ravel(), (nz - 1 - dd).ravel()], 1).astype(float))
        Tw, res = to_template(chain_pull(S, P), Xb, At, T2.shape)
        tv = nib.affines.apply_affine(iAt, Tw)
        win = (slice(r0, r1_), slice(k0, k1)); wsh = dd.shape
        t2 = np.full((nz, nx), np.nan); t2[win] = ndi.map_coordinates(T2, tv.T, order=1, cval=0.0).reshape(wsh)
        bad = (res > 0.01).reshape(wsh); t2[win][bad] = np.nan; bad_in = int((bad & reg[win]).sum())
        q = np.rint(tv).astype(int); okq = np.all((q >= 0) & (q < np.array(T2.shape)), 1); q[~okq] = 0
        bm = np.zeros((nz, nx), bool); bm[win] = (brain[q[:, 0], q[:, 1], q[:, 2]] & okq).reshape(wsh)
        lab3 = {}
        for g, cls, col, lw in GROUPS:
            G = np.zeros(T2.shape, bool)
            for c in cls:
                v = carS[sid].get(c)
                if v is not None and len(v):
                    G[v[:, 0], v[:, 1], v[:, 2]] = True
            lab3[g] = np.zeros((nz, nx), bool); lab3[g][win] = (G[q[:, 0], q[:, 1], q[:, 2]] & okq).reshape(wsh)
        on_reg = reg[win].ravel()
        ty = (float(Tw[on_reg, 1].min()), float(Tw[on_reg, 1].max()))
        print(f"{S} {sid}: window {wsh}, inverse residual median {np.median(res):.2e} max {res.max():.2e} mm, {int(bad.sum())} px > 0.01 mm; "
              f"({bad_in} in the section); registered section at template y {ty[0]:.2f}..{ty[1]:.2f} mm", flush=True)
        prepped.append(dict(S=S, sid=sid, why=why, his=his, ph=ph, tb=ph.max(-1) > 8, t2=t2, bm=bm, reg=reg, lab=lab, lab3=lab3, px_b=px_b,
                            cb=cb, half=half, sls=sls, root=root, ty=ty, res=res, bad=int(bad.sum()), bad_in=bad_in))
    tot_h = TOP + sum(pw * d["half"][0] / d["half"][1] for d in prepped) + (len(prepped) - 1) * (GAP + HEAD) + GAP + t_h + MG
    fig = plt.figure(figsize=(FS.FULL_W_IN, tot_h / 25.4)); ycur = TOP
    ROWLAB, FIRST = [], []
    for r_i, d in enumerate(prepped):
        S, sid, why, half, sls, px_b, cb = (d[k] for k in ("S", "sid", "why", "half", "sls", "px_b", "cb"))
        ph_mm = pw * half[0] / half[1]
        y0 = ycur; ycur += ph_mm + GAP + HEAD
        for c_i in range(3):
            ax = FS.axes_mm(fig, LEFT + c_i * (pw + GAP), y0, pw, ph_mm)
            if c_i == 0:
                show(ax, np.clip(d["his"] / max(np.percentile(d["his"][d["his"] > 0], 99.5), 1), 0, 1), px_b / 2, cb, half, cmap="gray",
                     vmin=0, vmax=1)
            elif c_i == 1:
                show(ax, np.clip(d["ph"] / max(np.percentile(d["ph"][d["tb"]], 99.5), 1), 0, 1), px_b, cb, half)
            else:
                show(ax, d["t2"], px_b, cb, half, cmap="gray", vmin=0, vmax=np.nanpercentile(d["t2"][d["bm"]], 99))
            for g, cls, col, lw in GROUPS:
                contour(ax, (d["lab3"] if c_i == 2 else d["lab"])[g], px_b, col, lw)
            contour(ax, d["reg"], px_b, OUTLINE, 0.6)
            if r_i == 0:                                                       # the column titles over the first row's label line
                FS.header(ax, titles[c_i], dx=0, dy=HEAD / pt + TITLE_GAP_PT)
            if c_i == 0:
                nsec = int(sid.split("_")[-1][1:3])
                photo = f"photograph {sls[0]}" if len(sls) == 1 else "photographs " + ", ".join(map(str, sls[:-1])) + f" and {sls[-1]}"
                ROWLAB.append(FS.header(ax, f"{FS.label(S)}, section {nsec}, {photo}", dx=0, dy=2.0))   # the row's label, above it
            if r_i == 0:
                FIRST.append(ax)
            if c_i == 2:                                                       # zoomed: the row's bar on its label's line, top right
                FS.scalebar_title(ax, 1.0, SB_MM, ROWLAB[-1])                   # (world-mm axes: 1 unit = 1 mm)
        if r_i == 0:                                                           # orientation letters once (neurological): row 1,
            FS.dirs_row(FIRST[0], FIRST[-1], "coronal", pad=DPAD, color=FS.FG, y=0.2)
        rows.append(dict(case=FS.label(S), section=sid, why=why, slices=sls, window_mm=[round(2 * half[0], 1), round(2 * half[1], 1)],
                         arm=d["root"].name, section_template_y_mm=[round(v, 2) for v in d["ty"]],
                         inverse_residual_mm=dict(median=float(f"{np.median(d['res']):.2e}"), max=float(f"{d['res'].max():.2e}"),
                                                  pixels_over_0p01=d["bad"], of_them_in_the_section=d["bad_in"])))
    rend = fig.canvas.get_renderer()                                           # the row labels and the letters stay on the page
    assert min(t.get_window_extent(rend).x0 for t in ROWLAB) / fig.dpi * 25.4 >= 0.5, "a row label runs off the page's left edge"
    l_ = [t for t in FIRST[0].texts if t.get_text() == "L"][0].get_window_extent(rend)
    assert not l_.overlaps(ROWLAB[0].get_window_extent(rend)), "the L touches the first row's label"
    r_ = [t for t in FIRST[-1].texts if t.get_text() == "R"][0].get_window_extent(rend)
    assert r_.x1 / fig.dpi * 25.4 <= W - 0.1, "the R runs off the page"
    lax = FS.axes_mm(fig, MG, ycur - HEAD, W - 2 * MG, t_h); lax.axis("off"); lax.set_xlim(0, 100); lax.set_ylim(0, 1)   # the key, a gutter under
    items = [(g[0].upper() + g[1:], col, lw, "-") for g, cls, col, lw in GROUPS] + [("The registered section's outline", OUTLINE, 0.6, "-")]
    rnd = fig.canvas.get_renderer(); inv = lax.transData.inverted()
    x, yrow = 0.0, 0.5
    for g, col, lw, ls in items:                                           # each entry placed after the measured end of the previous label
        t = lax.text(x + 4, yrow, g, va="center", color=FS.FG, fontsize=FS.MIN_PT)
        x1 = inv.transform(t.get_window_extent(rnd).get_points())[1, 0]
        if x1 > 99.5 and yrow == 0.5:                                       # does not fit: wrap the rest to a second row
            t.remove(); x, yrow = 1.0, -0.6; lax.set_ylim(-1.1, 1)
            t = lax.text(x + 4, yrow, g, va="center", color=FS.FG, fontsize=FS.MIN_PT); x1 = inv.transform(t.get_window_extent(rnd).get_points())[1, 0]
        lax.plot([x, x + 3], [yrow, yrow], color=col, lw=max(lw, 0.6), ls=ls); x = x1 + 3.0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    FS.save(fig, a.out, [Path(f) for f in inputs], cmd="$PY_TIRL code/figures/s_tracer_fidelity.py " + " ".join(argv or sys.argv[1:]),
            extra=dict(rows=rows), md5=False)
    plt.close(fig); print(a.out, flush=True)


if __name__ == "__main__":
    main()
