#!/usr/bin/env python3
"""The paper's supplementary figure "From the chart to the template, one section per case": does the chain keep the tracer labels
where the anatomist charted them?

You need
  the LINC histology dataset (--linc-root, default $LINC_RAW/linc_000003: rawdata/sub-<ID>/micr/, the darkfield stacks and the
  annotators' masks/), and the products of record under $LINC_PRODUCTS (settings/internal/products_figures.yaml): the pushed tracer sheets
  (push_sheets), the tracer labels in the template (tracer_template), the section maps (section_maps), the template's T2 and brain
  mask, the histology registration each brain's labels went through (settings/internal/products_blockface.yaml labels_histology_arm), the
  blockface volumes and the transforms of record (MM1-MM3: blockface -> b0 -> template; MF8: through the blockface template).
  settings/env/tirl.yml ($PY_TIRL), the environment the paper's figure was made in.

What it does
  One row per case, at its injection level (the section with the most injection-site pixels; MM3, without an annotated site, the
  section with the most label), three panels at one scale, each centered on its own tissue and drawn in the anatomical convention
  (the subject's left on the left, dorsal up; no case mirrored):
    (i)   the chart (level 6 of the annotators' masks) on its darkfield section, the slide image flipped left-right; dashed gray: the
          chart's own outline; dotted: the pieces that were registered;
    (ii)  the blockface photograph of the slice the section was matched to, with the section's tissue outline after its full
          registration (cyan) and the labels transformed with it; MF8's two frontal lobes were matched to adjacent photographs
          (0.75 mm apart) and are drawn together;
    (iii) the labels of that section after the transform into the MRI template (each template voxel pulled through the chain of
          record into the section's sheet), projected along y onto the template's T2 at the section's median y.
  Classes: bundles (dense and moderate, and MM2's single bundle class), light bundles, terminal fields (all three density classes),
  injection site; single-fiber strokes are not drawn.

You get
  --out (default figures/s_tracer_fidelity.png) and its provenance.json (the rows: case, section, why, blockface slices, template y,
  window, registration folder).

    $PY_TIRL code/figures/s_tracer_fidelity.py [--linc-root <LINC dataset 000003>] [--out figures/s_tracer_fidelity.png]

Ported from the analysis project's paper/fig_tracer_fidelity_v2_r058.py (--set r058d --layout s3) and the helpers it imported from
paper/fig_tracer_fidelity_r058.py.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "code"))
import figstyle as FS                                           # noqa: E402
from shared import paths, xfm                                     # noqa: E402
from histology import sections as io                       # noqa: E402

Image.MAX_IMAGE_PIXELS = None
CASES = ["MR243", "MR252", "MR256", "MF191"]
TV = 0.25
GROUPS = [("bundles", ("Fiber_dense_bundle", "Fiber_moderate_bundle", "Bundle"), "#D55E00", 0.9),
          ("light bundles", ("Fiber_light_bundle",), "#E69F00", 0.5),
          ("terminal fields", ("Terminal_dense", "Terminal_moderate", "Terminal_light"), "#56B4E9", 0.5),
          ("injection site", ("Injection_site",), "#ffffff", 0.9)]


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
    ap.add_argument("--linc-root", default=None, help="the LINC histology dataset 000003 (default $LINC_RAW/linc_000003)")
    ap.add_argument("--out", default="figures/s_tracer_fidelity.png")
    a = ap.parse_args(argv)
    P = paths.products("products_figures")
    arms = paths.products("products_blockface")["labels_histology_arm"]
    linc = Path(a.linc_root) if a.linc_root else paths.raw_root() / "linc_000003"
    FS.setup()
    import matplotlib.pyplot as plt
    ref = nib.load(str(P["template_t2"])); At = ref.affine; T2 = load(P["template_t2"]).astype(float)
    brain = load(P["template_mask"]) > 0.5
    rows_out, cache, inputs = [], {}, []
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
        cache[S] = (car, carS, files, classes, A, pj, sec_sl, yinfo)
        rows_out.append((S, pick, why))
    W = FS.FULL_W_IN * 25.4; LEFT, GAP, TOP = 27.0, 3.0, 9.0
    pw = (W - LEFT - 2 * GAP - 2.0) / 3
    disp = lambda v: np.flipud(np.swapaxes(v, 0, 1))  # noqa: E731
    rows, prepped = [], []
    for S, sid, why in rows_out:
        car, carS, files, classes, A, pj, sec_sl, yinfo = cache[S]
        root = paths.products_root() / arms[S]                                 # the histology registration the labels went through
        bf_root = root / f"sub-{S}" / "blockface"
        zp = io.image_stacks(str(linc / "rawdata" / f"sub-{S}" / "micr"), S)[0]
        srows = {r["section"]: r for r in io.load_map(str(P["section_maps"] / f"{S}_{io.stain_of(zp)}.tsv"))}
        mdir = linc / "rawdata" / f"sub-{S}" / "micr" / "masks"
        pieces = sorted([p for p in pj["pieces"] if p["section"] == sid], key=lambda p: -sum(v["px_L2"] for v in p["classes"].values()))
        j = pieces[0]["bf_idx"]; sls = sorted(sec_sl[sid]); r = srows[sid]
        # (i) darkfield, level 6
        img = io.Section(zp, r["z"], r["h0"], r["w0"]).level(6)[:, ::-1]
        px_i = 0.4397 * 64 / 1000
        ms = {}
        for g, cls, col, lw in GROUPS:
            m = np.zeros(img.shape[:2], bool)
            for c in cls:
                p_ = mdir / f"{c}.ome.zarr"
                if p_.exists():
                    x = io.Section(str(p_), r["z"], r["h0"], r["w0"]).level(6); m |= ((x[..., 0] if x.ndim == 3 else x) > 0)[:, ::-1]
            ms[g] = m
        ol = io.Section(str(mdir / "Outline.ome.zarr"), r["z"], r["h0"], r["w0"]).level(6); ol = ((ol[..., 0] if ol.ndim == 3 else ol) > 0)[:, ::-1]
        pm = np.zeros(img.shape[:2], bool)
        for pp in sorted(glob(str(bf_root / "_work" / "histo_masks" / (sid + "_*.png")))):
            x = np.asarray(Image.open(pp)) > 0; h, w_ = min(x.shape[0], pm.shape[0]), min(x.shape[1], pm.shape[1]); pm[:h, :w_] |= x[:h, :w_]
        pm = pm[:, ::-1]
        ti = img.max(-1) > 3
        # (ii) photograph + the section's registered tissue outline (full registration) + labels
        rgb = load(P["data"] / f"sub-{S}" / "blockface" / f"{S}_blockface_L2.nii.gz")
        px_b = float(abs(A[0, 0]))
        ph = disp(rgb[:, j, :, :]).astype(float); tb = ph.max(-1) > 8
        reg = np.zeros(ph.shape[:2], bool)
        for p in pieces:
            H_ = np.asarray(Image.open(bf_root / "06_histo2bf" / "tirl" / p["piece"] / "moving4_nonlinear.png")); H_ = H_.max(-1) if H_.ndim == 3 else H_
            fm = ((H_ > 0) & (H_ < 254.5))[::-1, ::-1]                          # slide -> store orientation (level 0)
            K = 2                                                                  # store level 0 (the TIRL canvas) -> level 1 = the L2 grid
            hh, ww = fm.shape[0] // K * K, fm.shape[1] // K * K
            fl = fm[:hh, :ww].reshape(hh // K, K, ww // K, K).max((1, 3))
            reg_full = np.zeros(ph.shape[:2], bool)
            rr, cc = min(fl.shape[0], ph.shape[0]), min(fl.shape[1], ph.shape[1])
            reg_full[:rr, :cc] = fl[:rr, :cc]
            reg |= np.flipud(reg_full)                                             # store rows -> display (dorsal up)
        # (iii) template
        jy = int(round(yinfo[sid])); sl3 = disp(T2[:, jy, :]); bm = disp(brain[:, jy, :])
        ci, ei = tissue_extent_mm(ti, px_i); cb, eb = tissue_extent_mm(tb, px_b); ct, et = tissue_extent_mm(bm, TV)
        half = (float(max(ei[0], eb[0], et[0])) / 2 + 1.0, float(max(ei[1], eb[1], et[1])) / 2 + 1.0)   # (rows, cols) half extents, mm
        prepped.append(dict(S=S, sid=sid, why=why, img=img, px_i=px_i, ms=ms, ol=ol, pm=pm, ti=ti, ph=ph, tb=tb, px_b=px_b, reg=reg, sl3=sl3, bm=bm,
                            ci=ci, cb=cb, ct=ct, half=half, sls=sls, classes=classes, files=files, carS=carS, yinfo=yinfo, root=root))
    tot_h = TOP + sum(pw * d["half"][0] / d["half"][1] + 9.0 for d in prepped) + 14
    fig = plt.figure(figsize=(FS.FULL_W_IN, tot_h / 25.4)); ycur = TOP
    for r_i, d in enumerate(prepped):
        S, sid, why, half, sls, classes, files, carS, yinfo = (d[k] for k in ("S", "sid", "why", "half", "sls", "classes", "files", "carS", "yinfo"))
        ph_mm = pw * half[0] / half[1]
        y0 = ycur; ycur += ph_mm + 9.0
        ax = FS.axes_mm(fig, LEFT, y0, pw, ph_mm)
        show(ax, np.clip(d["img"] / max(np.percentile(d["img"][d["ti"]], 99.5), 1), 0, 1), d["px_i"], d["ci"], half)
        contour(ax, d["ol"], d["px_i"], FS.GRAY, 0.4, "--"); contour(ax, d["pm"], d["px_i"], FS.FG, 0.3, ":")
        for g, cls, col, lw in GROUPS:
            contour(ax, d["ms"][g], d["px_i"], col, lw)
        FS.scalebar(ax, 1.0, 5)
        ax.annotate(f"{FS.label(S)} {sid.split('_')[-1][:3]}\n{why}\nslice{'s' if len(sls) > 1 else ''} {', '.join(map(str, sls))}"
                    f"\ny = {world_y(At, yinfo[sid]):.1f} mm", xy=(0, 0.5), xycoords="axes fraction", xytext=(-3, 0), textcoords="offset points",
                    ha="right", va="center", color=FS.FG, fontsize=FS.MIN_PT, annotation_clip=False)
        if r_i == 0:
            FS.header(ax, "(i) chart on its darkfield section", dx=0, dy=2)
        ax = FS.axes_mm(fig, LEFT + pw + GAP, y0, pw, ph_mm)
        show(ax, np.clip(d["ph"] / max(np.percentile(d["ph"][d["tb"]], 99.5), 1), 0, 1), d["px_b"], d["cb"], half)
        contour(ax, d["reg"], d["px_b"], "#00e5ff", 0.6)
        for g, cls, col, lw in GROUPS:
            m = np.zeros(d["ph"].shape[:2], bool)
            for c in cls:
                if c in classes:
                    for jj in sls:
                        m |= disp(load(files[classes.index(c)])[:, jj, :] > 0)
            contour(ax, m, d["px_b"], col, lw)
        FS.scalebar(ax, 1.0, 5)
        if r_i == 0:
            FS.header(ax, "(ii) blockface photograph, section registered", dx=0, dy=2)
        ax = FS.axes_mm(fig, LEFT + 2 * (pw + GAP), y0, pw, ph_mm)
        show(ax, d["sl3"], TV, d["ct"], half, cmap="gray", vmin=0, vmax=np.percentile(d["sl3"][d["bm"]], 99))
        for g, cls, col, lw in GROUPS:
            m = np.zeros(T2.shape[::2], bool)
            for c in cls:
                v = carS[sid].get(c)
                if v is not None and len(v):
                    m[v[:, 0], v[:, 2]] = True
            contour(ax, disp(m), TV, col, lw)
        FS.scalebar(ax, 1.0, 5)
        if r_i == 0:
            FS.header(ax, "(iii) template T2, after the carry", dx=0, dy=2)
        rows.append(dict(case=FS.label(S), section=sid, why=why, slices=sls, template_y_mm=round(world_y(At, yinfo[sid]), 2),
                         window_mm=[round(2 * half[0], 1), round(2 * half[1], 1)], arm=d["root"].name))
    lax = FS.axes_mm(fig, 4, ycur + 1, W - 8, 8); lax.axis("off"); lax.set_xlim(0, 100); lax.set_ylim(0, 1)
    items = [(g, col, lw, "-") for g, cls, col, lw in GROUPS] + [("registered section", "#00e5ff", 0.6, "-"), ("chart outline", FS.GRAY, 0.4, "--"),
                                                                    ("registered pieces", FS.FG, 0.3, ":")]
    rnd = fig.canvas.get_renderer(); inv = lax.transData.inverted()
    x, yrow = 1.0, 0.5
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
