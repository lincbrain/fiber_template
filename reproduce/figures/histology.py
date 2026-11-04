#!/usr/bin/env python3
"""The paper's figure "Histology registered to the blockface" (TIRL 3.6.2), with the marker tubes hidden from the registration.

    python reproduce/figures/histology.py --jp2-root <LINC source data>/sourcedata [--out figures/histology.png]

Needs glymur (the raw histology is JPEG 2000). --jp2-root is the LINC source-data tree that holds sub-<ID>/micr/<section>.jp2.
  a  one section registered to its blockface slice: the MM3 piece (with a detected tube hole) whose nonlinear tissue Dice is closest
     to that brain's median (numbers_of_record/registration_qc.json, family histo2bf): the blockface slice with the histology's tissue
     outline (white), and the registered darkfield histology (contrast-stretched). Flipped left-right into the neurological convention
     (the photographs show the subject's left on the image right); dorsal up; loose blockface fragments without histology removed.
  b  marker tubes at near-native resolution (JP2 level 2, ~1.8 um/px, 3 mm windows), three fixed quantiles (25th, 50th, 90th) of the
     pooled offsets: for each, the found piece whose offset is closest to that quantile; + = the hole-only blockface tube center carried
     into the histology through that piece's tubes-hidden TIRL chain, o = the hole found in the histology (its recorded area; both
     white, black halo). The offset is in the header.
  c  ECDF of the tube offset of the six brains, tubes hidden from the registration (every piece in which the hole was found; blockface
     tube center from the hole only); the median as a thin vertical line, checked against $LINC_NUMBERS/tube_offsets_r054.json
     (variant hole_hidden; reproduce/numbers/tube_offsets_r054.py).
  d  per-brain medians (the same record).
Reads (configs/products_mri.yaml): tube_landmarks/{preinp,r054hole}_<ID>_native.json, histo_arm_visible (panel a's TIRL piece),
numbers_of_record/registration_qc.json, $LINC_NUMBERS/tube_offsets_r054.json, the JP2s. Writes the PNG and its provenance.json.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
import figstyle as S                                           # noqa: E402
from linc import paths                                         # noqa: E402

Image.MAX_IMAGE_PIXELS = None
MAIN = ["MR243", "MR252", "MR256", "MF278", "MF275", "MF191"]  # the six histology brains (LINC IDs)
A_SUBJ = "MR256"                                               # panel (a)'s brain
PX_MM, FIX_MM = 0.4397e-3, 0.034                               # JP2 level-0 pixel (mm); TIRL fixed/moving grid (mm)
JP2_LEVEL = 4                                                  # (b): read every 4th level-0 pixel (JP2 level 2)
JP2 = "sub-{S}/micr/{sid}.jp2"                                 # under --jp2-root
TIRL = "sub-{S}/blockface/06_histo2bf/tirl/{piece}"            # under histo_arm_visible
TUBE_VARIANT = "hole_hidden"                                   # the paper's tube set
QUANT = (25, 50, 90)
WIN_MM = 3.0                                                   # (b): window side
ECDF_XMAX = 1.0                                                # (c): to the 1 mm search radius (the curve reaches 1)
SCALEBARS_MM = (5, 0.5)                                        # (a), (b)
LEFT, RIGHT, GAP, TOP = 7.0, 7.0, 3.0, 6.0                     # mm


def stretch(rgb, m, top=99.5):
    lo, hi = np.percentile(rgb[m], [1, top], axis=0)
    return np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1)


def largest(m):
    lab, n = ndi.label(m)
    return lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1) if n else m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jp2-root", required=True, help="LINC source-data tree holding sub-<ID>/micr/<section>.jp2")
    ap.add_argument("--out", default="figures/histology.png")
    a = ap.parse_args()
    PR = paths.products("products_mri")
    NUM, TUBE = PR["numbers_of_record"], PR["tube_landmarks"]
    TJP = paths.numbers_root() / "tube_offsets_r054.json"
    pieces = lambda tag, S_: {r["piece"]: r for r in json.loads((TUBE / f"{tag}_{S_}_native.json").read_text())["pieces"]}  # noqa: E731
    ins = []
    # ---- a: representative piece ----
    rq = [r for r in json.loads((NUM / "registration_qc.json").read_text()) if r.get("family") == "histo2bf" and r["subject"] == A_SUBJ]
    med = float(np.median([r["metrics"]["dice_nonlinear"] for r in rq]))
    tp = pieces("preinp", A_SUBJ)
    ra = min((r for r in rq if tp.get(r["tag"], {}).get("found")), key=lambda r: abs(r["metrics"]["dice_nonlinear"] - med))
    P = PR["histo_arm_visible"] / TIRL.format(S=A_SUBJ, piece=ra["tag"])
    fx = np.asarray(Image.open(P / "fixed.png"), np.float32) / 255
    rgb = np.asarray(Image.open(P / "moving4_nonlinear_rgb.png").convert("RGB"), np.float32)
    mx = rgb.max(-1)
    tis = largest(ndi.binary_fill_holes(ndi.binary_opening((mx > 6) & (mx < 250), iterations=2)))
    hs = stretch(rgb, tis, top=99.8) * tis[..., None]                   # 99.8: bright dorsal spots not clipped to saturation
    fxd = fx * ((fx > 0.02) & ndi.binary_dilation(tis, iterations=12))
    ys, xs = np.where((fxd > 0) | tis); pad = 25
    cr = (slice(max(ys.min() - pad, 0), ys.max() + pad), slice(max(xs.min() - pad, 0), xs.max() + pad))
    flip = lambda a_: a_[:, ::-1]                                       # noqa: E731
    soft = lambda m: np.clip(ndi.gaussian_filter(m.astype(np.float32), 1.0), 0, 1)   # noqa: E731  antialiased tissue edge (~1 px)
    fxd = fx * soft(fxd > 0); hs = hs * soft(tis)[..., None]
    pa = [flip(fxd[cr]), flip(hs[cr])]; outl = flip(tis[cr])
    ins += [P / "fixed.png", P / "moving4_nonlinear_rgb.png", NUM / "registration_qc.json"]
    # ---- c: the six-brain distribution, tubes hidden, hole-only centers ----
    TJ = json.loads(TJP.read_text())["data"]
    e_inp, cand = [], []
    for S_ in MAIN:
        for r_ in pieces("r054hole", S_).values():
            if r_.get("bf_tube") and r_["hole"]["found"]:
                e_inp.append(r_["hole"]["error_mm"]); cand.append((S_, dict(r_["hole"], sid=r_["sid"], piece=r_["piece"])))
        ins.append(TUBE / f"r054hole_{S_}_native.json")
    m_inp = TJ["pooled"][TUBE_VARIANT]["median_mm"]
    assert len(e_inp) == TJ["pooled"][TUBE_VARIANT]["n"] and round(float(np.median(e_inp)), 3) == m_inp, "distribution != JSON"
    # ---- b: fixed quantiles of the offsets ----
    qv = np.percentile(e_inp, QUANT)
    ex = [min(cand, key=lambda c: abs(c[1]["error_mm"] - q)) for q in qv]
    # ---- figure ----
    import glymur
    S.setup()
    W = S.FULL_W_IN * 25.4
    h_px, w_px = pa[0].shape[:2]
    pw = (W - LEFT - RIGHT - 2 * GAP) / 3                         # (a)/(b) fill the page width (true scale in the provenance)
    scale_a = pw / (w_px * FIX_MM); ph = h_px * FIX_MM * scale_a
    bw = bh = pw                                                  # square windows, same width as (a)'s panels
    CH = 30.0
    H = TOP + ph + 16.0 + bh + 12.0 + CH + 12.0
    fig = plt.figure(figsize=(S.FULL_W_IN, H / 25.4))
    x0 = (W - (3 * pw + 2 * GAP)) / 2
    xa = (W - (2 * pw + GAP)) / 2                                 # (a): two panels, centered
    heads = ["blockface photograph, registered section outlined", "registered histology"]
    halo = [pe.withStroke(linewidth=2.2, foreground="black")]
    for k, img in enumerate(pa):
        ax = S.axes_mm(fig, xa + k * (pw + GAP), TOP, pw, ph)
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        if k == 0:
            S.outline(ax, outl, lw=0.6)
        ax.set_xticks([]); ax.set_yticks([]); [s_.set_visible(False) for s_ in ax.spines.values()]
        S.header(ax, heads[k], dx=0, dy=2)
        if k == 0:
            S.letter(ax, "a", dx=-16, dy=2); S.scalebar(ax, 1 / FIX_MM, SCALEBARS_MM[0], y_frac=0.05)
            ax.annotate("L", xy=(0, 0.5), xycoords="axes fraction", xytext=(-2, 0), textcoords="offset points", ha="right",
                        va="center", color=S.GRAY, annotation_clip=False)
        if k == len(pa) - 1:
            ax.annotate("R", xy=(1, 0.5), xycoords="axes fraction", xytext=(2, 0), textcoords="offset points", ha="left",
                        va="center", color=S.GRAY, annotation_clip=False)
    half = int(WIN_MM / 2 / PX_MM); yb = TOP + ph + 16.0; f = JP2_LEVEL
    for k, ((S_, r), q, qq) in enumerate(zip(ex, qv, QUANT)):
        jp = Path(a.jp2_root) / JP2.format(S=S_, sid=r["sid"])
        j2k = glymur.Jp2k(str(jp)); ins.append(jp)
        y0, x0_ = r["bf_tube_l0"]; hy, hx = r["hole_l0"]
        r0, c0 = int(max(0, y0 - half)), int(max(0, x0_ - half))
        w = j2k[r0:int(y0 + half):f, c0:int(x0_ + half):f][:, :, :3].astype(np.float32)
        m = w.max(-1) > 0
        ax = S.axes_mm(fig, x0 + k * (pw + GAP), yb, bw, bh)
        ax.imshow((stretch(w, m) if m.any() else w / 255)[:, ::-1])          # same left-right flip as (a)
        wd = w.shape[1]
        ax.plot(wd - 1 - (x0_ - c0) / f, (y0 - r0) / f, "+", color=S.FG, ms=10, mew=1.4, path_effects=halo)
        rad = np.sqrt(max(r.get("hole_area_mm2", 0.0), 1e-6) / np.pi) / (PX_MM * f)     # the detected hole: its recorded area
        ax.add_patch(Circle((wd - 1 - (hx - c0) / f, (hy - r0) / f), rad, fill=False, ec=S.FG, lw=1.1, path_effects=halo))
        ax.set_xticks([]); ax.set_yticks([]); [s_.set_visible(False) for s_ in ax.spines.values()]
        S.header(ax, f"{S.label(S_)}, {qq}th percentile: {r['error_mm']:.2f} mm", dx=0, dy=2)
        if k == 0:
            S.letter(ax, "b", dx=-16, dy=12); S.scalebar(ax, 1 / (PX_MM * f), SCALEBARS_MM[1], y_frac=0.04)
            S.header(ax, "tubes hidden from the registration", dx=0, dy=12)
    yc = yb + bh + 12.0; wc = (3 * pw + 2 * GAP) * 0.54                  # (c) wide enough for (d)'s horizontal brain labels
    ax = S.axes_mm(fig, x0 + 12.0, yc, wc - 12.0, CH)
    s = np.sort(e_inp); ax.step(s, np.arange(1, len(s) + 1) / len(s), where="post", color=S.FG, lw=1.1, ls="-", label="tubes hidden")
    ax.axvline(np.median(e_inp), color=S.FG, lw=0.6, ls=":")
    ax.annotate(f"{np.median(e_inp):.2f}", xy=(np.median(e_inp), 1), xycoords=("data", "axes fraction"), xytext=(0, 2),
                textcoords="offset points", ha="center", va="bottom", color=S.FG, annotation_clip=False)
    ax.set_xlabel("tube offset (mm)"); ax.set_ylabel("fraction of components")
    ax.set_xlim(0, ECDF_XMAX); ax.set_ylim(0, 1.02); ax.spines[["top", "right"]].set_visible(False)
    S.letter(ax, "c", dx=-16 - 12.0 / 25.4 * 72, dy=2)                 # aligned with a / b
    # (d) per-brain medians
    ps = TJ["per_subject"]
    axd = S.axes_mm(fig, x0 + wc + 16.0, yc, (3 * pw + 2 * GAP) - wc - 16.0, CH)
    xs_ = np.arange(len(MAIN))
    hid = [ps[S_][TUBE_VARIANT]["median_mm"] for S_ in MAIN]
    axd.plot(xs_, hid, "o", color=S.FG, ms=3.8, mec="none", zorder=3)
    axd.set_xticks(xs_); axd.set_xticklabels([S.label(S_) for S_ in MAIN])
    axd.set_ylabel("median offset (mm)"); axd.set_ylim(0, 1.15 * max(hid)); axd.set_xlim(-0.6, len(MAIN) - 0.4)
    axd.spines[["top", "right"]].set_visible(False)
    S.letter(axd, "d", dx=-28, dy=2)
    S.save(fig, str(a.out), ins + [TJP], cmd=" ".join(["python"] + sys.argv),
           extra=dict(panel_a_piece=ra["tag"], panel_a_dice=ra["metrics"]["dice_nonlinear"], panel_a_subject_median_dice=med,
                      panel_b=[dict(subject=S_, piece=r["piece"], quantile=qq, quantile_mm=round(float(q), 3), offset_mm=r["error_mm"])
                               for (S_, r), q, qq in zip(ex, qv, QUANT)],
                      per_brain_median_mm={S_: ps[S_][TUBE_VARIANT]["median_mm"] for S_ in MAIN},
                      median_tubes_hidden_mm=m_inp, n_pieces=len(e_inp), scale_a=round(scale_a, 3), scalebars_mm=list(SCALEBARS_MM)))
    plt.close(fig)
    print("->", a.out, "a:", ra["tag"], "b:", [(S_, r["piece"], r["error_mm"]) for S_, r in ex])


if __name__ == "__main__":
    main()
