#!/usr/bin/env python3
"""Supplementary S1, synthesis of the withheld slabs: the synthesis network's paper (Gonzales et al. 2026,
ISBI) Fig. 3, re-rendered in this paper's house style with this paper's names.

    python code/figures/s_synthesis_depth.py [--out figures/s_synthesis_depth.pdf]

  (a) SSIM per 1 mm past the simulated junction plane, mean over the brains whose slab reaches that deep (bins with >= 4 brains; n per
      bin above the plot): the synthesis by networks that never saw the brain (six-fold cross-validation, solid) and the same network
      trained on the slab itself (dashed: a reference for how well the photographs can be reproduced), with direct labels
      ($LINC_NUMBERS/r057_s1_synthesis.json isbi.extra.per_mm). The other synthesis methods' curves are not drawn (the lean
      Supplement, 6 Nov 2026).
  (b) the withheld slab of the brain at the median SSIM and of the brain with the lowest (the cross-validation evaluations,
      completion_eval/d3_cv<k>_A_s0.json, and their saved predictions in completion_eval/pred/), never seen by its network: photographs
      and synthesis, sagittal (through the left hemisphere) and coronal (middle of the slab) sections on the brain's section-aligned
      grid, one physical scale; photographed voxels in their own color, synthesized voxels colored with the brain's luminance -> color
      lookup (code/blockface/synthesis/views.py); dashed line: the simulated junction plane; ticks: the other section's plane.
      Both rows on one grid (11 Nov 2026, the author): each cell as large as the largest image of its column, the junction planes in
      one column line, the images centered. The two brains are also what this rule gives, written before looking at the images: of the
      two brains nearest the median SSIM and the three lowest, the pair whose slabs differ least in depth (MM5 6.50 mm, MM7 5.25 mm).
Reads $LINC_NUMBERS/r057_s1_synthesis.json (code/figures/paper_numbers/r057_s1_synthesis.py) and the products of settings/internal/products_completion.yaml
(completion_inputs canon_d3, completion_eval). Writes the PDF and a PNG beside it.
Ported from the analysis project's paper/scripts/figS_synthesis_depth.py; panel (b)'s grid changed 11 Nov 2026.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO / "code"))
import figstyle as FS  # noqa: E402
from shared import paths  # noqa: E402
from blockface.synthesis.data import y_dep_volume  # noqa: E402
from blockface.synthesis.views import Flat, brain_colors, on_black  # noqa: E402

RETAINED_MM = 12.0
H_MM = 52.0
SERIES = [("in_sample", "trained on the slab", FS.FG, (0, (2.2, 1.4)), 0.9),
          ("completion", "synthesis", FS.FG, "-", 1.5)]


def load(f):
    return np.asanyarray(nib.load(f).dataobj).astype(np.float32)


def pick_brains(EV):
    ss, ins = {}, []
    for k in range(6):
        f = EV / f"d3_cv{k}_A_s0.json"; ins.append(str(f))
        for S, v in json.load(open(f))["subjects"].items():
            ss[S] = v["deployed"]["ssim"]
    med = float(np.median(list(ss.values())))
    S_med = min(ss, key=lambda S: (abs(ss[S] - med), ss[S]))
    S_low = min(ss, key=ss.get)
    return (S_med, S_low), dict(median=med, ssim={S: ss[S] for S in (S_med, S_low)}), ins


def slab_views(S, ROOT, EV):
    """the withheld slab of brain S (the cross-validation fold that held it out) on its flat grid: photographs and synthesis"""
    F = Flat(ROOT / S)
    k = int([kk for kk, v in json.load(open(ROOT / "folds.json"))["folds"].items() if S in v][0])
    D = ROOT / S / "fine"
    g_real, rgb = load(f"{D}/bf.nii.gz"), load(f"{D}/bf_rgb.nii.gz")
    brain, real = load(f"{D}/brain.nii.gz") > 0, load(f"{D}/real_mask.nii.gz") > 0
    real &= brain
    q1, q99 = json.load(open(EV / f"d3_cv{k}_A_s0.json"))["subjects"][S]["target_norm"]
    dpp = EV / "pred" / f"d3_cv{k}_A_s0_{S}_deployed.nii.gz"
    g_dep = load(dpp) * (q99 - q1) + q1
    zf, n = F.z_f, F.n
    cut = zf + n
    ydep = y_dep_volume(F.rec, brain.shape)
    ret = real & (np.rint(ydep) >= cut)
    disp, photo, pred = brain_colors(g_real, rgb, ret)
    y0, y1 = zf, cut + int(round(RETAINED_MM / 0.25))
    blk = (slice(None), slice(y0, y1), slice(None))
    fb = {name: F.sample(v.astype(np.float32), blk, order=o) for name, v, o in
          (("brain", brain, 0), ("real", real, 0), ("g_real", g_real, 1), ("g_dep", g_dep, 1))}
    fb["rgb"] = F.sample(rgb, blk, order=1)
    B, Rm = fb["brain"] > 0.5, fb["real"] > 0.5
    yy = np.arange(y0, y1)[None, :, None]
    xs = np.where(B.any(axis=(1, 2)))[0]
    xsag = int(round(xs.min() + 0.3 * (xs.max() - xs.min())))
    ph_m = Rm
    co_m = B & ((yy < cut) | Rm)
    slab3 = np.broadcast_to(yy < cut, B.shape)

    def colored(sl, which):
        g_r, g_d, r_ = fb["g_real"][sl], fb["g_dep"][sl], fb["rgb"][sl]
        if which == "photo":
            return photo(g_r, r_)
        out = photo(g_d, r_)
        m_ = slab3[sl]
        out[m_] = pred(g_d)[m_]
        return out

    to_img = lambda im: np.swapaxes(im, 0, 1)[::-1]                    # noqa: E731  rows = superior -> inferior
    yc = n // 2
    cxs = np.where(B[:, yc, :].any(1))[0]; czs = np.where(B[:, yc, :].any(0))[0]
    zs = np.where((ph_m | co_m)[xsag].any(axis=0))[0]
    z0, z1 = min(zs.min(), czs.min()) - 2, max(zs.max(), czs.max()) + 3
    sl = (xsag, slice(None), slice(z0, z1))
    sag = dict(photo=on_black(to_img(colored(sl, "photo")), to_img(ph_m[sl])),
               comp=on_black(to_img(colored(sl, "comp")), to_img(co_m[sl])), cut_col=cut - y0, cor_col=yc + 0.5)
    x0 = cxs.min() - 2
    sl = (slice(x0, cxs.max() + 3), yc, slice(z0, z1))
    cor = dict(photo=on_black(to_img(colored(sl, "photo")), to_img(ph_m[sl])),
               comp=on_black(to_img(colored(sl, "comp")), to_img(co_m[sl])), sag_col=xsag - x0 + 0.5)
    ins = [str(p) for p in (dpp, D / "bf.nii.gz", D / "bf_rgb.nii.gz", D / "brain.nii.gz", D / "real_mask.nii.gz", ROOT / S / "seam.json")]
    return dict(sag=sag, cor=cor), dict(x_sag_flat=xsag, y_coronal_flat=y0 + yc, z_f=zf, cut=cut, slab_mm=n * 0.25, fold=k), ins


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figures/s_synthesis_depth.pdf")
    a = ap.parse_args()
    P = paths.products("products_completion")
    ROOT, EV = P["completion_inputs"] / "canon_d3", P["completion_eval"]
    FS.setup()
    rp = paths.numbers_root() / "r057_s1_synthesis.json"; pm = json.load(open(rp))["data"]["isbi"]["extra"]["per_mm"]
    lp = REPO / "settings" / "specimen_labels.json"; lab = json.load(open(lp))["labels"]
    BRAINS, pick, ins = pick_brains(EV)
    ins += [str(rp), str(lp)]
    views, meta = {}, dict(pick=pick)
    for S in BRAINS:
        views[S], meta[S], i_ = slab_views(S, ROOT, EV)
        ins += i_
        print(lab[S], S, meta[S], flush=True)
    W = FS.FULL_W_IN * 25.4
    fig = plt.figure(figsize=(FS.FULL_W_IN, H_MM / 25.4))
    # ---- (b): two rows (brains) x [sagittal photo, coronal photo | sagittal synthesis, coronal synthesis], one scale
    # one grid for both rows (voxels of the flat grid, one physical scale): every cell as large as the largest image of its column, the
    # junction planes of the sagittal sections in one column line, the coronal sections centered, each image centered vertically
    top, rgap, bot, lab_w, g_in, g_mid = 6.5, 1.5, 4.0, 5.0, 0.6, 3.0
    hrow = max(views[S]["cor"]["photo"].shape[0] for S in BRAINS)
    cut_max = max(views[S]["sag"]["cut_col"] for S in BRAINS)
    wcol = dict(sag=cut_max + max(views[S]["sag"]["photo"].shape[1] - views[S]["sag"]["cut_col"] for S in BRAINS),
                cor=max(views[S]["cor"]["photo"].shape[1] for S in BRAINS))
    s = (H_MM - top - rgap * (len(BRAINS) - 1) - bot) / (len(BRAINS) * hrow)   # mm / voxel
    right_w = lab_w + 2 * g_in + g_mid + s * 2 * (wcol["sag"] + wcol["cor"])
    xb = W - right_w
    y = top
    last_ax = None
    for r, S in enumerate(BRAINS):
        v = views[S]; rh = hrow * s
        cx = xb + lab_w
        for key in ("photo", "comp"):
            for view in ("sag", "cor"):
                im = v[view][key]; h_, w_ = im.shape[0] * s, im.shape[1] * s
                dx = (cut_max - v["sag"]["cut_col"]) * s if view == "sag" else (wcol["cor"] * s - w_) / 2
                ax = FS.axes_mm(fig, cx + dx, y + (rh - h_) / 2, w_, h_)
                ax.imshow(im, interpolation="antialiased"); ax.axis("off")
                hh, ww = im.shape[:2]
                if view == "sag":
                    c = v["sag"]["cut_col"]
                    ax.plot([c - 0.5, c - 0.5], [0.5, hh - 1.5], color=FS.FG, lw=0.5, ls=(0, (1.6, 1.1)))
                    a_ = v["sag"]["cor_col"]
                    for yy_ in ((-0.5, 3.0), (hh - 4.0, hh - 0.5)):
                        ax.plot([a_, a_], yy_, color=FS.FG, lw=0.8, solid_capstyle="butt")
                else:
                    a_ = v["cor"]["sag_col"]
                    for yy_ in ((-0.5, 3.0), (hh - 4.0, hh - 0.5)):
                        ax.plot([a_, a_], yy_, color=FS.GRAY, lw=0.6, solid_capstyle="butt")
                ax.set_xlim(-0.5, ww - 0.5); ax.set_ylim(hh - 0.5, -0.5)
                dpt = dx * 72 / 25.4                                       # labels stay at the cell's edge, not the image's
                if view == "sag" and key == "photo":
                    ax.annotate(lab[S], xy=(0, 0.5), xycoords="axes fraction", xytext=(-6 - dpt, 0), textcoords="offset points",
                                rotation=90, ha="center", va="center", fontsize=FS.MIN_PT, fontweight="bold", color=FS.FG,
                                annotation_clip=False)
                if r == 0 and view == "sag":
                    FS.header(ax, "Photographs" if key == "photo" else "Synthesis", dx=-dpt, fontweight="bold")
                if r == 0 and view == "sag" and key == "photo":
                    FS.letter(ax, "b", dx=-14 - dpt, dy=3)
                last_ax = (ax, ww)
                cx += wcol[view] * s + g_in
            cx += g_mid - g_in
        y += rh + rgap
    ax, ww = last_ax                                                      # 5 mm scale bar under the last image
    FS.scalebar(ax, 1.0 / 0.25, 5.0, y_frac=0.08, x0_px=ww - 0.5 - 5.0 / 0.25)
    # ---- (a) SSIM per mm past the junction plane, direct labels
    bins = sorted(pm, key=lambda k_: float(k_.split("-")[0]))
    keep = [b for b in bins if pm[b]["completion"]["n"] >= 4]
    xs = np.array([float(b.split("-")[0]) + 0.5 for b in keep])
    left_w = xb - 14.0
    axa = FS.axes_mm(fig, 9.0, 7.5, left_w - 22.0, H_MM - 7.5 - 9.0)
    ends = []
    for key, name, col, ls, lw in SERIES:
        ys = np.array([pm[b][key]["mean"] if pm[b].get(key, {}).get("n") else np.nan for b in keep])
        axa.plot(xs, ys, color=col, ls=ls, lw=lw, marker="o", ms=1.9, mew=0, zorder=3)
        ends.append([ys[-1], name, col, ls, lw])
    ends.sort(key=lambda e: -e[0])
    for i in range(1, len(ends)):
        ends[i][0] = min(ends[i][0], ends[i - 1][0] - 0.075)
    for yv, name, col, ls, lw in ends:                                 # a sample of the line's own style keys each label
        axa.plot([xs[-1] + 0.25, xs[-1] + 0.85], [yv, yv], color=col, ls=ls, lw=lw, clip_on=False, solid_capstyle="butt")
        axa.annotate(name, xy=(xs[-1] + 0.85, yv), xytext=(3, 0), textcoords="offset points", fontsize=FS.MIN_PT, color=col,
                     ha="left", va="center", annotation_clip=False)
    for x_, b in zip(xs, keep):
        axa.annotate(str(pm[b]["completion"]["n"]), xy=(x_, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 2),
                     textcoords="offset points", fontsize=FS.MIN_PT, color=FS.GRAY, ha="center", va="bottom",
                     annotation_clip=False)
    axa.annotate("n", xy=(0, 1.0), xycoords="axes fraction", xytext=(-4, 2), textcoords="offset points", fontsize=FS.MIN_PT,
                 color=FS.GRAY, ha="right", va="bottom", style="italic", annotation_clip=False)
    axa.set_xlim(0, xs[-1] + 0.6); axa.set_ylim(0, 0.9)
    axa.set_xticks(range(0, int(xs[-1] + 1.5))); axa.set_yticks(np.arange(0, 0.91, 0.2))
    axa.set_ylabel("SSIM", labelpad=1.5); axa.set_xlabel("mm past the junction plane", labelpad=1.0)
    for sp in ("top", "right"):
        axa.spines[sp].set_visible(False)
    FS.letter(axa, "a", dx=-22, dy=10)
    meta["labels"] = {S: lab[S] for S in BRAINS}
    cmd = " ".join(["code/figures/s_synthesis_depth.py"] + sys.argv[1:])
    FS.save(fig, a.out, ins, cmd, extra=dict(meta={k: v for k, v in meta.items() if k != "pick"},
                                            pick={lab[S]: pick["ssim"][S] for S in BRAINS}, median_ssim=pick["median"], scale_bar_mm=5))
    fig.savefig(str(Path(a.out).with_suffix(".png")), dpi=FS.DPI, facecolor=fig.get_facecolor())
    print("->", a.out, f"{W:.0f} x {H_MM:.1f} mm")


if __name__ == "__main__":
    main()
