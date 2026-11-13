#!/usr/bin/env python3
"""The paper's figure "Between-brain deviation after construction" (Results "Between-brain variability").

(a) The mean absolute deviation of the four brains from their mean, for three variants (T2 only, SANDI only, T2 + SANDI + tensor), in
    T2 (z-units), the soma fraction (percentage points; the neurite fraction gives the same value) and the
    principal direction (degrees), on the coronal section through the anterior commissure, in the voxels covered by all inputs (the
    template's T2 core); brain voxels outside the core dark gray; one color scale per measure; white outline = the template brain;
    L and R beside every row (figstyle.orient_letters: 1 mm from the brain, at the middle of the row); one 5 mm scale bar, its length
    above it, top right at the column titles' level (figstyle.scalebar_title, the standard place).
(b) The core mean of every one of the seven variants; open = in sample (the variant was built with that channel), filled = out of
    sample; rings = the three variants shown in (a).
Reads (settings/internal/products_mri.yaml): the deviation maps mri_runs/_variability/<arm>_four__<T2|FRAC1|DTI>.nii.gz (written by
code/figures/paper_numbers/variability_four_brains_r061.py), its summary numbers_of_record/variability_four_brains_r061.json, the anterior commissure of
numbers_of_record/acpc.json, and template_mri/linc_mri_{T2,core_t2,brainmask}.nii.gz.
    python code/figures/variability.py [--out figures/variability.png]"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.transforms as mtrans

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "code"))
import figstyle as S                                           # noqa: E402
from shared import paths                                         # noqa: E402

ARMS = ["rb_t2_four", "rb_micro_four", "rb_dti_four", "rb_t2micro_four", "rb_t2dti_four", "rb_dtimicro_four", "rb_final_four"]     # the seven variants (Table 2 order)
LABEL = {"rb_t2_four": "T2", "rb_micro_four": "SANDI", "rb_dti_four": "Tensor", "rb_t2micro_four": "T2 + SANDI", "rb_t2dti_four": "T2 + tensor",
         "rb_dtimicro_four": "Tensor + SANDI", "rb_final_four": "T2 + SANDI + tensor"}
SHOW = ("rb_t2_four", "rb_micro_four", "rb_final_four")         # the three variants in (a)
FIG_METRICS = [("T2", "T2 |Δz|"), ("FRAC1", "soma = neurite |Δ| (pp)"), ("DTI", "V1 angle (°)")]
XUNIT = {"T2": "z-units", "FRAC1": "Percentage points", "DTI": "Degrees"}
CHAN = {"T2": "T2", "FRAC1": "SANDI", "DTI": "tensor"}         # a measure is in sample for variants built with that channel
HEADS = {"T2": "T2 (z-units)", "FRAC1": "Soma fraction (percentage points)", "DTI": "Angular deviation (°)"}
SUMMARY = "variability_four_brains_r061"                        # the seven-variant summary record
MAPS = "_variability"                                           # folder of the deviation maps under mri_runs
SCALEBAR_MM, PAD_PX = 5, 3


def load(p):
    return np.asanyarray(nib.load(str(p)).dataobj).astype(np.float64)


def in_sample(arm, k):
    lab = LABEL[arm]
    return lab == "T2 + SANDI + tensor" or CHAN[k].lower() in lab.lower()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figures/variability.png")
    a = ap.parse_args()
    P = paths.products("products_mri")
    TPL, OUTD, NUM = P["template_mri"], P["mri_runs"] / MAPS, P["numbers_of_record"]
    ref = nib.load(str(TPL / "linc_mri_T2.nii.gz"))
    core = load(TPL / "linc_mri_core_t2.nii.gz") > 0
    ac = json.loads((NUM / "acpc.json").read_text())["data"]["definitive"]["AC"]
    yac = int(round((np.linalg.inv(ref.affine) @ np.append(ac, 1))[1]))
    summ = json.loads((NUM / f"{SUMMARY}.json").read_text())["data"]["arms"]
    maps = {arm: {k: load(OUTD / f"{arm}__{k}.nii.gz") for k, _ in FIG_METRICS} for arm in SHOW}

    S.setup()
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    show, arms = list(SHOW), ARMS
    brain = load(TPL / "linc_mri_brainmask.nii.gz") > 0.5
    sect = lambda v: S.section(v, "coronal", yac)                                   # noqa: E731
    cs, bs = sect(core) > 0.5, sect(brain.astype(float)) > 0.5
    ij = np.argwhere(bs)
    r0, r1, c0, c1 = ij[:, 0].min() - PAD_PX, ij[:, 0].max() + PAD_PX + 1, ij[:, 1].min() - PAD_PX, ij[:, 1].max() + PAD_PX + 1
    vx = abs(ref.affine[0, 0]); tw, th = (c1 - c0) * vx, (r1 - r0) * vx
    nm = len(FIG_METRICS); W = S.FULL_W_IN * 25.4
    # the space (the author, 12 Nov 2026: "optimize space"): one gap between the columns and between the rows, nothing between the
    # panels needing more; the left margin as wide as the row labels or (b)'s variant names need; (b) one gap below the color
    # bars' tick labels; the page margin below
    GAP, B_IN = 1.5, 2.0                                           # B_IN: (b)'s plots inset in their column (mm)
    RIGHT = S.MARGIN_MM + S.orient_room("R")                       # the R beside every row
    f_ = plt.figure(); r_ = f_.canvas.get_renderer(); tw_ = lambda t: f_.text(0, 0, t, fontsize=S.MIN_PT).get_window_extent(r_).width / f_.dpi * 25.4  # noqa: E731
    line_h = f_.text(0, 0, f"{SCALEBAR_MM} mm", fontsize=S.MIN_PT).get_window_extent(r_).height / f_.dpi * 25.4
    TOP = max(6.0, S.MARGIN_MM + (2.0 + 1.2 + 2.0) / 72 * 25.4 + line_h)   # the bar on the titles' baseline, its length above it
    need_rows = S.MARGIN_MM + max(tw_(l_) for l_ in ("T2", "SANDI", "T2 + SANDI", "+ tensor")) + 16 / 72 * 25.4
    need_b = S.MARGIN_MM + max(tw_(LABEL[a_]) for a_ in ARMS) + (2.5 + 2.0) / 72 * 25.4 - B_IN     # tick length + pad
    plt.close(f_)
    LEFT = max(need_rows, need_b)
    scale = (W - LEFT - RIGHT - (nm - 1) * GAP) / (nm * tw)
    pw, ph = tw * scale, th * scale
    CBG, CBH, BH = 4.0, 1.6, 27.0
    ycb = TOP + len(show) * ph + (len(show) - 1) * GAP + CBG

    def draw(yb, H):
        """The figure, (b)'s plots at yb (mm), H mm tall; returns it, the color bars' and (b)'s axes."""
        fig = plt.figure(figsize=(S.FULL_W_IN, H / 25.4)); cbs, bs_ = [], []; grid, heads = {}, {}
        for j, (k, _l) in enumerate(FIG_METRICS):
            x = LEFT + j * (pw + GAP)
            vmax = float(np.percentile(np.concatenate([maps[arm][k][core] for arm in show]), 99))
            cm = plt.get_cmap("magma").copy(); cm.set_bad((0, 0, 0, 0))      # transparent: the gray not-evaluated layer shows through
            for i, arm in enumerate(show):
                ax = S.axes_mm(fig, x, TOP + i * (ph + GAP), pw, ph)
                ne = np.where(bs & ~cs, 1.0, np.nan)[r0:r1, c0:c1]          # brain voxels outside the core: dark gray
                ax.imshow(ne, cmap=ListedColormap(["#3a3a3a"]), vmin=0, vmax=1, interpolation="nearest")
                m = np.where(cs, sect(maps[arm][k]), np.nan)[r0:r1, c0:c1]
                im = ax.imshow(m, cmap=cm, vmin=0, vmax=vmax, interpolation_stage="rgba")
                S.outline(ax, bs[r0:r1, c0:c1], lw=0.4)
                ax.set_xticks([]); ax.set_yticks([]); [s_.set_visible(False) for s_ in ax.spines.values()]
                grid[(i, j)] = ax
                if i == 0:
                    heads[j] = S.header(ax, HEADS[k], dx=0, dy=2)
                    if j == 0:
                        S.letter(ax, "a", dx=-12, dy=2)
                if j == 0:
                    ax.annotate(LABEL[arm].replace("T2 + SANDI + tensor", "T2 + SANDI\n+ tensor"), xy=(0, 0.5), xycoords="axes fraction",
                                xytext=(-16, 0), textcoords="offset points", ha="right", va="center", color=S.FG, annotation_clip=False)
            cax = S.axes_mm(fig, x + 0.1 * pw, ycb, 0.8 * pw, CBH); cbs.append(cax)
            cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.outline.set_edgecolor(S.FRAME); cb.ax.tick_params(length=2, pad=1)
            cb.ax.xaxis.set_major_formatter("{x:.1f}" if k == "T2" else "{x:.0f}")
            ax = S.axes_mm(fig, x + B_IN, yb, pw - 2 * B_IN, BH); bs_.append(ax)  # (b): core mean of every variant
            ys = np.arange(len(arms))[::-1]
            vals = [summ[arm][k]["core_mean"] for arm in arms]
            for arm, y, v in zip(arms, ys, vals):
                ax.plot(v, y, "o", mfc="none" if in_sample(arm, k) else S.WITHOUT, mec=S.WITHOUT, ms=3.4, mew=0.8)
            for arm, y, v in zip(arms, ys, vals):
                if arm in show:
                    ax.plot(v, y, "o", mfc="none", mec=S.FG, ms=7, mew=0.8)
            ax.set_yticks(ys); ax.set_yticklabels([LABEL[x_] for x_ in arms] if j == 0 else [])
            ax.set_ylim(-0.6, len(arms) - 0.4); ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="x", alpha=0.5)
            ax.set_xlabel(XUNIT[k])
            lo, hi = min(vals), max(vals); d = (hi - lo) or abs(hi) * 0.05
            ax.set_xlim(lo - 0.15 * d, hi + 0.15 * d)
            if j == 0:
                S.letter(ax, "b", dx=-(LEFT + B_IN - S.MARGIN_MM) / 25.4 * 72, dy=2)   # at the page margin
        S.orient_letters(fig, [(grid[(i, 0)], grid[(i, nm - 1)], "coronal") for i in range(len(show))])   # L and R on every row
        S.scalebar_title(grid[(0, nm - 1)], 1 / vx, SCALEBAR_MM, heads[nm - 1])     # ONE bar: top right, at the titles' level
        return fig, cbs, bs_

    def box(fig, axs):
        r_ = fig.canvas.get_renderer(); Hf = fig.get_size_inches()[1] * 25.4
        bb = mtrans.Bbox.union([a_.get_tightbbox(r_) for a_ in axs])
        return Hf - bb.y1 / fig.dpi * 25.4, Hf - bb.y0 / fig.dpi * 25.4
    yb0 = ycb + CBH + 14.0; fig, cbs, bs_ = draw(yb0, yb0 + BH + 12.0)        # drawn once to measure, then packed
    cb_bot = box(fig, cbs)[1]; b_top, b_bot = box(fig, bs_); plt.close(fig)
    yb = cb_bot + GAP + (yb0 - b_top); H = yb + (b_bot - yb0) + S.MARGIN_MM
    fig, cbs, bs_ = draw(yb, H)
    S.save(fig, a.out, [NUM / f"{SUMMARY}.json", NUM / "acpc.json"] + [OUTD / f"{arm}__{k}.nii.gz" for arm in show for k, _ in FIG_METRICS],
           cmd="python code/figures/variability.py " + " ".join(sys.argv[1:]),
           extra=dict(print_mm_per_tissue_mm=round(scale, 3), section="coronal through the anterior commissure", scalebar_mm=SCALEBAR_MM,
                      shown=show, cmap="magma"))
    plt.close(fig); print("->", a.out, "scale", round(scale, 3))


if __name__ == "__main__":
    main()
