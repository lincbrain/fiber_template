#!/usr/bin/env python3
"""The paper's figure "Between-brain deviation after construction" (Results "Between-brain variability").

(a) The mean absolute deviation of the eight warped inputs from their own template, for three variants (T2 only, SANDI only, the
    four-channel template), in T2 (z-units), the soma fraction (percentage points; the neurite fraction gives the same value) and the
    principal direction (degrees), on the coronal section through the anterior commissure, in the voxels covered by all inputs (the
    template's T2 core); brain voxels outside the core dark gray; one color scale per measure; white outline = the template brain.
(b) The core mean of every one of the seven variants; open = in sample (the variant was built with that channel), filled = out of
    sample; rings = the three variants shown in (a).
Reads (configs/products_mri.yaml): the deviation maps mri_runs/_variability_rebuild_all/<arm>__<T2|FRAC1|DTI>.nii.gz (written by
linc/evaluation/variability.py), the seven-variant summary numbers_of_record/variability_rebuild.json, the anterior commissure of
numbers_of_record/acpc.json, and template_mri/linc_mri_{T2,core_t2,brainmask}.nii.gz.
    python reproduce/figures/variability.py [--out figures/variability.png]"""
import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
import figstyle as S                                           # noqa: E402
from linc import paths                                         # noqa: E402

ARMS = ["rb_t2", "rb_micro", "rb_dti", "rb_t2micro", "rb_t2dti", "rb_dtimicro", "rb_final"]     # the seven variants (Table 2 order)
LABEL = {"rb_t2": "T2", "rb_micro": "SANDI", "rb_dti": "tensor", "rb_t2micro": "T2 + SANDI", "rb_t2dti": "T2 + tensor",
         "rb_dtimicro": "tensor + SANDI", "rb_final": "four-channel template"}
SHOW = ("rb_t2", "rb_micro", "rb_final")                        # the three variants in (a)
FIG_METRICS = [("T2", "T2 |Δz|"), ("FRAC1", "soma = neurite |Δ| (pp)"), ("DTI", "V1 angle (°)")]
XUNIT = {"T2": "z-units", "FRAC1": "percentage points", "DTI": "degrees"}
CHAN = {"T2": "T2", "FRAC1": "SANDI", "DTI": "tensor"}         # a measure is in sample for variants built with that channel
HEADS = {"T2": "T2 (z-units)", "FRAC1": "soma fraction (percentage points)", "DTI": "angular deviation (°)"}
SUMMARY = "variability_rebuild"                                 # the seven-variant summary record
MAPS = "_variability_rebuild_all"                               # folder of the deviation maps under mri_runs
SCALEBAR_MM, PAD_PX = 5, 3


def load(p):
    return np.asanyarray(nib.load(str(p)).dataobj).astype(np.float64)


def in_sample(arm, k):
    lab = LABEL[arm]
    return lab == "four-channel template" or CHAN[k] in lab


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
    import matplotlib.ticker as tk
    from matplotlib.colors import ListedColormap
    show, arms = list(SHOW), ARMS
    brain = load(TPL / "linc_mri_brainmask.nii.gz") > 0.5
    sect = lambda v: S.section(v, "coronal", yac)                                   # noqa: E731
    cs, bs = sect(core) > 0.5, sect(brain.astype(float)) > 0.5
    ij = np.argwhere(bs)
    r0, r1, c0, c1 = ij[:, 0].min() - PAD_PX, ij[:, 0].max() + PAD_PX + 1, ij[:, 1].min() - PAD_PX, ij[:, 1].max() + PAD_PX + 1
    vx = abs(ref.affine[0, 0]); tw, th = (c1 - c0) * vx, (r1 - r0) * vx
    nm = len(FIG_METRICS); W = S.FULL_W_IN * 25.4
    LEFT, RIGHT, GAPC, GAPR, TOP = 27.0, 4.0, 2.0, 1.5, 6.0
    scale = min(S.SCALE["whole"], (W - LEFT - RIGHT - (nm - 1) * GAPC) / (nm * tw))
    pw, ph = tw * scale, th * scale
    CBG, CBH, BG_, BH = 4.0, 1.6, 8.0, 27.0
    H = TOP + len(show) * ph + (len(show) - 1) * GAPR + CBG + CBH + 6.0 + BG_ + BH + 9.0
    fig = plt.figure(figsize=(S.FULL_W_IN, H / 25.4))
    for j, (k, _l) in enumerate(FIG_METRICS):
        x = LEFT + j * (pw + GAPC)
        vmax = float(np.percentile(np.concatenate([maps[arm][k][core] for arm in show]), 99))
        cm = plt.get_cmap("magma").copy(); cm.set_bad((0, 0, 0, 0))          # transparent: the gray not-evaluated layer shows through
        for i, arm in enumerate(show):
            ax = S.axes_mm(fig, x, TOP + i * (ph + GAPR), pw, ph)
            ne = np.where(bs & ~cs, 1.0, np.nan)[r0:r1, c0:c1]              # brain voxels outside the core: dark gray
            ax.imshow(ne, cmap=ListedColormap(["#3a3a3a"]), vmin=0, vmax=1, interpolation="nearest")
            m = np.where(cs, sect(maps[arm][k]), np.nan)[r0:r1, c0:c1]
            im = ax.imshow(m, cmap=cm, vmin=0, vmax=vmax, interpolation_stage="rgba")
            S.outline(ax, bs[r0:r1, c0:c1], lw=0.4)
            ax.set_xticks([]); ax.set_yticks([]); [s_.set_visible(False) for s_ in ax.spines.values()]
            if i == 0:
                S.header(ax, HEADS[k], dx=0, dy=2)
                if j == 0:
                    S.letter(ax, "a", dx=-12, dy=2)
            if j == 0 and i == 0:
                ax.annotate("L", xy=(0, 0.5), xycoords="axes fraction", xytext=(-2, 0), textcoords="offset points", ha="right",
                            va="center", color=S.GRAY, annotation_clip=False)
            if j == 0:
                ax.annotate(LABEL[arm].replace("four-channel template", "four-channel\ntemplate"), xy=(0, 0.5), xycoords="axes fraction",
                            xytext=(-16, 0), textcoords="offset points", ha="right", va="center", color=S.FG, annotation_clip=False)
                if i == len(show) - 1:
                    S.scalebar(ax, 1 / vx, SCALEBAR_MM, y_frac=0.05)
            if j == nm - 1 and i == 0:
                ax.annotate("R", xy=(1, 0.5), xycoords="axes fraction", xytext=(2, 0), textcoords="offset points", ha="left",
                            va="center", color=S.GRAY, annotation_clip=False)
        ycb = TOP + len(show) * ph + (len(show) - 1) * GAPR + CBG
        cax = S.axes_mm(fig, x + 0.1 * pw, ycb, 0.8 * pw, CBH)
        cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.outline.set_edgecolor(S.GRAY); cb.ax.tick_params(length=2, pad=1)
        cb.ax.xaxis.set_major_locator(tk.MaxNLocator(5, integer=(k != "T2")))          # integer ticks where labels are integers
        cb.ax.xaxis.set_major_formatter("{x:.1f}" if k == "T2" else "{x:.0f}")
        ax = S.axes_mm(fig, x + 2.0, ycb + CBH + 6.0 + BG_, pw - 4.0, BH)            # (b): core mean of every variant
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
            S.letter(ax, "b", dx=-70, dy=2)
    S.save(fig, a.out, [NUM / f"{SUMMARY}.json", NUM / "acpc.json"] + [OUTD / f"{arm}__{k}.nii.gz" for arm in show for k, _ in FIG_METRICS],
           cmd="python reproduce/figures/variability.py " + " ".join(sys.argv[1:]),
           extra=dict(print_mm_per_tissue_mm=round(scale, 3), section="coronal through the anterior commissure", scalebar_mm=SCALEBAR_MM,
                      shown=show, cmap="magma"))
    plt.close(fig); print("->", a.out, "scale", round(scale, 3))


if __name__ == "__main__":
    main()
