#!/usr/bin/env python3
"""The paper's figure "The LGN and STN of three brains in the template" (Results, after the LGN and STN table; earlier a
supplementary figure).

    python code/figures/lgnstn.py [--out figures/lgnstn.pdf]

Rows: LGN and STN, left and right hemisphere. Columns: coronal, axial and sagittal sections of the template's T2 through the mean
centroid of the labels shown (window +-HALF_MM): the three brains' labels of record (demons; products
lgnstn_labels/<ID>/<ID>_<struct>_mri-template_mask_demons) outlined in COLORS, and the SARM nucleus (products
lgnstn_slab/SARM_<struct>_mri-template.nii.gz, as code/figures/paper_numbers/lgnstn_budget_r059.py writes it) dashed white. A legend
above names them. Orientation as figstyle.section: coronal and axial, the left of the brain on the left; sagittal, anterior right;
superior up. Values (centroid distances) go in the caption (lgnstn_budget_r059). One column wide (figstyle.COL_W_IN).
Writes the PDF and a PNG beside it. Ported from the analysis project's paper/scripts/figS_lgnstn_r059.py.
10 Nov 2026 (requested at the 10 Nov meeting): only the former last column (all three brains and SARM, the same section centers); the
three pair columns dropped; MM3 golden yellow instead of figstyle's green, which was hard to tell from MM1's blue; a legend.
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
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO / "code"))
import figstyle as FS  # noqa: E402
from shared import paths  # noqa: E402

SUBS = ["MR243", "MR252", "MR256"]
GOLD = "#FFC20A"   # saturated golden yellow: reads against the dashed white SARM outline (figstyle's pale yellow does not)
COLORS = dict(MR243=FS.BRAIN["MR243"], MR252=FS.BRAIN["MR252"], MR256=GOLD)   # MM1 blue, MM2 pink, MM3 golden yellow
LW = 1.0
ROWS = [("lhLGN", "LGN, left"), ("rhLGN", "LGN, right"), ("lhSTN", "STN, left"), ("rhSTN", "STN, right")]
HALF_MM = {"LGN": 4.5, "STN": 3.5}
PLANES = ("coronal", "axial", "sagittal")
VOX = 0.25


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figures/lgnstn.pdf")
    a = ap.parse_args()
    P = paths.products("products_completion")
    FS.setup()
    lp = REPO / "settings" / "specimen_labels.json"; lab = json.load(open(lp))["labels"]
    t2p = P["template_mri"] / "linc_mri_T2.nii.gz"
    t2i = nib.load(t2p); T2 = np.asarray(t2i.dataobj).astype(np.float32); A = t2i.affine
    assert np.allclose(A[:3, :3], np.diag([VOX] * 3)), "template not axis-aligned RAS"
    lps = {(S, st): P["lgnstn_labels"] / S / f"{S}_{st}_mri-template_mask_demons.nii.gz" for S in SUBS for st, _ in ROWS}
    rps = {st: P["lgnstn_slab"] / f"SARM_{st}_mri-template.nii.gz" for st, _ in ROWS}
    L = {k: np.asarray(nib.load(p).dataobj) > 0 for k, p in lps.items()}
    R = {k: np.asarray(nib.load(p).dataobj) > 0 for k, p in rps.items()}
    ins = [str(t2p)] + [str(p) for p in lps.values()] + [str(p) for p in rps.values()] + [str(lp)]
    gap_in, gap_row, left, right, top, bottom = 0.5, 2.2, 6.0, 1.0, 10.0, 3.0
    W = FS.COL_W_IN * 25.4
    cell = (W - left - right - 2 * gap_in) / 3.0
    H = top + len(ROWS) * cell + (len(ROWS) - 1) * gap_row + bottom
    fig = plt.figure(figsize=(FS.COL_W_IN, H / 25.4))
    centers = {}
    for r_, (st, rname) in enumerate(ROWS):
        nuc = st[2:]; h = int(round(HALF_MM[nuc] / VOX))
        y0 = top + r_ * (cell + gap_row)
        ms = [L[S, st] for S in SUBS] + [R[st]]   # the labels shown: the three brains and SARM
        cen = np.rint(np.mean([np.argwhere(m).mean(0) for m in ms], axis=0)).astype(int)
        centers[st] = (cen * VOX + A[:3, 3]).round(2).tolist()
        for p_, plane in enumerate(PLANES):
            x0 = left + p_ * (cell + gap_in)
            ax = FS.axes_mm(fig, x0, y0, cell, cell); ax.axis("off")
            k = FS.PLANE_AXIS[plane]
            box = [slice(max(cen[i] - h, 0), cen[i] + h + 1) for i in range(3)]
            box[k] = slice(None)
            img = FS.section(T2[tuple(box)], plane, cen[k])
            lo, hi = np.percentile(img, [1, 99.5])
            ax.imshow(np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1), cmap="gray", vmin=0, vmax=1)
            for S in SUBS:
                m2 = FS.section(L[S, st][tuple(box)].astype(float), plane, cen[k])
                if m2.any():
                    FS.outline(ax, m2, color=COLORS[S], lw=LW)
            m2 = FS.section(R[st][tuple(box)].astype(float), plane, cen[k])
            if m2.any():
                FS.outline(ax, m2, color=FS.FG, lw=LW, linestyles="dashed")
            if r_ == 0:
                ax.annotate(plane, xy=(0.5, 1), xycoords="axes fraction", xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=FS.MIN_PT, color=FS.GRAY, annotation_clip=False)
            if p_ == 0:
                ax.annotate(rname, xy=(0, 0.5), xycoords="axes fraction", xytext=(-3, 0), textcoords="offset points",
                            rotation=90, ha="right", va="center", fontsize=FS.MIN_PT, color=FS.FG, annotation_clip=False)
            if r_ == len(ROWS) - 1 and p_ == 2:
                FS.scalebar(ax, 1 / VOX, 2.0, y_frac=0.06)
    hl = [Line2D([], [], color=COLORS[S], lw=LW) for S in SUBS] + [Line2D([], [], color=FS.FG, lw=LW, linestyle="--")]
    fig.legend(hl, [lab[S] for S in SUBS] + ["SARM"], loc="upper center", ncol=4, frameon=False, fontsize=FS.MIN_PT,
               handlelength=2.6, columnspacing=1.6, borderaxespad=0, bbox_transform=fig.transFigure,
               bbox_to_anchor=((left + 1.5 * cell + gap_in) / W, 1 - 1.0 / H))
    cmd = " ".join(["code/figures/lgnstn.py"] + sys.argv[1:])
    FS.save(fig, a.out, ins, cmd, extra=dict(half_window_mm=HALF_MM, section_centers_world_mm=centers,
                                            colors=dict({lab[S]: COLORS[S] for S in SUBS}, SARM=f"{FS.FG}, dashed"),
                                            scale_bar_mm=2))
    fig.savefig(str(Path(a.out).with_suffix(".png")), dpi=FS.DPI, facecolor=fig.get_facecolor())
    print("->", a.out, f"{W:.0f} x {H:.1f} mm")


if __name__ == "__main__":
    main()
