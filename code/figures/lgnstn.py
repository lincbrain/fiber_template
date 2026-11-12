#!/usr/bin/env python3
"""The paper's figure "The LGN and STN of three brains in the template" (Results, after the LGN and STN table; earlier a
supplementary figure).

    python code/figures/lgnstn.py [--out figures/lgnstn.pdf]

Rows: LGN and STN. Columns, in two groups: the left hemisphere's coronal, axial and sagittal sections of the template's FA, then the
right hemisphere's, each through the mean centroid of the labels shown for that nucleus and hemisphere (window +-HALF_MM): the three brains' labels of record (demons; products
lgnstn_labels/<ID>/<ID>_<struct>_mri-template_mask_demons, outlined by the 0.5 contour of their probability maps) in COLORS, and the SARM nucleus (products
lgnstn_slab/SARM_<struct>_mri-template.nii.gz, as code/figures/paper_numbers/lgnstn_budget_r059.py writes it) dashed white. A legend
above names them, the LGN row's 2 mm scale bar at its right; the STN row's, at its own scale, under that row (the LGN windows are 9.25 mm
across, the STN's 7.25 mm).
Orientation as figstyle.section: coronal and axial, the left of the brain on the left; sagittal, anterior right;
superior up. Values (centroid distances) go in the caption (lgnstn_budget_r059). Full width, 180 mm.
Writes the PDF and a PNG beside it. Ported from the analysis project's paper/scripts/figS_lgnstn_r059.py.
10 Nov 2026 (requested at the 10 Nov meeting): only the former last column (all three brains and SARM, the same section centers); the
three pair columns dropped; MM3 golden yellow instead of figstyle's green, which was hard to tell from MM1's blue; a legend.
11 Nov 2026 (the author: "change the layout"): horizontal and compact, 2 rows (LGN, STN) x 6 columns grouped by hemisphere, 180 mm
wide; the same section centers, colors and panels' size.
11 Nov 2026 (the author, after checks found no registration step that misplaces the labels): the background is the template's FA,
where both nuclei show as dark bodies between bright fiber tracts (T2 shows them faintly); each brain's outline is the 0.5 contour of
its carried probability map (lgnstn_labels <ID>_<struct>_mri-template_prob_demons; the mask of record is prob > 0.5), drawn with
sub-voxel interpolation instead of voxel steps; SARM's binary nucleus is smoothed by the anti-aliasing rule of record (FWHM 0.25 mm)
before its 0.5 contour. The same layout, section centers, colors and window.
11 Nov 2026 (AY's comment, the author's request): orientation letters once, on the left hemisphere's view headers: L and R on the
coronal and axial views, P and A on the sagittal views (as figstyle.section draws them: neurological, anterior up on axial, anterior
right on sagittal); the axial view's A on one more header line above and its P below the bottom row, because the panels sit 0.5 mm
apart. The header band is 2.54 mm taller (60 px at 600 ppi) and the group headers one line higher; nothing else moves.
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
NUCS = ("LGN", "STN")                                          # the figure's rows
HEMIS = (("lh", "left"), ("rh", "right"))                      # its two groups of columns
FIG_W = 180.0                                                  # mm
HALF_MM = {"LGN": 4.5, "STN": 3.5}
PLANES = ("coronal", "axial", "sagittal")
LETTERS = dict(coronal=("L", "R"), axial=("L", "R"), sagittal=("P", "A"))   # left and right of each view, as figstyle.section draws it
VOX = 0.25
SCALE_BAR_MM = 2.0                                             # one bar per row (the rows' windows differ): LGN on the legend's row, STN under its row
BAR_BELOW_MM = 3.5                                             # the STN bar this far under its panel: its length fits between
BAR_Y_MM = 3.6                                                 # the bar's height from the figure's top: its length above it, inside


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figures/lgnstn.pdf")
    a = ap.parse_args()
    P = paths.products("products_completion")
    FS.setup()
    lp = REPO / "settings" / "specimen_labels.json"; lab = json.load(open(lp))["labels"]
    bgp = P["template_mri"] / "linc_mri_FA.nii.gz"                      # the background: FA
    bgi = nib.load(bgp); BG = np.asarray(bgi.dataobj).astype(np.float32); A = bgi.affine
    assert np.allclose(A[:3, :3], np.diag([VOX] * 3)), "template not axis-aligned RAS"
    lps = {(S, st): P["lgnstn_labels"] / S / f"{S}_{st}_mri-template_mask_demons.nii.gz" for S in SUBS for st, _ in ROWS}
    pps = {(S, st): P["lgnstn_labels"] / S / f"{S}_{st}_mri-template_prob_demons.nii.gz" for S in SUBS for st, _ in ROWS}
    rps = {st: P["lgnstn_slab"] / f"SARM_{st}_mri-template.nii.gz" for st, _ in ROWS}
    L = {k: np.asarray(nib.load(p).dataobj) > 0 for k, p in lps.items()}
    R = {k: np.asarray(nib.load(p).dataobj) > 0 for k, p in rps.items()}
    from scipy.ndimage import gaussian_filter
    PR = {k: np.asarray(nib.load(p).dataobj).astype(np.float32) for k, p in pps.items()}                 # outlines: prob 0.5 (= the masks)
    RS = {k: gaussian_filter(v.astype(np.float32), 0.25 / 2.3548 / VOX) for k, v in R.items()}            # SARM: FWHM 0.25 mm, then 0.5
    ins = [str(bgp)] + [str(p) for p in lps.values()] + [str(p) for p in pps.values()] + [str(p) for p in rps.values()] + [str(lp)]
    gap_in, gap_grp, gap_row, left, right, top, bottom = 0.5, 3.0, 2.2, 5.0, 1.0, 13.5 + 2.54, BAR_BELOW_MM + 1.0
    W = FIG_W
    cell = (W - left - right - 4 * gap_in - gap_grp) / 6.0
    H = top + len(NUCS) * cell + (len(NUCS) - 1) * gap_row + bottom
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    centers = {}
    for r_, nuc in enumerate(NUCS):
        h = int(round(HALF_MM[nuc] / VOX))
        y0 = top + r_ * (cell + gap_row)
        for g_, (hemi, hname) in enumerate(HEMIS):
            st = hemi + nuc
            ms = [L[S, st] for S in SUBS] + [R[st]]   # the labels shown: the three brains and SARM
            cen = np.rint(np.mean([np.argwhere(m).mean(0) for m in ms], axis=0)).astype(int)
            centers[st] = (cen * VOX + A[:3, 3]).round(2).tolist()
            for p_, plane in enumerate(PLANES):
                x0 = left + g_ * (3 * cell + 2 * gap_in + gap_grp) + p_ * (cell + gap_in)
                ax = FS.axes_mm(fig, x0, y0, cell, cell); ax.axis("off")
                k = FS.PLANE_AXIS[plane]
                box = [slice(max(cen[i] - h, 0), cen[i] + h + 1) for i in range(3)]
                box[k] = slice(None)
                img = FS.section(BG[tuple(box)], plane, cen[k])
                lo, hi = np.percentile(img, [1, 99.5])
                ax.imshow(np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1), cmap="gray", vmin=0, vmax=1)
                for S in SUBS:
                    m2 = FS.section(PR[S, st][tuple(box)], plane, cen[k])
                    if m2.any():
                        FS.outline(ax, m2, color=COLORS[S], lw=LW)
                m2 = FS.section(RS[st][tuple(box)], plane, cen[k])
                if m2.any():
                    FS.outline(ax, m2, color=FS.FG, lw=LW, linestyles="dashed")
                if r_ == 0:
                    ax.annotate(plane, xy=(0.5, 1), xycoords="axes fraction", xytext=(0, 2), textcoords="offset points",
                                ha="center", va="bottom", fontsize=FS.MIN_PT, color=FS.GRAY, annotation_clip=False)
                    if g_ == 0:                                       # orientation letters, once: the left group's view headers
                        lt, rt = LETTERS[plane]
                        for s_, x_, dx_, ha_ in ((lt, 0, 4, "left"), (rt, 1, -4, "right")):     # 4 pt in from the corners: one view's letters
                            ax.annotate(s_, xy=(x_, 1), xycoords="axes fraction", xytext=(dx_, 2), textcoords="offset points",
                                        ha=ha_, va="bottom", fontsize=FS.MIN_PT, color=FS.GRAY, annotation_clip=False)
                        if plane == "axial":                          # its A one header line up (its P under the bottom row)
                            ax.annotate("A", xy=(0.5, 1), xycoords="axes fraction", xytext=(0, 10.5), textcoords="offset points",
                                        ha="center", va="bottom", fontsize=FS.MIN_PT, color=FS.GRAY, annotation_clip=False)
                    if p_ == 1:                                       # the group's header, above its view labels
                        ax.annotate(hname, xy=(0.5, 1), xycoords="axes fraction", xytext=(0, 19), textcoords="offset points",
                                    ha="center", va="bottom", fontsize=FS.MIN_PT, color=FS.FG, annotation_clip=False)
                if g_ == 0 and plane == "axial" and r_ == len(NUCS) - 1:
                    ax.annotate("P", xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -2), textcoords="offset points",
                                ha="center", va="top", fontsize=FS.MIN_PT, color=FS.GRAY, annotation_clip=False)
                if r_ == len(NUCS) - 1 and g_ == len(HEMIS) - 1 and p_ == 2:   # the bottom row's own bar, at its scale, under its last panel
                    FS.scalebar(ax, 1 / VOX, SCALE_BAR_MM, y_frac=BAR_BELOW_MM / cell, x0_px=2 * h + 0.5 - SCALE_BAR_MM / VOX)
                if g_ == 0 and p_ == 0:
                    ax.annotate(nuc, xy=(0, 0.5), xycoords="axes fraction", xytext=(-3, 0), textcoords="offset points",
                                rotation=90, ha="right", va="center", fontsize=FS.MIN_PT, color=FS.FG, annotation_clip=False)
    # the top row's 2 mm bar on the legend's row, at the top right (the author, 12 Nov 2026), at that row's scale (the LGN's window; the
    # STN's is narrower, HALF_MM, and has its own bar under the bottom row), right-aligned with that row's last panel
    n0 = 2 * int(round(HALF_MM[NUCS[0]] / VOX)) + 1
    bx = FS.axes_mm(fig, left + (len(HEMIS) - 1) * (3 * cell + 2 * gap_in + gap_grp) + 2 * (cell + gap_in), BAR_Y_MM, cell, 0.01)
    bx.axis("off"); bx.set_xlim(-0.5, n0 - 0.5); bx.set_ylim(0, 1)
    FS.scalebar(bx, 1 / VOX, SCALE_BAR_MM, y_frac=0.0, x0_px=n0 - 0.5 - SCALE_BAR_MM / VOX)
    hl = [Line2D([], [], color=COLORS[S], lw=LW) for S in SUBS] + [Line2D([], [], color=FS.FG, lw=LW, linestyle="--")]
    fig.legend(hl, [lab[S] for S in SUBS] + ["SARM"], loc="upper center", ncol=4, frameon=False, fontsize=FS.MIN_PT,
               handlelength=2.6, columnspacing=1.6, borderaxespad=0, bbox_transform=fig.transFigure,
               bbox_to_anchor=(0.5, 1 - 0.8 / H))
    cmd = " ".join(["code/figures/lgnstn.py"] + sys.argv[1:])
    FS.save(fig, a.out, ins, cmd, extra=dict(half_window_mm=HALF_MM, section_centers_world_mm=centers,
                                            colors=dict({lab[S]: COLORS[S] for S in SUBS}, SARM=f"{FS.FG}, dashed"),
                                            scale_bar_mm=SCALE_BAR_MM, scale_bars={NUCS[0]: "top right, the legend's row",
                                                                                    NUCS[-1]: "bottom right, under its row"}))
    fig.savefig(str(Path(a.out).with_suffix(".png")), dpi=FS.DPI, facecolor=fig.get_facecolor())
    print("->", a.out, f"{W:.0f} x {H:.1f} mm")


if __name__ == "__main__":
    main()
