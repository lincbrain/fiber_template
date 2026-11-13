#!/usr/bin/env python3
"""The paper's figure "The LGN and STN of three brains in the template" (Results, after the LGN and STN table; earlier a
supplementary figure).

    python code/figures/lgnstn.py [--out figures/lgnstn.pdf]

Rows: LGN and STN. Columns, in two groups: the left hemisphere's coronal, axial and sagittal sections of the template's FA, then the
right hemisphere's, each through the mean centroid of the labels shown for that nucleus and hemisphere (window +-HALF_MM): the three brains' labels of record (demons; products
lgnstn_labels/<ID>/<ID>_<struct>_mri-template_mask_demons, outlined by the 0.5 contour of their probability maps) in COLORS, and the SARM nucleus (products
lgnstn_slab/SARM_<struct>_mri-template.nii.gz, as code/figures/paper_numbers/lgnstn_budget_r059.py writes it) dashed white. A legend
above names them, the 2 mm scale bar at its right; both rows share one window (9.25 mm across), so the bar holds for both.
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
12 Nov 2026 (the author: space and sentence case, for every figure): one gutter (figstyle GUTTER_MM) between all six views and
between the rows (before, 0.5 mm inside a hemisphere's group, 3 mm between the groups and 2.2 mm between the rows), MARGIN_MM from
the outermost ink to the page's edges; the 2 mm bar in its standard place, top right, its length's baseline on the legend's
(figstyle.scalebar_title), right-aligned with the top row's last view; headers in sentence case (Left, Right, Coronal, Axial,
Sagittal). The same sections, windows, colors, letters and legend.
12 Nov 2026 (the author: one rule for orientation letters in every figure, figstyle.orient_letters): each view's letters left and
right of it, at the first row's middle, ORIENT_GAP_MM from the view, on the left hemisphere's first row only (coronal and axial L
and R, sagittal P and A); none above or below a view (the axial A and P go: anterior is up, as the caption states). Two letters
share a gutter, so every gutter between the views is widened to the least that holds them (each letter its gap from its own view,
and one gap between the two), all equal; the row labels move out of the first row's L. The header band loses the axial A's line.
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
HEMIS = (("lh", "Left"), ("rh", "Right"))                      # its two groups of columns
FIG_W = 180.0                                                  # mm
HALF_MM = {"LGN": 4.5, "STN": 4.5}   # one window for both rows (the author, 12 Nov 2026), so one scale bar holds for all: the smallest
                                     # half-window that holds every outline of both nuclei in every view (3.38 mm, LGN) plus the margin
                                     # both windows had (1.12 mm); before, STN 3.5 mm
PLANES = ("coronal", "axial", "sagittal")
LETTERS = dict(coronal=("L", "R"), axial=("L", "R"), sagittal=("P", "A"))   # left and right of each view, as figstyle.section draws it
VOX = 0.25
SCALE_BAR_MM = 2.0                                             # one bar for both rows (one window size), on the legend's line
HDR_PT = (2.0, 10.5)                                           # the header lines above the top row (pt): views, groups


def ink_asc(s_, fs=FS.MIN_PT):
    """mm of a text's ink above its baseline, drawn at figstyle's DPI"""
    g = plt.figure(figsize=(4, 1), dpi=FS.DPI, facecolor="k"); t = g.text(0.05, 0.3, s_, fontsize=fs, color="w", va="baseline")
    g.canvas.draw(); a_ = np.asarray(g.canvas.buffer_rgba())[..., 0]; plt.close(g)
    return (0.7 * a_.shape[0] - np.where((a_ > 60).any(1))[0].min()) / FS.DPI * 25.4


def text_mm(fig):
    """a function (text) -> (width, height, descent) mm of a figstyle.MIN_PT text as drawn"""
    from matplotlib.font_manager import FontProperties
    r = fig.canvas.get_renderer(); fp = FontProperties(size=FS.MIN_PT)
    def T(s_):
        w, h, d = r.get_text_width_height_descent(s_, fp, ismath=False)
        return w / fig.dpi * 25.4, h / fig.dpi * 25.4, d / fig.dpi * 25.4
    return T


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
    W, GUT, MG, pt = FIG_W, FS.GUTTER_MM, FS.MARGIN_MM, 25.4 / 72
    T = text_mm(plt.figure(figsize=(W / 25.4, 1.0), dpi=FS.DPI)); plt.close("all")   # figstyle's DPI, as it is saved
    _, t_h, t_d = T("lp")                                          # a text line's box as matplotlib lays it out (its "lp")
    OG, room = FS.ORIENT_GAP_MM, FS.orient_room                    # the orientation letters: gap; room beside a view (gap + ink)
    lets = [LETTERS[p_] for p_ in PLANES]                           # each view's (left, right) letters, on the left group's first row
    need = [room(lets[p_][1]) + OG + room(lets[p_ + 1][0]) for p_ in range(len(PLANES) - 1)]   # two letters in a gutter, one gap apart
    need.append(room(lets[-1][1]) + OG)                            # the group gutter: the last view's right letter alone
    GUTV = max([GUT] + need)                                       # every gutter between the views: equal, the least that holds them
    rl = room(lets[0][0]) + OG                                     # the rotated row labels: one gap left of the first view's left letter
    left, right = MG + t_h + rl, MG
    rise = max(ink_asc("MM1"), (2.0 + 1.2) * pt + ink_asc(f"{SCALE_BAR_MM:g} mm"))   # the legend line's ink above its baseline: the bar's
    base_l = MG + rise                                              # length, 2 pt over the 1.2 pt bar on the baseline (scalebar_title)
    top = base_l + GUT - (t_h - t_d - ink_asc("Left")) + t_h + HDR_PT[1] * pt   # the group headers' ink one gutter under the line
    bottom = MG
    cell = (W - left - right - 5 * GUTV) / 6.0
    H = top + len(NUCS) * cell + (len(NUCS) - 1) * GUT + bottom
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    centers, pairs = {}, []
    for r_, nuc in enumerate(NUCS):
        h = int(round(HALF_MM[nuc] / VOX))
        y0 = top + r_ * (cell + GUT)
        for g_, (hemi, hname) in enumerate(HEMIS):
            st = hemi + nuc
            ms = [L[S, st] for S in SUBS] + [R[st]]   # the labels shown: the three brains and SARM
            cen = np.rint(np.mean([np.argwhere(m).mean(0) for m in ms], axis=0)).astype(int)
            centers[st] = (cen * VOX + A[:3, 3]).round(2).tolist()
            for p_, plane in enumerate(PLANES):
                x0 = left + (3 * g_ + p_) * (cell + GUTV)
                ax = FS.axes_mm(fig, x0, y0, cell, cell); ax.axis("off")
                if r_ == 0 and g_ == len(HEMIS) - 1 and p_ == len(PLANES) - 1:
                    ax_tr = ax                                        # the top row's last view: the bar's edge and scale
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
                    ax.annotate(plane.capitalize(), xy=(0.5, 1), xycoords="axes fraction", xytext=(0, HDR_PT[0]), textcoords="offset points",
                                ha="center", va="bottom", fontsize=FS.MIN_PT, color=FS.GRAY, annotation_clip=False)
                    if g_ == 0:                                       # orientation letters, once: beside the left group's first row
                        pairs.append((ax, ax, LETTERS[plane]))
                    if p_ == 1:                                       # the group's header, above its view labels
                        ax.annotate(hname, xy=(0.5, 1), xycoords="axes fraction", xytext=(0, HDR_PT[1]), textcoords="offset points",
                                    ha="center", va="bottom", fontsize=FS.MIN_PT, color=FS.FG, annotation_clip=False)
                if g_ == 0 and p_ == 0:
                    ax.annotate(nuc, xy=(0, 0.5), xycoords="axes fraction", xytext=(-rl / pt, 0), textcoords="offset points",
                                rotation=90, ha="right", va="center", fontsize=FS.MIN_PT, color=FS.FG, annotation_clip=False)
    hl = [Line2D([], [], color=COLORS[S], lw=LW) for S in SUBS] + [Line2D([], [], color=FS.FG, lw=LW, linestyle="--")]
    leg = fig.legend(hl, [lab[S] for S in SUBS] + ["SARM"], loc="upper center", ncol=4, frameon=False, fontsize=FS.MIN_PT,
                     handlelength=2.6, columnspacing=1.6, borderpad=0, borderaxespad=0, bbox_transform=fig.transFigure,
                     bbox_to_anchor=(0.5, 1 - (base_l - (t_h - t_d)) / H))       # its texts' boxes' top: the baseline at base_l
    FS.orient_letters(fig, [pairs])                                 # one level: the first row's middle (figstyle's gap, size, color)
    fig.canvas.draw()                                               # the legend laid out: its texts' places are final
    FS.scalebar_title(ax_tr, 1 / VOX, SCALE_BAR_MM, leg.get_texts()[-1])   # one bar for both rows (one window), top right
    cmd = " ".join(["code/figures/lgnstn.py"] + sys.argv[1:])
    FS.save(fig, a.out, ins, cmd, extra=dict(half_window_mm=HALF_MM, section_centers_world_mm=centers,
                                            colors=dict({lab[S]: COLORS[S] for S in SUBS}, SARM=f"{FS.FG}, dashed"),
                                            scale_bar_mm=SCALE_BAR_MM, scale_bar_place="top right, its length on the legend's baseline (both rows)",
                                            layout_mm=dict(width=W, height=round(H, 2), view=round(cell, 2), gutter=GUT, view_gutter=round(GUTV, 2), margin=MG)))
    fig.savefig(str(Path(a.out).with_suffix(".png")), dpi=FS.DPI, facecolor=fig.get_facecolor())
    print("->", a.out, f"{W:.0f} x {H:.1f} mm")


if __name__ == "__main__":
    main()
