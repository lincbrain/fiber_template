#!/usr/bin/env python3
"""The paper's figure "The LGN and STN of three brains in the template" (Results, after the LGN and STN table; earlier a
supplementary figure).

    python reproduce/figures/lgnstn.py [--out figures/lgnstn.pdf]

Rows: LGN and STN, left and right hemisphere. Columns: the three brain pairs (MM1-MM2, MM1-MM3, MM2-MM3) and a reference column (the
three brains against the SARM nucleus). Each cell: coronal, axial and sagittal sections of the template's T2 through the mean centroid
of the labels shown (window +-HALF_MM), the labels of record (demons; products lgnstn_labels/<ID>/<ID>_<struct>_mri-template_mask_demons)
outlined in each brain's fixed color (figstyle.BRAIN), the SARM nucleus (products lgnstn_slab/SARM_<struct>_mri-template.nii.gz, as
reproduce/numbers/lgnstn_budget_r059.py writes it) dashed. Orientation as figstyle.section: coronal and axial, the left of the brain on
the left; sagittal, anterior right; superior up. Values (centroid distances) go in the caption (lgnstn_budget_r059).
Writes the PDF and a PNG beside it. Ported from the analysis project's paper/scripts/figS_lgnstn_r059.py (the figure unchanged).
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
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO))
import figstyle as FS  # noqa: E402
from linc import paths  # noqa: E402

SUBS = ["MR243", "MR252", "MR256"]
PAIRS = [("MR243", "MR252"), ("MR243", "MR256"), ("MR252", "MR256")]
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
    lp = REPO / "docs" / "specimen_labels.json"; lab = json.load(open(lp))["labels"]
    t2p = P["template_mri"] / "linc_mri_T2.nii.gz"
    t2i = nib.load(t2p); T2 = np.asarray(t2i.dataobj).astype(np.float32); A = t2i.affine
    assert np.allclose(A[:3, :3], np.diag([VOX] * 3)), "template not axis-aligned RAS"
    lps = {(S, st): P["lgnstn_labels"] / S / f"{S}_{st}_mri-template_mask_demons.nii.gz" for S in SUBS for st, _ in ROWS}
    rps = {st: P["lgnstn_slab"] / f"SARM_{st}_mri-template.nii.gz" for st, _ in ROWS}
    L = {k: np.asarray(nib.load(p).dataobj) > 0 for k, p in lps.items()}
    R = {k: np.asarray(nib.load(p).dataobj) > 0 for k, p in rps.items()}
    ins = [str(t2p)] + [str(p) for p in lps.values()] + [str(p) for p in rps.values()] + [str(lp)]
    cols = [(f"{lab[a_]}, {lab[b_]}", [a_, b_], False) for a_, b_ in PAIRS] + [("all three, SARM", SUBS, True)]
    gap_in, gap_col, gap_row, left, top = 0.5, 3.0, 2.2, 13.0, 9.0
    W = FS.FULL_W_IN * 25.4
    cell = (W - left - 3 * gap_col - 4 * 2 * gap_in) / 12.0
    H = top + len(ROWS) * cell + (len(ROWS) - 1) * gap_row + 2.0
    fig = plt.figure(figsize=(FS.FULL_W_IN, H / 25.4))
    centers = {}
    for r_, (st, rname) in enumerate(ROWS):
        nuc = st[2:]; h = int(round(HALF_MM[nuc] / VOX))
        y0 = top + r_ * (cell + gap_row)
        for c_, (cname, brains, with_ref) in enumerate(cols):
            ms = [L[S, st] for S in brains] + ([R[st]] if with_ref else [])
            cen = np.rint(np.mean([np.argwhere(m).mean(0) for m in ms], axis=0)).astype(int)
            centers[f"{st}|{cname}"] = (cen * VOX + A[:3, 3]).round(2).tolist()
            for p_, plane in enumerate(PLANES):
                x0 = left + c_ * (3 * cell + 2 * gap_in + gap_col) + p_ * (cell + gap_in)
                ax = FS.axes_mm(fig, x0, y0, cell, cell); ax.axis("off")
                k = FS.PLANE_AXIS[plane]
                box = [slice(max(cen[i] - h, 0), cen[i] + h + 1) for i in range(3)]
                box[k] = slice(None)
                img = FS.section(T2[tuple(box)], plane, cen[k])
                lo, hi = np.percentile(img, [1, 99.5])
                ax.imshow(np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1), cmap="gray", vmin=0, vmax=1)
                for S in brains:
                    m2 = FS.section(L[S, st][tuple(box)].astype(float), plane, cen[k])
                    if m2.any():
                        FS.outline(ax, m2, color=FS.BRAIN[S], lw=0.8)
                if with_ref:
                    m2 = FS.section(R[st][tuple(box)].astype(float), plane, cen[k])
                    if m2.any():
                        FS.outline(ax, m2, color=FS.FG, lw=0.8, linestyles="dashed")
                if r_ == 0:
                    ax.annotate(plane, xy=(0.5, 1), xycoords="axes fraction", xytext=(0, 2), textcoords="offset points",
                                ha="center", va="bottom", fontsize=FS.MIN_PT, color=FS.GRAY, annotation_clip=False)
                if r_ == 0 and p_ == 0:
                    FS.header(ax, cname, dx=0, dy=11, fontweight="bold")
                if c_ == 0 and p_ == 0:
                    ax.annotate(rname, xy=(0, 0.5), xycoords="axes fraction", xytext=(-3, 0), textcoords="offset points",
                                rotation=90, ha="right", va="center", fontsize=FS.MIN_PT, color=FS.FG, annotation_clip=False)
                if r_ == len(ROWS) - 1 and c_ == 3 and p_ == 2:
                    FS.scalebar(ax, 1 / VOX, 2.0, y_frac=0.06)
    cmd = " ".join(["reproduce/figures/lgnstn.py"] + sys.argv[1:])
    FS.save(fig, a.out, ins, cmd, extra=dict(half_window_mm=HALF_MM, section_centers_world_mm=centers,
                                            colors={lab[S]: FS.BRAIN[S] for S in SUBS}, scale_bar_mm=2))
    fig.savefig(str(Path(a.out).with_suffix(".png")), dpi=FS.DPI, facecolor=fig.get_facecolor())
    print("->", a.out, f"{W:.0f} x {H:.1f} mm")


if __name__ == "__main__":
    main()
