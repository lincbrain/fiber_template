#!/usr/bin/env python3
"""The paper's figure "The multimodal template": the template's channels (NMT v2 symmetric frame).

    python code/figures/template.py [--out figures/template.png]

Rows: sagittal at x = +2 mm (off the interhemispheric fissure; the template is exactly symmetric, so -2 mm is its mirror image),
coronal and axial through the anterior commissure (acpc.json of the numbers of record). Columns: T2-weighted, soma fraction, neurite
fraction, FA, MD, principal diffusion direction weighted by FA, and the blockface channel (the blockface template's grey channel
mapped into the MRI template by the registration d2bs8_s5sym, inside its own mask). The b0 image is not shown: it is not a headline
channel (it stays in the release as the registration target for new brains).
Orientation: figstyle.section() (neurological; sagittal anterior-right; axial anterior-up). One true scale for every panel (print mm
per tissue mm, the largest that fits the columns across 178 mm; written to the provenance), each panel cropped to the brain + 0.75 mm.
No row labels: the caption names the planes and the direction letters show the view (the author, 12 Nov 2026). Direction letters
outside the first and last panels of each row; one space (GAP) between columns and between rows; the left margin as wide as the
letters need; one 10 mm scale bar, its length above it, at the bottom left (--bar strip: a strip under the color bars as tall as the
bar needs with 1.5 mm to the labels; --bar corner: the empty place nearest the bottom left inside the layout), placed by
figstyle.scalebar_free (measured, 1.5 mm clear of tissue and of every text); colorbars (and the direction-color key) below the columns. Windows: 0-1; T2 and the blockface channel normalized to their in-brain 99.5th percentile; MD 2nd-99.5th
percentile of the tissue mode (voxels above a quarter of the median, brain eroded 0.5 mm; um^2/ms). Display bilinear.
Reads (settings/internal/products_mri*.yaml): template_mri/linc_mri_{brainmask,T2,frac1,frac2,FA,MD,DEC}.nii.gz,
template2template/d2bs8_s5sym/d2bs8_s5sym{,_mask}_in_mri_template.nii.gz, numbers_of_record/acpc.json.
Writes the PNG and <name>.provenance.json beside it.
"""
import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "code"))
import figstyle as S                                           # noqa: E402
from shared import paths                                         # noqa: E402

PREFIX = "linc_mri"                                            # the MRI channels' file prefix
BF_MAP = "d2bs8_s5sym"                                         # the registration that maps the blockface template into the MRI template
CH = [("T2", "T2-weighted", "gray"), ("frac1", "Soma fraction", "turbo"),
      ("frac2", "Neurite fraction", "turbo"), ("FA", "FA", "gray"), ("MD", "MD", "gray"), ("DEC", "Principal direction", None),
      ("BF", "Blockface channel", "gray")]
FIXED = {"frac1": (0, 1), "frac2": (0, 1), "FA": (0, 1)}
CBLAB = {"T2": "Normalized", "frac1": "Fraction", "frac2": "Fraction", "FA": "FA", "MD": "µm²/ms", "BF": "Normalized"}
BF_GAP = 2.5                                                   # extra gap (mm) before the mapped blockface column
PAD_MM, X_SAG = 0.75, 2.0                                      # crop margin (mm); sagittal plane (NMT v2 x, mm)
SCALEBAR_MM = 10
DEC_GAMMA = 1.6                                                # display gamma of the direction colors (read at print size)
RIGHT, GAP, TOP, CB_H, BOTTOM = 3.5, 1.2, 5.0, 9.0, 1.0      # mm; GAP: the one space between columns and between rows
LETTER_OFF_PT = 2                                              # the direction letters' distance from the panels (pt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figures/template.png")
    ap.add_argument("--bar", choices=["strip", "corner"], default="strip",
                    help="strip: the bar at the bottom-left, in a strip under the color bars (the length's height + 1.5 mm); "
                         "corner: the measured empty place nearest the bottom left inside the layout (no strip)")
    a = ap.parse_args()
    PR = {**paths.products("products_mri"), **paths.products("products_mri_figures")}
    import matplotlib
    matplotlib.use("Agg")
    S.setup()
    import matplotlib.pyplot as plt
    T = Path(PR["template_mri"]); P = lambda k: T / f"{PREFIX}_{k}.nii.gz"          # noqa: E731
    load = lambda k: S.canonical(nib.load(str(P(k))))                             # noqa: E731
    acpc = Path(PR["numbers_of_record"]) / "acpc.json"
    brain, A = load("brainmask"); brain = brain > 0.5
    vx = abs(A[0, 0]); px_mm = 1 / vx
    ac = json.loads(acpc.read_text())["data"]["definitive"]["AC"]
    iac = np.round(np.linalg.inv(A) @ np.append(ac, 1))[:3].astype(int)
    xs = (X_SAG - A[0, 3]) / A[0, 0]; xs0 = int(np.floor(xs)); wx = xs - xs0          # x = +2 mm by linear interpolation

    def sect(v, pl):
        if pl == "sagittal":
            return (1 - wx) * S.section(v, pl, xs0).astype(np.float32) + wx * S.section(v, pl, xs0 + 1).astype(np.float32)
        return S.section(v, pl, iac[S.PLANE_AXIS[pl]]).astype(np.float32)
    planes = ["sagittal", "coronal", "axial"]
    pos = dict(sagittal=f"x = {X_SAG:.1f} mm", coronal=f"y = {(A @ np.append(iac, 1))[1]:.1f} mm",
               axial=f"z = {(A @ np.append(iac, 1))[2]:.1f} mm")
    boxes = {}
    for pl in planes:                                          # brain bbox + pad, in display pixels
        ij = np.argwhere(sect(brain.astype(np.float32), pl) > 0.5); p = int(round(PAD_MM * px_mm))
        boxes[pl] = (ij[:, 0].min() - p, ij[:, 0].max() + p + 1, ij[:, 1].min() - p, ij[:, 1].max() + p + 1)
    ext = {pl: ((b[3] - b[2]) * vx, (b[1] - b[0]) * vx) for pl, b in boxes.items()}   # (width, height) mm of tissue
    ncol = len(CH)
    colw_t = max(w for w, _ in ext.values())                   # widest panel (tissue mm) sets the column width
    f_ = plt.figure(); r_ = f_.canvas.get_renderer()             # the left margin: the widest row label, its offset, the page margin
    wlab = max(f_.text(0, 0, S.DIRS[pl][0], fontsize=S.MIN_PT).get_window_extent(r_).width / f_.dpi * 25.4 for pl in planes)
    bar_h = (f_.text(0, 0, f"{SCALEBAR_MM} mm", fontsize=S.MIN_PT).get_window_extent(r_).height / f_.dpi * 25.4
             + (2.0 + 1.2) / 72 * 25.4)                           # the bar and its length: the text, its 2 pt gap, the line
    plt.close(f_)
    LEFT = S.MARGIN_MM + wlab + LETTER_OFF_PT / 72 * 25.4                 # the left letters only (no row labels: the caption names the planes)
    scale = (S.FULL_W_IN * 25.4 - LEFT - RIGHT - (ncol - 1) * GAP - BF_GAP) / (ncol * colw_t)
    colw = colw_t * scale
    rowh = [ext[pl][1] * scale for pl in planes]
    BOT = BOTTOM + (bar_h + S.CLEAR_MM if a.bar == "strip" else 0.0)
    H = TOP + sum(rowh) + GAP * len(planes) + CB_H + BOT
    fig = plt.figure(figsize=(S.FULL_W_IN, H / 25.4))
    inputs = [P("brainmask"), acpc]
    for j, (key, title, cmap) in enumerate(CH):
        if key == "BF":
            bfd = Path(PR["template2template"]) / BF_MAP
            bimg, bmsk = bfd / f"{BF_MAP}_in_mri_template.nii.gz", bfd / f"{BF_MAP}_mask_in_mri_template.nii.gz"
            v, _ = S.canonical(nib.load(str(bimg))); v = v.astype(np.float32); inputs += [bimg, bmsk]
            v[~(S.canonical(nib.load(str(bmsk)))[0] > 0.5)] = np.nan                  # the blockface channel: its own mask
        else:
            v, _ = load(key); v = v.astype(np.float32); inputs.append(P(key))
            v[~brain] = np.nan                                     # display only the template brain mask
        rgb = key == "DEC"
        if rgb:
            v = np.nan_to_num(v / 255.0)
        else:
            lo, hi = np.nanpercentile(v[brain], [1, 99.5])
            if key in ("T2", "BF"):
                v = v / hi; lo, hi = 0.0, 1.0
            elif key == "MD":                                      # window from tissue with values (rim voxels without inputs are 0)
                v = v * 1000.0; core = binary_erosion(brain, iterations=2); w = v[core]; w = w[w > 0.25 * np.median(w)]
                lo, hi = np.percentile(w, [2, 99.5])                  # MD is bimodal: a low rim tail and the tissue mode
            elif key in FIXED:
                lo, hi = FIXED[key]
        x_col = LEFT + j * (colw + GAP) + (BF_GAP if key == "BF" else 0.0)
        y = TOP; im = None
        for r, pl in enumerate(planes):
            r0, r1, c0, c1 = boxes[pl]
            sl = sect(v, pl) if not rgb else np.stack([sect(v[..., k], pl) for k in range(3)], -1)
            pw, ph = ext[pl][0] * scale, rowh[r]
            ax = S.axes_mm(fig, x_col + (colw - pw) / 2, y, pw, ph)
            pad = lambda arr: np.pad(arr, [(max(0, -r0), max(0, r1 - arr.shape[0])), (max(0, -c0), max(0, c1 - arr.shape[1]))]  # noqa: E731
                                     + [(0, 0)] * (arr.ndim - 2), constant_values=0 if rgb else np.nan)
            slp = pad(sl)[r0 + max(0, -r0):r1 + max(0, -r0), c0 + max(0, -c0):c1 + max(0, -c0)]
            if rgb:
                ax.imshow(np.clip(slp, 0, 1) ** (1 / DEC_GAMMA))
            else:                                                 # outside the mask = NaN, drawn as the background (turbo's 0 is not black)
                cm = plt.get_cmap(cmap).copy(); cm.set_bad(S.BG)
                im = ax.imshow(slp, cmap=cm, vmin=lo, vmax=hi)
            ax.set_xticks([]); ax.set_yticks([]); [s_.set_visible(False) for s_ in ax.spines.values()]
            if r == 0:
                S.header(ax, title, dx=0, dy=2)
            if j == 0:
                ax.annotate(S.DIRS[pl][0], xy=(0, 0.5), xycoords="axes fraction", xytext=(-2, 0), textcoords="offset points",
                            ha="right", va="center", color=S.GRAY, annotation_clip=False)      # left letter before the first column
            if j == ncol - 1:                                     # right letter after the last column
                ax.annotate(S.DIRS[pl][1], xy=(1, 0.5), xycoords="axes fraction", xytext=(2 + (colw - pw) / 2 / 25.4 * 72, 0),
                            textcoords="offset points", ha="left", va="center", color=S.GRAY, annotation_clip=False)
            y += ph + GAP
        cax = S.axes_mm(fig, x_col + 0.18 * colw, y + 1.0, 0.64 * colw, 1.6)    # narrower: neighboring tick labels stay apart
        if rgb:
            cax.set_axis_off()
            for k, (txt, c) in enumerate((("Left–right", "#ff4040"), ("Anterior–posterior", "#40d040"), ("Dorsal–ventral", "#8fb0ff"))):
                cax.annotate(txt, xy=(0.5, 1.2 - 1.6 * k), xycoords="axes fraction", ha="center", va="top", color=c, annotation_clip=False)
        else:
            cb = fig.colorbar(im, cax=cax, orientation="horizontal")
            cb.outline.set_edgecolor(S.GRAY); cb.ax.tick_params(length=2, pad=1)
            if key == "MD":                                                # labeled ticks at the two ends of the window
                cb.set_ticks([lo, hi]); cb.ax.set_xticklabels([f"{lo:.2f}", f"{hi:.2f}"])
            else:                                                          # 0-1 bars: short labels "0", "0.5", "1"
                cb.set_ticks([0, 0.5, 1]); cb.ax.set_xticklabels(["0", "0.5", "1"])
            cb.set_label(CBLAB[key], labelpad=1)
    S.scalebar_free(fig, SCALEBAR_MM, scale, (S.MARGIN_MM, LEFT + colw, TOP, H - BOTTOM))   # ONE scale bar: the first column's empty
                                                                                         # corner nearest the bottom left
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    S.save(fig, str(out), inputs, cmd=" ".join(["python"] + sys.argv),
           extra=dict(print_mm_per_tissue_mm=round(scale, 4), sections=pos, scalebar_mm=SCALEBAR_MM, scalebar_place=a.bar,
                      height_mm=round(H, 1),
                      orientation="neurological; sagittal anterior-right; axial anterior-up (figstyle.section)"))
    plt.close(fig)
    print(f"-> {out}  scale {scale:.3f} print mm per tissue mm, height {H:.1f} mm, bar {a.bar}; {pos}")


if __name__ == "__main__":
    main()
