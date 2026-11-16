#!/usr/bin/env python3
"""The paper's figure "The blockface template in MRI space".

    python code/figures/bf_template.py [--out figures/bf_template.png]

Columns: (a) grayscale, (b) color (RGB), (c) photographed fraction (<arm>_photo_share: the fraction of each voxel's value that comes from
photographs), over the dimmed grayscale template: each voxel blends orange (fraction 0) into the dimmed gray anatomy (fraction 1), so only
tissue that is not fully photographed is colored. Rows: sagittal, coronal and axial sections through one point, the sections of the
MRI channels figure (NMT v2 x = 2, y = 19.5, z = 15.5 mm, transformed into the blockface template by the link's pull map), in the stated
convention (sagittal anterior right; coronal and axial neurological; axial anterior up), the template's own 0.15 mm voxels (step 9;
nothing resampled; bilinear display). No row headers: the caption names the planes and the direction letters show the view (the author,
12 Nov 2026), left of (a) and right of (c) on one line per row, at the middle of the row's brains, 1 mm from them
(figstyle.orient_letters); the sections' NMT v2 coordinates (mm), carried through the template's link of record, are in the provenance. Each view is cropped to its own section's brain box + 0.5
mm, and one print scale makes the three columns and the color bar fill the page width (or, when the figure would be taller, its
max_height_mm), with one gutter between the columns and between the rows; one 10 mm scale bar, its length above it, at the bottom right.
The color bar (--colorbar): right = vertical, in its own column beside (c), the bar at the figure's bottom-right corner beside the last
row; below = horizontal under (c), the freed column given to the images, the bar in the empty place nearest the bottom right
(figstyle.scalebar_free).
Inputs: the template of record and its link (settings/internal/products_blockface.yaml `bftemplate_step9`, `template_link`; step 9 keeps
step 8's frame, so step 8's link applies).
Display parameters: settings/figures.yaml `bf_template`.
Ported from the analysis project's paper/scripts/fig7_bf_template.py, the path the paper's figure was made with
(--simple --no-d --fill on design 2b's step 8 and its link); the other variants of that script are not ported. Since 15 Nov 2026 it shows
the step-9 template (the author adopted it).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO / "code"))
import figstyle as FS  # noqa: E402
from shared import paths  # noqa: E402
from blockface.registration import xfmtools as X  # noqa: E402

CFG = yaml.safe_load((REPO / "settings" / "figures.yaml").read_text())["bf_template"]
ARM = paths.blockface_template_names()["own"]   # the template of record's file prefix (settings/blockface_template.yaml step9)
NMT_SECTIONS = tuple(CFG["nmt_sections_mm"])  # the MRI channels figure's sections (NMT v2 mm)
MARGIN = tuple(CFG["margin_mm"])              # mm of black around the brain box (x, y, z) ...
POST = CFG["post_mm"]                         # ... and extra mm behind the occipital pole (the 3D box the views are cut from)
FILL_MARGIN = CFG["fill_margin_mm"]           # each view: its own section's brain box + this
CB_GAP, CB_GAP_BELOW, CB_T = 2.5, 1.5, 2.2      # the color bar: its gap beside (c), its gap under (c), its thickness (mm)
JOIN = "0.5 contour of photo_share within 1 mm of predicted tissue >= ~1 mm thick (opening x2, dilation x4)"   # provenance text, verbatim


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figures/bf_template.png")
    ap.add_argument("--colorbar", choices=["right", "below"], default="right",
                    help="the photographed-fraction color bar: vertical beside (c) (right) or horizontal under (c) (below)")
    a = ap.parse_args()
    prod = paths.products("products_blockface")
    link_rel = yaml.safe_load((REPO / "settings" / "internal" / "products_blockface.yaml").read_text())["template_link"]
    T, link = prod["bftemplate_step9"], prod["template_link"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FS.setup()
    load = lambda n: nib.load(str(T / f"{ARM}_{n}.nii.gz"))              # noqa: E731

    g = load("grey"); A = g.affine
    assert nib.aff2axcodes(A) == ("R", "A", "S") and np.allclose(A[:3, :3], np.diag(np.diag(A[:3, :3]))), "expects a RAS grid"
    vox = float(A[0, 0])
    B = np.asanyarray(load("brainmask").dataobj) > 0                  # shown inside the template's brain mask (coverage >= half the
    G0 = np.asanyarray(g.dataobj).astype(np.float32)                  # members); outside it are sparse low-coverage voxels
    G = G0 * B
    RGB = np.clip(np.asanyarray(load("rgb").dataobj).astype(np.float32), 0, 1) * B[..., None]
    PS = np.asanyarray(load("photo_share").dataobj).astype(np.float32)
    # NMT v2 -> blockface template through the link's pull map
    xs, yc, za = (float(v) for v in np.asarray(X.load_map(str(link / "bftemplate__to__mri-template"), "full")(
        np.array([NMT_SECTIONS], float)))[0])
    nmt = FS.to_nmt(str(link), [[xs, yc, za]])[0]                     # the sections' intersection, NMT v2 mm (template link)
    assert np.abs(nmt - np.array(NMT_SECTIONS)).max() < 0.05, f"link round trip {nmt} vs {NMT_SECTIONS}"
    ijk = lambda axis, w: int(round((w - A[axis, 3]) / A[axis, axis]))  # noqa: E731
    i, j, k = ijk(0, xs), ijk(1, yc), ijk(2, za)
    bb = np.argwhere(B); m = [int(round(v / vox)) for v in MARGIN]
    lo = np.maximum(bb.min(0) - m, 0); lo[1] = max(lo[1] - int(round(POST / vox)), 0)
    hi = np.minimum(bb.max(0) + 1 + m, B.shape)
    crop = lambda V: V[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]          # noqa: E731
    idx = dict(sagittal=i - lo[0], coronal=j - lo[1], axial=k - lo[2])
    view0 = lambda V, plane: FS.section(crop(V), plane, idx[plane])    # noqa: E731
    size, sl2 = {}, {}
    m2 = int(round(FILL_MARGIN / vox))
    for pl in ("sagittal", "coronal", "axial"):                        # each view cropped to its own section's brain box
        bs_ = view0(B, pl); rr_ = np.where(bs_.any(1))[0]; cc_ = np.where(bs_.any(0))[0]
        r0_, r1_ = max(rr_[0] - m2, 0), min(rr_[-1] + 1 + m2, bs_.shape[0]); c0_, c1_ = max(cc_[0] - m2, 0), min(cc_[-1] + 1 + m2, bs_.shape[1])
        sl2[pl] = (slice(r0_, r1_), slice(c0_, c1_)); size[pl] = ((c1_ - c0_) * vox, (r1_ - r0_) * vox)
    view = lambda V, plane: view0(V, plane)[sl2[plane]]                # noqa: E731
    # the space (the author, 12 Nov 2026): one gutter between the columns and between a row and the next row's header, the left
    # margin as wide as the direction letters need, the page margin below the last row
    f_ = plt.figure(); r_ = f_.canvas.get_renderer()
    hl = f_.text(0, 0, "a", fontsize=FS.LETTER_PT, fontweight="bold").get_window_extent(r_).height / f_.dpi * 25.4
    txt = lambda s_: f_.text(0, 0, s_, fontsize=FS.MIN_PT).get_window_extent(r_)                       # noqa: E731
    tick_mm = (2.0 + 3.5 + 4.0) / 72 * 25.4                          # tick length, tick pad, label pad (pt)
    if a.colorbar == "right":   # the color bar's column: its gap, width, ticks, tick labels and title (rotated), and the page margin
        RIGHT = (CB_GAP + CB_T + tick_mm + txt("0.5").width / f_.dpi * 25.4 + txt("Photographed fraction").height / f_.dpi * 25.4
                 + FS.MARGIN_MM)
        cbh = 0.0
    else:                        # under (c): its gap, thickness, ticks, tick labels and title; the right letters as the left ones
        RIGHT = FS.MARGIN_MM + FS.orient_room("AR")
        cbh = CB_GAP_BELOW + CB_T + tick_mm + (txt("0.5").height + txt("Photographed fraction").height) / f_.dpi * 25.4
    plt.close(f_)
    GAP = FS.GUTTER_MM
    TOP = FS.MARGIN_MM + 2.0 / 72 * 25.4 + hl                          # the column headers 2 pt above the first row
    LEFT = FS.MARGIN_MM + FS.orient_room("PL")                       # the left letters, 1 mm from the first column's brain
    ROWGAP = GAP                                                       # no row headers: the caption names the planes
    ws = (FS.FULL_W_IN / FS.MM - LEFT - 2 * GAP - RIGHT) / (3 * max(v[0] for v in size.values()))  # columns + color bar fill the page
    hsum = sum(size[pl][1] for pl in ("sagittal", "coronal", "axial"))
    ws = min(ws, (CFG["max_height_mm"] - TOP - 2 * ROWGAP - cbh - FS.MARGIN_MM) / hsum)            # ... or its height binds
    colw = max(v[0] for v in size.values()) * ws
    y0 = {}; y = TOP
    for plane in ("sagittal", "coronal", "axial"):
        y0[plane] = y; y += size[plane][1] * ws + ROWGAP
    y_end = y - ROWGAP                                                 # the last row's bottom
    fig = plt.figure(figsize=(FS.FULL_W_IN, (y_end + cbh + FS.MARGIN_MM) * FS.MM))
    cols = [("a", "Grayscale"), ("b", "Color"), ("c", "Photographed fraction")]
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    ORANGE = np.array(to_rgb(FS.COL["pred"])); DIM = (0.18, 0.62)            # (c): orange -> the dimmed gray anatomy
    cmap = LinearSegmentedColormap.from_list("frac", [FS.COL["pred"], (0.42, 0.42, 0.42)]); axes = {}
    for r, plane in enumerate(("sagittal", "coronal", "axial")):
        w, h = (v * ws for v in size[plane])
        for c, (let, cname) in enumerate(cols):
            x = LEFT + c * (colw + GAP) + (colw - w) / 2               # panels centered in their column
            ax = FS.axes_mm(fig, x, y0[plane], w, h); axes[(plane, c)] = ax
            if c == 0:
                ax.imshow(view(G, plane), cmap="gray", vmin=0, vmax=1)
            elif c == 1:
                ax.imshow(view(RGB, plane))
            else:
                f_ = np.clip(view(PS, plane), 0, 1)[..., None]; gr = (DIM[0] + (DIM[1] - DIM[0]) * view(G, plane))[..., None]
                ax.imshow((f_ * gr + (1 - f_) * ORANGE) * view(B, plane)[..., None])
            ax.set_axis_off()
            if r == 0:
                ax.annotate(let, xy=(0, 1), xycoords="axes fraction", xytext=(0, 2), textcoords="offset points",
                            fontsize=FS.LETTER_PT, fontweight="bold", ha="left", va="bottom", annotation_clip=False)
                ax.annotate(cname, xy=(0, 1), xycoords="axes fraction", xytext=(11, 2), textcoords="offset points",
                            fontsize=FS.MIN_PT, ha="left", va="bottom", annotation_clip=False)
    FS.orient_letters(fig, [(axes[(pl, 0)], axes[(pl, 2)], pl) for pl in ("sagittal", "coronal", "axial")])   # one line per row,
    pc = axes[("coronal", 2)].get_position(); fw, fh = fig.get_size_inches() * 25.4                        # its brains' middle
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    if a.colorbar == "right":
        cx = (LEFT + 2 * (colw + GAP) + colw + CB_GAP) / fw
        cax = fig.add_axes([cx, pc.y0 + 0.1 * pc.height, CB_T / fw, 0.8 * pc.height])
        cb = fig.colorbar(sm, cax=cax, ticks=[0, 0.5, 1])
        cb.outline.set_linewidth(0.4); cb.ax.tick_params(width=0.5, length=2); cb.ax.set_yticklabels(["0", "0.5", "1"])
    else:                                                              # under (c), centered, as wide as Fig. 1's color bars
        xc0 = LEFT + 2 * (colw + GAP)
        cax = FS.axes_mm(fig, xc0 + 0.18 * colw, y_end + CB_GAP_BELOW, 0.64 * colw, CB_T)
        cb = fig.colorbar(sm, cax=cax, ticks=[0, 0.5, 1], orientation="horizontal")
        cb.outline.set_linewidth(0.4); cb.ax.tick_params(width=0.5, length=2); cb.ax.set_xticklabels(["0", "0.5", "1"])
    cb.set_label("Photographed fraction")
    if a.colorbar == "right":                                        # ONE scale bar: the bottom-right corner, beside the last row
        FS.scalebar_mm(fig, fw - FS.MARGIN_MM, y_end - 0.3, 10.0, ws, anchor="right")   # (the color bar's column is empty there)
    else:                                                              # the empty place nearest the bottom right
        FS.scalebar_free(fig, 10.0, ws, (FS.MARGIN_MM, fw - FS.MARGIN_MM, TOP, fh - FS.MARGIN_MM), corner="bottom right")
    ins =[str(T / f"{ARM}_{n}.nii.gz") for n in ("grey", "rgb", "brainmask", "photo_share")]
    FS.save(fig, a.out, ins + [str(link / "mri-template__to__bftemplate" / "maps.json")],
            " ".join(["code/figures/bf_template.py"] + sys.argv[1:]),
            extra=dict(template=ARM, slices_world_mm=dict(sagittal_x=xs, coronal_y=yc, axial_z=za), symmetry=None,
                       slices_nmt_v2_mm=[round(float(v), 2) for v in nmt], nmt_via=link_rel,
                       print_scale=ws, no_d=True, fill=True, join=JOIN, simple=True, ventral_band=None, colorbar=a.colorbar,
                       height_mm=round(fh, 1)))
    print(a.out, "| NMT v2 of the sections' intersection:", np.round(nmt, 2), "| print scale", round(ws, 4), "| height", round(fh, 1),
          "mm | color bar", a.colorbar)


if __name__ == "__main__":
    main()
