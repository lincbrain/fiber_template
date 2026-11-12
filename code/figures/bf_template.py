#!/usr/bin/env python3
"""The paper's figure "The blockface template in MRI space".

    python code/figures/bf_template.py [--out figures/bf_template.png]

Columns: (a) grayscale, (b) color (RGB), (c) photographed fraction (<arm>_photo_share: the fraction of each voxel's value that comes from
photographs), over the dimmed grayscale template: each voxel blends orange (fraction 0) into the dimmed gray anatomy (fraction 1), so only
tissue that is not fully photographed is colored. Rows: sagittal, coronal and axial sections through one point, the sections of the
MRI channels figure (NMT v2 x = 2, y = 19.5, z = 15.5 mm, transformed into the blockface template by the link's pull map), in the stated
convention (sagittal anterior right; coronal and axial neurological; axial anterior up), the template's own 0.25 mm voxels (nothing
resampled; bilinear display). Row headers give the NMT v2 coordinates (mm) of the sections, carried through the template's link of
record. Each view is cropped to its own section's brain box + 0.5 mm, and one print scale makes the three columns and the color bar
fill the page width; one 10 mm scale bar, its length above it, at the figure's bottom-right corner. Inputs: the template of record and its link (settings/internal/products_blockface.yaml `bftemplate_step8`, `template_link`).
Display parameters: settings/figures.yaml `bf_template`.
Ported from the analysis project's paper/scripts/fig7_bf_template.py, the path the paper's figure was made with
(--simple --no-d --fill on design 2b's step 8 and its link); the other variants of that script are not ported.
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
ARM = "d2bs8"                                 # the template of record's file prefix (design 2b, step 8)
NMT_SECTIONS = tuple(CFG["nmt_sections_mm"])  # the MRI channels figure's sections (NMT v2 mm)
MARGIN = tuple(CFG["margin_mm"])              # mm of black around the brain box (x, y, z) ...
POST = CFG["post_mm"]                         # ... and extra mm behind the occipital pole (the 3D box the views are cut from)
FILL_MARGIN = CFG["fill_margin_mm"]           # each view: its own section's brain box + this
CBW = CFG["colorbar_mm"]                      # the color bar, its ticks and its title (mm)
SB_MARGIN = 2.0                                # the scale bar's distance from the figure's right and bottom edges (mm)
JOIN = "0.5 contour of photo_share within 1 mm of predicted tissue >= ~1 mm thick (opening x2, dilation x4)"   # provenance text, verbatim


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figures/bf_template.png")
    a = ap.parse_args()
    prod = paths.products("products_blockface")
    link_rel = yaml.safe_load((REPO / "settings" / "internal" / "products_blockface.yaml").read_text())["template_link"]
    T, link = prod["bftemplate_step8"], prod["template_link"]
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
    mm = lambda v: f"{v:.1f}".replace("-", "−")                   # noqa: E731
    heads = dict(sagittal=f"sagittal, x = {mm(nmt[0])} mm", coronal=f"coronal, y = {mm(nmt[1])} mm",
                 axial=f"axial, z = {mm(nmt[2])} mm")

    ws = (FS.FULL_W_IN / FS.MM - 7.0 - 2 * 3.0 - CBW) / (3 * max(v[0] for v in size.values()))   # columns + color bar fill the page
    colw = max(v[0] for v in size.values()) * ws; GAP, LEFT, ROWGAP, TOP = 3.0, 7.0, 6.0, 9.5
    y0 = {}; y = TOP
    for plane in ("sagittal", "coronal", "axial"):
        y0[plane] = y; y += size[plane][1] * ws + ROWGAP
    fig = plt.figure(figsize=(FS.FULL_W_IN, (y - ROWGAP + 6.0) * FS.MM))
    cols = [("a", "grayscale"), ("b", "color"), ("c", "photographed fraction")]
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
            if c == 0:
                FS.header(ax, heads[plane], dx=0, dy=2)
            if r == 0:
                ax.annotate(let, xy=(0, 1), xycoords="axes fraction", xytext=(0, 13), textcoords="offset points",
                            fontsize=FS.LETTER_PT, fontweight="bold", ha="left", va="bottom", annotation_clip=False)
                ax.annotate(cname, xy=(0, 1), xycoords="axes fraction", xytext=(11, 13), textcoords="offset points",
                            fontsize=FS.MIN_PT, ha="left", va="bottom", annotation_clip=False)
        FS.dirs_row(axes[(plane, 0)], axes[(plane, 2)], plane, y=0.85 if plane == "sagittal" else 0.5)
    pc = axes[("coronal", 2)].get_position(); fw, fh = fig.get_size_inches() * 25.4
    cx = (LEFT + 2 * (colw + GAP) + colw + 2.5) / fw
    cax = fig.add_axes([cx, pc.y0 + 0.1 * pc.height, 2.2 / fw, 0.8 * pc.height])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)), cax=cax, ticks=[0, 0.5, 1])
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(width=0.5, length=2); cb.ax.set_yticklabels(["0", "0.5", "1"])
    cb.set_label("photographed fraction")
    FS.scalebar_mm(fig, fw - SB_MARGIN, fh - SB_MARGIN, 10.0, ws, anchor="right")   # ONE scale bar, at the figure's bottom-right corner
    ins =[str(T / f"{ARM}_{n}.nii.gz") for n in ("grey", "rgb", "brainmask", "photo_share", "real_eff")]
    FS.save(fig, a.out, ins + [str(link / "mri-template__to__bftemplate" / "maps.json")],
            " ".join(["code/figures/bf_template.py"] + sys.argv[1:]),
            extra=dict(template=ARM, slices_world_mm=dict(sagittal_x=xs, coronal_y=yc, axial_z=za), symmetry=None,
                       slices_nmt_v2_mm=[round(float(v), 2) for v in nmt], nmt_via=link_rel,
                       print_scale=ws, no_d=True, fill=True, join=JOIN, simple=True, ventral_band=None))
    print(a.out, "| NMT v2 of the sections' intersection:", np.round(nmt, 2), "| print scale", round(ws, 4))


if __name__ == "__main__":
    main()
