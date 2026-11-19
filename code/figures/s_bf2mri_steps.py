#!/usr/bin/env python3
"""Supplementary figure: the three steps that register a photographed brain's blockface volume to its own b0 image (Methods, the
registration of each brain to its own MRI), for one brain in one sagittal plane, every panel on the same world axes (the b0's frame, mm).

  a  the b0 image, masked to the brain (the 0.5 mm b0 that FLIRT registered to)
  b  step 1, FLIRT (7 then 9 degrees of freedom, correlation ratio, weighted to the photographed slab): the blockface volume through
     the 9-dof matrix (nonlinear_<variant>/<ID>_bf_rigid, 0.25 mm), with the b0's outline
  c  step 2, the crop: the b0 image cropped to the photographed slab, between the two end faces mapped by the 9-dof result
     (<variant>/<ID>_mri_fixed_<variant>); orange, the cut planes (the border between the kept and the removed part of the brain);
     dashed (figstyle's ON_IMG2, secondary ink on images), the whole brain's outline, so the part removed by the cut is the empty part
  d  step 3, the deformable registration (anatomix + MIND-SSC features, FireANTs): the blockface volume after it
     (nonlinear_<variant>/<ID>_bf_nonlin), with the outline of the cut b0 (nonlinear_<variant>/<ID>_mri, 0.25 mm)
Gray scales are linear, 0 to the 99.5th percentile inside each image's own tissue; nothing is smoothed, cut or filled for display.
Outlines are each mask's 0.5 contour on its own grid. One 10 mm bar for all four panels (one world scale), in the last panel's empty
bottom-right corner (under the frontal lobe), its length above it (the author, 12 Nov 2026; before, in (a)'s).
Space (the author, 12 Nov 2026): the view box is the b0 brain's extent here plus BOX_VOX voxels; one gutter (figstyle GUTTER_MM)
between the panels and MARGIN_MM from the outermost ink (the direction letters, the panel letters) to the page's edges.
Orientation letters (the author, 12 Nov 2026; figstyle.orient_letters): P and A beside the row, ORIENT_GAP_MM from the tissue, at the
middle of the row's tissue; the panels widen into the room the letters leave inside the outer panels.

You need (settings/internal/products_blockface.yaml data): sub-<ID>/mri/<variant>/<ID>_b0_masked, <ID>_mri_fixed_<variant>, and
sub-<ID>/mri/nonlinear_<variant>/<ID>_bf_rigid, <ID>_bf_nonlin, <ID>_mri. The brain, its registration variant and the plane:
settings/figures.yaml bf2mri_steps (the plane must be a voxel plane of both grids, 0.5 mm and 0.25 mm: for MM2 4.25 or 4.75 mm).

    python code/figures/s_bf2mri_steps.py --out figures/s_bf2mri_steps.pdf      (and the PNG beside it, and its provenance)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "code"))
import figstyle as FS  # noqa: E402
from shared import paths  # noqa: E402

CFG = yaml.safe_load((HERE.parents[1] / "settings" / "figures.yaml").read_text())["bf2mri_steps"]
LINE = FS.OKABE["sky"]
SB_MM = 10
BOX_VOX = 2                                                         # the view box: the b0 brain's extent plus this many 0.5 mm voxels


def plane(path, x):
    """The sagittal plane at world x of a RAS volume on a diagonal grid: (array [z, y] for origin='lower', imshow extent in world mm, x sampled)."""
    import nibabel as nib
    im = nib.load(str(path)); A = im.affine
    assert nib.aff2axcodes(A) == ("R", "A", "S") and np.allclose(A[:3, :3], np.diag(np.diag(A[:3, :3]))), path
    i = int(round((x - A[0, 3]) / A[0, 0]))
    d = np.asarray(im.dataobj[i], np.float32)
    if d.ndim == 3:
        d = d.mean(-1)
    dy, dz = A[1, 1], A[2, 2]
    y0, z0 = A[1, 3], A[2, 3]
    ext = [y0 - dy / 2, y0 + dy * (d.shape[0] - 0.5), z0 - dz / 2, z0 + dz * (d.shape[1] - 0.5)]
    return d.T, ext, float(A[0, 3] + A[0, 0] * i)


def gray(d):
    m = d > 0
    return np.ma.masked_where(~m, np.clip(d / np.percentile(d[m], 99.5), 0, 1))


def _centers(d, ext):
    ny, nz = d.shape[1], d.shape[0]
    return (np.linspace(ext[0], ext[1], ny + 1)[:-1] + (ext[1] - ext[0]) / ny / 2,
            np.linspace(ext[2], ext[3], nz + 1)[:-1] + (ext[3] - ext[2]) / nz / 2)


def contour(ax, d, ext, **kw):
    y, z = _centers(d, ext)
    ax.contour(y, z, (d > 0).astype(float), levels=[0.5], **kw)


def cut_line(ax, kept, whole, ext, **kw):
    """The cut planes in this plane: the border between the kept and the removed part of the brain (no contour against the background)."""
    y, z = _centers(kept, ext)
    ax.contour(y, z, np.where(whole, (kept & whole).astype(float), np.nan), levels=[0.5], **kw)


def text_room(fig):
    """(side, top): side, the (left, right) room of the outer direction letters beside the tissue (figstyle.orient_room, mm); top,
    the panel letters' height with figstyle.letter's 3 pt offset (the tallest ink above the panels, mm)"""
    r = fig.canvas.get_renderer(); mm = lambda px: px / fig.dpi * 25.4  # noqa: E731
    side = (FS.orient_room(FS.DIRS["sagittal"][0]), FS.orient_room(FS.DIRS["sagittal"][1]))   # P and A: gap + ink (figstyle)
    top = mm(fig.text(0, 0, "abcd", fontsize=FS.LETTER_PT, fontweight="bold").get_window_extent(r).height) + 3.0 / 72 * 25.4
    for t in list(fig.texts):
        t.remove()
    return side, top


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figures/s_bf2mri_steps.pdf")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    FS.setup()
    import matplotlib.pyplot as plt
    S, V, X = CFG["subject"], CFG["variant"], float(CFG["x_mm"])
    M = paths.products("products_blockface")["data"] / f"sub-{S}" / "mri"
    P = dict(b0=M / V / f"{S}_b0_masked.nii.gz", fixed=M / V / f"{S}_mri_fixed_{V}.nii.gz",
             rigid=M / f"nonlinear_{V}" / f"{S}_bf_rigid.nii.gz", nonlin=M / f"nonlinear_{V}" / f"{S}_bf_nonlin.nii.gz",
             fixed25=M / f"nonlinear_{V}" / f"{S}_mri.nii.gz")
    b0, e_b0, xs = plane(P["b0"], X)
    fx, e_fx, _ = plane(P["fixed"], X)
    rg, e_rg, xr = plane(P["rigid"], X)
    nl, e_nl, _ = plane(P["nonlin"], X)
    f25, e_f25, _ = plane(P["fixed25"], X)
    assert abs(xs - xr) < 1e-4, f"the 0.5 mm panels lie at x = {xs}, the 0.25 mm panels at {xr}: choose a plane of both grids"
    zz, yy = np.nonzero(b0 > 0)                                     # one view box for all panels: the b0 brain's extent here + BOX_VOX
    dy, dz = (e_b0[1] - e_b0[0]) / b0.shape[1], (e_b0[3] - e_b0[2]) / b0.shape[0]
    Y0, Y1 = e_b0[0] + dy * (yy.min() - BOX_VOX), e_b0[0] + dy * (yy.max() + 1 + BOX_VOX)
    Z0, Z1 = e_b0[2] + dz * (zz.min() - BOX_VOX), e_b0[2] + dz * (zz.max() + 1 + BOX_VOX)
    ncol, gap, MG = 4, FS.GUTTER_MM, FS.MARGIN_MM
    W = FS.FULL_W_IN / FS.MM
    side, top = text_room(plt.figure(figsize=(W * FS.MM, 1.0))); plt.close("all")
    hdr = MG + top
    panels = [("a", "b0 image", b0, e_b0, []),
              ("b", "1  FLIRT (7, then 9 dof)", rg, e_rg, [(b0, e_b0, dict(colors=[LINE], linewidths=0.6))]),
              ("c", "2  Cropped to photographed slab", fx, e_fx, [(b0, e_b0, dict(colors=[FS.ON_IMG2], linewidths=0.4, linestyles="dashed"))]),
              ("d", "3  Deformable registration", nl, e_nl, [(f25, e_f25, dict(colors=[LINE], linewidths=0.6))])]

    def draw(sl, sr):
        """the row with sl and sr mm beside it (inside the margins): (figure, panels, height mm)"""
        tw = (W - 2 * MG - sl - sr - gap * (ncol - 1)) / ncol
        th = tw * (Z1 - Z0) / (Y1 - Y0)
        H = hdr + th + MG
        fig = plt.figure(figsize=(W * FS.MM, H * FS.MM))
        axs = []
        for k, p_ in enumerate(panels):
            axs.append(panel(fig, k, p_, MG + sl + k * (tw + gap), tw, th))
        return fig, axs, H

    def panel(fig, k, p_, x0, tw, th):
        letter, title, img, ext, outs = p_
        ax = FS.axes_mm(fig, x0, hdr, tw, th)
        ax.set_facecolor(FS.BG)
        ax.imshow(gray(img), extent=ext, origin="lower", cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        for d, e, kw in outs:
            contour(ax, d, e, **kw)
        ax.set_xlim(Y0, Y1); ax.set_ylim(Z0, Z1); ax.set_aspect("equal"); ax.axis("off")
        if letter == "c":
            cut_line(ax, fx > 0, b0 > 0, e_b0, colors=[FS.COL["pred"]], linewidths=1.0)
        FS.letter(ax, letter); FS.header(ax, title)
        if k == ncol - 1:                                             # one bar for all (one scale): the figure's bottom right
            FS.scalebar_in(ax, 1.0, SB_MM, corner="bottom right")         # world-mm axes: 1 unit = 1 mm
        return ax

    sl, sr = side
    for _ in range(4):                                                # the P and A sit by the tissue, which starts inside its panel:
        fig, axs, H = draw(sl, sr)                                    # the panels widen into that room, so the outer letters keep
        cb = FS.content_boxes(fig, [axs[0], axs[-1]])                 # MARGIN_MM from the page's edges
        ol, or_ = (cb[axs[0]][0] - axs[0].get_position().x0) * W, (axs[-1].get_position().x1 - cb[axs[-1]][1]) * W
        if abs(side[0] - ol - sl) < 0.1 and abs(side[1] - or_ - sr) < 0.1:
            break
        plt.close(fig); sl, sr = max(side[0] - ol, 0.0), max(side[1] - or_, 0.0)
    FS.orient_letters(fig, [(axs[0], axs[-1], "sagittal")])           # one level, the row's tissue middle (the author, 12 Nov 2026)
    out = Path(a.out).with_suffix(".pdf")
    FS.save(fig, str(out), [str(p) for p in P.values()], " ".join(["code/figures/s_bf2mri_steps.py"] + sys.argv[1:]),
            extra=dict(subject=S, label=FS.label(S), variant=V, x_mm_b0_grid=xs, x_mm_025_grid=xr, scale_bar_mm=SB_MM,
                       size_mm=[round(W, 1), round(H, 1)],
                       display="linear gray, 0 to the 99.5th percentile inside each image's tissue; outlines = each mask's 0.5 contour on its "
                               "own grid; nearest-neighbor display"))
    fig.savefig(str(out.with_suffix(".png")), dpi=FS.DPI, facecolor=fig.get_facecolor())
    print("->", out, f"{W:.0f} x {H:.1f} mm")


if __name__ == "__main__":
    main()
