#!/usr/bin/env python3
"""The blockface template registered to the MRI template, and the composed registrations (the paper's figure "The blockface template
registered into the MRI template").

    python code/figures/link.py [--out figures/link.png]

The registration of record is the direct symmetric b0 <-> b0 registration between the blockface template (step 8, d2bs8) and the
MRI template, exported as a pull map (template2template/d2bs8_s5sym/bftemplate__to__mri-template). The composed registrations are the
average over the brains of each brain's construction warp composed with its own registration to the MRI template (bftemplate_cv/s5_compose_d2bs8_r05_guard/link_ras_disp.nii.gz). No landmarks.
  a  mid-sagittal (x = 0) and coronal (through the anterior commissure, y = 19.6 mm) sections: the MRI template (T2-weighted) and the
     blockface template's grey channel pulled into it by the registration; on every panel the same two lines, drawn from the MRI
     template only (sky blue: brain-mask outline; orange: gray/white boundary = a two-class (Otsu) split of the T2 inside the mask).
  b  per-voxel distance between the registration and the composed registrations (mm), on the same sections, inside the MRI brain mask.
Two columns, each as wide as its widest cell (its images and the letters beside them; (b)'s coronal with its color bar), one gutter
between them and between the rows, at the print scale that fills the page's width (lowered if the figure would pass MAX_H); coronal
panels centered in their column; the color bar right of the coronal distance map; the 5 mm scale bar, its length above it, at the
column titles' level, top right (figstyle.scalebar_title: it holds for every panel); orientation letters left and right of each row
(figstyle.orient_letters: 1 mm from the brain, one line per row; (b)'s two views each with its pair on the row's line).
Reads (settings/internal/products_mri*.yaml): template_mri/linc_mri_{T2,brainmask}.nii.gz, bftemplate_step8/d2bs8_grey.nii.gz, the two maps
above, numbers_of_record/s5_link_d2bs8.json (recorded as an input: the caption's values). Writes the PNG and its provenance.json.
"""
import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "code"))
import figstyle as S                                           # noqa: E402
from shared import paths                                         # noqa: E402
from blockface.registration import xfmtools as X                    # noqa: E402

BF_MAP = "d2bs8_s5sym"                                         # the registration of record (template2template export)
COMPOSITION = "s5_compose_d2bs8_r05_guard"                     # the composed registrations (bftemplate_cv)
GREY = "d2bs8_grey.nii.gz"                                     # the blockface template's grey channel (bftemplate_step8)
Y_COR = 19.6                                                   # coronal section through the AC (acpc.json, record frame)
VOX = 0.25                                                     # MRI template grid (mm)
CROP_MM = 2.0                                                  # every panel of a row cropped to the MRI brain-mask box + this
DIST_MAX_MM = 2.0                                              # distance color bar maximum
SCALEBAR_MM = 5
GW_SMOOTH = 1.0                                                # Gaussian sigma (voxels) of the T2 before the gray/white contour
PAGE_W, TOP, GAPR = 178.0, 9.0, 9.0                           # mm; TOP, GAPR: where the rows are first drawn, to be measured
MAX_H = 185.0                                                  # the figure's height at most (mm): the page's 231 mm less its caption
CBAR_DX = 7.0                                                  # mm from the coronal distance map to its color bar
TITLES = ("MRI template (T2-weighted)\n", "Blockface template\nregistration")


def plane(kind, img, A):
    """World points of a section on the image's own grid (x = 0 sagittal, or y = Y_COR coronal), plus its 2-D shape and mm extent."""
    sh = img.shape[:3]; ax = lambda k: np.arange(sh[k]) * A[k, k] + A[k, 3]  # noqa: E731
    if kind == "sag":
        yy, zz = np.meshgrid(ax(1), ax(2), indexing="ij"); P = np.stack([np.zeros_like(yy), yy, zz], -1)
        return P.reshape(-1, 3), yy.shape, (ax(1)[0], ax(1)[-1], ax(2)[0], ax(2)[-1])
    xx, zz = np.meshgrid(ax(0), ax(2), indexing="ij"); P = np.stack([xx, np.full_like(xx, Y_COR), zz], -1)
    return P.reshape(-1, 3), xx.shape, (ax(0)[0], ax(0)[-1], ax(2)[0], ax(2)[-1])


def sample(vol, aff, P, order=1):
    V = np.linalg.inv(aff); v = (P @ V[:3, :3].T + V[:3, 3]).T
    return map_coordinates(vol, v, order=order, cval=0)


def otsu(v, bins=256):
    """Two-class Otsu threshold of the values v."""
    h, e = np.histogram(v, bins); c = 0.5 * (e[1:] + e[:-1]); w = np.cumsum(h); mu = np.cumsum(h * c)
    return float(c[np.argmax((mu[-1] / w[-1] * w - mu) ** 2 / np.maximum(w * (w[-1] - w), 1e-12))])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figures/link.png")
    a = ap.parse_args()
    PR = {**paths.products("products_mri"), **paths.products("products_mri_figures")}
    T2P, MSKP = PR["template_mri"] / "linc_mri_T2.nii.gz", PR["template_mri"] / "linc_mri_brainmask.nii.gz"
    GREYP = PR["bftemplate_step8"] / GREY
    LINK = PR["template2template"] / BF_MAP / "bftemplate__to__mri-template"
    COMP = PR["bftemplate_cv"] / COMPOSITION
    NUM = PR["numbers_of_record"] / "s5_link_d2bs8.json"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S.setup()
    t2i = nib.load(str(T2P)); T2 = np.asanyarray(t2i.dataobj).astype(np.float32)
    M = (np.asanyarray(nib.load(str(MSKP)).dataobj) > 0.5).astype(np.float32)
    gi = nib.load(str(GREYP)); G = np.asanyarray(gi.dataobj).astype(np.float32)
    link = X.load_map(str(LINK), "full")
    D = nib.load(str(COMP / "link_ras_disp.nii.gz")); Dd = np.asanyarray(D.dataobj).astype(np.float32); Dd = Dd.reshape(Dd.shape[:3] + (3,))
    gw = otsu(T2[M > 0.5])
    rows = {}
    for kind in ("sag", "cor"):
        P, shp, ext = plane(kind, t2i, t2i.affine)
        m = sample(M, t2i.affine, P) > 0.5
        xs = np.asarray(link(P)); xc = P + np.stack([sample(Dd[..., k], D.affine, P) for k in range(3)], 1)
        rows[kind] = dict(shp=shp, ext=ext, m=m.reshape(shp), t2=sample(T2, t2i.affine, P).reshape(shp),
                          sym=sample(G, gi.affine, xs).reshape(shp),
                          dist=np.where(m, np.linalg.norm(xs - xc, axis=1), np.nan).reshape(shp))
    for r in rows.values():                                    # the same crop for all panels of a row
        ij = np.argwhere(r["m"]); lo, hi = ij.min(0), ij.max(0); px = int(round(CROP_MM / VOX))
        lo = np.maximum(lo - px, 0); hi = np.minimum(hi + px, np.array(r["shp"]) - 1); r["crop"] = np.s_[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1]
        r["mm"] = ((hi[0] - lo[0] + 1) * VOX, (hi[1] - lo[1] + 1) * VOX)
    # ---- layout (the author, 12 Nov 2026: "optimize space"): two columns, each as wide as its widest cell (images and the letters
    # beside them; (b)'s coronal with its color bar), one gutter between them and between the rows, the page margins outside; the
    # print scale the largest that fills the width, lowered if the figure would be taller than MAX_H
    import matplotlib.transforms as mtrans
    G_, M_ = S.GUTTER_MM, S.MARGIN_MM

    def render(scale, Xc=None, T=None, Hf=None):
        hs, hc = rows["sag"]["mm"][1] * scale, rows["cor"]["mm"][1] * scale
        Xc = Xc or (PAGE_W / 4, 3 * PAGE_W / 4)                           # the two columns' centers (mm)
        T = T or (TOP, TOP + hs + GAPR, TOP + hs + GAPR + hc + GAPR + 4)  # the three rows' tops (mm)
        Hf = Hf or T[2] + max(hs, hc) + 9
        fig = plt.figure(figsize=(S.FULL_W_IN, Hf / 25.4), facecolor=S.BG)
        col, row, first, pairs = ([], []), ([], [], []), {}, ([], [], [])
        for r_, (kind, plane_name) in enumerate((("sag", "sagittal"), ("cor", "coronal"))):
            r = rows[kind]; w, h = r["mm"][0] * scale, r["mm"][1] * scale
            for j, key in enumerate(("t2", "sym")):
                ax = S.axes_mm(fig, Xc[j] - w / 2, T[r_], w, h)
                a_ = r[key][r["crop"]]; mm = r["m"][r["crop"]]
                vmax = np.percentile(a_[mm], 99.5) if mm.any() else 1
                ax.imshow(a_.T, cmap="gray", origin="lower", vmin=0, vmax=vmax, interpolation="nearest", aspect="equal")
                ax.contour(mm.T.astype(float), [0.5], colors=[S.OKABE["sky"]], linewidths=0.5)
                g = ndi.gaussian_filter(r["t2"][r["crop"]], GW_SMOOTH) * mm
                ax.contour(g.T, [gw], colors=[S.OKABE["orange"]], linewidths=0.5)
                ax.set_xticks([]); ax.set_yticks([]); [s_.set_visible(False) for s_ in ax.spines.values()]
                if kind == "sag":
                    ax.set_title(TITLES[j], color=S.FG, fontsize=S.MIN_PT, pad=2)
                if j == 0:
                    first[kind] = ax
                last = ax; col[j].append(ax); row[r_].append(ax)
            pairs[r_].append((first[kind], last, plane_name))       # direction letters on the row's outer edges only
        S.letter(first["sag"], "a", dx=-14, dy=10)
        b_axes, head = [], None
        for k, (kind, plane_name) in enumerate((("sag", "sagittal"), ("cor", "coronal"))):
            r = rows[kind]; w, h = r["mm"][0] * scale, r["mm"][1] * scale
            ax = S.axes_mm(fig, Xc[k] - w / 2, T[2], w, h)
            im = ax.imshow(r["dist"][r["crop"]].T, cmap="magma", origin="lower", vmin=0, vmax=DIST_MAX_MM, interpolation="nearest",
                           aspect="equal")
            ax.contour(r["m"][r["crop"]].T.astype(float), [0.5], colors=[S.OKABE["sky"]], linewidths=0.5)
            ax.set_xticks([]); ax.set_yticks([]); [s_.set_visible(False) for s_ in ax.spines.values()]
            pairs[2].append((ax, ax, plane_name))
            if k == 0:
                S.letter(ax, "b", dx=-14, dy=4)
                head = S.header(ax, "Distance between the registration and the composed registrations", dx=-3, dy=4)
                head.set_in_layout(False)                              # it runs over both columns: not a column's width
            col[k].append(ax); row[2].append(ax); b_axes.append(ax)
        wcor = rows["cor"]["mm"][0] * scale
        cax = S.axes_mm(fig, Xc[1] + wcor / 2 + CBAR_DX, T[2] + 0.15 * max(hs, hc), 3.0, 0.7 * max(hs, hc))
        cb = fig.colorbar(im, cax=cax); cb.set_label("Distance (mm)", color=S.FG, fontsize=S.MIN_PT)
        cb.ax.tick_params(colors=S.FG, labelsize=S.MIN_PT); cb.outline.set_edgecolor(S.FRAME)
        col[1].append(cax); row[2].append(cax)
        return fig, col, row, b_axes, head, pairs

    def ext(fig, axs, extra=()):
        r_ = fig.canvas.get_renderer(); Hf = fig.get_size_inches()[1] * 25.4
        bb = mtrans.Bbox.union([a_.get_tightbbox(r_) for a_ in axs] + [t.get_window_extent(r_) for t in extra])
        return bb.x0 / fig.dpi * 25.4, bb.x1 / fig.dpi * 25.4, Hf - bb.y1 / fig.dpi * 25.4, Hf - bb.y0 / fig.dpi * 25.4

    def lext(fig, j, col, pairs):
        """Column j's extent with the room its orientation letters take beside its images (orient_room; they are placed after the
        layout)."""
        x0, x1, y0, y1 = ext(fig, col[j]); r_ = fig.canvas.get_renderer(); f = lambda v: v / fig.dpi * 25.4    # noqa: E731
        for axl, axr, pl in (p_ for row_ in pairs for p_ in row_):
            if axl in col[j]:
                x0 = min(x0, f(axl.get_window_extent(r_).x0) - S.orient_room(S.DIRS[pl][0]))
            if axr in col[j]:
                x1 = max(x1, f(axr.get_window_extent(r_).x1) + S.orient_room(S.DIRS[pl][1]))
        return x0, x1, y0, y1

    def measure(scale):
        fig, col, row, _, head, pairs = render(scale)
        X0 = (PAGE_W / 4, 3 * PAGE_W / 4)
        sp = [lext(fig, j, col, pairs) for j in range(2)]
        T0 = (TOP, TOP + rows["sag"]["mm"][1] * scale + GAPR, TOP + (rows["sag"]["mm"][1] + rows["cor"]["mm"][1]) * scale + 2 * GAPR + 4)
        rx = [ext(fig, row[k], (head,) if k == 2 else ()) for k in range(3)]
        plt.close(fig)
        W_ = [x1 - x0 for x0, x1, _, _ in sp]; off = [X0[j] - sp[j][0] for j in range(2)]          # column center from its left
        Xc = (M_ + off[0], M_ + W_[0] + G_ + off[1])
        above = [T0[k] - rx[k][2] for k in range(3)]; below = [rx[k][3] - T0[k] for k in range(3)]
        t1 = M_ + above[0]; t2 = t1 + below[0] + G_ + above[1]; t3 = t2 + below[1] + G_ + above[2]
        return W_[0] + G_ + W_[1], Xc, (t1, t2, t3), t3 + below[2] + M_

    s0 = 1.0; w0 = measure(s0)[0]; w1 = measure(1.2)[0]
    scale = s0 + (PAGE_W - 2 * M_ - w0) / ((w1 - w0) / 0.2)            # the content is affine in the scale: fill the width
    wd, Xc, T, Hf = measure(scale)
    if Hf > MAX_H:                                                     # the height binds
        H0 = measure(0.9 * scale)[3]; scale = 0.9 * scale + (MAX_H - H0) / ((Hf - H0) / (0.1 * scale)); wd, Xc, T, Hf = measure(scale)
    fig, col, row, b_axes, head, pairs = render(scale, Xc, T, Hf)
    S.orient_letters(fig, [pairs[0][0], pairs[1][0], pairs[2]])        # one line per row, 1 mm from the brain
    S.scalebar_title(col[1][0], 1 / VOX, SCALEBAR_MM, col[1][0].title)   # top right, at the column titles' level: for every panel
    hs, hc = rows["sag"]["mm"][1] * scale, rows["cor"]["mm"][1] * scale
    S.save(fig, str(a.out), [T2P, MSKP, GREYP, LINK / "maps.json", LINK / "full_d.nii.gz", COMP / "link_ras_disp.nii.gz", NUM],
           cmd=" ".join(["python"] + sys.argv),
           extra=dict(print_mm_per_tissue_mm=round(scale, 3), scalebar_mm=SCALEBAR_MM,
                      sections=dict(sagittal="x = 0 mm", coronal=f"y = {Y_COR} mm (AC)"), gray_white_level=gw,
                      distance_colorbar_max_mm=DIST_MAX_MM, link=str(LINK), composition=str(COMP)))
    plt.close(fig); print("->", a.out, "scale", round(scale, 3))


if __name__ == "__main__":
    main()
