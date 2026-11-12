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
Two equal columns (the sagittal width) at up to the whole-brain print scale; the free width is split evenly into left margin, gap and
right margin; coronal panels centered in their column; the color bar right of the coronal distance map.
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
SB_BELOW = 3.4                                                 # the scale bar's line under (b)'s first panel (mm): its length fits above
GW_SMOOTH = 1.0                                                # Gaussian sigma (voxels) of the T2 before the gray/white contour
PAGE_W, RIGHT_M, TOP, GAPR = 178.0, 8.0, 9.0, 9.0              # mm; RIGHT_M: room for the A direction letter when fixing the scale
SCALE_LEFT, SCALE_GAP = 9.0, 3.0                               # mm: the margins the print scale is fitted with
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
    scale = min(S.SCALE["whole"], (PAGE_W - SCALE_LEFT - SCALE_GAP - RIGHT_M) / (2 * rows["sag"]["mm"][0]))
    hs, hc = rows["sag"]["mm"][1] * scale, rows["cor"]["mm"][1] * scale
    C = rows["sag"]["mm"][0] * scale; LEFT = GAP = (PAGE_W - 2 * C) / 3      # two equal columns, free width split evenly
    H = TOP + hs + GAPR + hc + GAPR + 4 + max(hs, hc) + 9
    fig = plt.figure(figsize=(S.FULL_W_IN, H / 25.4), facecolor=S.BG)
    y = TOP; first = {}
    for kind, plane_name in (("sag", "sagittal"), ("cor", "coronal")):
        r = rows[kind]; w, h = r["mm"][0] * scale, r["mm"][1] * scale
        for j, key in enumerate(("t2", "sym")):
            ax = S.axes_mm(fig, LEFT + j * (C + GAP) + (C - w) / 2, y, w, h)
            a_ = r[key][r["crop"]]; mm = r["m"][r["crop"]]
            vmax = np.percentile(a_[mm], 99.5) if mm.any() else 1
            ax.imshow(a_.T, cmap="gray", origin="lower", vmin=0, vmax=vmax, interpolation="nearest", aspect="equal")
            ax.contour(mm.T.astype(float), [0.5], colors=[S.OKABE["sky"]], linewidths=0.5)
            g = ndi.gaussian_filter(r["t2"][r["crop"]], GW_SMOOTH) * mm
            ax.contour(g.T, [gw], colors=[S.OKABE["orange"]], linewidths=0.5)
            ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
            if kind == "sag":
                ax.set_title(TITLES[j], color=S.FG, fontsize=S.MIN_PT, pad=2)
            if j == 0:
                first[kind] = ax
            last = ax
        S.dirs_row(first[kind], last, plane_name)              # direction letters on the row's outer edges only
        y += h + GAPR
    S.letter(first["sag"], "a", dx=-14, dy=10)
    y += 4
    for k, (kind, plane_name) in enumerate((("sag", "sagittal"), ("cor", "coronal"))):
        r = rows[kind]; w, h = r["mm"][0] * scale, r["mm"][1] * scale
        ax = S.axes_mm(fig, LEFT + k * (C + GAP) + (C - w) / 2, y, w, h)
        im = ax.imshow(r["dist"][r["crop"]].T, cmap="magma", origin="lower", vmin=0, vmax=DIST_MAX_MM, interpolation="nearest",
                       aspect="equal")
        ax.contour(r["m"][r["crop"]].T.astype(float), [0.5], colors=[S.OKABE["sky"]], linewidths=0.5)
        ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
        S.dirs(ax, plane_name)
        if k == 0:
            S.letter(ax, "b", dx=-14, dy=4)
            S.header(ax, "Distance between the registration and the composed registrations", dx=-3, dy=4)
            S.scalebar(ax, 1 / VOX, SCALEBAR_MM, y_frac=SB_BELOW / h)      # its length above it, clear of the frame
    cax = S.axes_mm(fig, LEFT + C + GAP + (C + rows["cor"]["mm"][0] * scale) / 2 + CBAR_DX, y + 0.15 * max(hs, hc), 3.0,
                    0.7 * max(hs, hc))
    cb = fig.colorbar(im, cax=cax); cb.set_label("distance (mm)", color=S.FG, fontsize=S.MIN_PT)
    cb.ax.tick_params(colors=S.FG, labelsize=S.MIN_PT); cb.outline.set_edgecolor(S.GRAY)
    S.save(fig, str(a.out), [T2P, MSKP, GREYP, LINK / "maps.json", LINK / "full_d.nii.gz", COMP / "link_ras_disp.nii.gz", NUM],
           cmd=" ".join(["python"] + sys.argv),
           extra=dict(print_mm_per_tissue_mm=round(scale, 3), scalebar_mm=SCALEBAR_MM,
                      sections=dict(sagittal="x = 0 mm", coronal=f"y = {Y_COR} mm (AC)"), gray_white_level=gw,
                      distance_colorbar_max_mm=DIST_MAX_MM, link=str(LINK), composition=str(COMP)))
    plt.close(fig); print("->", a.out, "scale", round(scale, 3))


if __name__ == "__main__":
    main()
