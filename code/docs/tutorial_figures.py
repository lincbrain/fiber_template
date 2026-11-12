#!/usr/bin/env python3
"""The figures of the step-by-step tutorials (docs/tutorials/ of the data tree), one function per figure, in the paper's figure style
(code/figures/figstyle.py: black background, letters, headers and scale bars outside the images, sentence case; command names, set in
the figures as code, are the one exception to sentence case).

Every figure reads only files of the data tree (and the release's settings/histology.yaml, for the sections' pixel size) and is written
only to the tree's docs/tutorials/figures/, as <name>.png (200 dpi, 178 mm wide) with <name>.provenance.json beside it. The example
brain is MR243 (MM1 in the paper); the MRI template's figures use MF283 (MF1), one of its four brains. The histology figures read the
JPEG 2000 sections, so they need glymur, which $PY_TIRL has. Each figure takes seconds, the histology ones up to a minute.

    $PY_TIRL code/docs/tutorial_figures.py --tree $LINC_DATA_TREE                 # every figure
    $PY_TIRL code/docs/tutorial_figures.py --tree $LINC_DATA_TREE --only mri_05_b0_in_template
    $PY_TIRL code/docs/tutorial_figures.py --list                                 # the figures, in order

--tree defaults to $LINC_DATA_TREE (settings/roots.sh sets it). --out defaults to <tree>/docs/tutorials/figures.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO / "code" / "figures"))
import figstyle as FS  # noqa: E402

DPI = 200                                     # the tutorials' figures (FS.DPI is the paper's 600)
W_MM = 178.0                                  # full page width
LET_MM = 3.0                                  # room for a direction letter beside a panel
S = "MR243"                                   # the example brain (MM1)
T = "MF283"                                   # a brain of the MRI template (MF1)
T4 = ("MF283", "MF284", "MF285", "MF286")     # the MRI template's four brains
SECTION, SECTION_N, SECTION_SLICE = "mr243LY_c18_s36nDF", 36, 107
#   the example section, through the LGN. Its blockface slice comes from the match table
#   (inputs/manual_annotations/MR243_section_match.txt: sections 16-40 match photograph 3 n + 4758, here 4866) minus the first
#   photograph's number, 4759 (settings/brains/MR243.yaml idx_range): 4866 - 4759 = 107, in the cutting order.
PIECES, PIECES_N, PIECES_SLICE = "mr243LY_c18_s05nDF", 5, 14
#   a section registered as two components (left and right hemispheres): sections 1-15 match 3 n + 4758 for both, 4773 - 4759 = 14
TRACER_SLAB_VOX = 10                          # the tracer figure's template panels: labels within +-10 voxels (2.5 mm) of the plane
REGIONS_SLICE = 134                           # a blockface slice where all four labels of the regions volume are present
JP2_STEP = 64                                 # read every 64th pixel of a section (a reduced level of the JPEG 2000)
FIGS = {}                                     # name -> function, in the tutorials' order
CODE_TEXT = set()                             # command names drawn in the figures: exempt from the sentence-case rule


def figure(fn):
    FIGS[fn.__name__] = fn
    return fn


# ---- reading -----------------------------------------------------------------------------------------------------------------
class Tree:
    def __init__(self, root):
        self.root = Path(root)
        self.read = []                        # every file a figure read (its provenance)

    def __call__(self, rel):
        p = self.root / rel
        if not p.exists():
            raise FileNotFoundError(f"{p}: not in the data tree")
        self.read.append(str(p))
        return p

    def nii(self, rel):
        """(data, affine) in canonical RAS voxel order."""
        import nibabel as nib
        return FS.canonical(nib.load(str(self(rel))))

    def img(self, rel):
        import nibabel as nib
        return nib.load(str(self(rel)))


def window(img, mask=None, lo=1, hi=99):
    """img scaled to [0, 1] between its lo and hi percentiles inside mask (or inside img > 0)."""
    m = (mask > 0.5) if mask is not None else (img > 0)
    v = img[m] if m.any() else img.ravel()
    a, b = np.percentile(v, [lo, hi])
    return np.clip((img - a) / max(b - a, 1e-9), 0, 1)


def bbox(mask2d, margin_px):
    r = np.where(mask2d.any(1))[0]; c = np.where(mask2d.any(0))[0]
    return (max(r[0] - margin_px, 0), min(r[-1] + margin_px + 1, mask2d.shape[0]),
            max(c[0] - margin_px, 0), min(c[-1] + margin_px + 1, mask2d.shape[1]))


def crop(a, box):
    return a[box[0]:box[1], box[2]:box[3]]


def pull_itk(field, field_aff, src, src_aff, plane, index, order=1):
    """One plane of `src` pulled onto the field's grid through an ITK displacement field (LPS mm, the convention antsApplyTransforms
    reads; the data tree's <ID>_*_to_template.nii.gz): p_src = p + d(p). `field` (canonical RAS order, last axis the vector) and `src`
    in canonical RAS order. Returns the plane in section()'s display order (linear interpolation, as antsApplyTransforms -n Linear)."""
    from scipy import ndimage as ndi
    k = FS.PLANE_AXIS[plane]
    d = np.take(field, int(index), axis=k)
    ax = [a for a in range(3) if a != k]
    g = np.meshgrid(np.arange(d.shape[0]), np.arange(d.shape[1]), indexing="ij")
    ijk = np.zeros(d.shape[:2] + (3,)); ijk[..., ax[0]], ijk[..., ax[1]], ijk[..., k] = g[0], g[1], index
    ras = ijk @ field_aff[:3, :3].T + field_aff[:3, 3]
    q = (ras * [-1, -1, 1] + d) * [-1, -1, 1]
    vox = (q - src_aff[:3, 3]) @ np.linalg.inv(src_aff[:3, :3]).T
    out = ndi.map_coordinates(src, vox.reshape(-1, 3).T, order=order, cval=0.0).reshape(d.shape[:2])
    return np.flipud(out.T)


def itk_field(tr, rel):
    """An ITK displacement field of the data tree in canonical RAS order: (field (X, Y, Z, 3) in LPS mm, affine)."""
    import nibabel as nib
    F = nib.as_closest_canonical(tr.img(rel))
    return np.asanyarray(F.dataobj)[:, :, :, 0, :].astype(np.float32), F.affine


def rgb01(a):
    return np.clip(a.astype(np.float32) / (255.0 if a.dtype == np.uint8 else max(float(a.max()), 1e-9)), 0, 1)


def fa_slice(tensor6, plane, index):
    """FA of one plane of a 6-component tensor (FSL dtifit order: xx, xy, xz, yy, yz, zz)."""
    t = np.take(tensor6, int(index), axis=FS.PLANE_AXIS[plane]).astype(np.float64)
    M = np.stack([np.stack([t[..., 0], t[..., 1], t[..., 2]], -1), np.stack([t[..., 1], t[..., 3], t[..., 4]], -1),
                  np.stack([t[..., 2], t[..., 4], t[..., 5]], -1)], -2)
    ev = np.linalg.eigvalsh(M)
    md = ev.mean(-1, keepdims=True); den = np.sqrt((ev ** 2).sum(-1))
    fa = np.where(den > 0, np.sqrt(1.5 * ((ev - md) ** 2).sum(-1)) / np.where(den > 0, den, 1), 0)
    return np.flipud(np.clip(fa, 0, 1).T)


# ---- the blockface grid of MR243 (outputs/MR243_blockface_L2: R, P, S order, so slice j is the j-th section cut) ---------------
def bf_slice(tr, rel, j):
    """Coronal slice j (cutting order) of a volume on MR243's L2 blockface grid, in display order (superior up, the subject's left on
    the left); RGB as floats in [0, 1]."""
    import nibabel as nib
    im = tr.img(rel)
    assert nib.aff2axcodes(im.affine) == ("R", "P", "S"), f"{rel}: expected R, P, S axes (slice = cutting order)"
    a = np.asanyarray(im.dataobj[:, j])
    a = np.flipud(np.swapaxes(a, 0, 1))
    return (rgb01(a) if a.ndim == 3 else a), abs(im.affine[0, 0])


def regions_on_bf(tr, j):
    """The regions volume (intermediate/histology/MR243_regions_t2, made at a 400 um section step on a 0.15 mm grid) on blockface
    slice j: its section axis is scaled into the calibrated frame by 366 / 400 about the world origin (settings/histology.yaml
    regions.made_at_spacing_um; code/histology/linc-histology does the same), then sampled nearest. Display order."""
    import nibabel as nib
    from scipy import ndimage as ndi
    import yaml
    ph = tr.img(f"brains/sub-{S}/outputs/{S}_blockface_L2.nii.gz"); R = tr.img(f"brains/sub-{S}/intermediate/histology/{S}_regions_t2.nii.gz")
    made = yaml.safe_load((REPO / "settings" / "histology.yaml").read_text())["regions"]["made_at_spacing_um"][S]
    vz_cal = abs(ph.affine[1, 1]) * 1000.0
    fz = vz_cal / float(made["vz"])                               # MR243's in-plane factor is 1 (made at 20 um, calibrated 20 um)
    nx, nz = ph.shape[0], ph.shape[2]
    ii, kk = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
    vox = np.stack([ii, np.full_like(ii, j), kk, np.ones_like(ii)], -1).reshape(-1, 4).T
    w = ph.affine @ vox
    w[1] /= fz                                                     # the calibrated world -> the made-at world (section axis)
    v = np.linalg.inv(R.affine) @ w
    lab = ndi.map_coordinates(np.asanyarray(R.dataobj), v[:3], order=0, cval=0).reshape(nx, nz)
    return np.flipud(lab.T)


def section_px_mm():
    import yaml
    return float(yaml.safe_load((REPO / "settings" / "histology.yaml").read_text())["propose"]["section_pixel_um"]) / 1000.0


def section_thumb(tr, name=SECTION, step=JP2_STEP):
    """A section as scanned, read at a reduced level (every `step`-th pixel), and its pixel size (mm)."""
    try:
        import glymur
    except ImportError:
        sys.exit("the histology figures read the JPEG 2000 sections: run this script with $PY_TIRL (it has glymur)")
    j = glymur.Jp2k(str(tr(f"brains/sub-{S}/inputs/histology/sections/{name}.jp2")))
    a = rgb01(j[::step, ::step])
    tissue = a.max(-1) > 8 / 255                                   # the dark field: tissue is what is not black
    hi = np.percentile(a[tissue], 99.5) if tissue.any() else 1.0
    return np.clip(a / max(hi, 1e-6), 0, 1), section_px_mm() * step     # brightened for display (one scale for all three channels)


# ---- layout ------------------------------------------------------------------------------------------------------------------
TOP_MM, HEAD_MM, BAR_MM = 2.0, 5.0, 6.5      # the page's top margin, a panel's letter and header line, a scale bar's room below


def fit_scale(widths_mm, gutter=FS.GUTTER_MM, extra_mm=0.0, letters=True):
    """The one print scale at which a row of panels (tissue widths, mm) fills the page."""
    room = W_MM - 2 * FS.MARGIN_MM - gutter * (len(widths_mm) - 1) - extra_mm - (2 * LET_MM if letters else 0)
    return room / sum(widths_mm)


def row(fig, dims, y_mm, scale, gutter=FS.GUTTER_MM, x0=None, letters=True):
    """Axes for panels `dims` = [(h_mm, w_mm), ...] (tissue mm), left to right, tops at y_mm, at print scale `scale`."""
    x = (FS.MARGIN_MM + (LET_MM if letters else 0)) if x0 is None else x0
    axs = []
    for h, w in dims:
        ax = FS.axes_mm(fig, x, y_mm, w * scale, h * scale); ax.set_axis_off(); axs.append(ax); x += w * scale + gutter
    return axs


def show(ax, img, px_mm, cmap="gray", vmin=0, vmax=1, interpolation="bilinear"):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interpolation)
    ax.set_xlim(-0.5, img.shape[1] - 0.5); ax.set_ylim(img.shape[0] - 0.5, -0.5)
    ax.px_mm = px_mm
    return ax


def head(ax, letter, text=None):
    FS.letter(ax, letter)
    if text:
        FS.header(ax, text)


def bar_below(ax, px_per_mm, mm, y_frac=None, gap_mm=3.0):
    """figstyle's scale bar, gap_mm under the panel's bottom edge whatever the panel's height (y_frac is ignored)."""
    h_mm = ax.get_position().height * ax.figure.get_size_inches()[1] * 25.4
    return FS.scalebar(ax, px_per_mm, mm, y_frac=gap_mm / h_mm)


def code_text(fig, x_mm, y_mm, s, ha="center", va="bottom", **kw):
    """A command name in the figure (monospace, bold and half a point larger, so that its underscores survive a viewer's downscaling;
    exempt from the sentence-case rule)."""
    W, H = (v * 25.4 for v in fig.get_size_inches())
    CODE_TEXT.add(s)
    return fig.text(x_mm / W, 1 - y_mm / H, s, ha=ha, va=va, fontsize=FS.MIN_PT + 0.5, color=FS.FG, family="monospace", fontweight="bold", **kw)


def arrow(fig, p0_mm, p1_mm, head=True):
    """A plain arrow between two points of the page (mm from the top-left corner); head=False: a line."""
    from matplotlib.patches import FancyArrowPatch
    W, H = (v * 25.4 for v in fig.get_size_inches())
    f = lambda p: (p[0] / W, 1 - p[1] / H)
    fig.add_artist(FancyArrowPatch(f(p0_mm), f(p1_mm), transform=fig.transFigure, arrowstyle="-|>" if head else "-", mutation_scale=6, lw=0.8,
                                   color=FS.FG, shrinkA=0, shrinkB=0))


def box_mm(fig, ax):
    """An axes' box in page mm: (x0, y0, x1, y1), y down."""
    W, H = (v * 25.4 for v in fig.get_size_inches()); p = ax.get_position()
    return p.x0 * W, (1 - p.y1) * H, p.x1 * W, (1 - p.y0) * H


def arrow_between(fig, ax0, ax1):
    a, b = box_mm(fig, ax0), box_mm(fig, ax1)
    y = (max(a[1], b[1]) + min(a[3], b[3])) / 2
    arrow(fig, (a[2] + 1.0, y), (b[0] - 1.0, y))


def new_fig(h_mm):
    import matplotlib.pyplot as plt
    return plt.figure(figsize=(W_MM * FS.MM, h_mm * FS.MM))


def save(fig, out, tree, name):
    """figstyle's checks (all but sentence case for the command names), then the PNG at 200 dpi and its provenance."""
    import matplotlib.pyplot as plt
    from shared import provenance
    FS.tiles(fig)
    bad = [b for b in FS.check(fig) if not (b.startswith("not in sentence case") and any(repr(c[:40]) in b for c in CODE_TEXT))]
    if bad:
        raise ValueError(f"{name}: house rules broken:\n  " + "\n  ".join(bad))
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{name}.png"
    fig.savefig(str(png), dpi=DPI, facecolor=fig.get_facecolor())
    provenance.write_provenance(str(png), inputs=sorted(set(tree.read)) + [os.path.abspath(__file__), os.path.abspath(FS.__file__)],
                                cmd=" ".join(sys.argv), md5=False,
                                extra=dict(figure=name, dpi=DPI, width_mm=round(fig.get_size_inches()[0] * 25.4, 1), house_rules="ok"))
    plt.close(fig)


# ---- MR243's b0 and the template ---------------------------------------------------------------------------------------------
def mri_example(tr):
    b0, ba = tr.nii(f"brains/sub-{S}/inputs/mri/{S}_b0.nii.gz")
    m, _ = tr.nii(f"brains/sub-{S}/inputs/manual_annotations/{S}_b0_brain_mask.nii.gz")
    tb, ta = tr.nii("template/outputs/template_b0.nii.gz")
    tm, _ = tr.nii("template/outputs/template_brain_mask.nii.gz")
    f, fa = itk_field(tr, f"brains/sub-{S}/outputs/{S}_b0_to_template.nii.gz")
    return b0, ba, m, tb, ta, tm, f, fa


def template_planes(tm):
    """The template's planes through its brain's center (canonical voxel indices); the sagittal 6 mm left of it (one hemisphere)."""
    c = np.round(np.argwhere(tm > 0.5).mean(0)).astype(int)
    return dict(sagittal=int(c[0]) - 24, coronal=int(c[1]), axial=int(c[2]))


def native_planes(ba, f, fa, tp):
    """A native image's planes through the native point the template's plane center comes from (both show the same anatomy)."""
    p = np.array([tp["sagittal"], tp["coronal"], tp["axial"]])
    q = ((fa[:3, :3] @ p + fa[:3, 3]) * [-1, -1, 1] + f[tuple(p)]) * [-1, -1, 1]
    ijk = np.linalg.inv(ba[:3, :3]) @ (q - ba[:3, 3])
    return dict(sagittal=int(round(ijk[0])), coronal=int(round(ijk[1])), axial=int(round(ijk[2])))


def one_row(panels, gutter=FS.GUTTER_MM, extra_mm=0.0, letters=True, bar_room=True):
    """A one-row figure: panels = [(image, px_mm), ...]; returns (fig, axes)."""
    dims = [(p[0].shape[0] * p[1], p[0].shape[1] * p[1]) for p in panels]
    sc = fit_scale([w for h, w in dims], gutter=gutter, extra_mm=extra_mm, letters=letters)
    fig = new_fig(TOP_MM + HEAD_MM + max(h for h, w in dims) * sc + (BAR_MM if bar_room else FS.MARGIN_MM))
    axs = row(fig, dims, TOP_MM + HEAD_MM, sc, gutter=gutter, letters=letters)
    for ax, (img, px) in zip(axs, panels):
        show(ax, img, px)
    return fig, axs, sc


# ---- the MRI tutorial (01_mri.md) --------------------------------------------------------------------------------------------
@figure
def mri_01_flow(tr):
    """The MRI part: the four brains' T2-weighted images make the template (top); a brain's b0 is registered to it (bottom)."""
    tm, ta = tr.nii("template/outputs/template_brain_mask.nii.gz")
    tp = template_planes(tm)
    top = []
    for B in T4:
        d, a = tr.nii(f"brains/sub-{B}/outputs/{B}_t2_T2w.nii.gz"); mk, _ = tr.nii(f"brains/sub-{B}/inputs/manual_annotations/{B}_t2_brain_mask.nii.gz")
        f, fa = itk_field(tr, f"brains/sub-{B}/outputs/{B}_t2_to_template.nii.gz")
        j = native_planes(a, f, fa, tp)["coronal"]
        im, m2 = FS.section(window(d, mk), "coronal", j), FS.section(mk, "coronal", j)
        bx = bbox(m2 > 0.5, int(2 / abs(a[0, 0]))); top.append((crop(im * (m2 > 0.5), bx), abs(a[0, 0])))
    t2, _ = tr.nii("template/outputs/template_T2w.nii.gz")
    m2 = FS.section(tm, "coronal", tp["coronal"]); bx = bbox(m2 > 0.5, 8)
    top.append((crop(FS.section(window(t2, tm), "coronal", tp["coronal"]) * (m2 > 0.5), bx), 0.25))
    b0, ba, m, tb, ta, tm, f, fa = mri_example(tr)
    npl = native_planes(ba, f, fa, tp)
    n_im, n_m = FS.section(window(b0, m), "coronal", npl["coronal"]), FS.section(m, "coronal", npl["coronal"])
    nb = bbox(n_m > 0.5, int(2 / abs(ba[0, 0])))
    mv = pull_itk(f, fa, b0, ba, "coronal", tp["coronal"])
    bot = [(crop(n_im, nb), abs(ba[0, 0]), crop(n_m, nb)), (crop(window(mv, mv > 0), bx), 0.25, crop(m2, bx)),
           (crop(FS.section(window(tb, tm), "coronal", tp["coronal"]), bx), 0.25, crop(m2, bx))]
    g = 3.0; ga = 8.0                                               # the gutter; the gutter that holds an arrow
    wt = [p[0].shape[1] * p[1] for p in top]
    sc = (W_MM - 2 * FS.MARGIN_MM - 2 * LET_MM - 3 * g - ga) / sum(wt)
    ht = max(p[0].shape[0] * p[1] for p in top) * sc; hb = max(p[0].shape[0] * p[1] for p in bot) * sc
    fig = new_fig(TOP_MM + HEAD_MM + ht + HEAD_MM + 2.0 + hb + BAR_MM)
    x = FS.MARGIN_MM + LET_MM; axs_t = []
    for i, (img, px) in enumerate(top):
        ax = FS.axes_mm(fig, x, TOP_MM + HEAD_MM, img.shape[1] * px * sc, img.shape[0] * px * sc); ax.set_axis_off(); show(ax, img, px)
        axs_t.append(ax); x += img.shape[1] * px * sc + (ga if i == 3 else g)
    head(axs_t[0], "a", "The four brains' T2-weighted images ..."); head(axs_t[4], "b", "... make the template")
    arrow_between(fig, axs_t[3], axs_t[4])
    y1 = TOP_MM + HEAD_MM + ht + HEAD_MM + 2.0; x = FS.MARGIN_MM + LET_MM; axs_b = []
    for i, (img, px, ol) in enumerate(bot):
        ax = FS.axes_mm(fig, x, y1, img.shape[1] * px * sc, img.shape[0] * px * sc); ax.set_axis_off(); show(ax, img, px)
        FS.outline(ax, ol, lw=0.5); axs_b.append(ax); x += img.shape[1] * px * sc + ga
    for a0, a1 in zip(axs_b[:-1], axs_b[1:]):
        arrow_between(fig, a0, a1)
    for ax, l, t in zip(axs_b, "cde", (f"{S}'s b0, its mask", "In the template", "The template's b0")):
        head(ax, l, t)
    FS.dirs_row(axs_t[0], axs_t[-1], "coronal", pad=1.5); FS.dirs_row(axs_b[0], axs_b[-1], "coronal", pad=1.5)
    bar_below(axs_b[0], 1 / axs_b[0].px_mm, 10, y_frac=0.08)
    return fig


@figure
def mri_02_input(tr):
    """The inputs: MF283's T2-weighted image, FA (from its tensor) and soma fraction, each in its own space, and MR243's b0 (coronal)."""
    P = []
    d, a = tr.nii(f"brains/sub-{T}/outputs/{T}_t2_T2w.nii.gz"); mk, _ = tr.nii(f"brains/sub-{T}/inputs/manual_annotations/{T}_t2_brain_mask.nii.gz")
    j = int(round(np.argwhere(mk > 0.5)[:, 1].mean())); m2 = FS.section(mk, "coronal", j); bx = bbox(m2 > 0.5, 4)
    P.append((crop(FS.section(window(d, mk), "coronal", j) * (m2 > 0.5), bx), abs(a[0, 0]), f"{T}: T2-weighted"))
    t6, a = tr.nii(f"brains/sub-{T}/outputs/{T}_tensor_dti.nii.gz"); mk, _ = tr.nii(f"brains/sub-{T}/inputs/manual_annotations/{T}_tensor_brain_mask.nii.gz")
    j = int(round(np.argwhere(mk > 0.5)[:, 1].mean())); m2 = FS.section(mk, "coronal", j); bx = bbox(m2 > 0.5, 8)
    P.append((crop(fa_slice(t6, "coronal", j) * (m2 > 0.5), bx), abs(a[0, 0]), "FA, from its tensor"))
    d, a = tr.nii(f"brains/sub-{T}/outputs/{T}_micro_soma_fraction.nii.gz"); mk, _ = tr.nii(f"brains/sub-{T}/inputs/manual_annotations/{T}_micro_brain_mask.nii.gz")
    j = int(round(np.argwhere(mk > 0.5)[:, 1].mean())); m2 = FS.section(mk, "coronal", j); bx = bbox(m2 > 0.5, 4)
    P.append((crop(np.clip(FS.section(d, "coronal", j), 0, 1) * (m2 > 0.5), bx), abs(a[0, 0]), "Soma fraction"))
    b0, ba = tr.nii(f"brains/sub-{S}/inputs/mri/{S}_b0.nii.gz"); mk, _ = tr.nii(f"brains/sub-{S}/inputs/manual_annotations/{S}_b0_brain_mask.nii.gz")
    j = int(round(np.argwhere(mk > 0.5)[:, 1].mean())); m2 = FS.section(mk, "coronal", j); bx = bbox(m2 > 0.5, 3)
    P.append((crop(FS.section(window(b0, mk), "coronal", j), bx), abs(ba[0, 0]), f"{S}: b0"))
    g = 3.0; ga = 7.0
    dims = [(p[0].shape[0] * p[1], p[0].shape[1] * p[1]) for p in P]
    sc = (W_MM - 2 * FS.MARGIN_MM - 2 * LET_MM - 2 * g - ga - LET_MM * 2) / sum(w for h, w in dims)
    fig = new_fig(TOP_MM + HEAD_MM + max(h for h, w in dims) * sc + BAR_MM)
    x = FS.MARGIN_MM + LET_MM; axs = []
    for i, ((img, px, t), (h, w)) in enumerate(zip(P, dims)):
        ax = FS.axes_mm(fig, x, TOP_MM + HEAD_MM, w * sc, h * sc); ax.set_axis_off(); show(ax, img, px); head(ax, "abcd"[i], t)
        axs.append(ax); x += w * sc + (ga + 2 * LET_MM if i == 2 else g)
    FS.dirs_row(axs[0], axs[2], "coronal", pad=1.5); FS.dirs(axs[3], "coronal", pad=1.5)
    bar_below(axs[0], 1 / axs[0].px_mm, 10, y_frac=0.08)                  # one print scale: one bar
    return fig


@figure
def mri_03_template_channels(tr):
    """The MRI template's channels in one coronal plane: T2-weighted, b0, FA, the principal direction, neurite and soma fractions; the
    brain mask's outline on the first."""
    tm, _ = tr.nii("template/outputs/template_brain_mask.nii.gz")
    j = template_planes(tm)["coronal"]; m2 = FS.section(tm, "coronal", j); bx = bbox(m2 > 0.5, 8)
    P = []
    for title, rel, kind in (("T2-weighted", "template_T2w", "win"), ("b0", "template_b0", "win"), ("FA", "template_FA", "01"),
                             ("Principal direction", "template_DEC", "rgb"), ("Neurite fraction", "template_neurite_fraction", "01"),
                             ("Soma fraction", "template_soma_fraction", "01")):
        d, _ = tr.nii(f"template/outputs/{rel}.nii.gz")
        im = FS.section(window(d, tm) if kind == "win" else (rgb01(d) if kind == "rgb" else np.clip(d, 0, 1)), "coronal", j)
        im = im * ((m2 > 0.5)[..., None] if im.ndim == 3 else (m2 > 0.5))
        P.append((crop(im, bx), 0.25, title))
    fig, axs, _ = one_row([(p[0], p[1]) for p in P])
    for i, (ax, p) in enumerate(zip(axs, P)):
        head(ax, "abcdef"[i], p[2])
    FS.outline(axs[0], crop(m2, bx), lw=0.5)
    FS.dirs_row(axs[0], axs[-1], "coronal", pad=1.5)
    bar_below(axs[0], 4, 10, y_frac=0.08)
    return fig


@figure
def mri_04_prep(tr):
    """What linc-mri prep makes, on MF283's b0 (the files of record of its preparation): the b0, the brain mask, the masked b0, the bias
    field N4 found, and the bias-corrected brain (the masked b0 divided by the field, then rescaled)."""
    b, a = tr.nii(f"brains/sub-{T}/intermediate/prepare/dwi/b0.nii.gz")
    bb, _ = tr.nii(f"brains/sub-{T}/intermediate/prepare/dwi/b0_brain.nii.gz")
    bias, _ = tr.nii(f"brains/sub-{T}/intermediate/prepare/dwi/b0_brain_n4_bias.nii.gz")
    mk, _ = tr.nii(f"brains/sub-{T}/inputs/manual_annotations/{T}_tensor_brain_mask.nii.gz")
    j = int(round(np.argwhere(mk > 0.5)[:, 1].mean())); px = abs(a[0, 0])
    m2 = FS.section(mk, "coronal", j); bx = bbox(m2 > 0.5, int(3 / px))
    raw = FS.section(window(b, b > np.percentile(b, 50)), "coronal", j)
    br = FS.section(window(bb, mk), "coronal", j)
    bi = FS.section(bias, "coronal", j); bi = np.where(m2 > 0.5, bi, 0)
    v = bi[m2 > 0.5]; bi = np.clip((bi - v.min()) / max(v.max() - v.min(), 1e-9), 0, 1) * (m2 > 0.5)
    corr = np.where(m2 > 0.5, FS.section(bb, "coronal", j) / np.where(FS.section(bias, "coronal", j) > 0, FS.section(bias, "coronal", j), 1), 0)
    corr = window(corr, m2)
    P = [(crop(raw, bx), px), (crop(raw, bx), px), (crop(br, bx), px), (crop(bi, bx), px), (crop(corr, bx), px)]
    fig, axs, _ = one_row(P)
    FS.outline(axs[1], crop(m2, bx), lw=0.5)
    for i, (ax, t) in enumerate(zip(axs, ("The b0", "Its brain mask", "Masked", "Bias field (N4)", "Bias-corrected"))):
        head(ax, "abcde"[i], t)
    FS.dirs_row(axs[0], axs[-1], "coronal", pad=1.5)
    bar_below(axs[0], 1 / px, 10, y_frac=0.08)
    return fig


@figure
def mri_05_b0_in_template(tr):
    """The registration's result: MR243's b0 pulled into the template through outputs/MR243_b0_to_template (top) and the template's
    b0 (bottom), the template brain's outline on both, three planes."""
    b0, ba, m, tb, ta, tm, f, fa = mri_example(tr)
    tp = template_planes(tm); tw = window(tb, tm)
    rows = [[], []]
    for pl in ("sagittal", "coronal", "axial"):
        mv = pull_itk(f, fa, b0, ba, pl, tp[pl]); mk = FS.section(tm, pl, tp[pl]); bx = bbox(mk > 0.5, 8)
        rows[0].append((pl, crop(window(mv, mv > 0), bx), crop(mk, bx)))
        rows[1].append((pl, crop(FS.section(tw, pl, tp[pl]), bx), crop(mk, bx)))
        r = np.corrcoef(mv[mk > 0.5], FS.section(tb, pl, tp[pl])[mk > 0.5])[0, 1]
        print(f"  {pl}: r (the moved b0, the template's b0) inside the template brain = {r:.3f}")
    dims = [(p[1].shape[0] * 0.25, p[1].shape[1] * 0.25) for p in rows[0]]
    g = 2 * LET_MM + 1.0; sc = fit_scale([w for h, w in dims], gutter=g)
    hrow = max(h for h, w in dims) * sc
    fig = new_fig(TOP_MM + 2 * (HEAD_MM + hrow) + BAR_MM)
    heads = (f"{S}'s b0 in the template", "The template's b0")
    for k in range(2):
        axs = row(fig, dims, TOP_MM + HEAD_MM + k * (HEAD_MM + hrow), sc, gutter=g)
        for i, (ax, (pl, im, mk)) in enumerate(zip(axs, rows[k])):
            show(ax, im, 0.25); FS.outline(ax, mk, lw=0.5)
            head(ax, "abcdef"[3 * k + i], f"{heads[k]}, {pl}" if i == 0 else pl.capitalize()); FS.dirs(ax, pl, pad=1.5)
        if k == 1:
            bar_below(axs[0], 4, 10, y_frac=0.08)
    return fig


@figure
def mri_06_outline_check(tr):
    """The outline check, a pass and a warning: MR243 (PASS) and MR253 (WARN, ventral overshoot), each brain's b0 mask moved into the
    template (dashed) over the template's b0 and its brain outline (solid), one sagittal plane 2 mm off the midline and one coronal
    plane; the dotted orange line is the template brain's lowest axial slice, below which the check reports tissue but does not gate it."""
    tb, ta = tr.nii("template/outputs/template_b0.nii.gz"); tm, _ = tr.nii("template/outputs/template_brain_mask.nii.gz")
    tp = template_planes(tm); tw = window(tb, tm)
    tp_s = dict(tp, sagittal=int(np.round(np.argwhere(tm > 0.5)[:, 0].mean())) - 8)      # 2 mm off the midline: the brainstem in view
    rows = []
    for B in (S, "MR253"):
        m, ma = tr.nii(f"brains/sub-{B}/inputs/manual_annotations/{B}_b0_brain_mask.nii.gz")
        f, fa = itk_field(tr, f"brains/sub-{B}/outputs/{B}_b0_to_template.nii.gz")
        P = []
        for pl in ("sagittal", "coronal"):
            k = tp_s[pl]; mk = FS.section(tm, pl, k); bx = bbox(mk > 0.5, 16)
            mv = pull_itk(f, fa, m.astype(np.float32), ma, pl, k) > 0.5
            P.append((pl, crop(FS.section(tw, pl, k), bx), crop(mk, bx), crop(mv, bx), bx))
        rows.append((B, P))
    zlow = int(np.where((tm > 0.5).any(axis=(0, 1)))[0][0])            # the template brain's lowest axial slice (its flat brainstem cut)
    row_low = tm.shape[2] - 1 - zlow                                   # its row in section()'s display (superior up)
    import json
    verdict = {}
    for B, _ in rows:
        q = json.load(open(tr(f"brains/sub-{B}/intermediate/registration/b0-native__to__mri-template/meta.json")))["qc"]
        verdict[B] = q["outline"] + (f", {q['outline_reasons'][0].split(' p99')[0]}" if q.get("outline_reasons") else "")
    dims = [(p[1].shape[0] * 0.25, p[1].shape[1] * 0.25) for p in rows[0][1]] * 2
    g = 2 * LET_MM + 1.0; sc = fit_scale([w for h, w in dims], gutter=g)
    fig = new_fig(TOP_MM + HEAD_MM + max(h for h, w in dims) * sc + BAR_MM)
    axs = row(fig, dims, TOP_MM + HEAD_MM, sc, gutter=g)
    i = 0
    for B, P in rows:
        for j, (pl, im, mk, mv, bx) in enumerate(P):
            ax = axs[i]; show(ax, im, 0.25); FS.outline(ax, mk, lw=0.5)
            FS.outline(ax, mv, color=FS.ON_IMG2, lw=0.6, linestyles="dashed")
            ax.axhline(row_low - bx[0] + 0.5, color=FS.COL["pred"], lw=0.5, ls=(0, (1, 1.5)))    # below it: reported, not gated
            head(ax, "abcd"[i], f"{B}: {verdict[B]}" if j == 0 else pl.capitalize()); FS.dirs(ax, pl, pad=1.5); i += 1
    bar_below(axs[0], 4, 10)
    return fig


# ---- the blockface tutorial (02_blockface.md) -------------------------------------------------------------------------------
def provenance_view(lum, tissue, predicted):
    """The paper's provenance colors (its blockface template figure, panel c): photographed tissue in dimmed gray (its luminance),
    predicted tissue orange (figstyle COL["pred"])."""
    from matplotlib.colors import to_rgb
    gray = (0.18 + 0.44 * np.clip(lum, 0, 1))[..., None] * np.ones(3)
    return np.where(predicted[..., None], np.asarray(to_rgb(FS.COL["pred"])), gray) * tissue[..., None]


def two_rows(panels, heads):
    """A two-row figure: panels[k] = [(image, px_mm, plane), ...] for row k (the same planes and crops in both rows)."""
    dims = [(p[0].shape[0] * p[1], p[0].shape[1] * p[1]) for p in panels[0]]
    g = 2 * LET_MM + 1.0; sc = fit_scale([w for h, w in dims], gutter=g)
    hrow = max(h for h, w in dims) * sc
    fig = new_fig(TOP_MM + 2 * (HEAD_MM + hrow) + BAR_MM)
    axs = []
    for k in range(2):
        ax_k = row(fig, dims, TOP_MM + HEAD_MM + k * (HEAD_MM + hrow), sc, gutter=g)
        for i, (ax, (img, px, pl)) in enumerate(zip(ax_k, panels[k])):
            show(ax, img, px); head(ax, "abcdef"[len(panels[0]) * k + i], heads[k][i]); FS.dirs(ax, pl, pad=1.5)
        axs.append(ax_k)
    return fig, axs


@figure
def blockface_06_completed(tr):
    """MR252's completed volume (outputs/MR252_completed) in a sagittal and a coronal plane (top), and the same planes in the paper's
    provenance colors (bottom): photographed gray, predicted orange (outputs/MR252_synthesized_mask). In the sagittal plane the border
    between the two is the junction plane; the coronal plane lies 5 mm caudal of it, in predicted tissue."""
    B = "MR252"
    v, A = tr.nii(f"brains/sub-{B}/outputs/{B}_completed.nii.gz")
    sm, _ = tr.nii(f"brains/sub-{B}/outputs/{B}_synthesized_mask.nii.gz")
    tissue, px = v > 0, abs(A[0, 0]); sm = sm > 0
    vw = window(v, tissue, 0.5, 99.5)
    i = int(round(np.argwhere(tissue)[:, 0].mean() - 4.5 / px))        # 4.5 mm left of the brain's center (the paper's pipeline figure)
    frac = sm.sum((0, 2)) / np.maximum(tissue.sum((0, 2)), 1)           # the predicted share of each coronal plane
    ys = np.where(tissue.any((0, 2)))[0]; j_junction = int(ys.min())     # the junction: from the caudal end, the last plane mostly predicted
    while j_junction + 1 <= ys.max() and frac[j_junction + 1] > 0.5:
        j_junction += 1
    j = j_junction - int(round(5.0 / px))
    P = [[], []]
    for pl, k in (("sagittal", i), ("coronal", j)):
        t2 = FS.section(tissue.astype(np.float32), pl, k) > 0.5; bx = bbox(t2, int(1.0 / px))
        lum = FS.section(vw, pl, k) * t2
        P[0].append((crop(lum, bx), px, pl))
        P[1].append((crop(provenance_view(lum, t2, FS.section(sm.astype(np.float32), pl, k) > 0.5), bx), px, pl))
    print(f"  {B}: sagittal index {i}, junction plane y index {j_junction}, coronal {j} (5 mm caudal)")
    fig, axs = two_rows(P, (("Completed volume, sagittal", "Coronal, 5 mm caudal of the junction"),
                            ("Photographed (gray) and predicted (orange)", "Coronal")))
    bar_below(axs[1][0], 1 / px, 10, y_frac=0.08)
    return fig


@figure
def blockface_07_template(tr):
    """The blockface template's channel (template/outputs/template_blockface_rgb, on the MRI template's grid) in three planes (top), and
    the same planes in the provenance colors (bottom): gray where its brains were photographed, orange where none was
    (template_blockface_synthesized_mask)."""
    rgb, _ = tr.nii("template/outputs/template_blockface_rgb.nii.gz")
    bm, _ = tr.nii("template/outputs/template_blockface_mask.nii.gz")
    sm, _ = tr.nii("template/outputs/template_blockface_synthesized_mask.nii.gz")
    tp = template_planes(bm); rgb = np.clip(rgb.astype(np.float32), 0, 1)
    lum_all = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    hi = float(np.percentile(lum_all[bm > 0.5], 99.5))
    P = [[], []]
    for pl in ("sagittal", "coronal", "axial"):
        k = tp[pl]; t2 = FS.section(bm, pl, k) > 0.5; bx = bbox(t2, 4)
        col = np.stack([FS.section(rgb[..., c], pl, k) for c in range(3)], -1) * t2[..., None]
        lum = FS.section(lum_all, pl, k) / hi * t2
        P[0].append((crop(col, bx), 0.25, pl))
        P[1].append((crop(provenance_view(lum, t2, FS.section(sm, pl, k) > 0.5), bx), 0.25, pl))
    fig, axs = two_rows(P, (("Blockface template, sagittal", "Coronal", "Axial"),
                            ("Photographed (gray) and predicted (orange)", "Coronal", "Axial")))
    bar_below(axs[1][0], 4, 10, y_frac=0.08)
    return fig


# ---- the histology tutorial (03_histology.md) --------------------------------------------------------------------------------
def template_lgn_plane(tr, k=1):
    lt, _ = tr.nii(f"brains/sub-{S}/outputs/{S}_lgn_stn_template.nii.gz")
    return lt, int(round(np.argwhere(lt == k)[:, 1].mean()))


def template_both_plane(tr):
    """The template's coronal plane that cuts all four nuclei (the most voxels of the least-cut one): the LGN's anterior end meets the
    STN's posterior end there."""
    lt, _ = tr.nii(f"brains/sub-{S}/outputs/{S}_lgn_stn_template.nii.gz")
    n = np.stack([(lt == k).sum(axis=(0, 2)) for k in (1, 2, 3, 4)])
    return lt, int(np.argmax(n.min(0)))


@figure
def histology_01_flow(tr):
    """The histology part: a section as scanned, the photograph of its own section, the section registered onto it, and the LGN (solid)
    and STN (dashed) drawn on the sections, carried into the template, on the coronal plane that cuts all four."""
    sec, spx = section_thumb(tr)
    ph, px = bf_slice(tr, f"brains/sub-{S}/outputs/{S}_blockface_L2.nii.gz", SECTION_SLICE)
    hi, _ = bf_slice(tr, f"brains/sub-{S}/outputs/{S}_histology_blockface.nii.gz", SECTION_SLICE)
    bx = bbox(ph.sum(-1) > 0, int(2 / px))
    tb, _ = tr.nii("template/outputs/template_b0.nii.gz"); tm, _ = tr.nii("template/outputs/template_brain_mask.nii.gz")
    lt, jt = template_both_plane(tr)
    print(f"  the template's coronal plane through all four nuclei: {jt}")
    m2 = FS.section(tm, "coronal", jt); tbx = bbox(m2 > 0.5, 8)
    P = [(sec, spx), (crop(ph, bx), px), (crop(hi, bx), px), (crop(FS.section(window(tb, tm), "coronal", jt) * (m2 > 0.5), tbx), 0.25)]
    fig, axs, _ = one_row(P, gutter=8.0, letters=False)
    for i, (ax, t) in enumerate(zip(axs, ("A section as scanned", "Its own photograph", "Registered onto it", "LGN and STN in the template"))):
        head(ax, "abcd"[i], t)
    tl = crop(FS.section(lt, "coronal", jt), tbx)
    for k in (1, 2):
        FS.outline(axs[3], tl == k, color=FS.BRAIN[S], lw=0.6)
    for k in (3, 4):
        FS.outline(axs[3], tl == k, color=FS.BRAIN[S], lw=0.6, linestyles="dashed")
    for a0, a1 in zip(axs[:-1], axs[1:]):
        arrow_between(fig, a0, a1)
    bar_below(axs[0], 1 / spx, 10, y_frac=0.08)                            # one print scale: one bar
    return fig


@figure
def histology_02_input(tr):
    """The input: section 36 as scanned (JPEG 2000, read at a reduced level; dark field) beside the blockface photograph of its own
    section (slice 107 of the cutting order)."""
    sec, spx = section_thumb(tr)
    ph, px = bf_slice(tr, f"brains/sub-{S}/outputs/{S}_blockface_L2.nii.gz", SECTION_SLICE)
    bx = bbox(ph.sum(-1) > 0, int(2 / px))
    fig, axs, _ = one_row([(sec, spx), (crop(ph, bx), px)], gutter=8.0)
    head(axs[0], "a", f"Section {SECTION_N} as scanned"); head(axs[1], "b", f"Its photograph, slice {SECTION_SLICE}")
    FS.dirs(axs[1], "coronal", pad=1.5)
    bar_below(axs[0], 1 / spx, 10, y_frac=0.06)                            # one print scale: one bar
    return fig


REGION_COLORS = ["black", "#d9d9d9", "#8c8c8c", "#4d4d4d", FS.COL["pred"]]
REGION_NAMES = ["Left hemisphere (1)", "Right hemisphere (2)", "Cerebellum and brainstem (3)", "Marker tubes (4)"]


def region_key(fig, x_mm, y_mm):
    from matplotlib.patches import Patch
    W, H = (v * 25.4 for v in fig.get_size_inches())
    fig.legend(handles=[Patch(facecolor=c, edgecolor="none", label=n) for c, n in zip(REGION_COLORS[1:], REGION_NAMES)], loc="upper left",
               bbox_to_anchor=(x_mm / W, 1 - y_mm / H), frameon=False, handlelength=1.0, handleheight=1.0, borderaxespad=0)


@figure
def histology_03_regions(tr):
    """The regions volume of MR243 (intermediate/histology/MR243_regions_t2) on blockface slice 134, where all four labels are present,
    beside the photograph of that slice with the regions' outlines."""
    from matplotlib.colors import ListedColormap
    r, ra = tr.nii(f"brains/sub-{S}/intermediate/histology/{S}_regions_t2.nii.gz")
    jj = REGIONS_SLICE
    ph, px = bf_slice(tr, f"brains/sub-{S}/outputs/{S}_blockface_L2.nii.gz", jj)
    lab = regions_on_bf(tr, jj)
    bx = bbox((ph.sum(-1) > 0) | (lab > 0), int(2 / px))
    fig, axs, _ = one_row([(crop(lab, bx), px), (crop(ph, bx), px)], gutter=8.0, extra_mm=50)
    axs[0].images[0].remove(); show(axs[0], crop(lab, bx), px, cmap=ListedColormap(REGION_COLORS), vmin=-0.5, vmax=4.5, interpolation="nearest")
    for k in (1, 2, 3, 4):
        FS.outline(axs[1], crop(lab, bx) == k, lw=0.4, color=FS.ON_IMG if k < 4 else FS.COL["pred"])
    head(axs[0], "a", f"Regions on slice {jj}"); head(axs[1], "b", "On its photograph")
    FS.dirs_row(axs[0], axs[1], "coronal", pad=1.5)
    x1 = box_mm(fig, axs[1])[2] + LET_MM + 3.0
    region_key(fig, x1, TOP_MM + HEAD_MM)
    bar_below(axs[0], 1 / px, 10, y_frac=0.08)
    return fig


@figure
def histology_04_components(tr):
    """One section's components: section 5 is registered as two, its left and right hemispheres, each to its own part of the photograph
    of slice 14 (the regions volume divides the photograph)."""
    sec, spx = section_thumb(tr, PIECES)
    ph, px = bf_slice(tr, f"brains/sub-{S}/outputs/{S}_blockface_L2.nii.gz", PIECES_SLICE)
    lab = regions_on_bf(tr, PIECES_SLICE)
    t = ph.sum(-1) > 0; bx = bbox(t, int(2 / px))
    from scipy import ndimage as ndi
    left = ndi.binary_dilation(lab == 1, iterations=2) & t; right = ndi.binary_dilation(lab == 2, iterations=2) & t & ~left
    L = crop(np.where(left[..., None], ph, 0), bbox(left, int(1 / px))); R = crop(np.where(right[..., None], ph, 0), bbox(right, int(1 / px)))
    fig, axs, _ = one_row([(sec, spx), (L, px), (R, px)], gutter=8.0)
    head(axs[0], "a", f"Section {PIECES_N} as scanned"); head(axs[1], "b", "Component 1: left"); head(axs[2], "c", "Component 2: right")
    FS.dirs_row(axs[1], axs[2], "coronal", pad=1.5)
    bar_below(axs[0], 1 / spx, 10, y_frac=0.08)                            # one print scale: one bar
    return fig


@figure
def histology_05_registered(tr):
    """Section 36 after registration, zoomed on its left half: the photograph, the registered section, and the registered section's
    tissue outline on the photograph (the check by eye)."""
    from scipy import ndimage as ndi
    ph, px = bf_slice(tr, f"brains/sub-{S}/outputs/{S}_blockface_L2.nii.gz", SECTION_SLICE)
    hi, _ = bf_slice(tr, f"brains/sub-{S}/outputs/{S}_histology_blockface.nii.gz", SECTION_SLICE)
    t = hi.max(-1) > 0.02
    w = t.shape[1]; left = t.copy(); left[:, w // 2:] = False
    bx = bbox(left | ((ph.sum(-1) > 0) & (np.arange(w)[None, :] < w // 2)), int(1 / px))
    tis = ndi.binary_fill_holes(ndi.binary_opening(crop(t, bx), iterations=2))
    fig, axs, _ = one_row([(crop(ph, bx), px), (crop(hi, bx), px), (crop(ph, bx), px)])
    FS.outline(axs[2], tis, lw=0.5)
    for i, (ax, tt) in enumerate(zip(axs, ("Photograph", "Registered section", "Its outline on the photograph"))):
        head(ax, "abc"[i], tt)
    FS.dirs_row(axs[0], axs[-1], "coronal", pad=1.5)
    bar_below(axs[0], 1 / px, 5, y_frac=0.08)
    return fig


@figure
def histology_06_volume(tr):
    """The registered sections as a volume on the blockface grid, one sagittal plane through the left LGN: the photographs, the
    registered histology, and which slices hold a registered section (white) or are interpolated between two (orange)."""
    from matplotlib.colors import to_rgb
    from scipy import ndimage as ndi
    ph = tr.img(f"brains/sub-{S}/outputs/{S}_blockface_L2.nii.gz"); hi = tr.img(f"brains/sub-{S}/outputs/{S}_histology_blockface.nii.gz")
    pv = tr.img(f"brains/sub-{S}/intermediate/histology/provenance_L2.nii.gz"); lb = tr.img(f"brains/sub-{S}/outputs/{S}_lgn_stn_blockface.nii.gz")
    i = int(round(np.argwhere(np.asanyarray(lb.dataobj) == 1)[:, 0].mean()))
    disp = lambda a: np.flipud(np.swapaxes(a, 0, 1))[:, ::-1]      # superior up; the L2 y axis runs posterior, so anterior to the right
    vy, vz = abs(ph.affine[1, 1]), abs(ph.affine[2, 2]); f = vy / vz
    p = ndi.zoom(rgb01(disp(np.asanyarray(ph.dataobj[i]))), (1, f, 1), order=0)
    h = ndi.zoom(rgb01(disp(np.asanyarray(hi.dataobj[i]))), (1, f, 1), order=0)
    v = ndi.zoom(disp(np.asanyarray(pv.dataobj[i])), (1, f), order=0)
    bx = bbox(p.sum(-1) > 0, int(2 / vz))
    prov = np.zeros(v.shape + (3,), np.float32); prov[v == 1] = to_rgb(FS.ON_IMG); prov[v == 2] = to_rgb(FS.COL["pred"])
    fig, axs, _ = one_row([(crop(p, bx), vz), (crop(h, bx), vz), (crop(prov, bx), vz)])
    for k, (ax, tt) in enumerate(zip(axs, ("Photographs", "Registered histology", "Registered (white), interpolated (orange)"))):
        ax.images[0].set_interpolation("nearest"); head(ax, "abc"[k], tt)
    FS.dirs_row(axs[0], axs[-1], "sagittal", pad=1.5)
    bar_below(axs[0], 1 / vz, 10, y_frac=0.08)
    return fig


@figure
def histology_07_tracer(tr):
    """MR243's tracer labels on the blockface (the slice with the most bundle voxels, over the registered section) and in the template
    (a sagittal and an axial plane through the pathway's center, the labels within 2.5 mm of the plane): the injection site, the
    terminal fields, and the bundles, joined in the template into one pathway (charted and inferred)."""
    from matplotlib.colors import to_rgb
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    c = to_rgb(FS.TRACER[S]); g2 = to_rgb(FS.ON_IMG2)
    inj_t, _ = tr.nii(f"brains/sub-{S}/outputs/{S}_tracer_injection_site_template.nii.gz")
    ter_t, _ = tr.nii(f"brains/sub-{S}/outputs/{S}_tracer_terminals_template.nii.gz")
    con_t, _ = tr.nii(f"brains/sub-{S}/outputs/{S}_tracer_bundles_connected_template.nii.gz")
    t2, _ = tr.nii("template/outputs/template_T2w.nii.gz"); tm, _ = tr.nii("template/outputs/template_brain_mask.nii.gz")
    bun_b = np.asanyarray(tr.img(f"brains/sub-{S}/outputs/{S}_tracer_bundles_blockface.nii.gz").dataobj)
    jb = int(np.argmax((bun_b > 0).sum(axis=(0, 2))))                   # the slice with the most bundle voxels
    hi, px = bf_slice(tr, f"brains/sub-{S}/outputs/{S}_histology_blockface.nii.gz", jb)
    ph, _ = bf_slice(tr, f"brains/sub-{S}/outputs/{S}_blockface_L2.nii.gz", jb)
    ib, _ = bf_slice(tr, f"brains/sub-{S}/outputs/{S}_tracer_injection_site_blockface.nii.gz", jb)
    tb_, _ = bf_slice(tr, f"brains/sub-{S}/outputs/{S}_tracer_terminals_blockface.nii.gz", jb)
    bb_, _ = bf_slice(tr, f"brains/sub-{S}/outputs/{S}_tracer_bundles_blockface.nii.gz", jb)
    bx = bbox(ph.sum(-1) > 0, int(2 / px))
    cen = np.round(np.argwhere(con_t > 0).mean(0)).astype(int)          # the pathway's center in the template
    slab = TRACER_SLAB_VOX
    tpan = []
    for pl in ("sagittal", "axial"):
        k = FS.PLANE_AXIS[pl]; i0 = int(cen[k])
        proj = lambda v: FS.section(np.take(v, range(i0 - slab, i0 + slab + 1), axis=k).max(axis=k, keepdims=True), pl, 0)
        m2 = FS.section(tm, pl, i0); tbx = bbox(m2 > 0.5, 8)
        tpan.append((pl, crop(FS.section(window(t2, tm), pl, i0) * (m2 > 0.5), tbx), *(crop(proj(v), tbx) for v in (inj_t, ter_t, con_t))))
    print(f"  blockface slice with the most bundle voxels: {jb}; the template's planes: x {cen[0]}, z {cen[2]} (slab +-{slab} voxels)")
    dims = [(hi[bx[0]:bx[1]].shape[0] * px, (bx[3] - bx[2]) * px)] + [(p[1].shape[0] * 0.25, p[1].shape[1] * 0.25) for p in tpan]
    g = 2 * LET_MM + 1.0; sc = fit_scale([w for h, w in dims], gutter=g)
    hrow = max(h for h, w in dims) * sc
    fig = new_fig(TOP_MM + HEAD_MM + hrow + BAR_MM + 6.0)
    axs = row(fig, dims, TOP_MM + HEAD_MM, sc, gutter=g)
    show(axs[0], crop(hi, bx), px)
    for ax, p in zip(axs[1:], tpan):
        show(ax, p[1], 0.25)

    def over(ax, inj, ter, bun, con=None):
        ov = np.zeros(inj.shape + (4,), np.float32)
        ov[ter > 0] = (*g2, 0.55)
        if con is not None:
            ov[con == 2] = (*c, 0.4); ov[con == 1] = (*c, 0.9)
        else:
            ov[bun > 0] = (*c, 0.9)
        ax.imshow(ov, interpolation="nearest"); FS.outline(ax, inj > 0, color=FS.ON_IMG, lw=0.7)
    over(axs[0], crop(ib, bx), crop(tb_, bx), crop(bb_, bx))
    for ax, p in zip(axs[1:], tpan):
        over(ax, p[2], p[3], None, p[4])
    head(axs[0], "a", f"Blockface, slice {jb}"); head(axs[1], "b", "Template, sagittal"); head(axs[2], "c", "Axial")
    for ax, pl in zip(axs, ("coronal", "sagittal", "axial")):
        FS.dirs(ax, pl, pad=1.5)
    hs = [Line2D([], [], color=FS.ON_IMG, lw=0.7, label="Injection site"), Patch(facecolor=g2, alpha=0.55, edgecolor="none", label="Terminal fields"),
          Patch(facecolor=c, alpha=0.9, edgecolor="none", label="Bundles (charted)"), Patch(facecolor=c, alpha=0.4, edgecolor="none", label="Joins (inferred)")]
    W, H = (v * 25.4 for v in fig.get_size_inches())
    fig.legend(handles=hs, loc="upper right", bbox_to_anchor=(1 - FS.MARGIN_MM / W, 1 - (TOP_MM + HEAD_MM + hrow + 2.0) / H), ncol=4,
               frameon=False, handlelength=1.2, borderaxespad=0)
    bar_below(axs[0], 1 / px, 10)                                          # one print scale: one bar
    return fig


@figure
def histology_08_lgn_stn(tr):
    """The LGN and STN of MR243 (solid lines LGN, dashed STN): on the blockface, over the registered section 36, and in the template,
    through each nucleus."""
    hi, px = bf_slice(tr, f"brains/sub-{S}/outputs/{S}_histology_blockface.nii.gz", SECTION_SLICE)
    lb, _ = bf_slice(tr, f"brains/sub-{S}/outputs/{S}_lgn_stn_blockface.nii.gz", SECTION_SLICE)
    tb, _ = tr.nii("template/outputs/template_b0.nii.gz"); tm, _ = tr.nii("template/outputs/template_brain_mask.nii.gz")
    lt, jl = template_lgn_plane(tr, 1); _, js = template_lgn_plane(tr, 3)
    bx = bbox(hi.max(-1) > 0.02, int(2 / px)); tw = window(tb, tm)
    P = [(crop(hi, bx), px, crop(lb, bx))]
    for j in (jl, js):
        m2 = FS.section(tm, "coronal", j); b2 = bbox(m2 > 0.5, 8)
        P.append((crop(FS.section(tw, "coronal", j) * (m2 > 0.5), b2), 0.25, crop(FS.section(lt, "coronal", j), b2)))
    fig, axs, _ = one_row([(p[0], p[1]) for p in P])
    for i, (ax, p, tt) in enumerate(zip(axs, P, (f"On the blockface, slice {SECTION_SLICE}", "In the template, through the LGN",
                                                 "Through the STN"))):
        head(ax, "abc"[i], tt)
        for k in (1, 2):
            FS.outline(ax, p[2] == k, color=FS.BRAIN[S], lw=0.6)
        for k in (3, 4):
            FS.outline(ax, p[2] == k, color=FS.BRAIN[S], lw=0.6, linestyles="dashed")
    FS.dirs_row(axs[0], axs[-1], "coronal", pad=1.5)
    bar_below(axs[0], 1 / px, 10, y_frac=0.08)                             # one print scale: one bar
    return fig


QC_CASES = (("mr243LY_c18_s15nDF_0001", 44, 1), ("mr243LY_c18_s68nDF_0003", 155, 3))
#   the registration check's best and worst components of MR243 (paper/numbers/registration_qc_histo2bf_h9.json, the 12 Nov rows):
#   section 15's left hemisphere (photograph slice 44) and section 68's cerebellum (slice 155); (tag, blockface slice, regions label)


@figure
def histology_09_qc(tr):
    """The registration check on two components of MR243: one that passes (section 15, left hemisphere) and one that fails (section 68,
    cerebellum), each as the photograph with the registered section's tissue outline; the Dice coefficients from the check's record."""
    import json
    from scipy import ndimage as ndi
    rec = json.load(open(tr("paper/numbers/registration_qc_histo2bf_h9.json")))
    rec = rec.get("data", rec)
    rows = {r["tag"]: r for r in rec if r["subject"] == S and r["time"].startswith("2026-11-12")}
    P = []
    for tag, j, k in QC_CASES:
        q = rows[tag]; assert q["bf_slice"] == j, (tag, q["bf_slice"], j)
        ph, px = bf_slice(tr, f"brains/sub-{S}/outputs/{S}_blockface_L2.nii.gz", j)
        hi, _ = bf_slice(tr, f"brains/sub-{S}/outputs/{S}_histology_blockface.nii.gz", j)
        lab = regions_on_bf(tr, j)
        part = ndi.binary_dilation(lab == k, iterations=3)
        bx = bbox(part & (ph.sum(-1) > 0), int(1.5 / px))
        tis = ndi.binary_fill_holes(ndi.binary_opening(crop((hi.max(-1) > 0.02) & part, bx), iterations=2))
        P.append((crop(ph, bx), crop(np.where(part[..., None], hi, 0), bx), tis, q))
    dims = []
    for p in P:
        dims += [(p[0].shape[0] * px, p[0].shape[1] * px)] * 2
    g = FS.GUTTER_MM; sc = fit_scale([w for h, w in dims], gutter=g, extra_mm=6.0)
    fig = new_fig(TOP_MM + 2 * HEAD_MM + max(h for h, w in dims) * sc + BAR_MM)
    x = FS.MARGIN_MM + LET_MM; axs = []
    for i, (h, w) in enumerate(dims):
        ax = FS.axes_mm(fig, x, TOP_MM + 2 * HEAD_MM, w * sc, h * sc); ax.set_axis_off(); axs.append(ax)
        x += w * sc + (g + 6.0 if i == 1 else g)
    for c, (ph_, hi_, tis, q) in enumerate(P):
        a0, a1 = axs[2 * c], axs[2 * c + 1]
        show(a0, hi_, px); show(a1, ph_, px); FS.outline(a1, tis, lw=0.5)
        m = q["metrics"]; part_name = {1: "left hemisphere", 2: "right hemisphere", 3: "cerebellum"}[QC_CASES[c][2]]
        W, H = (v * 25.4 for v in fig.get_size_inches())
        fig.text(box_mm(fig, a0)[0] / W, 1 - (TOP_MM + HEAD_MM - 1.0) / H, f"Section {int(q['tag'].split('_s')[1][:2])}, {part_name}: "
                 f"{'passes' if q['gate'] == 'pass' else 'fails'}, Dice {m['dice_nonlinear']:.2f}", ha="left", va="baseline",
                 fontsize=FS.MIN_PT, color=FS.FG)
        head(a0, "ac"[c], "Registered"); head(a1, "bd"[c], "On its photograph")
    FS.dirs(axs[0], "coronal", pad=1.5)
    bar_below(axs[0], 1 / px, 5)
    return fig


# ---- the overview (00_start_here.md) -----------------------------------------------------------------------------------------
@figure
def overview_01_chain(tr):
    """The chain of spaces, with the command between each pair: a section as scanned (histo-piece), its photograph (bf-native), the
    brain's b0 (b0-native), the MRI template (mri-template, T2-weighted); below, the blockface template's channel (bf-template, shown on
    the MRI template's grid), the route for a brain without MRI."""
    sec, spx = section_thumb(tr)
    ph, px = bf_slice(tr, f"brains/sub-{S}/outputs/{S}_blockface_L2.nii.gz", SECTION_SLICE)
    bx = bbox(ph.sum(-1) > 0, int(1 / px))
    b0, ba, m, tb, ta, tm, f, fa = mri_example(tr)
    tp = template_planes(tm); npl = native_planes(ba, f, fa, tp)
    n_m = FS.section(m, "coronal", npl["coronal"]); nb = bbox(n_m > 0.5, 2)
    t2, _ = tr.nii("template/outputs/template_T2w.nii.gz")
    m2 = FS.section(tm, "coronal", tp["coronal"]); tbx = bbox(m2 > 0.5, 4)
    rgb, _ = tr.nii("template/outputs/template_blockface_rgb.nii.gz"); bm, _ = tr.nii("template/outputs/template_blockface_mask.nii.gz")
    bm2 = FS.section(bm, "coronal", tp["coronal"])
    top = [(sec, spx, "Section (histo-piece)"), (crop(ph, bx), px, "Photograph (bf-native)"),
           (crop(FS.section(window(b0, m), "coronal", npl["coronal"]), nb), abs(ba[0, 0]), "b0 (b0-native)"),
           (crop(FS.section(window(t2, tm), "coronal", tp["coronal"]) * (m2 > 0.5), tbx), 0.25, "MRI template (mri-template)")]
    low = (crop(rgb01(FS.section(rgb, "coronal", tp["coronal"])) * (bm2 > 0.5)[..., None], tbx), 0.25, "Blockface template (bf-template)")
    ga = 22.0                                                        # gutters that hold an arrow and its command
    sc = (W_MM - 2 * FS.MARGIN_MM - 3 * ga) / sum(p[0].shape[1] * p[1] for p in top)
    ht = max(p[0].shape[0] * p[1] for p in top) * sc; hl = low[0].shape[0] * low[1] * sc
    gap_v = 12.0
    fig = new_fig(TOP_MM + HEAD_MM + ht + gap_v + HEAD_MM + hl + BAR_MM)
    x = FS.MARGIN_MM; axs = []
    for img, ppx, t in top:
        h, w = img.shape[0] * ppx * sc, img.shape[1] * ppx * sc
        ax = FS.axes_mm(fig, x, TOP_MM + HEAD_MM + (ht - h) / 2, w, h); ax.set_axis_off(); show(ax, img, ppx)
        axs.append(ax); x += w + ga
    for k, (ax, (img, ppx, t)) in enumerate(zip(axs, top)):
        last = k == len(top) - 1
        ax.annotate(t, xy=(1.0 if last else 0.5, 1), xycoords="axes fraction", xytext=(0, 3), textcoords="offset points",
                    ha="right" if last else "center", va="bottom", fontsize=FS.MIN_PT, color=FS.FG, annotation_clip=False)
    b1 = box_mm(fig, axs[1]); w_l, h_l = low[0].shape[1] * low[1] * sc, hl
    xl = (b1[0] + b1[2]) / 2 - w_l / 2; yl = TOP_MM + HEAD_MM + ht + gap_v + HEAD_MM
    axl = FS.axes_mm(fig, xl, yl, w_l, h_l); axl.set_axis_off(); show(axl, low[0], low[1])
    axl.annotate(low[2], xy=(0.5, 1), xycoords="axes fraction", xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                 fontsize=FS.MIN_PT, color=FS.FG, annotation_clip=False)
    cmds = ("linc-histology", "linc-register mri", "linc-register mri-template")
    for a0, a1, cmd in zip(axs[:-1], axs[1:], cmds):
        p, q = box_mm(fig, a0), box_mm(fig, a1)
        y = TOP_MM + HEAD_MM + ht / 2
        arrow(fig, (p[2] + 1.0, y), (q[0] - 1.0, y))
        lines = cmd.split(" ", 1)
        for k, s in enumerate(reversed(lines)):
            code_text(fig, (p[2] + q[0]) / 2, y - 1.2 - 3.0 * k, s)
    # the route without MRI: bf-native down to bf-template, then on to mri-template
    yb = box_mm(fig, axs[1])[3]
    arrow(fig, ((b1[0] + b1[2]) / 2, yb + 1.0), ((b1[0] + b1[2]) / 2, yl - HEAD_MM - 0.5))
    code_text(fig, (b1[0] + b1[2]) / 2 + 1.5, (yb + yl - HEAD_MM) / 2 + 1.2, "linc-register blockface-template", ha="left", va="center")
    p, q = box_mm(fig, axl), box_mm(fig, axs[3])
    yy = (p[1] + p[3]) / 2; x_end, y_end = (q[0] + q[2]) / 2, q[3] + 1.0
    arrow(fig, (p[2] + 1.0, yy), (x_end, yy), head=False); arrow(fig, (x_end, yy), (x_end, y_end))     # right, then up to the MRI template
    code_text(fig, (p[2] + x_end) / 2, yy + 1.2, "code/template/build_link.py", va="top")
    bar_below(axs[0], 1 / spx, 10, y_frac=0.08)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", default=os.environ.get("LINC_DATA_TREE"), help="the data tree (default $LINC_DATA_TREE)")
    ap.add_argument("--out", help="where the figures go (default <tree>/docs/tutorials/figures)")
    ap.add_argument("--only", nargs="+", metavar="NAME", help="these figures only (--list names them)")
    ap.add_argument("--list", action="store_true", help="list the figures and exit")
    a = ap.parse_args()
    if a.list:
        for n, fn in FIGS.items():
            print(f"{n:28s} {fn.__doc__.split(chr(10))[0]}")
        return
    if not a.tree:
        sys.exit("give --tree <data tree> (or source settings/roots.sh, which sets $LINC_DATA_TREE)")
    names = a.only or list(FIGS)
    bad = [n for n in names if n not in FIGS]
    if bad:
        sys.exit(f"unknown figure(s) {bad}: --list names them")
    out = Path(a.out or Path(a.tree) / "docs" / "tutorials" / "figures")
    FS.setup()
    for n in names:
        t0 = time.time(); tr = Tree(a.tree)
        fig = FIGS[n](tr)
        save(fig, out, tr, n)
        print(f"{n}: {out / (n + '.png')} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
