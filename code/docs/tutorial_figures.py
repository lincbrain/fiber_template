#!/usr/bin/env python3
"""The figures of the step-by-step tutorials (docs/tutorials/ of this repository), one function per figure, in the paper's figure style
(code/figures/figstyle.py: black background, letters, headers and scale bars outside the images, sentence case; command names, set in
the figures as code, are the one exception to sentence case).

Every figure reads only files of the data tree (and the release's settings/histology.yaml, for the sections' pixel size). It is written
to the repository's docs/tutorials/figures/<name>.png (200 dpi, 178 mm wide, no text chunks), and its provenance, which holds this
machine's paths, to outputs/docs/figures_provenance/<name>.provenance.json (outputs/ is not tracked). The example
brain is MR243 (MM1 in the paper); the MRI template's figures use MF283 (MF1), one of its four brains. The histology figures read the
JPEG 2000 sections, so they need glymur, which $PY_TIRL has. Each figure takes seconds, the histology ones up to a minute.

    $PY_TIRL code/docs/tutorial_figures.py --tree $LINC_DATA_TREE                 # every figure
    $PY_TIRL code/docs/tutorial_figures.py --tree $LINC_DATA_TREE --only mri_05_b0_in_template
    $PY_TIRL code/docs/tutorial_figures.py --list                                 # the figures, in order

--tree defaults to $LINC_DATA_TREE (settings/roots.sh sets it). --out (a trial render elsewhere) writes the provenance beside the PNGs.

The house rules (the author, 12 Nov 2026), drawn by one layout engine (grid() and its helpers, below): one print scale per figure, as
large as the page lets, tight crops and equal gutters; each panel's letter at its top left and its title centered over it; a brain named
by its Specimen ID and its paper ID, "MR243 (MM1)" (docs/brain_identifiers.md); no plane names where the orientation letters show the
view (the captions name the planes and say that axial views have anterior up); orientation letters by figstyle.orient_letters; one
scale bar per figure, in an empty corner (figstyle.scalebar_free); arrows named by their command where that helps. MR252's photographs
are shown mirrored left-right into the neurological convention (the camera sees the subject's left on the image's right), as in the
paper's reconstruction figure. save() also refuses texts that overlap or leave the page.
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

FIG_OUT = REPO / "docs" / "tutorials" / "figures"                   # the PNGs, tracked with the tutorials
PROV_OUT = REPO / "outputs" / "docs" / "figures_provenance"         # their provenance: machine paths, so local only
DPI = 200                                    # the tutorials' figures (FS.DPI is the paper's 600)
W_MM = 178.0                                  # full page width

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
# One engine for every figure (the author's house rules, 12 Nov 2026): rows of panels at ONE print scale per figure, as large as the
# page lets; tight crops (crop padding is empty space); one gutter (figstyle GUTTER_MM) between panels and rows, wider only where an
# arrow or a pair of orientation letters needs it; each panel's letter at its top left and its title centered over it (a title over a
# group of panels follows the group's letter); orientation letters by figstyle.orient_letters (left and right of the images only, 1 mm
# from the tissue, one line per image row); one scale bar per figure, its length above it, placed by measurement in an empty corner
# (figstyle.scalebar_free), never in added room.
G, MG = FS.GUTTER_MM, FS.MARGIN_MM
BASE_MM = 1.4                                 # a title line's baseline above the panels' top
TITLE_MM = 4.6                                # the band of the title line nearest the panels (a 9 pt bold letter's cap above it)
LINE_MM = 3.4                                 # each further title line
CROP_MM = 0.3                                 # tight crops: the tissue plus this
CMD_MM = 3.4                                  # one line of a command band


def cropm(px):
    """The crop margin, pixels: CROP_MM of tissue, at least one pixel (so that an outline at the tissue's edge is whole)."""
    return max(1, int(round(CROP_MM / px)))


class Pn:
    """One image panel: img (2-D, or RGB), px (tissue mm per pixel), view (a plane of figstyle.DIRS: its orientation letters; None:
    none), letter and title (on the row's title line `tline`, 1 = the line nearest the panels); lg: its letter group (panels of one
    group share one pair; default: a run of panels in one view); gap: the gutter before it (mm; widened if the letters need more);
    the rest goes to imshow (cmap, vmin, vmax, interpolation)."""

    def __init__(self, img, px, view=None, letter=None, title=None, lg=None, gap=None, tline=1, **kw):
        self.img, self.px, self.view, self.letter, self.title = img, float(px), view, letter, title
        self.lg, self.gap, self.tline, self.kw = lg, gap, tline, kw

    @property
    def w(self):
        return self.img.shape[1] * self.px

    @property
    def h(self):
        return self.img.shape[0] * self.px


class Row:
    """A row of panels: `lines` title lines above it; valign "top" or "center" (a flow, its arrows at the middle); below: room under
    it (mm: a command band, a key); under: for each panel, the panel of the row above it is centered under (instead of a packed row)."""

    def __init__(self, panels, lines=1, valign="top", below=0.0, under=None, above=0.0):
        self.panels, self.lines, self.valign, self.below, self.under, self.above = panels, lines, valign, below, under, above


def room(s):
    """The print room of an orientation letter beside the tissue: its 1 mm gap and its ink (figstyle.orient_room)."""
    return FS.orient_room(s)


def _groups(r):
    k, prev = 0, None
    for p in r.panels:
        if p.view is None:
            p.grp = None
        elif p.lg is not None:
            p.grp = ("lg", p.lg)
        elif prev is not None and prev.view == p.view and prev.lg is None and prev.grp is not None:
            p.grp = prev.grp
        else:
            k += 1; p.grp = ("auto", k)
        prev = p


def _gaps(r):
    out = []
    for a, b in zip(r.panels[:-1], r.panels[1:]):
        split = a.grp != b.grp
        need = ((room(FS.DIRS[a.view][1]) if a.view and split else 0.0) + (room(FS.DIRS[b.view][0]) if b.view and split else 0.0)
                + (1.0 if a.view and b.view and split else 0.0))
        out.append(max(G if b.gap is None else b.gap, need))
    return out


def grid(rows, sc=None, reserve_right=0.0, bottom=0.0):
    """Lay the rows out at one print scale (the largest at which every packed row fits the page) and draw the panels and their letters
    and titles. Returns (fig, sc); each panel gets .ax, .x, .y (page mm, its top left) and .row."""
    for r in rows:
        _groups(r); r.gaps = _gaps(r)
        f, l = r.panels[0], r.panels[-1]
        r.left = MG + (room(FS.DIRS[f.view][0]) if f.view else 0.0)
        r.right = MG + (room(FS.DIRS[l.view][1]) if l.view else 0.0) + reserve_right
    if sc is None:
        sc = min((W_MM - r.left - r.right - sum(r.gaps)) / sum(p.w for p in r.panels) for r in rows if r.under is None)
    y, prev = MG, None
    for r in rows:
        y += r.above + (TITLE_MM + LINE_MM * (r.lines - 1) if r.lines else 0.0)
        r.top, r.hrow = y, max(p.h for p in r.panels) * sc
        x = r.left
        for i, (p, g) in enumerate(zip(r.panels, [0.0] + r.gaps)):
            if r.under is not None:
                q = prev.panels[r.under[i]]; p.x = q.x + (q.w - p.w) * sc / 2
            else:
                x += g; p.x = x; x += p.w * sc
            p.y = r.top + ((r.hrow - p.h * sc) / 2 if r.valign == "center" else 0.0)
            p.row = r
        assert max(p.x + p.w * sc for p in r.panels) <= W_MM - MG + 1e-6, "a row runs off the page"
        y, prev = r.top + r.hrow + r.below + G, r
    fig = new_fig(y - G + MG + bottom)
    fig._sc = sc
    for r in rows:
        for p in r.panels:
            p.ax = FS.axes_mm(fig, p.x, p.y, p.w * sc, p.h * sc); p.ax.set_axis_off(); show(p.ax, p.img, p.px, **p.kw)
        for p in r.panels:
            if p.letter or p.title:
                head_at(fig, p.x, p.x + p.w * sc, base(r, p.tline), p.letter, p.title)
    return fig, sc


def base(r, line=1):
    """The baseline (page mm) of a row's title line (1: the nearest the panels)."""
    return r.top - BASE_MM - LINE_MM * (line - 1)


def page(fig):
    return tuple(v * 25.4 for v in fig.get_size_inches())


def txt(fig, x_mm, y_mm, s, ha="left", va="baseline", fs=FS.MIN_PT, color=FS.FG, **kw):
    W, H = page(fig)
    return fig.text(x_mm / W, 1 - y_mm / H, s, ha=ha, va=va, fontsize=fs, color=color, **kw)


def ink(s, fs=FS.MIN_PT, bold=False, mono=False):
    """A one-line text's ink, (x0, x1) in mm from its origin."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    fp = FontProperties(family="monospace" if mono else FS.FONT, weight="bold" if bold else "normal")
    e = TextPath((0, 0), s, size=fs, prop=fp).get_extents()
    return e.x0 / 72 * 25.4, e.x1 / 72 * 25.4


def head_at(fig, x0, x1, y, letter=None, title=None, align="center"):
    """A bold panel letter at x0 and a title on the baseline y (mm): the title centered over x0..x1 by its ink, or, if that would run
    into the letter (or align="left"), 1.5 mm after the letter."""
    xr = x0
    if letter:
        t = txt(fig, x0, y, letter, fs=FS.LETTER_PT, fontweight="bold")
        t.set_gid(FS.PANEL_LETTER_GID)
        xr = x0 + ink(letter, FS.LETTER_PT, bold=True)[1] + 1.5
    if title:
        a, b = ink(title)
        xc = (x0 + x1) / 2 - (b - a) / 2 - a
        txt(fig, xc if (align == "center" and xc + a >= xr) else xr - a, y, title)


def span(fig, panels, title, line, letter=None, align="center"):
    """A title over a group of panels (on the row's title line `line`), centered over the group, or after the group's letter."""
    sc, p0 = fig._sc, panels[0]
    x0 = p0.x
    if letter is None and p0.letter and p0.tline == line:              # the group's first panel has its letter on this line
        x0 += ink(p0.letter, FS.LETTER_PT, bold=True)[1] + 1.5
    head_at(fig, x0, max(p.x + p.w * sc for p in panels), base(p0.row, line), letter, title, align)


def bracket(fig, panels, label, line, letter=None):
    """A label over a group of panels (a subset of its row), on the row's title line `line`, inside a thin gray bracket that spans the
    group (after its letter, if any), so that it does not read as one panel's title."""
    sc, p0 = fig._sc, panels[0]
    y = base(p0.row, line); x0 = p0.x; x1 = max(p.x + p.w * sc for p in panels)
    if letter:
        t = txt(fig, x0, y, letter, fs=FS.LETTER_PT, fontweight="bold"); t.set_gid(FS.PANEL_LETTER_GID)
        x0 += ink(letter, FS.LETTER_PT, bold=True)[1] + 1.5
    a, b = ink(label); xm = (x0 + x1) / 2
    txt(fig, xm - (b - a) / 2 - a, y, label)
    ym = y - 0.9                                                          # the bracket at the label's mid-height
    for xa, xb in ((x0, xm - (b - a) / 2 - 1.5), (xm + (b - a) / 2 + 1.5, x1)):
        arrow(fig, (xa, ym), (xb, ym), head=False, color=FS.GRAY)
    for xe in (x0, x1):
        arrow(fig, (xe, ym), (xe, ym + 1.2), head=False, color=FS.GRAY)


def orient(fig, rows):
    """The orientation letters: one pair per letter group, one line per row (figstyle.orient_letters)."""
    out = []
    for r in rows:
        pairs = []
        for g in dict.fromkeys(p.grp for p in r.panels if p.grp is not None):
            ps = [p for p in r.panels if p.grp == g]
            pairs.append((ps[0].ax, ps[-1].ax, ps[0].view))
        if pairs:
            out.append(pairs)
    if out:
        FS.orient_letters(fig, out)


def bar(fig, sc, mm=10.0, corners=("bottom left", "bottom right", "top right", "top left"), reach=(40.0, 15.0), region=None):
    """The figure's one scale bar, `mm` long at the print scale sc, its length above it, in an empty corner of the page: the first of
    `corners` whose corner box (reach = its width and height, mm) has an empty place, 2 mm clear of everything drawn (ink; check()
    measures 1.5 mm from the texts' boxes), nearest that corner (figstyle.scalebar_free). region: one box instead (page mm, x0, x1, y0,
    y1). Call it last."""
    W, H = page(fig)
    for c in corners:
        if region is not None:
            reg = region
        else:
            x0, x1 = (MG, MG + reach[0]) if "left" in c else (W - MG - reach[0], W - MG)
            y0, y1 = (0.3, reach[1]) if "top" in c else (H - reach[1], H - MG)
            reg = (x0, x1, max(y0, 0.3), min(y1, H - MG))
        try:
            return FS.scalebar_free(fig, mm, sc, reg, corner=c, clear_mm=2.0)
        except ValueError:
            continue
    raise ValueError(f"no empty corner for the {mm:g} mm bar")


def box(p, sc):
    """A panel's box in page mm: (x0, y0, x1, y1), y down."""
    return p.x, p.y, p.x + p.w * sc, p.y + p.h * sc


def show(ax, img, px_mm, cmap="gray", vmin=0, vmax=1, interpolation="bilinear"):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interpolation)
    ax.set_xlim(-0.5, img.shape[1] - 0.5); ax.set_ylim(img.shape[0] - 0.5, -0.5)
    ax.px_mm = px_mm
    return ax


def code_text(fig, x_mm, y_mm, s, ha="center", va="bottom", **kw):
    """A command name in the figure (monospace, bold and half a point larger, so that its underscores survive a viewer's downscaling;
    exempt from the sentence-case rule)."""
    CODE_TEXT.add(s)
    return txt(fig, x_mm, y_mm, s, ha=ha, va=va, fs=FS.MIN_PT + 0.5, family="monospace", fontweight="bold", **kw)


def code_lines(fig, x_mm, y_mm, lines, up=True):
    """Command lines centered on x_mm: stacked upward from the baseline y_mm (up=True: above an arrow), or downward from y_mm (a band
    under a row). A line that starts with "~" is plain words (secondary gray), not a command."""
    out = []
    seq = list(reversed(lines)) if up else list(lines)
    for k, s in enumerate(seq):
        yy = y_mm - k * CMD_MM if up else y_mm + k * CMD_MM
        if s.startswith("~"):
            out.append(txt(fig, x_mm, yy, s[1:], ha="center", va="bottom" if up else "top", color=FS.GRAY))
        elif s:
            out.append(code_text(fig, x_mm, yy, s, va="bottom" if up else "top"))
    return out


def mixed(fig, xc, y, parts, gap=1.2):
    """One line of plain words and command names, centered on xc (mm) on the baseline y: parts = [(text, is_command), ...]."""
    ws = [ink(t, FS.MIN_PT + 0.5, bold=True, mono=True) if c else ink(t) for t, c in parts]
    x = xc - (sum(b - a for a, b in ws) + gap * (len(parts) - 1)) / 2
    for (t, c), (a, b) in zip(parts, ws):
        (code_text(fig, x - a, y, t, ha="left", va="baseline") if c else txt(fig, x - a, y, t)); x += b - a + gap


def arrow(fig, p0_mm, p1_mm, head=True, color=FS.FG):
    """A plain arrow between two points of the page (mm from the top-left corner); head=False: a line."""
    from matplotlib.patches import FancyArrowPatch
    W, H = page(fig)
    f = lambda p: (p[0] / W, 1 - p[1] / H)                                # noqa: E731
    fig.add_artist(FancyArrowPatch(f(p0_mm), f(p1_mm), transform=fig.transFigure, arrowstyle="-|>" if head else "-", mutation_scale=6,
                                   lw=0.8, color=color, shrinkA=0, shrinkB=0))


def arrow_gutter(fig, a, b, sc, y=None):
    """An arrow across the gutter between panels a and b, 1 mm from each, at height y (mm; default the row's middle)."""
    y = (a.row.top + a.row.hrow / 2) if y is None else y
    arrow(fig, (a.x + a.w * sc + 1.0, y), (b.x - 1.0, y))
    return y


def key(fig, handles, x_mm, y_mm, loc="upper left", ncol=1, **kw):
    """A key (legend) at page mm (x, y), frameless."""
    W, H = page(fig)
    return fig.legend(handles=handles, loc=loc, bbox_to_anchor=(x_mm / W, 1 - y_mm / H), ncol=ncol, frameon=False, borderaxespad=0,
                      handlelength=kw.pop("handlelength", 1.6), handletextpad=0.6, columnspacing=1.6, **kw)


def name(S):
    """A brain's name in the figures: its Specimen ID, the repository's name, and the paper's (docs/brain_identifiers.md)."""
    return f"{S} ({FS.label(S)})"


def new_fig(h_mm):
    import matplotlib.pyplot as plt
    return plt.figure(figsize=(W_MM * FS.MM, h_mm * FS.MM))


def _overlaps(fig):
    """Texts (and keys) that overlap one another or leave the page: the cut-off or overlapping text figstyle.check does not look for."""
    from matplotlib.text import Text
    fig.canvas.draw(); r = fig.canvas.get_renderer(); fb = fig.bbox
    ts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()
          and not any(t in lg.get_texts() for lg in fig.legends)]
    items = [(t.get_text()[:30], t.get_window_extent(r)) for t in ts] + [("key", lg.get_window_extent(r)) for lg in fig.legends]
    bad = []
    for i, (s, e) in enumerate(items):
        if e.x0 < fb.x0 - 0.5 or e.y0 < fb.y0 - 0.5 or e.x1 > fb.x1 + 0.5 or e.y1 > fb.y1 + 0.5:
            bad.append(f"text off the page: {s!r}")
        for s2, e2 in items[i + 1:]:
            if e.x0 < e2.x1 - 0.5 and e.x1 > e2.x0 + 0.5 and e.y0 < e2.y1 - 0.5 and e.y1 > e2.y0 + 0.5:
                bad.append(f"texts overlap: {s!r} and {s2!r}")
    return bad


def save(fig, out, prov_out, tree, name):
    """figstyle's checks (all but sentence case for the command names) and the overlap check, then the PNG at 200 dpi, with no text
    chunks (no paths or host names in a public file), and its provenance in prov_out."""
    import json
    import matplotlib.pyplot as plt
    from shared import provenance
    FS.tiles(fig)
    bad = [b for b in FS.check(fig) if not (b.startswith("not in sentence case") and any(repr(c[:40]) in b for c in CODE_TEXT))]
    bad += _overlaps(fig)
    if bad:
        raise ValueError(f"{name}: house rules broken:\n  " + "\n  ".join(bad))
    out.mkdir(parents=True, exist_ok=True); prov_out.mkdir(parents=True, exist_ok=True)
    png = out / f"{name}.png"
    fig.savefig(str(png), dpi=DPI, facecolor=fig.get_facecolor(), metadata={"Software": None})
    d = provenance.provenance_dict(str(png), inputs=sorted(set(tree.read)) + [os.path.abspath(__file__), os.path.abspath(FS.__file__)],
                                   cmd=" ".join(sys.argv), md5=False,
                                   extra=dict(figure=name, dpi=DPI, width_mm=round(fig.get_size_inches()[0] * 25.4, 1), house_rules="ok"))
    (prov_out / f"{name}.provenance.json").write_text(json.dumps(d, indent=1, default=str))
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




# ---- the MRI tutorial (01_mri.md) --------------------------------------------------------------------------------------------
GA = 23.0                                     # an arrow gutter that holds a two-line command above the arrow ("linc-register")


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
        bx = bbox(m2 > 0.5, cropm(abs(a[0, 0]))); top.append(Pn(crop(im * (m2 > 0.5), bx), abs(a[0, 0]), "coronal", title=name(B)))
    t2, _ = tr.nii("template/outputs/template_T2w.nii.gz")
    m2 = FS.section(tm, "coronal", tp["coronal"]); bx = bbox(m2 > 0.5, 1)
    top.append(Pn(crop(FS.section(window(t2, tm), "coronal", tp["coronal"]) * (m2 > 0.5), bx), 0.25, "coronal", gap=17.0))
    b0, ba, m, tb, ta, tm, f, fa = mri_example(tr)
    npl = native_planes(ba, f, fa, tp)
    n_im, n_m = FS.section(window(b0, m), "coronal", npl["coronal"]), FS.section(m, "coronal", npl["coronal"])
    nb = bbox(n_m > 0.5, cropm(abs(ba[0, 0])))
    mv = pull_itk(f, fa, b0, ba, "coronal", tp["coronal"])
    bot = [Pn(crop(n_im, nb), abs(ba[0, 0]), "coronal", "c", f"{name(S)}: b0, its mask"),
           Pn(crop(window(mv, mv > 0), bx), 0.25, "coronal", "d", "Its b0, in the template", gap=GA),
           Pn(crop(FS.section(window(tb, tm), "coronal", tp["coronal"]), bx), 0.25, "coronal", "e", "The template's b0")]
    rows = [Row(top, lines=2, valign="center"), Row(bot, valign="center")]
    fig, sc = grid(rows)
    bracket(fig, top[:4], "The template's four brains, T2-weighted, each in its own space", 2, letter="a")
    head_at(fig, top[4].x, top[4].x + top[4].w * sc, base(rows[0], 2), "b", "The MRI template")
    for p, ol in zip(bot, (crop(n_m, nb), crop(m2, bx), crop(m2, bx))):
        FS.outline(p.ax, ol, lw=0.5)
    y = arrow_gutter(fig, top[3], top[4], sc)
    txt(fig, (top[3].x + top[3].w * sc + top[4].x) / 2, y - 1.2, "Groupwise\nregistration", ha="center", va="bottom", color=FS.GRAY,
        multialignment="center", linespacing=1.1)
    y = arrow_gutter(fig, bot[0], bot[1], sc)
    code_lines(fig, (bot[0].x + bot[0].w * sc + bot[1].x) / 2, y - 1.2, ["linc-register", "mri-template"])
    orient(fig, rows)
    bar(fig, sc, corners=("bottom right", "bottom left"))
    return fig


@figure
def mri_02_input(tr):
    """The inputs: MF283's T2-weighted image, FA (from its tensor) and soma fraction, each in its own space, and MR243's b0 (coronal)."""
    P = []
    d, a = tr.nii(f"brains/sub-{T}/outputs/{T}_t2_T2w.nii.gz"); mk, _ = tr.nii(f"brains/sub-{T}/inputs/manual_annotations/{T}_t2_brain_mask.nii.gz")
    j = int(round(np.argwhere(mk > 0.5)[:, 1].mean())); m2 = FS.section(mk, "coronal", j); bx = bbox(m2 > 0.5, cropm(abs(a[0, 0])))
    P.append(Pn(crop(FS.section(window(d, mk), "coronal", j) * (m2 > 0.5), bx), abs(a[0, 0]), "coronal", "a", "T2-weighted"))
    t6, a = tr.nii(f"brains/sub-{T}/outputs/{T}_tensor_dti.nii.gz"); mk, _ = tr.nii(f"brains/sub-{T}/inputs/manual_annotations/{T}_tensor_brain_mask.nii.gz")
    j = int(round(np.argwhere(mk > 0.5)[:, 1].mean())); m2 = FS.section(mk, "coronal", j); bx = bbox(m2 > 0.5, cropm(abs(a[0, 0])))
    P.append(Pn(crop(fa_slice(t6, "coronal", j) * (m2 > 0.5), bx), abs(a[0, 0]), "coronal", "b", "FA, from the tensor"))
    d, a = tr.nii(f"brains/sub-{T}/outputs/{T}_micro_soma_fraction.nii.gz"); mk, _ = tr.nii(f"brains/sub-{T}/inputs/manual_annotations/{T}_micro_brain_mask.nii.gz")
    j = int(round(np.argwhere(mk > 0.5)[:, 1].mean())); m2 = FS.section(mk, "coronal", j); bx = bbox(m2 > 0.5, cropm(abs(a[0, 0])))
    P.append(Pn(crop(np.clip(FS.section(d, "coronal", j), 0, 1) * (m2 > 0.5), bx), abs(a[0, 0]), "coronal", "c", "Soma fraction"))
    b0, ba = tr.nii(f"brains/sub-{S}/inputs/mri/{S}_b0.nii.gz"); mk, _ = tr.nii(f"brains/sub-{S}/inputs/manual_annotations/{S}_b0_brain_mask.nii.gz")
    j = int(round(np.argwhere(mk > 0.5)[:, 1].mean())); m2 = FS.section(mk, "coronal", j); bx = bbox(m2 > 0.5, cropm(abs(ba[0, 0])))
    P.append(Pn(crop(FS.section(window(b0, mk), "coronal", j), bx), abs(ba[0, 0]), "coronal", "d", "b0", lg="new brain"))
    rows = [Row(P, lines=2)]
    fig, sc = grid(rows)
    bracket(fig, P[:3], f"{name(T)}, one of the template's four brains", 2)
    bracket(fig, P[3:], f"{name(S)}, a new brain", 2)
    orient(fig, rows)
    bar(fig, sc)
    return fig


@figure
def mri_03_template_channels(tr):
    """The MRI template's channels in one coronal plane: T2-weighted, b0, FA, the principal direction, neurite and soma fractions; the
    brain mask's outline on the first; under the principal direction, the key to its colors (red left-right, green front-back, blue
    up-down)."""
    tm, _ = tr.nii("template/outputs/template_brain_mask.nii.gz")
    j = template_planes(tm)["coronal"]; m2 = FS.section(tm, "coronal", j); bx = bbox(m2 > 0.5, 1)
    P = []
    for i, (title, rel, kind) in enumerate((("T2-weighted", "template_T2w", "win"), ("b0", "template_b0", "win"), ("FA", "template_FA", "01"),
                                            ("Principal direction", "template_DEC", "rgb"), ("Neurite fraction", "template_neurite_fraction", "01"),
                                            ("Soma fraction", "template_soma_fraction", "01"))):
        d, _ = tr.nii(f"template/outputs/{rel}.nii.gz")
        im = FS.section(window(d, tm) if kind == "win" else (rgb01(d) if kind == "rgb" else np.clip(d, 0, 1)), "coronal", j)
        im = im * ((m2 > 0.5)[..., None] if im.ndim == 3 else (m2 > 0.5))
        P.append(Pn(crop(im, bx), 0.25, "coronal", "abcdef"[i], title))
    rows = [Row(P, lines=2, below=4.0)]
    fig, sc = grid(rows)
    span(fig, P, "The MRI template's channels", 2)
    FS.outline(P[0].ax, crop(m2, bx), lw=0.5)
    from matplotlib.patches import Patch                                 # (d)'s colors, under it: the principal direction's axis
    key(fig, [Patch(facecolor=c, edgecolor="none", label=t) for c, t in (((1, 0, 0), "Left–right"), ((0, 1, 0), "Front–back"),
                                                                          ((0, 0, 1), "Up–down"))],
        P[3].x + P[3].w * sc / 2, rows[0].top + rows[0].hrow + 1.2, loc="upper center", ncol=3, handlelength=1.0, handleheight=1.0)
    orient(fig, rows)
    bar(fig, sc, corners=("top right",))
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
    m2 = FS.section(mk, "coronal", j); bx = bbox(m2 > 0.5, cropm(px))
    raw = FS.section(window(b, b > np.percentile(b, 50)), "coronal", j)
    br = FS.section(window(bb, mk), "coronal", j)
    bi = FS.section(bias, "coronal", j); bi = np.where(m2 > 0.5, bi, 0)
    v = bi[m2 > 0.5]; bi = np.clip((bi - v.min()) / max(v.max() - v.min(), 1e-9), 0, 1) * (m2 > 0.5)
    corr = np.where(m2 > 0.5, FS.section(bb, "coronal", j) / np.where(FS.section(bias, "coronal", j) > 0, FS.section(bias, "coronal", j), 1), 0)
    corr = window(corr, m2)
    P = [Pn(crop(im, bx), px, "coronal", "abcde"[i], t) for i, (im, t) in
         enumerate(((raw, "The b0"), (raw, "Its brain mask"), (br, "Masked"), (bi, "Bias field (N4)"), (corr, "Bias-corrected")))]
    rows = [Row(P, lines=2)]
    fig, sc = grid(rows)
    FS.outline(P[1].ax, crop(m2, bx), lw=0.5)
    mixed(fig, (P[0].x + P[-1].x + P[-1].w * sc) / 2, base(rows[0], 2), [(f"{name(T)}: its b0 through", False), ("linc-mri prep", True)])
    orient(fig, rows)
    bar(fig, sc)
    return fig


@figure
def mri_05_b0_in_template(tr):
    """The registration's result: MR243's b0 pulled into the template through outputs/MR243_b0_to_template (top) and the template's
    b0 (bottom), the template brain's outline on both, three planes (sagittal 6 mm left of the midline)."""
    b0, ba, m, tb, ta, tm, f, fa = mri_example(tr)
    tp = template_planes(tm); tw = window(tb, tm)
    rows = [[], []]
    for pl in ("sagittal", "coronal", "axial"):
        mv = pull_itk(f, fa, b0, ba, pl, tp[pl]); mk = FS.section(tm, pl, tp[pl]); bx = bbox(mk > 0.5, 1)
        rows[0].append((Pn(crop(window(mv, mv > 0), bx), 0.25, pl), crop(mk, bx)))
        rows[1].append((Pn(crop(FS.section(tw, pl, tp[pl]), bx), 0.25, pl), crop(mk, bx)))
        r = np.corrcoef(mv[mk > 0.5], FS.section(tb, pl, tp[pl])[mk > 0.5])[0, 1]
        print(f"  {pl}: r (the moved b0, the template's b0) inside the template brain = {r:.3f}")
    R = [Row([p for p, _ in rw]) for rw in rows]
    fig, sc = grid(R)
    for k, (rw, letter, t) in enumerate(zip(rows, "ab", (f"{name(S)}: its b0, moved into the template", "The template's b0"))):
        for p, mk in rw:
            FS.outline(p.ax, mk, lw=0.5)
        span(fig, [rw[0][0]], t, 1, letter=letter, align="left")
    orient(fig, R)
    bar(fig, sc)
    return fig


@figure
def mri_06_outline_check(tr):
    """The outline check, a pass and a warning: MR243 (PASS) and MR253 (WARN, ventral overshoot), each brain's b0 mask moved into the
    template (dashed) over the template's b0 and its brain outline (solid), one sagittal plane 2 mm off the midline and one coronal
    plane; the dotted orange line is the template brain's lowest axial slice, below which the check reports tissue but does not gate it."""
    from matplotlib.lines import Line2D
    tb, ta = tr.nii("template/outputs/template_b0.nii.gz"); tm, _ = tr.nii("template/outputs/template_brain_mask.nii.gz")
    tp = template_planes(tm); tw = window(tb, tm)
    tp_s = dict(tp, sagittal=int(np.round(np.argwhere(tm > 0.5)[:, 0].mean())) - 8)      # 2 mm off the midline: the brainstem in view
    zlow = int(np.where((tm > 0.5).any(axis=(0, 1)))[0][0])            # the template brain's lowest axial slice (its flat brainstem cut)
    row_low = tm.shape[2] - 1 - zlow                                   # its row in section()'s display (superior up)
    import json
    P, O, verdict = [], [], {}
    for B in (S, "MR253"):
        q = json.load(open(tr(f"brains/sub-{B}/intermediate/registration/b0-native__to__mri-template/meta.json")))["qc"]
        verdict[B] = q["outline"] + (f", {q['outline_reasons'][0].split(' p99')[0]}" if q.get("outline_reasons") else "")
        m, ma = tr.nii(f"brains/sub-{B}/inputs/manual_annotations/{B}_b0_brain_mask.nii.gz")
        f, fa = itk_field(tr, f"brains/sub-{B}/outputs/{B}_b0_to_template.nii.gz")
        for pl in ("sagittal", "coronal"):
            k = tp_s[pl]; mk = FS.section(tm, pl, k); mv = pull_itk(f, fa, m.astype(np.float32), ma, pl, k) > 0.5
            bx = bbox((mk > 0.5) | mv | (np.arange(mk.shape[0])[:, None] == row_low) & (mk > 0.5).any(0)[None, :], 1)
            P.append(Pn(crop(FS.section(tw, pl, k), bx), 0.25, pl, lg=f"{B}{pl}")); O.append((crop(mk, bx), crop(mv, bx), bx))
    rows = [Row(P, below=4.0)]
    fig, sc = grid(rows)
    for p, (mk, mv, bx) in zip(P, O):
        FS.outline(p.ax, mk, lw=0.5); FS.outline(p.ax, mv, color=FS.ON_IMG2, lw=0.6, linestyles="dashed")
        p.ax.axhline(row_low - bx[0] + 0.5, color=FS.COL["pred"], lw=0.6, ls=(0, (1, 1.5)))    # below it: reported, not gated
    for i, (B, letter) in enumerate(zip((S, "MR253"), "ab")):
        span(fig, P[2 * i:2 * i + 2], f"{name(B)}: {verdict[B]}", 1, letter=letter, align="left")
    hs = [Line2D([], [], color=FS.ON_IMG, lw=0.6, label="Template brain's outline"),
          Line2D([], [], color=FS.ON_IMG2, lw=0.8, ls=(0, (3, 2)), label="The brain's b0 mask, moved into the template"),
          Line2D([], [], color=FS.COL["pred"], lw=0.8, ls=(0, (1, 1.5)), label="The template brain's lowest slice")]
    key(fig, hs, W_MM / 2, rows[0].top + rows[0].hrow + 1.2, loc="upper center", ncol=3)
    orient(fig, rows)
    bar(fig, sc)
    return fig


# ---- the blockface tutorial (02_blockface.md) -------------------------------------------------------------------------------
def provenance_view(lum, tissue, synthesized):
    """The paper's provenance colors (its blockface template figure, panel c): photographed tissue in dimmed gray (its luminance),
    synthesized tissue orange (figstyle COL["pred"])."""
    from matplotlib.colors import to_rgb
    gray = (0.18 + 0.44 * np.clip(lum, 0, 1))[..., None] * np.ones(3)
    return np.where(synthesized[..., None], np.asarray(to_rgb(FS.COL["pred"])), gray) * tissue[..., None]


BF = "MR252"                                  # the blockface tutorial's example brain (MM2)
BF_PHOTO_REDUCE = 8                           # photographs read at 1/8 (20 um -> 0.16 mm per pixel)


def bf_photo(tr, k, reduce=BF_PHOTO_REDUCE):
    """Photograph k (cutting order: inputs/photographs/<ID>_slice<k>.jpg) and its mask (manual_annotations/photograph_masks, PNG x50),
    reduced `reduce` times by block mean / any; (RGB [0, 1], mask bool, pixel mm)."""
    from PIL import Image
    ph = Image.open(tr(f"brains/sub-{BF}/inputs/photographs/{BF}_slice{k:04d}.jpg")).convert("RGB")
    W, H = ph.size
    a = rgb01(np.asarray(ph.reduce(reduce)))
    m = np.asarray(Image.open(tr(f"brains/sub-{BF}/inputs/manual_annotations/photograph_masks/{BF}_slice{k:04d}.png"))) > 0
    m = m[:H // reduce * reduce, :W // reduce * reduce].reshape(H // reduce, reduce, W // reduce, reduce).any((1, 3))
    return a[:m.shape[0], :m.shape[1]], m, 0.020 * reduce


def bf_vx_mm():
    import yaml
    c = yaml.safe_load((REPO / "settings" / "brains" / f"{BF}.yaml").read_text())
    return float((c.get("reconstruction_spacing_um") or {}).get("vx", c["vx"])) / 1000.0


def plane_world(img, plane, index):
    """World coordinates (canonical RAS) of the voxel centers of one plane, in figstyle.section's display order: (rows, cols, 3)."""
    sh = img.shape[:3]; k = FS.PLANE_AXIS[plane]
    ax = [a for a in range(3) if a != k]
    g = np.meshgrid(np.arange(sh[ax[0]]), np.arange(sh[ax[1]]), indexing="ij")
    ijk = np.zeros(g[0].shape + (3,)); ijk[..., ax[0]], ijk[..., ax[1]], ijk[..., k] = g[0], g[1], index
    w = ijk @ img.affine[:3, :3].T + img.affine[:3, 3]
    return np.stack([np.flipud(w[..., c].T) for c in range(3)], -1)


def sample_world(vol, aff, world, order=0):
    """vol (canonical RAS) sampled at world points (..., 3): nearest (order 0) or linear."""
    from scipy import ndimage as ndi
    v = (world.reshape(-1, 3) - aff[:3, 3]) @ np.linalg.inv(aff[:3, :3]).T
    return ndi.map_coordinates(vol, v.T, order=order, cval=0).reshape(world.shape[:-1])


def square_px(img, px_r, px_c):
    """A plane with pixels px_r x px_c mm resampled (linear) to square pixels of the smaller size; (img, px)."""
    from scipy import ndimage as ndi
    p = min(px_r, px_c)
    z = (px_r / p, px_c / p) + ((1,) if img.ndim == 3 else ())
    return ndi.zoom(img, z, order=1), p


def bf_volume_planes(tr):
    """MR252's L2 volume (RGB), brain mask and tubes (L4 grid, sampled nearest on the L2 voxels), canonical RAS, and planes through
    the tube voxel nearest the brain's center."""
    import nibabel as nib
    v = nib.as_closest_canonical(tr.img(f"brains/sub-{BF}/outputs/{BF}_blockface_L2.nii.gz"))
    mk = nib.as_closest_canonical(tr.img(f"brains/sub-{BF}/outputs/{BF}_brain_mask.nii.gz"))
    tb = nib.as_closest_canonical(tr.img(f"brains/sub-{BF}/outputs/{BF}_tubes.nii.gz"))
    M = np.asanyarray(mk.dataobj) > 0; Tb = np.asanyarray(tb.dataobj) > 0
    c = np.argwhere(M).mean(0); t = np.argwhere(Tb); t0 = t[np.argmin(((t - c) ** 2).sum(1))]
    w0 = mk.affine[:3, :3] @ t0 + mk.affine[:3, 3]                       # the tube voxel's world point -> the L2 planes through it
    i0 = np.round(np.linalg.inv(v.affine[:3, :3]) @ (w0 - v.affine[:3, 3])).astype(int)
    return v, mk, tb, M, Tb, dict(sagittal=int(i0[0]), coronal=int(i0[1]), axial=int(i0[2]))


def bf_plane(v, mk, tb, M, Tb, plane, index):
    """One plane of the L2 volume (RGB), its mask and tubes on the same pixels, at square pixels; (rgb, mask, tubes, px)."""
    k = FS.PLANE_AXIS[plane]
    arr = np.take(np.asanyarray(v.dataobj), index, axis=k)                # (a, b, 3)
    disp = np.stack([np.flipud(arr[..., c].T) for c in range(3)], -1)
    w = plane_world(v, plane, index)
    m2 = sample_world(M.astype(np.uint8), mk.affine, w) > 0; t2 = sample_world(Tb.astype(np.uint8), tb.affine, w) > 0
    z = np.abs(np.diag(v.affine)[:3]); ax = [a for a in range(3) if a != k]
    px_r, px_c = z[ax[1]], z[ax[0]]                                      # display rows run along the second in-plane axis
    img, p = square_px(rgb01(disp), px_r, px_c)
    m3 = square_px(m2.astype(np.float32), px_r, px_c)[0] > 0.5; t3 = square_px(t2.astype(np.float32), px_r, px_c)[0] > 0.5
    return img, m3, t3, p




def mirror(a):
    """A photograph of MR252 in the neurological convention: the camera sees the subject's left on the image's right (the paper's
    reconstruction figure decides it by correlation with the reconstructed section, photo_mirrored: true for MR252), so the figures flip
    it left-right, one uniform display operation, as that figure does."""
    return a[:, ::-1]


@figure
def blockface_01_flow(tr):
    """The blockface part with MR252: a photograph (mirrored to the neurological convention), the reconstructed volume, the volume
    registered onto its own b0, and the completed volume (sagittal, 4.5 mm left of the b0 brain's center); the commands between them."""
    import nibabel as nib
    k = 700
    ph, pm, ppx = bf_photo(tr, k); ph, pm = mirror(ph), mirror(pm); bx = bbox(pm, cropm(ppx))
    v, mk, tb, M, Tb, pl = bf_volume_planes(tr)
    vimg, vm, _, vpx = bf_plane(v, mk, tb, M, Tb, "sagittal", pl["sagittal"])
    vb = bbox(vm, cropm(vpx))
    nl = nib.as_closest_canonical(tr.img(f"paper/brains/sub-{BF}/mri/nonlinear_r1v3mr/{BF}_bf_nonlin_rgb.nii.gz"))
    b0 = nib.as_closest_canonical(tr.img(f"paper/brains/sub-{BF}/mri/r1v3mr/{BF}_b0_masked.nii.gz"))
    co = nib.as_closest_canonical(tr.img(f"brains/sub-{BF}/outputs/{BF}_completed.nii.gz"))
    xw = float((b0.affine[:3, :3] @ np.argwhere(np.asanyarray(b0.dataobj) > 0).mean(0) + b0.affine[:3, 3])[0]) - 4.5   # 4.5 mm left of center

    def sag(img_n, x_mm, rgb=False):
        i = int(round((x_mm - img_n.affine[0, 3]) / img_n.affine[0, 0]))
        a = np.take(np.asanyarray(img_n.dataobj), i, axis=0)
        return (np.stack([np.flipud(a[..., c].T) for c in range(3)], -1) if rgb else np.flipud(a.T)), i
    nl2, i_nl = sag(nl, xw, rgb=True); nl2 = rgb01(nl2)
    from scipy import ndimage as ndi
    bm = sample_world(ndi.binary_fill_holes(np.asanyarray(b0.dataobj) > 0).astype(np.uint8), b0.affine, plane_world(nl, "sagittal", i_nl)) > 0
    nbx = bbox(bm, 1)
    co2, _ = sag(co, xw); cm = co2 > 0; cbx = bbox(cm, 1)
    GB = 10.0                                                              # the arrow gutters (the commands go in the band under the row)
    P = [Pn(crop(ph * pm[..., None], bx), ppx, "coronal", "a", f"{name(BF)}: photograph {k}"),
         Pn(crop(vimg * vm[..., None], vb), vpx, "sagittal", "b", "Blockface volume", gap=GB),
         Pn(crop(nl2 * bm[..., None], nbx), 0.25, "sagittal", "c", "Registered to its b0", gap=GB),
         Pn(crop(window(co2, cm, 0.5, 99.5) * cm, cbx), 0.25, "sagittal", "d", "Completed", gap=GB)]
    rows = [Row(P, valign="center", below=1.5 + 3 * CMD_MM)]
    fig, sc = grid(rows)
    FS.outline(P[2].ax, crop(bm, nbx), lw=0.5)
    ya = rows[0].top + rows[0].hrow / 2 - 3.0                                # the arrows: above the orientation letters' line
    for a, b in zip(P[:-1], P[1:]):
        arrow_gutter(fig, a, b, sc, y=ya)
    y0 = rows[0].top + rows[0].hrow + 1.5
    for a, b, lines in zip(P[:-1], P[1:], (("linc-reconstruct all",), ("linc-register mri", "~Without MRI:", "linc-register blockface-template"),
                                           ("linc-complete prepare", "linc-complete"))):
        code_lines(fig, (a.x + a.w * sc + b.x) / 2, y0, lines, up=False)
    orient(fig, rows)
    bar(fig, sc)
    print(f"  photograph {k}, the volume's sagittal index {pl['sagittal']}, x = {xw:.2f} mm in the b0 frame")
    return fig


@figure
def blockface_02_input(tr):
    """The input: one photograph of MR252 (inputs/photographs) beside its brain mask (inputs/manual_annotations/photograph_masks), both
    mirrored to the neurological convention."""
    k = 700
    ph, pm, ppx = bf_photo(tr, k); ph, pm = mirror(ph), mirror(pm); bx = bbox(pm, int(1.0 / ppx))
    P = [Pn(crop(ph, bx), ppx, "coronal", "a", f"{name(BF)}: photograph {k}"),
         Pn(crop(pm.astype(np.float32), bx), ppx, "coronal", "b", "Its brain mask")]
    rows = [Row(P)]
    fig, sc = grid(rows)
    orient(fig, rows)
    bar(fig, sc)
    return fig


@figure
def blockface_03_jumps(tr):
    """MR252's largest camera jump (settings/brains/MR252.yaml jumps): the photograph after it with its outline (white) and the outline of
    the one before it (orange), as photographed (left) and after prepare (intermediate/prepare/02_blockface, the jump corrected; right);
    both mirrored to the neurological convention."""
    import yaml, zarr
    c = yaml.safe_load((REPO / "settings" / "brains" / f"{BF}.yaml").read_text())
    from scipy import ndimage as ndi
    best = None
    for a_, b_ in c["jumps"]:
        if b_ != a_ + 1:
            continue
        try:
            _, m0, _ = bf_photo(tr, a_, 16); _, m1, _ = bf_photo(tr, b_, 16)
        except FileNotFoundError:
            continue
        d = float(np.hypot(*np.subtract(ndi.center_of_mass(m0), ndi.center_of_mass(m1)))) * 0.32
        if best is None or d > best[0]:
            best = (d, a_, b_)
    d, a_, b_ = best
    ph0, m0, px = bf_photo(tr, a_); ph1, m1, _ = bf_photo(tr, b_)
    st = zarr.open_group(str(tr(f"brains/sub-{BF}/intermediate/prepare/02_blockface.ome.zarr")), mode="r")["0"]
    sm = zarr.open_group(str(tr(f"brains/sub-{BF}/intermediate/prepare/02_blockface_mask.ome.zarr")), mode="r")["0"]
    r = BF_PHOTO_REDUCE

    def prep(k):                                                           # the stack is stored reversed: [::-1, ::-1] is upright
        a = np.asarray(st[:, ::r, k, ::r]).transpose(1, 2, 0)[::-1, ::-1]; m = np.asarray(sm[0, ::r, k, ::r])[::-1, ::-1] > 0
        return rgb01(a), m
    q1, n1 = prep(b_); _, n0 = prep(a_)
    ph1, m0, m1, q1, n0, n1 = (mirror(x) for x in (ph1, m0, m1, q1, n0, n1))
    bx = bbox(m1 | m0, int(1.0 / px)); bq = bbox(n1 | n0, cropm(px))
    P = [Pn(crop(ph1, bx), px, "coronal", "a", "As photographed"), Pn(crop(q1, bq), px, "coronal", "b", "After prepare")]
    rows = [Row(P, lines=2)]
    fig, sc = grid(rows)
    span(fig, P, f"{name(BF)}, its largest camera jump: photograph {b_} (white outline) and {a_} before it (orange outline)", 2)
    for p, mb, ma in ((P[0], crop(m0, bx), crop(m1, bx)), (P[1], crop(n0, bq), crop(n1, bq))):
        FS.outline(p.ax, ma, lw=0.5)                                       # the photograph after the jump: white
        FS.outline(p.ax, mb, lw=0.8, color=FS.COL["pred"])                # the one before it: orange
    orient(fig, rows)
    bar(fig, sc)
    print(f"  jump [{a_}, {b_}]: its masks' centers {d:.2f} mm apart before prepare")
    return fig


@figure
def blockface_04_volume(tr):
    """MR252's volume (outputs/MR252_blockface_L2) in three planes through one fiducial marker, the brain mask's outline (white) and the
    fiducial markers (outputs/MR252_tubes, orange; filled in the volume)."""
    from matplotlib.lines import Line2D
    v, mk, tb, M, Tb, pl = bf_volume_planes(tr)
    P, O = [], []
    for i, nm in enumerate(("coronal", "sagittal", "axial")):
        img, m3, t3, p = bf_plane(v, mk, tb, M, Tb, nm, pl[nm])
        bx = bbox(m3, cropm(p)); P.append(Pn(crop(img, bx), p, nm, "abc"[i])); O.append((crop(m3, bx), crop(t3, bx)))
    rows = [Row(P)]
    fig, sc = grid(rows)
    span(fig, P[:1], f"{name(BF)}: blockface volume", 1, letter=None, align="left")
    for p, (m3, t3) in zip(P, O):
        FS.outline(p.ax, m3, lw=0.5); FS.outline(p.ax, t3, lw=0.8, color=FS.COL["pred"])
    hs = [Line2D([], [], color=FS.ON_IMG, lw=0.6, label="Brain mask"),
          Line2D([], [], color=FS.COL["pred"], lw=0.9, label="Fiducial markers (filled in the volume)")]
    key(fig, hs, P[1].x, P[1].y + P[1].h * sc + 3.0, ncol=1)
    orient(fig, rows)
    bar(fig, sc)
    print(f"  planes through a tube: {pl}")
    return fig


@figure
def blockface_05_on_b0(tr):
    """MR252's blockface registered onto its own b0 (the paper's products: mri/nonlinear_r1v3mr/MR252_bf_nonlin_rgb, on the b0's 0.25 mm
    grid) beside the b0 (mri/r1v3mr/MR252_b0_masked), sagittal 4.5 mm left of the b0 brain's center and coronal through the
    photographed slab; the b0 brain's outline on all four."""
    import nibabel as nib
    nl = nib.as_closest_canonical(tr.img(f"paper/brains/sub-{BF}/mri/nonlinear_r1v3mr/{BF}_bf_nonlin_rgb.nii.gz"))
    b0 = nib.as_closest_canonical(tr.img(f"paper/brains/sub-{BF}/mri/r1v3mr/{BF}_b0_masked.nii.gz"))
    from scipy import ndimage as ndi
    B0 = np.asanyarray(b0.dataobj).astype(np.float32); bm = ndi.binary_fill_holes(B0 > 0).astype(np.uint8)   # the brain (dark CSF is 0)
    N = rgb01(np.asanyarray(nl.dataobj)); photo = N.max(-1) > 0
    cw = b0.affine[:3, :3] @ np.argwhere(bm > 0).mean(0) + b0.affine[:3, 3]
    yw = float((nl.affine[:3, :3] @ np.argwhere(photo).mean(0) + nl.affine[:3, 3])[1])
    out = []
    for nm, wpt in (("sagittal", cw[0] - 4.5), ("coronal", yw)):
        k = FS.PLANE_AXIS[nm]
        i_n = int(round((wpt - nl.affine[k, 3]) / nl.affine[k, k]))
        wn = plane_world(nl, nm, i_n)
        b0_on = sample_world(B0, b0.affine, wn, order=1); m_on = sample_world(bm, b0.affine, wn) > 0
        rgb = np.stack([np.flipud(np.take(N, i_n, axis=k)[..., c].T) for c in range(3)], -1)
        bx = bbox(m_on, 1)
        out += [(crop(window(b0_on, m_on) * m_on, bx), crop(m_on, bx), nm), (crop(rgb, bx), crop(m_on, bx), nm)]
    heads = (f"{name(BF)}: b0", "Blockface, registered to it", "b0", "Blockface, registered to it")
    P = [Pn(img, 0.25, nm, "abcd"[i], heads[i]) for i, (img, m, nm) in enumerate(out)]
    rows = [Row(P)]
    fig, sc = grid(rows)
    for p, (img, m, nm) in zip(P, out):
        FS.outline(p.ax, m, lw=0.5)
    orient(fig, rows)
    bar(fig, sc)
    return fig


@figure
def blockface_06_completed(tr):
    """MR252's completed volume (outputs/MR252_completed) in a sagittal and a coronal plane (left column, a and b), and the same planes in
    the paper's provenance colors (right column, c and d): photographed gray, synthesized orange (outputs/MR252_synthesized_mask). In the
    sagittal plane the border between the two is the junction plane; the coronal plane lies 5 mm caudal of it, in synthesized tissue."""
    B = "MR252"
    v, A = tr.nii(f"brains/sub-{B}/outputs/{B}_completed.nii.gz")
    sm, _ = tr.nii(f"brains/sub-{B}/outputs/{B}_synthesized_mask.nii.gz")
    tissue, px = v > 0, abs(A[0, 0]); sm = sm > 0
    vw = window(v, tissue, 0.5, 99.5)
    i = int(round(np.argwhere(tissue)[:, 0].mean() - 4.5 / px))        # 4.5 mm left of the brain's center (the paper's pipeline figure)
    frac = sm.sum((0, 2)) / np.maximum(tissue.sum((0, 2)), 1)           # the synthesized share of each coronal plane
    ys = np.where(tissue.any((0, 2)))[0]; j_junction = int(ys.min())     # the junction: from the caudal end, the last plane mostly synthesized
    while j_junction + 1 <= ys.max() and frac[j_junction + 1] > 0.5:
        j_junction += 1
    j = j_junction - int(round(5.0 / px))
    R = []
    heads = {"sagittal": (f"{name(B)}: completed volume", "Photographed (gray), synthesized (orange)"), "coronal": ("In the synthesized part", None)}
    for pl, k, letters in (("sagittal", i, "ac"), ("coronal", j, "bd")):
        t2 = FS.section(tissue.astype(np.float32), pl, k) > 0.5; bx = bbox(t2, 1)
        lum = FS.section(vw, pl, k) * t2
        R.append([Pn(crop(lum, bx), px, pl, letters[0], heads[pl][0]),
                  Pn(crop(provenance_view(lum, t2, FS.section(sm.astype(np.float32), pl, k) > 0.5), bx), px, pl, letters[1], heads[pl][1])])
    print(f"  {B}: sagittal index {i}, junction plane y index {j_junction}, coronal {j} (5 mm caudal)")
    rows = [Row(R[0]), Row(R[1], under=[0, 1])]
    fig, sc = grid(rows)
    orient(fig, rows)
    bar(fig, sc)
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
    R = [[], []]
    for n, pl in enumerate(("sagittal", "coronal", "axial")):
        k = tp[pl]; t2 = FS.section(bm, pl, k) > 0.5; bx = bbox(t2, 1)
        col = np.stack([FS.section(rgb[..., c], pl, k) for c in range(3)], -1) * t2[..., None]
        lum = FS.section(lum_all, pl, k) / hi * t2
        R[0].append(Pn(crop(col, bx), 0.25, pl, "abc"[n]))
        R[1].append(Pn(crop(provenance_view(lum, t2, FS.section(sm, pl, k) > 0.5), bx), 0.25, pl, "def"[n]))
    rows = [Row(R[0]), Row(R[1])]
    fig, sc = grid(rows)
    span(fig, R[0][:1], "Blockface template", 1, letter=None, align="left")
    span(fig, R[1][:1], "Photographed (gray), synthesized (orange)", 1, letter=None, align="left")
    orient(fig, rows)
    bar(fig, sc)
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




def section_crop(sec, spx):
    """A scanned section cropped to its tissue (the window, never the pixels)."""
    return crop(sec, bbox(sec.max(-1) > 0.04, cropm(spx)))


@figure
def histology_01_flow(tr):
    """The histology part: a section as scanned, the photograph of its own section, the section registered onto it, and the LGN (solid)
    and STN (dashed) drawn on the sections, carried into the template, on the coronal plane that cuts all four; the commands."""
    sec, spx = section_thumb(tr)
    ph, px = bf_slice(tr, f"brains/sub-{S}/outputs/{S}_blockface_L2.nii.gz", SECTION_SLICE)
    hi, _ = bf_slice(tr, f"brains/sub-{S}/outputs/{S}_histology_blockface.nii.gz", SECTION_SLICE)
    bx = bbox((ph.sum(-1) > 0) | (hi.max(-1) > 0.02), cropm(px))
    tb, _ = tr.nii("template/outputs/template_b0.nii.gz"); tm, _ = tr.nii("template/outputs/template_brain_mask.nii.gz")
    lt, jt = template_both_plane(tr)
    print(f"  the template's coronal plane through all four nuclei: {jt}")
    m2 = FS.section(tm, "coronal", jt); tbx = bbox(m2 > 0.5, 1)
    GH = 10.0                                                              # the arrow gutters (the commands go in the band under the row)
    P = [Pn(section_crop(sec, spx), spx, "coronal", "a", f"Section {SECTION_N}, as scanned"),
         Pn(crop(ph, bx), px, "coronal", "b", "Its blockface photograph", gap=GH),
         Pn(crop(hi, bx), px, "coronal", "c", "The section, registered onto it"),
         Pn(crop(FS.section(window(tb, tm), "coronal", jt) * (m2 > 0.5), tbx), 0.25, "coronal", "d", "LGN and STN in the template", gap=GH)]
    rows = [Row(P, lines=2, valign="center", below=1.5 + 2 * CMD_MM + 3.0)]
    fig, sc = grid(rows)
    bracket(fig, P[:3], name(S), 2)
    tl = crop(FS.section(lt, "coronal", jt), tbx)
    for k in (1, 2):
        FS.outline(P[3].ax, tl == k, color=FS.BRAIN[S], lw=0.6)
    for k in (3, 4):
        FS.outline(P[3].ax, tl == k, color=FS.BRAIN[S], lw=0.6, linestyles="dashed")
    y0 = rows[0].top + rows[0].hrow + 1.5
    for a, b, lines in ((P[0], P[1], ("linc-histology all",)), (P[2], P[3], ("linc-labels push", "linc-labels carry"))):
        arrow_gutter(fig, a, b, sc)
        code_lines(fig, (a.x + a.w * sc + b.x) / 2, y0, lines, up=False)
    lgn_stn_key(fig, P[3].x + P[3].w * sc / 2, y0 + 2 * CMD_MM + 0.5, loc="upper center", ncol=2)
    orient(fig, rows)
    bar(fig, sc)
    return fig


def lgn_stn_key(fig, x_mm, y_mm, **kw):
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=FS.BRAIN[S], lw=0.8, label="LGN"), Line2D([], [], color=FS.BRAIN[S], lw=0.8, ls=(0, (3, 2)), label="STN")]
    return key(fig, hs, x_mm, y_mm, **kw)


@figure
def histology_02_input(tr):
    """The input: section 36 as scanned (JPEG 2000, read at a reduced level; dark field) beside the blockface photograph of its own
    section (slice 107 of the cutting order)."""
    sec, spx = section_thumb(tr)
    ph, px = bf_slice(tr, f"brains/sub-{S}/outputs/{S}_blockface_L2.nii.gz", SECTION_SLICE)
    bx = bbox(ph.sum(-1) > 0, cropm(px))
    P = [Pn(section_crop(sec, spx), spx, "coronal", "a", f"{name(S)}: section {SECTION_N}, as scanned"),
         Pn(crop(ph, bx), px, "coronal", "b", f"Its blockface photograph, slice {SECTION_SLICE}")]
    rows = [Row(P)]
    fig, sc = grid(rows)
    orient(fig, rows)
    bar(fig, sc)
    return fig


REGION_COLORS = ["black", "#d9d9d9", "#8c8c8c", "#4d4d4d", FS.COL["pred"]]
REGION_NAMES = ["Left hemisphere (1)", "Right hemisphere (2)", "Cerebellum and brainstem (3)", "Marker tubes (4)"]


def region_key(fig, x_mm, y_mm):
    from matplotlib.patches import Patch
    return key(fig, [Patch(facecolor=c, edgecolor="none", label=n) for c, n in zip(REGION_COLORS[1:], REGION_NAMES)], x_mm, y_mm,
               handlelength=1.0, handleheight=1.0)


@figure
def histology_03_regions(tr):
    """The regions volume of MR243 (intermediate/histology/MR243_regions_t2) on blockface slice 134, where all four labels are present,
    beside the photograph of that slice with the regions' outlines; the key gives each label's value in the file."""
    from matplotlib.colors import ListedColormap
    jj = REGIONS_SLICE
    ph, px = bf_slice(tr, f"brains/sub-{S}/outputs/{S}_blockface_L2.nii.gz", jj)
    lab = regions_on_bf(tr, jj)
    bx = bbox((ph.sum(-1) > 0) | (lab > 0), cropm(px))
    KW = ink(max(REGION_NAMES, key=lambda n: ink(n)[1]))[1] + 6.5 + G      # the key's width: its swatch and pads, its longest name, a gutter
    P = [Pn(crop(lab, bx), px, "coronal", "a", f"{name(S)}: regions, slice {jj}", cmap=ListedColormap(REGION_COLORS), vmin=-0.5, vmax=4.5,
            interpolation="nearest"),
         Pn(crop(ph, bx), px, "coronal", "b", "Their outlines on its photograph")]
    rows = [Row(P)]
    fig, sc = grid(rows, reserve_right=KW)
    for k in (1, 2, 3, 4):
        FS.outline(P[1].ax, crop(lab, bx) == k, lw=0.4, color=FS.ON_IMG if k < 4 else FS.COL["pred"])
    region_key(fig, W_MM - MG - KW + G, rows[0].top)
    orient(fig, rows)
    bar(fig, sc)
    return fig


@figure
def histology_04_components(tr):
    """One section's components: section 5 is registered as two, its left and right hemispheres, each to its own part of the photograph
    of slice 14 (the regions volume divides the photograph). The scanned section has no orientation letters: its two hemispheres were
    mounted apart."""
    sec, spx = section_thumb(tr, PIECES)
    ph, px = bf_slice(tr, f"brains/sub-{S}/outputs/{S}_blockface_L2.nii.gz", PIECES_SLICE)
    lab = regions_on_bf(tr, PIECES_SLICE)
    t = ph.sum(-1) > 0
    from scipy import ndimage as ndi
    left = ndi.binary_dilation(lab == 1, iterations=2) & t; right = ndi.binary_dilation(lab == 2, iterations=2) & t & ~left
    L = crop(np.where(left[..., None], ph, 0), bbox(left, cropm(px))); R = crop(np.where(right[..., None], ph, 0), bbox(right, cropm(px)))
    P = [Pn(section_crop(sec, spx), spx, None, "a", f"{name(S)}: section {PIECES_N}, as scanned"),
         Pn(L, px, "coronal", "b", "Component 1: left"), Pn(R, px, "coronal", "c", "Component 2: right")]
    rows = [Row(P, lines=2)]
    fig, sc = grid(rows)
    bracket(fig, P[1:], f"Its photograph, slice {PIECES_SLICE}, divided in two", 2)
    orient(fig, rows)
    bar(fig, sc)
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
    bx = bbox(left | ((ph.sum(-1) > 0) & (np.arange(w)[None, :] < w // 2)), cropm(px))
    tis = ndi.binary_fill_holes(ndi.binary_opening(crop(t, bx), iterations=2))
    P = [Pn(crop(ph, bx), px, "coronal", "a", "Photograph"), Pn(crop(hi, bx), px, "coronal", "b", "Registered section"),
         Pn(crop(ph, bx), px, "coronal", "c", "Its outline on the photograph")]
    rows = [Row(P, lines=2)]
    fig, sc = grid(rows)
    span(fig, P, f"{name(S)}: section {SECTION_N}, its left half", 2)
    FS.outline(P[2].ax, tis, lw=0.5)
    orient(fig, rows)
    bar(fig, sc, mm=5.0)
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
    disp = lambda a: np.flipud(np.swapaxes(a, 0, 1))[:, ::-1]      # noqa: E731  superior up; the L2 y axis runs posterior, so anterior right
    vy, vz = abs(ph.affine[1, 1]), abs(ph.affine[2, 2]); f = vy / vz
    p = ndi.zoom(rgb01(disp(np.asanyarray(ph.dataobj[i]))), (1, f, 1), order=0)
    h = ndi.zoom(rgb01(disp(np.asanyarray(hi.dataobj[i]))), (1, f, 1), order=0)
    v = ndi.zoom(disp(np.asanyarray(pv.dataobj[i])), (1, f), order=0)
    bx = bbox(p.sum(-1) > 0, cropm(vz))
    prov = np.zeros(v.shape + (3,), np.float32); prov[v == 1] = to_rgb(FS.ON_IMG); prov[v == 2] = to_rgb(FS.COL["pred"])
    P = [Pn(crop(im, bx), vz, "sagittal", "abc"[k], t, interpolation="nearest") for k, (im, t) in
         enumerate(((p, "Blockface volume"), (h, "Registered histology"), (prov, "Registered (white), interpolated (orange)")))]
    rows = [Row(P, lines=2)]
    fig, sc = grid(rows)
    span(fig, P, f"{name(S)}: the registered sections as a volume, through the left LGN", 2)
    orient(fig, rows)
    bar(fig, sc)
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
    bx = bbox((ph.sum(-1) > 0) | (hi.max(-1) > 0.02), cropm(px))
    cen = np.round(np.argwhere(con_t > 0).mean(0)).astype(int)          # the pathway's center in the template
    slab = TRACER_SLAB_VOX
    tpan = []
    for pl in ("sagittal", "axial"):
        k = FS.PLANE_AXIS[pl]; i0 = int(cen[k])
        proj = lambda v: FS.section(np.take(v, range(i0 - slab, i0 + slab + 1), axis=k).max(axis=k, keepdims=True), pl, 0)  # noqa: E731
        m2 = FS.section(tm, pl, i0); tbx = bbox(m2 > 0.5, 1)
        tpan.append((pl, crop(FS.section(window(t2, tm), pl, i0) * (m2 > 0.5), tbx), *(crop(proj(v), tbx) for v in (inj_t, ter_t, con_t))))
    print(f"  blockface slice with the most bundle voxels: {jb}; the template's planes: x {cen[0]}, z {cen[2]} (slab +-{slab} voxels)")
    P = [Pn(crop(hi, bx), px, "coronal", "a", f"{name(S)}: blockface, slice {jb}"),
         Pn(tpan[0][1], 0.25, "sagittal", "b"), Pn(tpan[1][1], 0.25, "axial", "c")]
    rows = [Row(P)]
    fig, sc = grid(rows)
    span(fig, P[1:], "In the template", 1, align="left")

    def over(ax, inj, ter, bun, con=None):
        ov = np.zeros(inj.shape + (4,), np.float32)
        ov[ter > 0] = (*g2, 0.55)
        if con is not None:
            ov[con == 2] = (*c, 0.4); ov[con == 1] = (*c, 0.9)
        else:
            ov[bun > 0] = (*c, 0.9)
        ax.imshow(ov, interpolation="nearest"); FS.outline(ax, inj > 0, color=FS.ON_IMG, lw=0.7)
    over(P[0].ax, crop(ib, bx), crop(tb_, bx), crop(bb_, bx))
    for p, tp_ in zip(P[1:], tpan):
        over(p.ax, tp_[2], tp_[3], None, tp_[4])
    hs = [Line2D([], [], color=FS.ON_IMG, lw=0.7, label="Injection site"), Patch(facecolor=g2, alpha=0.55, edgecolor="none", label="Terminal fields"),
          Patch(facecolor=c, alpha=0.9, edgecolor="none", label="Bundles (charted)"), Patch(facecolor=c, alpha=0.4, edgecolor="none", label="Joins (inferred)")]
    y_free = max(P[0].y + P[0].h * sc, P[1].y + P[1].h * sc) + 3.0      # under (a) and (b): the axial (c) sets the row's height
    key(fig, hs, P[0].x, y_free, ncol=2)
    orient(fig, rows)
    bar(fig, sc, region=(MG, P[2].x - 3.0, MG, page(fig)[1] - MG), corners=("bottom right", "bottom left"))
    return fig


@figure
def histology_08_lgn_stn(tr):
    """The LGN and STN of MR243 (solid lines LGN, dashed STN): on the blockface, over the registered section 36, and in the template,
    through each nucleus."""
    hi, px = bf_slice(tr, f"brains/sub-{S}/outputs/{S}_histology_blockface.nii.gz", SECTION_SLICE)
    lb, _ = bf_slice(tr, f"brains/sub-{S}/outputs/{S}_lgn_stn_blockface.nii.gz", SECTION_SLICE)
    tb, _ = tr.nii("template/outputs/template_b0.nii.gz"); tm, _ = tr.nii("template/outputs/template_brain_mask.nii.gz")
    lt, jl = template_lgn_plane(tr, 1); _, js = template_lgn_plane(tr, 3)
    bx = bbox(hi.max(-1) > 0.02, cropm(px)); tw = window(tb, tm)
    P = [Pn(crop(hi, bx), px, "coronal", "a", f"{name(S)}: blockface, slice {SECTION_SLICE}")]
    L = [crop(lb, bx)]
    for j, t in ((jl, "Template, through the LGN"), (js, "Template, through the STN")):
        m2 = FS.section(tm, "coronal", j); b2 = bbox(m2 > 0.5, 1)
        P.append(Pn(crop(FS.section(tw, "coronal", j) * (m2 > 0.5), b2), 0.25, "coronal", "bc"[len(P) - 1], t))
        L.append(crop(FS.section(lt, "coronal", j), b2))
    rows = [Row(P, below=4.0)]
    fig, sc = grid(rows)
    for p, lab in zip(P, L):
        for k in (1, 2):
            FS.outline(p.ax, lab == k, color=FS.BRAIN[S], lw=0.6)
        for k in (3, 4):
            FS.outline(p.ax, lab == k, color=FS.BRAIN[S], lw=0.6, linestyles="dashed")
    lgn_stn_key(fig, (P[1].x + P[2].x + P[2].w * sc) / 2, rows[0].top + rows[0].hrow + 1.2, loc="upper center", ncol=2)
    orient(fig, rows)
    bar(fig, sc)
    return fig


QC_CASES = (("mr243LY_c18_s15nDF_0001", 44, 1), ("mr243LY_c18_s68nDF_0003", 155, 3))
#   the registration check's best and worst components of MR243 (paper/numbers/registration_qc_histo2bf_h9.json, the 12 Nov rows):
#   section 15's left hemisphere (photograph slice 44) and section 68's cerebellum (slice 155); (tag, blockface slice, regions label)


@figure
def histology_09_qc(tr):
    """The registration check on two components of MR243: one that passes (section 15, left hemisphere) and one that fails (section 68,
    cerebellum), each as the registered section and as its tissue outline on the photograph; the Dice coefficients from the check's
    record."""
    import json
    from scipy import ndimage as ndi
    rec = json.load(open(tr("paper/numbers/registration_qc_histo2bf_h9.json")))
    rec = rec.get("data", rec)
    rows_ = {r["tag"]: r for r in rec if r["subject"] == S and r["time"].startswith("2026-11-12")}
    P, T_ = [], []
    for c_, (tag, j, k) in enumerate(QC_CASES):
        q = rows_[tag]; assert q["bf_slice"] == j, (tag, q["bf_slice"], j)
        ph, px = bf_slice(tr, f"brains/sub-{S}/outputs/{S}_blockface_L2.nii.gz", j)
        hi, _ = bf_slice(tr, f"brains/sub-{S}/outputs/{S}_histology_blockface.nii.gz", j)
        lab = regions_on_bf(tr, j)
        part = ndi.binary_dilation(lab == k, iterations=3)
        bx = bbox(part & (ph.sum(-1) > 0), cropm(px))
        tis = ndi.binary_fill_holes(ndi.binary_opening(crop((hi.max(-1) > 0.02) & part, bx), iterations=2))
        P += [Pn(crop(np.where(part[..., None], hi, 0), bx), px, "coronal", "ac"[c_], "Registered"),
              Pn(crop(ph, bx), px, "coronal", "bd"[c_], "On its photograph")]
        m = q["metrics"]; part_name = {1: "left hemisphere", 2: "right hemisphere", 3: "cerebellum"}[k]
        T_.append((tis, f"Section {int(q['tag'].split('_s')[1][:2])}, {part_name}: {'passes' if q['gate'] == 'pass' else 'fails'}, "
                        f"Dice {m['dice_nonlinear']:.2f}"))
    P[2].gap = G + 3.0                                                     # the two cases apart
    rows = [Row(P, lines=3)]
    fig, sc = grid(rows)
    span(fig, P, f"{name(S)}: a component that passes the check and one that fails it", 3)
    for c_, (tis, t) in enumerate(T_):
        FS.outline(P[2 * c_ + 1].ax, tis, lw=0.5)
        span(fig, P[2 * c_:2 * c_ + 2], t, 2, align="left")
    orient(fig, rows)
    bar(fig, sc, mm=5.0)
    return fig


# ---- the overview (00_start_here.md) -----------------------------------------------------------------------------------------
def space_text(fig, x_mm, y_mm, s):
    """A space's name under a tile's title (monospace, secondary gray; a name in the code, exempt from sentence case)."""
    CODE_TEXT.add(s)
    return txt(fig, x_mm, y_mm, s, ha="center", color=FS.GRAY, family="monospace")


@figure
def overview_01_chain(tr):
    """The chain of spaces, with the command between each pair: a section as scanned (histo-piece), its photograph (bf-native), the
    brain's b0 (b0-native), the MRI template (mri-template, T2-weighted); below, the blockface template's channel (bf-template, shown on
    the MRI template's grid) and the link between the two templates. Two routes: the MRI route through the brain's own b0 (top), and the
    blockface route through the blockface template, which bypasses the b0 (the route of every brain's histology labels, and the only
    one for a brain without MRI). Each command on one line: each top gutter is as wide as its command needs (at least GA); on the
    blockface route the gray names sit above the arrows and the commands below them."""
    sec, spx = section_thumb(tr)
    ph, px = bf_slice(tr, f"brains/sub-{S}/outputs/{S}_blockface_L2.nii.gz", SECTION_SLICE)
    bx = bbox(ph.sum(-1) > 0, cropm(px))
    b0, ba, m, tb, ta, tm, f, fa = mri_example(tr)
    tp = template_planes(tm); npl = native_planes(ba, f, fa, tp); bpx = abs(ba[0, 0])
    n_m = FS.section(m, "coronal", npl["coronal"]); nb = bbox(n_m > 0.5, cropm(bpx))
    t2, _ = tr.nii("template/outputs/template_T2w.nii.gz")
    m2 = FS.section(tm, "coronal", tp["coronal"]); tbx = bbox(m2 > 0.5, 1)
    rgb, _ = tr.nii("template/outputs/template_blockface_rgb.nii.gz"); bm, _ = tr.nii("template/outputs/template_blockface_mask.nii.gz")
    bm2 = FS.section(bm, "coronal", tp["coronal"]); bfx = bbox(bm2 > 0.5, 1)
    cmds = ("linc-histology", "linc-register mri", "linc-register mri-template")
    gap = [max(GA, (lambda e: e[1] - e[0])(ink(c, FS.MIN_PT + 0.5, bold=True, mono=True)) + 2.5) for c in cmds]   # one line above each arrow
    top = [Pn(section_crop(sec, spx), spx, "coronal"), Pn(crop(ph, bx), px, "coronal", gap=gap[0]),
           Pn(crop(FS.section(window(b0, m), "coronal", npl["coronal"]), nb), bpx, "coronal", gap=gap[1]),
           Pn(crop(FS.section(window(t2, tm), "coronal", tp["coronal"]) * (m2 > 0.5), tbx), 0.25, "coronal", gap=gap[2])]
    low = [Pn(crop(rgb01(FS.section(rgb, "coronal", tp["coronal"])) * (bm2 > 0.5)[..., None], bfx), 0.25, "coronal")]
    rows = [Row(top, lines=3, valign="center"), Row(low, lines=2, under=[2])]
    fig, sc = grid(rows)
    for p, r, (t, s) in zip(top + low, (rows[0],) * 4 + (rows[1],),
                            (("Histology section", "histo-piece"), ("Blockface photograph", "bf-native"), ("b0 (MRI)", "b0-native"),
                             ("MRI template", "mri-template"), ("Blockface template", "bf-template"))):
        cx = p.x + p.w * sc / 2
        txt(fig, cx, base(r, 2), t, ha="center"); space_text(fig, cx, base(r, 1), s)
    bracket(fig, top[:3], name(S), 3)                                     # the brain's three tiles: its name
    # the MRI route: section -> photograph -> b0 -> MRI template, the command over each arrow, the route's name under the second
    for k, (a_, b_, cmd) in enumerate(zip(top[:-1], top[1:], cmds)):
        y = arrow_gutter(fig, a_, b_, sc)
        code_lines(fig, (a_.x + a_.w * sc + b_.x) / 2, y - 1.2, [cmd])
        if k == 1:
            txt(fig, (a_.x + a_.w * sc + b_.x) / 2, y + 1.2, "MRI route", ha="center", va="top", color=FS.GRAY)
    # the blockface route: from the photograph down and right into the blockface template (bypassing the b0), then the link up into
    # the MRI template; both at a quarter of the tile's height, clear of the orientation letters at its middle; the names above the
    # arrows, the commands below them (the first one ends before the tile, clear of the line down from the photograph)
    q = low[0]; qx0, qy0, qx1, qy1 = box(q, sc)
    yr = qy0 + 0.25 * (qy1 - qy0)
    xs = top[1].x + top[1].w * sc / 2
    arrow(fig, (xs, top[1].y + top[1].h * sc + 1.0), (xs, yr), head=False); arrow(fig, (xs, yr), (qx0 - 1.0, yr))
    txt(fig, (xs + qx0) / 2, yr - 1.2, "Blockface route", ha="center", va="bottom", color=FS.GRAY)
    code_text(fig, qx0 - 1.5 - FS.orient_room("coronal"), yr + 1.2, "linc-register blockface-template", ha="right", va="top")   # clear of its L
    t3 = top[3]; xt = t3.x + t3.w * sc / 2
    arrow(fig, (qx1 + 1.0, yr), (xt, yr), head=False); arrow(fig, (xt, yr), (xt, t3.y + t3.h * sc + 1.0))
    txt(fig, (qx1 + xt) / 2, yr - 1.2, "Template link", ha="center", va="bottom", color=FS.GRAY)
    code_text(fig, (qx1 + xt) / 2, yr + 1.2, "build_link.py", ha="center", va="top")
    orient(fig, rows)
    bar(fig, sc, region=(MG, xs - 2.0, top[0].y + top[0].h * sc, page(fig)[1] - MG))
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", default=os.environ.get("LINC_DATA_TREE"), help="the data tree (default $LINC_DATA_TREE)")
    ap.add_argument("--out", help="a trial render elsewhere, provenance beside the PNGs (default: the repository's "
                                  "docs/tutorials/figures, provenance in outputs/docs/figures_provenance)")
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
    out, prov_out = (Path(a.out), Path(a.out)) if a.out else (FIG_OUT, PROV_OUT)
    FS.setup()
    for n in names:
        t0 = time.time(); tr = Tree(a.tree)
        fig = FIGS[n](tr)
        save(fig, out, prov_out, tr, n)
        print(f"{n}: {out / (n + '.png')} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
