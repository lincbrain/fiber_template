"""House style for the template paper's figures, v2 (30 Sep 2026): the author's ISBI treatment
(manuscript/docs/revision_brief_2026-09-30.md, Figures; docs/reviews/template_figure_review_2026-09-30.md).

- Black background for the WHOLE figure, as in the ISBI paper (isbi2027/code/fig*.py); off-white text, gray secondary text.
- Liberation Sans (Arial-metric), designed at print size: 178 mm full width (86 mm one column), text >= 7 pt, bold lower-case
  9 pt panel letters outside the images, short headers above the panels, American spelling, 600 ppi.
- NOTHING printed over images: headers above, direction letters and scale bars outside the frame, values in the caption but
  for the scale bars' lengths, printed above each bar (author, 12 Nov 2026). save() refuses a figure with text inside an
  image's axes or under 7 pt, or a scale bar's length over another text or outside the figure.
- ONE color lookup, one meaning per hue (COL, BRAIN):
    routes              direct = cyan-teal, MRI = tan-brown (v2.1, 1 Oct: moved away from MR256's green, the provenance
                        orange and MF191's yellow)
    tissue provenance   photographed = uncolored (neutral gray); not photographed (predicted, interpolated, filled) = orange
                        (v2.1: gray replaces v2's blue, so only what was NOT photographed carries a hue)
    brains / tracer cases  one fixed Okabe-Ito hue per brain (BRAIN), the same in every figure
    outlines, annotations  white; "without vs with" comparisons: gray vs white
  The default line cycle is white/grays, so any hue in a figure comes from this lookup.
- One orientation convention, stated once in the paper: neurological (the subject's left on the image's left); sagittal views
  anterior to the right; coronal and sagittal views superior up; axial views anterior up. section() returns that display
  array from a volume in canonical RAS order (canonical()); dirs() writes the direction letters outside the frame.
- True scale: place panels in physical mm (axes_mm) at one print scale per figure for whole brains (SCALE) and one for zooms;
  scale bars outside the tissue (scalebar): plain bars, each with its length centered above it, 2 pt over the bar.
- Display interpolation is bilinear for anatomy (rcParams; pass interpolation="nearest" for label maps); contours of masks
  are drawn at voxel accuracy (outline()).
- Slice positions are given in NMT v2 mm: the MRI template's world is the NMT v2 symmetric frame, and a blockface-template
  point is carried there by the template link of record (to_nmt()).
v1 (29 Sep, white background) is in _sessionbak/2026-09-30/L1/figstyle.py.v1_2303. Names kept from v1: setup, letter, header,
lr, scalebar, to_nmt, save, COL["A"/"B"/"photo"/"pred"/"band"/"b1"/"b2"], FULL_W_IN, DPI, MIN_PT, LETTER_PT.
"""
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the release repository
sys.path.insert(0, os.path.join(REPO, "code"))
MM = 1 / 25.4                                # inches per mm
FULL_W_IN = 178 * MM                         # full page width
COL_W_IN = 86 * MM                           # one column
DPI = 600
MIN_PT = 7
LETTER_PT = 9
FONT = "Liberation Sans"

# ---- the one color lookup ------------------------------------------------------------------------------------------------
BG = "black"
FG = "#f2f2f2"                               # text, outlines, annotations, "with" (the ISBI figures' white)
GRAY = "#b5b5b5"                             # secondary text, ticks, direction letters
DGRAY = "#5a5a5a"                            # faint axes, grids, reference lines
WITHOUT = "#8c8c8c"                          # "without" (vs "with" = FG)
OKABE = dict(orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442", blue="#0072B2", vermilion="#D55E00",
             purple="#CC79A7")
COL = dict(direct="#1fb5c4",                 # direct route, cyan-teal (v2: #2fa89a, too close to MR256's bluish green)
           mri="#b08458",                    # MRI route, tan-brown (v2: ocher #c9a25c, too close to orange and yellow)
           photo="#a6a6a6",                  # photographed tissue: uncolored, neutral gray (v2: blue #3f8fe0)
           pred=OKABE["orange"],             # tissue not photographed: predicted, interpolated or filled, orange
           outline=FG, annot=FG, without=WITHOUT, with_=FG)
BRAIN = dict(MR243=OKABE["sky"], MR252=OKABE["purple"], MR256=OKABE["green"], MF191=OKABE["yellow"],
             MF282=OKABE["vermilion"])       # fixed per brain, every figure; extend only with unused hues
TRACER = dict(MR243=OKABE["vermilion"], MR252=OKABE["green"], MF191=OKABE["yellow"], MR256=OKABE["blue"])   # the tracer cases' colors in
                                             # every tracer figure (Fig. 10 and its 3D scene, Fig. S4): MM1, MM2, MF8, MM3
COL.update(A=COL["direct"], B=COL["mri"], band=FG, b1=BRAIN["MR243"], b2=BRAIN["MR252"])    # v1 names
COL.update(uncorrected="#8c9cff", corrected=FG)   # v2.2 (2 Oct, the reconstruction figure's (e) at the author's request): photographs before
                                                # intensity correction, a light blue-violet distinct from every brain hue, the
                                                # routes and predicted orange; after correction = FG ("with")
LABELS_JSON = os.path.join(REPO, "settings", "specimen_labels.json")   # the paper's brain names (one mapping file)

# ---- print scales (print mm per tissue mm): one whole-brain scale per figure, plus at most one zoom scale ----------------
SCALE = dict(whole=1.0, strip=0.5, zoom=2.0)


def brain_color(S):
    if S not in BRAIN:
        raise KeyError(f"{S} has no fixed color in figstyle.BRAIN: add it there (one unused hue), never in a figure script")
    return BRAIN[S]


def label(S):
    """The paper's label for brain S from the one mapping file; the subject ID until that file exists."""
    if os.path.exists(LABELS_JSON):
        d = json.load(open(LABELS_JSON))
        return d.get("data", d)["labels"][S]                    # write_numbers layout ({"data": ..., "_provenance": ...})
    return S


def species(S):
    return "M. fascicularis" if S.startswith("MF") else "M. mulatta"


def setup():
    import matplotlib
    from cycler import cycler
    from matplotlib import font_manager
    for pattern in ("/usr/share/fonts/liberation-sans/LiberationSans-*.ttf",              # RHEL / Fedora
                    "/usr/share/fonts/truetype/liberation/LiberationSans-*.ttf"):         # Debian / Ubuntu (fonts-liberation)
        for f in sorted(glob.glob(pattern)):
            font_manager.fontManager.addfont(f)
    matplotlib.rcParams.update({
        "font.family": FONT, "font.size": MIN_PT, "axes.titlesize": MIN_PT, "axes.labelsize": MIN_PT,
        "xtick.labelsize": MIN_PT, "ytick.labelsize": MIN_PT, "legend.fontsize": MIN_PT, "figure.titlesize": MIN_PT,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "lines.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": DPI, "figure.dpi": 100,
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "savefig.edgecolor": BG,
        "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": GRAY, "xtick.color": GRAY, "ytick.color": GRAY,
        "xtick.labelcolor": FG, "ytick.labelcolor": FG, "axes.titlecolor": FG, "grid.color": DGRAY, "grid.linewidth": 0.4,
        "legend.frameon": False, "legend.labelcolor": FG, "patch.edgecolor": FG, "hatch.color": DGRAY,
        "axes.prop_cycle": cycler(color=[FG, GRAY, WITHOUT, DGRAY]),
        "image.interpolation": "bilinear", "image.resample": True, "image.cmap": "gray"})
    assert FONT in {f.name for f in font_manager.fontManager.ttflist}, f"{FONT} not registered (install the Liberation fonts)"


# ---- layout -----------------------------------------------------------------------------------------------------------------
def axes_mm(fig, x_mm, y_mm, w_mm, h_mm, **kw):
    """An axes placed in physical mm from the figure's top-left corner (true-scale layouts)."""
    W, H = (v * 25.4 for v in fig.get_size_inches())
    return fig.add_axes([x_mm / W, 1 - (y_mm + h_mm) / H, w_mm / W, h_mm / H], **kw)


def print_mm(extent_mm, scale="whole"):
    """Print size (mm) of a tissue extent (mm) at one of the figure's print scales."""
    return np.asarray(extent_mm, float) * (SCALE[scale] if isinstance(scale, str) else float(scale))


def letter(ax, s, dx=0.0, dy=3.0):
    """Bold lower-case panel letter at the top left, outside the axes (above its top-left corner)."""
    ax.annotate(s, xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy), textcoords="offset points", annotation_clip=False,
                fontsize=LETTER_PT, fontweight="bold", ha="left", va="bottom", color=FG)


def header(ax, s, dx=11.0, dy=3.0, **kw):
    """A short header above the panel, on the letter's line (regular weight), or alone with dx=0."""
    kw.setdefault("color", FG)
    ax.annotate(s, xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy), textcoords="offset points", annotation_clip=False,
                fontsize=kw.pop("fontsize", MIN_PT), ha=kw.pop("ha", "left"), va="bottom", **kw)


DIRS = dict(coronal=("L", "R", "S", "I"), sagittal=("P", "A", "S", "I"), axial=("L", "R", "A", "P"))


def dirs(ax, plane, vertical=False, pad=2.0, color=GRAY, fs=MIN_PT, y=0.5):
    """Direction letters OUTSIDE the frame, regular weight (first panel of a row): left/right of the frame at height y
    (axes fraction, default mid-height); with vertical=True also above/below it. Planes follow section()'s convention."""
    lft, rgt, top, bot = DIRS[plane]
    put = lambda s, xy, off, ha, va: ax.annotate(s, xy=xy, xycoords="axes fraction", xytext=off, textcoords="offset points",
                                                 ha=ha, va=va, fontsize=fs, color=color, annotation_clip=False)
    put(lft, (0, y), (-pad, 0), "right", "center"); put(rgt, (1, y), (pad, 0), "left", "center")
    if vertical:
        put(top, (0.5, 1), (0, pad), "center", "bottom"); put(bot, (0.5, 0), (0, -pad), "center", "top")


def dirs_row(ax_first, ax_last, plane, pad=2.0, color=GRAY, fs=MIN_PT, y=0.5):
    """A row of panels in one orientation: the left letter outside the first panel, the right letter outside the last
    (at height y, axes fraction)."""
    lft, rgt = DIRS[plane][:2]
    for ax, s, xy, off, ha in ((ax_first, lft, (0, y), (-pad, 0), "right"), (ax_last, rgt, (1, y), (pad, 0), "left")):
        ax.annotate(s, xy=xy, xycoords="axes fraction", xytext=off, textcoords="offset points", ha=ha, va="center",
                    fontsize=fs, color=color, annotation_clip=False)


def lr(ax, left, right, fs=MIN_PT, color=GRAY):
    """v1 name: the two horizontal direction letters, now outside the frame."""
    for xy, off, t, ha in (((0, 0.5), (-2, 0), left, "right"), ((1, 0.5), (2, 0), right, "left")):
        ax.annotate(t, xy=xy, xycoords="axes fraction", xytext=off, textcoords="offset points", ha=ha, va="center",
                    fontsize=fs, color=color, annotation_clip=False)


SCALEBAR_GID = "scalebar_length"                # marks a scale bar's length for check()


def scalebar(ax, px_per_mm, mm, y_frac=0.04, x0_px=None, color=FG, fs=MIN_PT, label=True, lw=1.2, end_px=None):
    """A plain horizontal bar of `mm` millimeters (image pixel units, imshow axes) BELOW the frame, outside the tissue, its
    length printed centered above it, 2 pt over the bar (label=False: the bar alone). The bar starts at x0_px (default: the
    left limit); with end_px, the bar and its length end there instead, by the wider of the two. save() refuses the length
    over an image, over another text, or outside the figure. Returns the length's text (or None)."""
    import matplotlib.transforms as mtrans
    tr = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
    sgn = 1 if ax.get_xlim()[1] > ax.get_xlim()[0] else -1
    L = mm * px_per_mm * sgn
    xl = sorted(ax.get_xlim()); x0 = x0_px if x0_px is not None else xl[0]
    if end_px is not None:
        x0 = end_px - L
    x1 = x0 + L
    ln, = ax.plot([x0, x1], [-y_frac, -y_frac], transform=tr, color=color, lw=lw, solid_capstyle="butt", clip_on=False)
    if not label:
        return None
    t = ax.annotate(f"{mm:g} mm", xy=((x0 + x1) / 2, -y_frac), xycoords=tr, xytext=(0, 2.0 + lw / 2), textcoords="offset points",
                    ha="center", va="baseline", fontsize=fs, color=color, annotation_clip=False)
    t.set_gid(SCALEBAR_GID); t.set_in_layout(False)              # a layout measured on tight boxes does not move for it
    if end_px is not None:                                       # the length wider than the bar: both move back by the excess
        e = t.get_window_extent(ax.figure.canvas.get_renderer()); inv = ax.transData.inverted()
        wd = abs(inv.transform((e.x1, 0))[0] - inv.transform((e.x0, 0))[0])
        dx = max(0.0, (wd - abs(L)) / 2) * sgn
        ln.set_xdata([x0 - dx, x1 - dx]); t.xy = ((x0 + x1) / 2 - dx, -y_frac)
    return t


def scalebar_mm(fig, x_mm, y_mm, mm, print_mm_per_mm, anchor="left", color=FG, fs=MIN_PT, lw=1.2):
    """A plain scale bar placed in figure mm (from the top-left corner, y down, as axes_mm), for a bar outside every panel: its
    line at height y_mm, `mm` tissue millimeters long at the print scale print_mm_per_mm, its length centered above it as
    scalebar() prints it. The bar and its length start at x_mm (anchor="left") or end there ("right"), by the wider of the two.
    Asserts the pair overlaps none of the figure's axes placed so far (draw it last). Returns the text and the pair's box
    (x0, x1, top, bottom; mm)."""
    from matplotlib.lines import Line2D
    W, H = (v * 25.4 for v in fig.get_size_inches())
    L = mm * print_mm_per_mm
    t = fig.text(0, 1 - (y_mm - (2.0 + lw / 2) / 72 * 25.4) / H, f"{mm:g} mm", ha="center", va="baseline", fontsize=fs, color=color,
                 gid=SCALEBAR_GID)
    r = fig.canvas.get_renderer()
    wu = max(L, t.get_window_extent(r).width / fig.dpi * 25.4)
    xc = x_mm + wu / 2 if anchor == "left" else x_mm - wu / 2
    t.set_x(xc / W)
    fig.add_artist(Line2D([(xc - L / 2) / W, (xc + L / 2) / W], [1 - y_mm / H] * 2, transform=fig.transFigure, color=color, lw=lw,
                          solid_capstyle="butt"))
    box = (xc - wu / 2, xc + wu / 2, H - t.get_window_extent(r).y1 / fig.dpi * 25.4, y_mm + lw / 2 / 72 * 25.4)
    for ax in fig.axes:
        p = ax.get_position()
        a0, a1, a2, a3 = p.x0 * W, p.x1 * W, (1 - p.y1) * H, (1 - p.y0) * H
        assert box[1] <= a0 or a1 <= box[0] or box[3] <= a2 or a3 <= box[2], f"the {mm:g} mm scale bar overlaps an axes {p}"
    return t, box


# ---- sections in the stated convention --------------------------------------------------------------------------------------
def canonical(img):
    """(data, affine) of a nibabel image in canonical RAS voxel order (x to the subject's right, y anterior, z superior)."""
    import nibabel as nib
    c = nib.as_closest_canonical(img)
    return np.asanyarray(c.dataobj), c.affine


PLANE_AXIS = dict(sagittal=0, coronal=1, axial=2)


def section(data, plane, index):
    """2-D display array (rows top->bottom; RGB kept as a last axis) of a canonical-RAS volume:
    sagittal: anterior right, superior up; coronal: subject's left on the left (neurological), superior up;
    axial: subject's left on the left, anterior up."""
    s = np.take(data, int(index), axis=PLANE_AXIS[plane])
    return np.flipud(np.swapaxes(s, 0, 1))


def world_index(affine, plane, mm):
    """The voxel index along a plane's axis at world coordinate `mm` (canonical affine)."""
    k = PLANE_AXIS[plane]
    return int(round((mm - affine[k, 3]) / affine[k, k]))


def outline(ax, mask2d, color=None, lw=0.5, **kw):
    """A mask's outline at voxel accuracy on an imshow of the same 2-D array."""
    ax.contour(np.asarray(mask2d, float), [0.5], colors=[color or COL["outline"]], linewidths=lw, **kw)


def provenance_cmap():
    """Photographed fraction: 0 = not photographed (orange) -> 1 = photographed (uncolored gray)."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("provenance", [COL["pred"], COL["photo"]])


def to_nmt(r6plus_dir, pts_world):
    """Blockface-template world points (N,3) -> NMT v2 mm through the template link of record."""
    from blockface.registration import xfmtools as X       # the release's transform tools
    m = X.load_map(os.path.join(r6plus_dir, "mri-template__to__bftemplate"), "full")
    return np.asarray(m(np.atleast_2d(np.asarray(pts_world, float))))


# ---- checks and saving ------------------------------------------------------------------------------------------------------
def check(fig):
    """The house rules a script can break silently: text under 7 pt; text inside an image's axes (titles, headers, letters
    and direction letters sit outside). Returns a list of problems."""
    from matplotlib.text import Text
    fig.canvas.draw(); r = fig.canvas.get_renderer(); bad = []
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text().strip() and t.get_fontsize() < MIN_PT - 1e-6:
            bad.append(f"text under {MIN_PT} pt ({t.get_fontsize():.1f} pt): {t.get_text()[:40]!r}")
    over = lambda e, b: e.x0 < b.x1 - 1 and e.x1 > b.x0 + 1 and e.y0 < b.y1 - 1 and e.y1 > b.y0 + 1
    boxes = [ax.get_window_extent(r) for ax in fig.axes if ax.images]
    for ax in fig.axes:
        cand = [t for t in ax.texts if t.get_visible() and t.get_text().strip()]
        if ax.get_legend() is not None and ax.get_legend().get_visible():
            cand.append(ax.get_legend())
        for t in cand:
            e = t.get_window_extent(r)
            if any(over(e, b) for b in boxes):
                bad.append(f"text over an image: {getattr(t, 'get_text', lambda: 'legend')()[:40]!r}")
    for t in fig.texts:
        if t.get_visible() and t.get_text().strip() and any(over(t.get_window_extent(r), b) for b in boxes):
            bad.append(f"figure text over an image: {t.get_text()[:40]!r}")
    for lg in fig.legends:
        if any(over(lg.get_window_extent(r), b) for b in boxes):
            bad.append("figure legend over an image")
    # a scale bar's length: inside the figure and clear of every other text (texts, titles, and the drawn axes' tick and axis labels)
    others = [t for t in fig.texts if t.get_visible() and t.get_text().strip()]
    for ax in fig.axes:
        others += [t for t in ax.texts if t.get_visible() and t.get_text().strip()]
        others += [t for t in (ax.title, ax._left_title, ax._right_title) if t.get_visible() and t.get_text().strip()]
        if ax.axison:
            others += [t for t in ax.get_xticklabels() + ax.get_yticklabels() + [ax.xaxis.label, ax.yaxis.label]
                       if t.get_visible() and t.get_text().strip()]
    fb = fig.bbox
    for t in others:
        if t.get_gid() != SCALEBAR_GID:
            continue
        e = t.get_window_extent(r)
        if e.x0 < fb.x0 or e.y0 < fb.y0 or e.x1 > fb.x1 or e.y1 > fb.y1:
            bad.append(f"scale bar length outside the figure: {t.get_text()!r}")
        for u in others:
            if u is not t and over(e, u.get_window_extent(r)):
                bad.append(f"scale bar length {t.get_text()!r} over the text {u.get_text()[:40]!r}")
    return bad


def save(fig, out, provenance_inputs, cmd, extra=None, md5=True, strict=True):
    from shared import provenance as timing
    bad = check(fig)
    if bad and strict:
        raise ValueError("figstyle.save: house rules broken:\n  " + "\n  ".join(bad))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=DPI, facecolor=fig.get_facecolor())
    timing.write_provenance(out, inputs=list(provenance_inputs) + [os.path.abspath(__file__)], cmd=cmd, md5=md5,
                            extra=dict(extra or {}, dpi=DPI, width_mm=round(fig.get_size_inches()[0] * 25.4, 1),
                                       house_rules=bad or "ok"))
