"""House style for the template paper's figures, v2 (30 Sep 2026): the author's ISBI treatment
(manuscript/docs/revision_brief_2026-09-30.md, Figures; docs/reviews/template_figure_review_2026-09-30.md).

- Black background for the WHOLE figure, as in the ISBI paper (isbi2027/code/fig*.py); off-white text, gray secondary text
  (THEME = "black", the paper's: the author, 12 Nov 2026). THEME = "white" is an unused option: a white page with black ink off
  the images, every image panel a tile on black (tiles()), ink on images in its own colors (ON_IMG), series colors >= 3:1.
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

# ---- the theme (the author, 12 Nov 2026: white figures; every image panel stays a tile on black) -------------------------
THEME = "black"                              # the paper's figures (the author, 12 Nov 2026: black kept); "white": an option, unused
TILE = "black"                               # every image panel's face, so masked-out areas stay black (tiles(), run by save())
ON_IMG = "#f2f2f2"                           # ink drawn ON images (outlines, overlays, markers): the same in both themes
ON_IMG2 = "#b5b5b5"                          # secondary ink ON images (dashed or de-emphasized outlines): the same in both themes
_THEMES = dict(                              # PAPER: the page and the plots' faces; INK: every text and line off the images;
    black=dict(PAPER="black", INK="#f2f2f2", INK2="#b5b5b5", FRAME="#b5b5b5", FAINT="#5a5a5a",    # INK2: secondary text;
               direct="#1fb5c4", mri="#b08458", pred="#E69F00", uncorrected="#8c9cff", route_text=None),   # FRAME: ticks, frames,
    white=dict(PAPER="white", INK="black", INK2="#666666", FRAME="black", FAINT="#c8c8c8",       # box edges, direction letters;
               # the series colors snapped to the nearest darker step of their own hue with >= 3:1 on white (OKLCH hue kept;
               # dataviz validate_palette, light mode): direct #1fb5c4 -> #01a4b2, mri #b08458 -> #b88148 (chroma to the 0.10
               # floor), pred #E69F00 -> #c68906, uncorrected #8c9cff -> #7e8def; the route labels are text, so text ink
               direct="#01a4b2", mri="#b88148", pred="#c68906", uncorrected="#7e8def", route_text="black"))    # FAINT: grids
_T = _THEMES[THEME]
PAPER, INK, INK2, FRAME, FAINT = (_T[k] for k in ("PAPER", "INK", "INK2", "FRAME", "FAINT"))

# ---- the one color lookup ------------------------------------------------------------------------------------------------
BG = TILE                                    # v2 name: the images' background (set_bad, panel faces)
FG = INK                                     # text, lines and annotations off the images, "with"
GRAY = INK2                                  # secondary text
DGRAY = FAINT                                # faint axes, grids, reference lines
WITHOUT = "#8c8c8c"                          # "without" (vs "with" = FG)
OKABE = dict(orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442", blue="#0072B2", vermilion="#D55E00",
             purple="#CC79A7")               # the raw Okabe-Ito set (drawn on images as it is)
COL = dict(direct=_T["direct"],              # direct route, cyan-teal (v2: #2fa89a, too close to MR256's bluish green)
           mri=_T["mri"],                    # MRI route, tan-brown (v2: ocher #c9a25c, too close to orange and yellow)
           photo="#a6a6a6",                  # photographed tissue: uncolored, neutral gray (v2: blue #3f8fe0); images only
           pred=_T["pred"],                  # tissue not photographed: predicted, interpolated or filled, orange
           outline=ON_IMG, annot=FG, without=WITHOUT, with_=FG,
           direct_text=_T["route_text"] or _T["direct"], mri_text=_T["route_text"] or _T["mri"])
BRAIN = dict(MR243=OKABE["sky"], MR252=OKABE["purple"], MR256=OKABE["green"], MF191=OKABE["yellow"],
             MF282=OKABE["vermilion"])       # fixed per brain, every figure; extend only with unused hues
TRACER = dict(MR243=OKABE["vermilion"], MR252=OKABE["green"], MF191=OKABE["yellow"], MR256=OKABE["blue"])   # the tracer cases' colors in
                                             # every tracer figure (Fig. 10 and its 3D scene, Fig. S4): MM1, MM2, MF8, MM3
COL.update(A=COL["direct"], B=COL["mri"], band=FG, b1=BRAIN["MR243"], b2=BRAIN["MR252"])    # v1 names
COL.update(uncorrected=_T["uncorrected"], corrected=FG)   # v2.2 (2 Oct, the reconstruction figure's (e) at the author's request): photographs before
                                                # intensity correction, a light blue-violet distinct from every brain hue, the
                                                # routes and predicted orange; after correction = FG ("with")
LABELS_JSON = os.path.join(REPO, "settings", "specimen_labels.json")   # the paper's brain names (one mapping file)

# ---- print scales (print mm per tissue mm): one whole-brain scale per figure, plus at most one zoom scale ----------------
SCALE = dict(whole=1.0, strip=0.5, zoom=2.0)
GUTTER_MM, MARGIN_MM = 3.0, 2.0              # the one space between panels (columns and rows) and at the page's edges (the author,
                                             # 12 Nov 2026: "optimize space ... less empty space"); scale bars go in empty corners


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
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER, "savefig.edgecolor": PAPER,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": FRAME, "xtick.color": FRAME, "ytick.color": FRAME,
        "xtick.labelcolor": INK, "ytick.labelcolor": INK, "axes.titlecolor": INK, "grid.color": FAINT, "grid.linewidth": 0.4,
        "legend.frameon": False, "legend.labelcolor": INK, "patch.edgecolor": INK, "hatch.color": FAINT,
        "axes.prop_cycle": cycler(color=[INK, INK2, WITHOUT, FAINT]),
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


PANEL_LETTER_GID = "panel_letter"              # the bold lower-case panel letters (exempt from the sentence-case check)


def letter(ax, s, dx=0.0, dy=3.0):
    """Bold lower-case panel letter at the top left, outside the axes (above its top-left corner)."""
    return ax.annotate(s, xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy), textcoords="offset points", annotation_clip=False,
                       fontsize=LETTER_PT, fontweight="bold", ha="left", va="bottom", color=FG, gid=PANEL_LETTER_GID)


def header(ax, s, dx=11.0, dy=3.0, **kw):
    """A short header above the panel, on the letter's line (regular weight), or alone with dx=0."""
    kw.setdefault("color", FG)
    return ax.annotate(s, xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy), textcoords="offset points", annotation_clip=False,
                       fontsize=kw.pop("fontsize", MIN_PT), ha=kw.pop("ha", "left"), va="bottom", **kw)


DIRS = dict(coronal=("L", "R", "S", "I"), sagittal=("P", "A", "S", "I"), axial=("L", "R", "A", "P"))


def dirs(ax, plane, vertical=False, pad=2.0, color=FRAME, fs=MIN_PT, y=0.5):
    """Direction letters OUTSIDE the frame, regular weight (first panel of a row): left/right of the frame at height y
    (axes fraction, default mid-height); with vertical=True also above/below it. Planes follow section()'s convention."""
    lft, rgt, top, bot = DIRS[plane]
    put = lambda s, xy, off, ha, va: ax.annotate(s, xy=xy, xycoords="axes fraction", xytext=off, textcoords="offset points",
                                                 ha=ha, va=va, fontsize=fs, color=color, annotation_clip=False)
    put(lft, (0, y), (-pad, 0), "right", "center"); put(rgt, (1, y), (pad, 0), "left", "center")
    if vertical:
        put(top, (0.5, 1), (0, pad), "center", "bottom"); put(bot, (0.5, 0), (0, -pad), "center", "top")


def dirs_row(ax_first, ax_last, plane, pad=2.0, color=FRAME, fs=MIN_PT, y=0.5):
    """A row of panels in one orientation: the left letter outside the first panel, the right letter outside the last
    (at height y, axes fraction)."""
    lft, rgt = DIRS[plane][:2]
    for ax, s, xy, off, ha in ((ax_first, lft, (0, y), (-pad, 0), "right"), (ax_last, rgt, (1, y), (pad, 0), "left")):
        ax.annotate(s, xy=xy, xycoords="axes fraction", xytext=off, textcoords="offset points", ha=ha, va="center",
                    fontsize=fs, color=color, annotation_clip=False)


def lr(ax, left, right, fs=MIN_PT, color=FRAME):
    """v1 name: the two horizontal direction letters, now outside the frame."""
    for xy, off, t, ha in (((0, 0.5), (-2, 0), left, "right"), ((1, 0.5), (2, 0), right, "left")):
        ax.annotate(t, xy=xy, xycoords="axes fraction", xytext=off, textcoords="offset points", ha=ha, va="center",
                    fontsize=fs, color=color, annotation_clip=False)


SCALEBAR_GID = "scalebar_length"                # marks a scale bar's length for check()
SCALEBAR_LINE_GID = "scalebar_line"             # ... and its line
CLEAR_MM = 1.5                                  # every bar and its length keep this much print space from every other text (the
                                                # author, 12 Nov 2026; check() asserts it for every bar, scalebar_free() places by it)


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
    ln, = ax.plot([x0, x1], [-y_frac, -y_frac], transform=tr, color=color, lw=lw, solid_capstyle="butt", clip_on=False,
                  gid=SCALEBAR_LINE_GID)
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


def scalebar_mm(fig, x_mm, y_mm, mm, print_mm_per_mm, anchor="left", color=FG, fs=MIN_PT, lw=1.2, panels=True):
    """A plain scale bar placed in figure mm (from the top-left corner, y down, as axes_mm), for a bar outside every panel: its
    line at height y_mm, `mm` tissue millimeters long at the print scale print_mm_per_mm, its length centered above it as
    scalebar() prints it. The bar and its length start at x_mm (anchor="left") or end there ("right"), by the wider of the two.
    Asserts the pair overlaps none of the figure's panels placed so far (axes with an image, or drawn axes; draw it last), unless
    panels=False (scalebar_free: a place measured empty, which may be an image's empty background).
    Returns the text and the pair's box (x0, x1, top, bottom; mm)."""
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
                          solid_capstyle="butt", gid=SCALEBAR_LINE_GID))
    box = (xc - wu / 2, xc + wu / 2, H - t.get_window_extent(r).y1 / fig.dpi * 25.4, y_mm + lw / 2 / 72 * 25.4)
    for ax in (fig.axes if panels else []):
        if not (ax.images or ax.axison):                         # an invisible layout axes (headers, keys) is not a panel
            continue
        p = ax.get_position()
        a0, a1, a2, a3 = p.x0 * W, p.x1 * W, (1 - p.y1) * H, (1 - p.y0) * H
        assert box[1] <= a0 or a1 <= box[0] or box[3] <= a2 or a3 <= box[2], f"the {mm:g} mm scale bar overlaps an axes {p}"
    return t, box


def scalebar_free(fig, mm, print_mm_per_mm, region_mm, corner="bottom left", clear_mm=CLEAR_MM, color=FG, fs=MIN_PT, lw=1.2,
                  dpi=150):
    """A plain scale bar (as scalebar_mm) in an EMPTY corner found by measurement, on the page or on an image's empty background:
    the figure as drawn so far (every artist; call it last) is rendered once at `dpi`; every pixel that is not the background
    (PAPER or TILE) is occupied, tissue, text and color bars alike; the occupied set is grown by clear_mm; the bar and its length go
    at the free position inside region_mm = (x0, x1, y0, y1) (figure mm from the top-left corner, y down) nearest that corner of the
    region. Raises if the region has no such position. Returns the text and the pair's box (x0, x1, top, bottom; mm)."""
    from matplotlib.colors import to_rgb
    from scipy import ndimage as ndi
    W, H = (v * 25.4 for v in fig.get_size_inches())
    dpi0 = fig.dpi; fig.set_dpi(dpi); fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(np.float32) / 255.0
    r = fig.canvas.get_renderer()
    t = fig.text(0, 0, f"{mm:g} mm", ha="center", va="baseline", fontsize=fs)
    e = t.get_window_extent(r); t.remove(); fig.set_dpi(dpi0)
    px = dpi / 25.4                                                       # pixels per print mm
    free = np.zeros(buf.shape[:2], bool)
    for c_ in {to_rgb(PAPER), to_rgb(TILE)}:
        free |= np.abs(buf - np.asarray(c_, np.float32)).max(-1) <= 0.08
    ok = ndi.distance_transform_edt(free) > clear_mm * px                  # free, and clear_mm from anything drawn
    L = mm * print_mm_per_mm; wu = max(L, e.width / px)                   # the pair: as wide as the wider of bar and length,
    up = (2.0 + lw / 2) / 72 * 25.4 + (e.y1 - e.y0) / px                  # from the length's top down to the bar's lower edge
    h = up + lw / 2 / 72 * 25.4
    wp, hp = int(np.ceil(wu * px)), int(np.ceil(h * px))
    S_ = np.pad(np.cumsum(np.cumsum((~ok).astype(np.int32), 0), 1), ((1, 0), (1, 0)))    # occupied count in any box: O(1)
    x0, x1, y0, y1 = (int(round(v * px)) for v in region_mm)
    best = None
    for yt in range(max(y0, 0), min(y1, ok.shape[0]) - hp + 1):           # yt: the pair's top row; xl: its left column
        for xl in range(max(x0, 0), min(x1, ok.shape[1]) - wp + 1):
            if S_[yt + hp, xl + wp] - S_[yt, xl + wp] - S_[yt + hp, xl] + S_[yt, xl]:
                continue
            dx = xl - x0 if "left" in corner else x1 - (xl + wp)
            dy = y1 - (yt + hp) if "bottom" in corner else yt - y0
            d = dx * dx + dy * dy
            if best is None or d < best[0]:
                best = (d, xl, yt)
    if best is None:
        raise ValueError(f"no place for the {mm:g} mm bar {clear_mm:g} mm clear of everything drawn in the region {region_mm}")
    _, xl, yt = best
    return scalebar_mm(fig, xl / px, yt / px + up, mm, print_mm_per_mm, anchor="left", color=color, fs=fs, lw=lw, panels=False)


def scalebar_title(ax, px_per_mm, mm, title, color=FG, fs=MIN_PT, lw=1.2, clear_mm=CLEAR_MM):
    """A zoomed panel's scale bar in its standard place (the author, 12 Nov 2026): at the panel title's level, top right; the bar on
    the title's baseline and its length above it (2 pt over the bar), the two right-aligned with the panel's right edge. `ax`: the
    panel whose right edge and scale the bar takes (imshow axes, px_per_mm data units per tissue mm); `title`: the panel's title (a
    Text). Placed in figure coordinates, so call it once the layout and every text near it are final. Asserts the bar and its length
    keep clear_mm (1.5 mm) from every other text of the figure. Returns the length's text."""
    from matplotlib.lines import Line2D
    from matplotlib.transforms import Bbox
    fig = ax.figure; r = fig.canvas.get_renderer(); pt = fig.dpi / 72
    e = title.get_window_extent(r)
    base = e.y0 + r.get_text_width_height_descent("lp", title.get_fontproperties(), ismath=False)[2]   # the title's baseline (px)
    L = abs(ax.transData.transform((mm * px_per_mm, 0))[0] - ax.transData.transform((0, 0))[0])        # the bar (px)
    right = ax.get_window_extent(r).x1
    others = [u for u in fig.texts if u.get_visible() and u.get_text().strip()]
    for a_ in fig.axes:
        others += [u for u in a_.texts if u.get_visible() and u.get_text().strip()]
    t = fig.text(0, 0, f"{mm:g} mm", fontsize=fs, color=color, ha="center", va="baseline", gid=SCALEBAR_GID)
    wu = max(L, t.get_window_extent(r).width); xc = right - wu / 2
    inv = fig.transFigure.inverted()
    yb = base + lw / 2 * pt                                                  # the bar sits on the title's baseline
    t.set_position(tuple(inv.transform((xc, yb + (2.0 + lw / 2) * pt))))
    (x0f, ybf), (x1f, _) = inv.transform((xc - L / 2, yb)), inv.transform((xc + L / 2, yb))
    fig.add_artist(Line2D([x0f, x1f], [ybf, ybf], transform=fig.transFigure, color=color, lw=lw, solid_capstyle="butt",
                          gid=SCALEBAR_LINE_GID))
    et = t.get_window_extent(r); c = clear_mm / 25.4 * fig.dpi
    pair = Bbox([[xc - wu / 2 - c, yb - lw / 2 * pt - c], [xc + wu / 2 + c, et.y1 + c]])
    for u in others:
        b = u.get_window_extent(r)
        assert pair.x1 <= b.x0 or b.x1 <= pair.x0 or pair.y1 <= b.y0 or b.y1 <= pair.y0, \
            f"the {mm:g} mm bar is closer than {clear_mm:g} mm to the text {u.get_text()[:30]!r}"
    return t


def scalebar_in(ax, px_per_mm, mm, corner="bottom left", inset_mm=0.8, color=FG, fs=MIN_PT, lw=1.2):
    """A plain scale bar INSIDE an image panel, in one of its corners ("top left", "top right", "bottom left", "bottom right")
    on the image's empty background, inset_mm (print) from the two edges, its length centered above it as scalebar() prints it
    (the wider of the two at the inset). It adds no room to the figure; save() refuses it over tissue. `mm` tissue millimeters
    at px_per_mm data units per mm (imshow axes). Returns the length's text."""
    import matplotlib.transforms as mtrans  # noqa: F401
    fig = ax.figure; r = fig.canvas.get_renderer()
    bb = ax.get_window_extent(r); pxmm = fig.dpi / 25.4; pt = fig.dpi / 72.0
    xl = ax.get_xlim(); L = abs(mm * px_per_mm / (xl[1] - xl[0]))          # the bar, in axes fraction
    t = ax.text(0, 0, f"{mm:g} mm", transform=ax.transAxes, ha="center", va="baseline", fontsize=fs, color=color,
                gid=SCALEBAR_GID, clip_on=False)
    t.set_in_layout(False)
    e = t.get_window_extent(r)
    tw, asc = e.width / bb.width, (e.y1 - bb.y0) / bb.height                 # the length's width and its height above the baseline
    W_ = max(L, tw); ix, iy = inset_mm * pxmm / bb.width, inset_mm * pxmm / bb.height
    gap = (2.0 + lw / 2) * pt / bb.height; half = lw / 2 * pt / bb.height
    xc = ix + W_ / 2 if "left" in corner else 1 - ix - W_ / 2
    if "bottom" in corner:
        yb = iy + half; base = yb + gap
    else:
        base = 1 - iy - asc; yb = base - gap
    t.set_position((xc, base))
    ax.plot([xc - L / 2, xc + L / 2], [yb, yb], transform=ax.transAxes, color=color, lw=lw, solid_capstyle="butt", clip_on=False,
            gid=SCALEBAR_LINE_GID)
    return t


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
    """The house rules a script can break silently: text under 7 pt; text or a scale bar over tissue (an image panel's empty
    background is allowed); a bar's length off the page or over another text; text below 3:1. Returns a list of problems."""
    from matplotlib.text import Text
    fig.canvas.draw(); r = fig.canvas.get_renderer(); bad = []
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text().strip() and t.get_fontsize() < MIN_PT - 1e-6:
            bad.append(f"text under {MIN_PT} pt ({t.get_fontsize():.1f} pt): {t.get_text()[:40]!r}")
    over = lambda e, b: e.x0 < b.x1 - 1 and e.x1 > b.x0 + 1 and e.y0 < b.y1 - 1 and e.y1 > b.y0 + 1
    boxes = [ax.get_window_extent(r) for ax in fig.axes if ax.images]
    # no text (and no scale bar) over tissue: what lies under each, inside an image panel, drawn once at 200 dpi with every text
    # and scale bar line hidden, must be the image's empty background (TILE) only
    from matplotlib.lines import Line2D
    from matplotlib.colors import to_rgb
    hide = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    bars = [ln for ln in fig.findobj(Line2D) if ln.get_gid() == SCALEBAR_LINE_GID and ln.get_visible()]
    dpi0 = fig.dpi; fig.set_dpi(200)
    for a_ in hide + bars:
        a_.set_visible(False)
    fig.canvas.draw(); buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(np.float32) / 255.0
    for a_ in hide + bars:
        a_.set_visible(True)
    fig.canvas.draw(); r = fig.canvas.get_renderer(); bg = np.asarray(to_rgb(TILE), np.float32)
    boxes = [ax.get_window_extent(r) for ax in fig.axes if ax.images]

    def on_tissue(e):
        for b in boxes:
            x0, x1, y0, y1 = max(e.x0, b.x0), min(e.x1, b.x1), max(e.y0, b.y0), min(e.y1, b.y1)
            if x1 - x0 > 1 and y1 - y0 > 1:
                sub = buf[max(int(buf.shape[0] - y1), 0):int(np.ceil(buf.shape[0] - y0)), max(int(x0), 0):int(np.ceil(x1))]
                if sub.size and (np.abs(sub - bg).max(-1) > 0.08).any():
                    return True
        return False
    for ax in fig.axes:
        cand = [t for t in ax.texts if t.get_visible() and t.get_text().strip()]
        if ax.get_legend() is not None and ax.get_legend().get_visible():
            cand.append(ax.get_legend())
        for t in cand:
            if on_tissue(t.get_window_extent(r)):
                bad.append(f"text over tissue: {getattr(t, 'get_text', lambda: 'legend')()[:40]!r}")
    for t in fig.texts:
        if t.get_visible() and t.get_text().strip() and on_tissue(t.get_window_extent(r)):
            bad.append(f"figure text over tissue: {t.get_text()[:40]!r}")
    for lg in fig.legends:
        if on_tissue(lg.get_window_extent(r)):
            bad.append("figure legend over tissue")
    for ln in bars:
        if on_tissue(ln.get_window_extent(r)):
            bad.append("scale bar over tissue")
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
    # every bar and its length: CLEAR_MM from every other text (texts, titles, tick and axis labels, legend entries); touching is
    # not overlapping, so a length that runs into a label ("10 mmNormalized") is caught here
    near_txt = others + [u for ax in fig.axes if ax.get_legend() is not None and ax.get_legend().get_visible()
                         for u in ax.get_legend().get_texts() if u.get_text().strip()] \
                      + [u for lg in fig.legends for u in lg.get_texts() if u.get_text().strip()]
    c = CLEAR_MM / 25.4 * fig.dpi
    near = lambda e, b: e.x0 - c < b.x1 and e.x1 + c > b.x0 and e.y0 - c < b.y1 and e.y1 + c > b.y0     # noqa: E731
    for t in [u for u in near_txt if u.get_gid() == SCALEBAR_GID]:
        e = t.get_window_extent(r)
        for u in near_txt:
            if u is not t and near(e, u.get_window_extent(r)):
                bad.append(f"scale bar length {t.get_text()!r} closer than {CLEAR_MM:g} mm to the text {u.get_text()[:40]!r}")
    for ln in bars:
        e = ln.get_window_extent(r)
        for u in near_txt:
            if u.get_gid() != SCALEBAR_GID and near(e, u.get_window_extent(r)):
                bad.append(f"a scale bar closer than {CLEAR_MM:g} mm to the text {u.get_text()[:40]!r}")
    # every text against what lies under its center, an image panel's black tile or the page: at least 3:1 (no black on a tile,
    # no white on white)
    for t in others:
        e = t.get_window_extent(r); cx, cy = (e.x0 + e.x1) / 2, (e.y0 + e.y1) / 2
        under = TILE if any(b.x0 <= cx <= b.x1 and b.y0 <= cy <= b.y1 for b in boxes) else PAPER
        if contrast(t.get_color(), under) < 3.0:
            bad.append(f"text {t.get_text()[:40]!r} in {t.get_color()} on {under} ({contrast(t.get_color(), under):.2f}:1)")
    # sentence case (the author, 12 Nov 2026): every label (texts, titles, axis and color bar labels, legend entries) starts with a
    # capital letter, unless it starts with a number or a symbol or with one of LOWERCASE_OK; panel letters, single letters
    # (orientation), tick labels and scale bar lengths are exempt
    labs = [t for t in fig.texts if t.get_visible() and t.get_text().strip()]
    for ax in fig.axes:
        labs += [t for t in ax.texts if t.get_visible() and t.get_text().strip()]
        labs += [t for t in (ax.title, ax._left_title, ax._right_title) if t.get_visible() and t.get_text().strip()]
        if ax.axison:
            labs += [t for t in (ax.xaxis.label, ax.yaxis.label) if t.get_visible() and t.get_text().strip()]
        if ax.get_legend() is not None:
            labs += [t for t in ax.get_legend().get_texts() if t.get_text().strip()]
    for lg in fig.legends:
        labs += [t for t in lg.get_texts() if t.get_text().strip()]
    for t in labs:
        if t.get_gid() not in (SCALEBAR_GID, PANEL_LETTER_GID) and not sentence_case_ok(t.get_text()):
            bad.append(f"not in sentence case: {t.get_text()[:40]!r}")
    return bad


LOWERCASE_OK = ("b0", "x =", "y =", "z =", "x ", "y ", "z ", "mm", "µm", "z-units", "°", "anatomix", "nnU-Net", "dof")


def sentence_case_ok(s):
    """True if a label is in sentence case: its first letter (after a leading sub-panel numeral "(i)" ... "(x)", and any leading
    dash, bullet or bracket) is upper case, or it starts with a number, a symbol or one of LOWERCASE_OK (symbols, units, names lower
    case by convention), or it is one letter."""
    import re
    s = re.sub(r"^\((i|ii|iii|iv|v|vi|vii|viii|ix|x)\)\s*", "", s.strip())   # a leading sub-panel numeral, "(i)", as a panel letter
    if len(s) <= 1:
        return True
    s = s.lstrip("—–-•(+ ")
    if not s or not s[0].isalpha() or s.startswith(LOWERCASE_OK):
        return True
    return s[0].isupper()


def contrast(c1, c2):
    """WCAG contrast ratio of two matplotlib colors."""
    from matplotlib.colors import to_rgb
    lum = lambda c: float(np.dot([0.2126, 0.7152, 0.0722], [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
                                                                for v in to_rgb(c)]))   # noqa: E731
    hi, lo = sorted((lum(c1), lum(c2)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


TILE_GID = "image_tile"


def tiles(fig):
    """Every image panel a tile on black: a TILE rectangle behind each axes that shows an image (its whole frame), so what the
    image leaves transparent or uncovered stays black on the white page. Idempotent; save() runs it (call it before any other
    savefig, e.g. a PDF written beside)."""
    from matplotlib.patches import Rectangle
    if TILE == PAPER:                                            # the black page: nothing to add, the files stay as they were
        return
    for ax in fig.axes:
        if ax.images and not any(p.get_gid() == TILE_GID for p in ax.patches):
            ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=TILE, edgecolor="none", zorder=-1e3,
                                   clip_on=False, gid=TILE_GID))


def save(fig, out, provenance_inputs, cmd, extra=None, md5=True, strict=True):
    from shared import provenance as timing
    tiles(fig)
    bad = check(fig)
    if bad and strict:
        raise ValueError("figstyle.save: house rules broken:\n  " + "\n  ".join(bad))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=DPI, facecolor=fig.get_facecolor())
    timing.write_provenance(out, inputs=list(provenance_inputs) + [os.path.abspath(__file__)], cmd=cmd, md5=md5,
                            extra=dict(extra or {}, dpi=DPI, width_mm=round(fig.get_size_inches()[0] * 25.4, 1),
                                       house_rules=bad or "ok"))
