#!/usr/bin/env python3
"""The paper's figure "Histology registered to the blockface" (TIRL 3.6.2), with the marker tubes hidden from the registration.

    python code/figures/histology.py --jp2-root <LINC source data>/sourcedata [--out figures/histology.png]

Needs glymur (the raw histology is JPEG 2000). --jp2-root is the LINC source-data tree that holds sub-<ID>/micr/<section>.jp2.
Layout (the author, 12 Nov 2026): two rows and four columns, the first wider; the blockface on the top row, the histology on the bottom.
Orientation letters (the author, 12 Nov 2026: "the histo doesn't have the L R"; figstyle.orient_letters): L and R beside each of
(a)'s two images, ORIENT_GAP_MM from its tissue, at its tissue's middle.
  a  (first column) one section registered to its blockface slice: a fixed, intact MM3 piece with a detected tube hole
     (settings/figures.yaml histology panel_a_piece; its nonlinear tissue Dice and MM3's median, numbers_of_record/
     registration_qc_histo2bf_h9.json, family histo2bf, go in the provenance; with the setting empty, the piece whose Dice is closest
     to that median): on top the blockface photograph (the registration's fixed image, the tubes visible) with the histology's tissue
     outline (white), below the registered darkfield histology (contrast-stretched); an orange arrowhead beside the slice's tube hole,
     in both images: the tube the offset record scores for this piece (tube_landmarks_h9 h9hole: the largest tube-mask component on
     the slice, its hole-only center, recomputed and checked as in b). Another tube's mask can end on the slice obliquely, over a
     streak and not a hole (MM3 s18: a 4-voxel tail at the end of a tube that stops 6 slices later); it is not marked.
     Flipped left-right into the neurological convention (the photographs show the subject's left on the image right); dorsal up;
     loose blockface fragments without histology removed. The 5 mm bar on (a)'s title line, top right of its column, as (b)'s (the
     author, 12 Nov 2026), at the blockface's own pixel (TIRL's fixed frame = half its
     L2 voxel, from the volume's header; TIRL's resolution_mm is a nominal size given to both images).
  b  (columns 2-4) marker tubes, three fixed quantiles (25th, 50th, 90th) of the pooled offsets: for each, the found piece whose
     offset is closest to that quantile, or the one settings/figures.yaml histology panel_b_pieces names (within PIN_RANKS ranks of
     the quantile). Top: the blockface photograph with the tube (the registration's fixed image, rebuilt for the whole slice from the
     store before inpainting, steps/<ID>_03_color.ome.zarr; checked against a's), centered on its hole-only blockface center (+; recomputed
     as figures.evaluation.tubes does and checked against the record's center_shift_fixed_px); bottom: the histology at near-native
     resolution (JP2 level 2, ~1.8 um/px), centered on that center carried into the histology through the piece's tubes-hidden TIRL
     chain (+), o = the hole found in the histology (its recorded area); both white, black halo. One window, WIN_MM of tissue, for
     all six; the offset above each histology window; one 0.5 mm bar, on (b)'s title line.
  c  ECDF of the tube offset of the six brains, tubes hidden from the registration (every piece in which the hole was found; blockface
     tube center from the hole only); the median as a thin vertical line, checked against $LINC_NUMBERS/tube_offsets_h9.json
     (variant hole_hidden; code/figures/paper_numbers/histology_h9.py).
  d  per-brain medians (the same record).
settings/figures.yaml histology panels: the panels drawn, [a, b] or all four (empty: all four; c and d go under the grid); the checks
against tube_offsets_h9 (the distribution's size and median) run either way.
Reads (settings/internal/products_mri.yaml): tube_landmarks_h9/{h9preinp,h9hole}_<ID>_native.json, products_blockface labels_histology_arm (panel
a's TIRL piece, in MM3's own arm), the blockface data of the four brains shown (<ID>_blockface_L2.nii.gz's header, <ID>_holes.nii.gz,
steps/<ID>_03_color.ome.zarr), numbers_of_record/registration_qc_histo2bf_h9.json, $LINC_NUMBERS/tube_offsets_h9.json, the JP2s. The registrations are
those with the per-component nonlinear stiffness (settings/histology.yaml tirl.stiff_nonlinear, 12 Nov 2026). Writes the PNG and its provenance.json.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "code"))
import figstyle as S                                           # noqa: E402
from shared import paths                                         # noqa: E402

Image.MAX_IMAGE_PIXELS = None
MAIN = ["MR243", "MR252", "MR256", "MF278", "MF275", "MF191"]  # the six histology brains (LINC IDs)
A_SUBJ = "MR256"                                               # panel (a)'s brain
PX_MM = 0.4397e-3                                              # JP2 level-0 pixel (mm)
SHIFT_TOL_PX = 0.25                                            # (b): the recomputed blockface center's shift vs the record's
ARROW_PT, ARROW_GAP_MM = 4.0, 0.3                              # (a): the arrowhead (pt) and its tip's gap to the tube (print mm)
JP2_LEVEL = 4                                                  # (b): read every 4th level-0 pixel (JP2 level 2)
JP2 = "sub-{S}/micr/{sid}.jp2"                                 # under --jp2-root
TIRL = "sub-{S}/blockface/06_histo2bf/tirl/{piece}"            # under the brain's labels_histology_arm
TUBE_VARIANT = "hole_hidden"                                   # the paper's tube set
QUANT = (25, 50, 90)
PIN_RANKS = 3                                                  # (b): a named piece lies within this many ranks of its quantile
WIN_MM = 3.0                                                   # (b): window side
ECDF_XMAX = 1.0                                                # (c): to the 1 mm search radius (the curve reaches 1)
SCALEBARS_MM = (5, 0.5)                                        # (a), (b)
L1_PT, L2_PT = 12.0, 2.0                                       # the header lines above a row: the title's, the sub-labels'


def stretch(rgb, m, top=99.5):
    lo, hi = np.percentile(rgb[m], [1, top], axis=0)
    return np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1)


def largest(m):
    lab, n = ndi.label(m)
    return lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1) if n else m


def text_mm(plt):
    """a function (text, size, **font) -> (width, height) mm as figstyle draws it (at its DPI: at the screen's 100, hinting
    widens a 7 pt text by up to 8 %); T.ink(text, size, **font): the ink's top above the text's box bottom (mm)"""
    fig = plt.figure(figsize=(7, 2), dpi=S.DPI); r = fig.canvas.get_renderer()
    def T(s, fs=S.MIN_PT, **kw):
        t = fig.text(0, 0, s, fontsize=fs, **kw); e = t.get_window_extent(r); t.remove()
        return e.width / fig.dpi * 25.4, e.height / fig.dpi * 25.4
    def ink(s, fs=S.MIN_PT, **kw):
        g = plt.figure(figsize=(7, 1), dpi=S.DPI, facecolor="k"); t = g.text(0.01, 0.1, s, fontsize=fs, color="w", va="bottom", **kw)
        g.canvas.draw(); a_ = np.asarray(g.canvas.buffer_rgba())[..., 0]; y0 = t.get_window_extent().y0; plt.close(g)
        return (a_.shape[0] - np.where((a_ > 60).any(1))[0].min() - y0) / S.DPI * 25.4
    def asc(fs=S.MIN_PT):
        from matplotlib.font_manager import FontProperties
        _, h_, d_ = r.get_text_width_height_descent("lp", FontProperties(size=fs), ismath=False)
        return (h_ - d_) / fig.dpi * 25.4
    T.ink, T.asc = ink, asc                                            # T.asc: a text line's ascent (mm), matplotlib's line pitch unit
    return T


def bf_data(S):
    """the brain's blockface folder (products_blockface data)"""
    return paths.products("products_blockface")["data"] / f"sub-{S}" / "blockface"


def fix_mm(S):
    """the TIRL fixed frame's pixel (mm): half the L2 voxel in-plane (the frame is the L2 slice upsampled 2 x, as
    figures.evaluation.tubes maps it), from the volume's header"""
    import nibabel as nib
    z = nib.load(str(bf_data(S) / f"{S}_blockface_L2.nii.gz")).header.get_zooms()
    assert abs(z[0] - z[2]) < 1e-6, (S, z)
    return float(z[0]) / 2


def bf_photo(S, idx):
    """the registration's fixed image of slice idx, the whole slice, tubes visible: the store before inpainting
    (steps/<ID>_03_color.ome.zarr, level 0) flipped in both axes into TIRL's fixed frame (rows, columns), grey and contrast as
    code/histology/register.fixed_images (settings/histology.yaml fixed.contrast); 0..1"""
    import zarr
    from PIL import ImageEnhance
    con = yaml.safe_load((HERE.parents[1] / "settings" / "histology.yaml").read_text())["fixed"]["contrast"]
    z = zarr.open_group(str(bf_data(S) / "steps" / f"{S}_03_color.ome.zarr"), mode="r")["0"]
    rgb = np.ascontiguousarray(np.asarray(z[:, :, idx, :]).transpose(1, 2, 0)[::-1, ::-1])
    return np.asarray(ImageEnhance.Contrast(Image.fromarray(rgb)).enhance(con).convert("L"), np.float32) / 255


def tube_centers(S, idx, g):
    """every marker tube of slice idx in TIRL's fixed frame, as figures.evaluation.tubes.one_piece finds the piece's (on the whole
    slice's tissue): [(mask_px, mask center, hole-only center, refined, hole_px)], centers (row, column)"""
    import nibabel as nib
    from scipy.ndimage import label, center_of_mass, binary_fill_holes, binary_dilation
    from figures.evaluation.tubes import otsu, RING_PX, MIN_KEEP
    A_d = nib.load(str(bf_data(S) / f"{S}_blockface_L2.nii.gz")).affine
    him = nib.load(str(bf_data(S) / f"{S}_holes.nii.gz")); Hv = np.asarray(him.dataobj) > 0
    g = g * 255; Hd, Wd = g.shape
    U, V = np.meshgrid(np.arange(Hd), np.arange(Wd), indexing="ij"); R, C = Hd - 1 - U, Wd - 1 - V
    vox = np.stack([(C - 0.5) / 2, np.full(C.shape, float(idx)), (R - 0.5) / 2, np.ones(C.shape)], -1)
    v = np.round((vox @ A_d.T) @ np.linalg.inv(him.affine).T)[..., :3].astype(int)
    ok = np.all((v >= 0) & (v < np.array(Hv.shape)), -1)
    tb = np.zeros((Hd, Wd), bool); tb[ok] = Hv[v[ok, 0], v[ok, 1], v[ok, 2]]
    tb &= binary_fill_holes(g > 0)
    lab, n = label(tb); out = []
    for k in range(1, n + 1):
        comp = lab == k; cm_mask = center_of_mass(comp)
        zone = binary_dilation(comp, iterations=RING_PX) & (g > 0)
        th = otsu(g[zone]) if zone.sum() > 10 else None
        hole = binary_fill_holes(comp & (g < th)) if th is not None else np.zeros_like(comp)
        lh, nh = label(hole)
        if nh:
            hole = lh == (int(np.argmax(np.bincount(lh.ravel())[1:])) + 1)
        refined = bool(nh) and hole.sum() >= MIN_KEEP * comp.sum()
        out.append((int(comp.sum()), cm_mask, center_of_mass(hole) if refined else cm_mask, refined, int(hole.sum()) if refined else int(comp.sum())))
    return out


def draw_cd(fig, S, e_inp, ps, x0, yc, wrow, CH):
    """(c) the ECDF of the offsets and (d) the per-brain medians, side by side under (b)"""
    wc = wrow * 0.54                                                  # (c) wide enough for (d)'s horizontal brain labels
    ax = S.axes_mm(fig, x0 + 12.0, yc, wc - 12.0, CH)
    s = np.sort(e_inp); ax.step(s, np.arange(1, len(s) + 1) / len(s), where="post", color=S.FG, lw=1.1, ls="-", label="tubes hidden")
    ax.axvline(np.median(e_inp), color=S.FG, lw=0.6, ls=":")
    ax.annotate(f"{np.median(e_inp):.2f}", xy=(np.median(e_inp), 1), xycoords=("data", "axes fraction"), xytext=(0, 2),
                textcoords="offset points", ha="center", va="bottom", color=S.FG, annotation_clip=False)
    ax.set_xlabel("tube offset (mm)"); ax.set_ylabel("fraction of components")
    ax.set_xlim(0, ECDF_XMAX); ax.set_ylim(0, 1.02); ax.spines[["top", "right"]].set_visible(False)
    S.letter(ax, "c", dx=-16 - 12.0 / 25.4 * 72, dy=2)                 # aligned with a / b
    # (d) per-brain medians
    axd = S.axes_mm(fig, x0 + wc + 16.0, yc, wrow - wc - 16.0, CH)
    xs_ = np.arange(len(MAIN))
    hid = [ps[S_][TUBE_VARIANT]["median_mm"] for S_ in MAIN]
    axd.plot(xs_, hid, "o", color=S.FG, ms=3.8, mec="none", zorder=3)
    axd.set_xticks(xs_); axd.set_xticklabels([S.label(S_) for S_ in MAIN])
    axd.set_ylabel("median offset (mm)"); axd.set_ylim(0, 1.15 * max(hid)); axd.set_xlim(-0.6, len(MAIN) - 0.4)
    axd.spines[["top", "right"]].set_visible(False)
    S.letter(axd, "d", dx=-28, dy=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jp2-root", required=True, help="LINC source-data tree holding sub-<ID>/micr/<section>.jp2")
    ap.add_argument("--out", default="figures/histology.png")
    a = ap.parse_args()
    PR = paths.products("products_mri")
    NUM, TUBE = PR["numbers_of_record"], PR["tube_landmarks_h9"]
    TJP = paths.numbers_root() / "tube_offsets_h9.json"
    pieces = lambda tag, S_: {r["piece"]: r for r in json.loads((TUBE / f"{tag}_{S_}_native.json").read_text())["pieces"]}  # noqa: E731
    ins = []
    # ---- a: representative piece ----
    rq = [r for r in json.loads((NUM / "registration_qc_histo2bf_h9.json").read_text())["data"] if r.get("family") == "histo2bf" and r["subject"] == A_SUBJ]
    med = float(np.median([r["metrics"]["dice_nonlinear"] for r in rq]))
    tp = pieces("h9preinp", A_SUBJ)
    CFG = yaml.safe_load((HERE.parents[1] / "settings" / "figures.yaml").read_text()).get("histology") or {}
    fixed = CFG.get("panel_a_piece")
    if fixed:
        ra = next((r for r in rq if r["tag"] == fixed), None)
        assert ra is not None and tp.get(fixed, {}).get("found"), f"panel_a_piece {fixed}: not a {A_SUBJ} piece with a found tube hole"
    else:
        ra = min((r for r in rq if tp.get(r["tag"], {}).get("found")), key=lambda r: abs(r["metrics"]["dice_nonlinear"] - med))
    P = paths.products_root() / paths.products("products_blockface")["labels_histology_arm"][A_SUBJ] / TIRL.format(S=A_SUBJ, piece=ra["tag"])
    fx = np.asarray(Image.open(P / "fixed.png"), np.float32) / 255     # the registration's fixed image: the tubes visible
    a_idx = tp[ra["tag"]]["bf_slice"]; fmm_a = fix_mm(A_SUBJ)
    a_all = tube_centers(A_SUBJ, a_idx, fx)                            # every tube-mask component on the slice
    a_rec = pieces("h9hole", A_SUBJ)[ra["tag"]]                         # the one the offset record scores (the paper's, tubes hidden)
    a_tubes = [t for t in a_all if t[0] == a_rec["mask_px"] and abs(np.hypot(*np.subtract(t[2], t[1])) - a_rec["center_shift_fixed_px"]) <= SHIFT_TOL_PX]
    assert len(a_tubes) == 1 and a_rec["hole"]["found"], f"panel a: the record's tube does not reproduce ({a_all})"
    d_ = np.abs(bf_photo(A_SUBJ, a_idx) - fx)[fx > 0]                    # (b)'s photographs come from the store: it must be this image
    assert (d_ > 3 / 255).mean() < 1e-3, f"panel a: steps/{A_SUBJ}_03_color is not the registration's fixed image ({(d_ > 3 / 255).mean():.4f})"
    rgb = np.asarray(Image.open(P / "moving4_nonlinear_rgb.png").convert("RGB"), np.float32)
    mx = rgb.max(-1)
    tis = largest(ndi.binary_fill_holes(ndi.binary_opening((mx > 6) & (mx < 250), iterations=2)))
    hs = stretch(rgb, tis, top=99.8) * tis[..., None]                   # 99.8: bright dorsal spots not clipped to saturation
    fxd = fx * ((fx > 0.02) & ndi.binary_dilation(tis, iterations=12))
    ys, xs = np.where((fxd > 0) | tis); pad = 3                          # 3 px (0.1 mm): the outline's line width, no more (space, 12 Nov 2026)
    cr = (slice(max(ys.min() - pad, 0), ys.max() + pad), slice(max(xs.min() - pad, 0), xs.max() + pad))
    flip = lambda a_: a_[:, ::-1]                                       # noqa: E731
    soft = lambda m: np.clip(ndi.gaussian_filter(m.astype(np.float32), 1.0), 0, 1)   # noqa: E731  antialiased tissue edge (~1 px)
    fxd = fx * soft(fxd > 0); hs = hs * soft(tis)[..., None]
    pa = [flip(fxd[cr]), flip(hs[cr])]; outl = flip(tis[cr])
    ins += [P / "fixed.png", P / "moving4_nonlinear_rgb.png", NUM / "registration_qc_histo2bf_h9.json"]
    ins += [bf_data(A_SUBJ) / f"{A_SUBJ}_blockface_L2.nii.gz", bf_data(A_SUBJ) / f"{A_SUBJ}_holes.nii.gz", bf_data(A_SUBJ) / "steps" / f"{A_SUBJ}_03_color.ome.zarr"]
    # ---- c: the six-brain distribution, tubes hidden, hole-only centers ----
    TJ = json.loads(TJP.read_text())["data"]
    e_inp, cand = [], []
    for S_ in MAIN:
        for r_ in pieces("h9hole", S_).values():
            if r_.get("bf_tube") and r_["hole"]["found"]:
                e_inp.append(r_["hole"]["error_mm"]); cand.append((S_, dict(r_["hole"], sid=r_["sid"], piece=r_["piece"])))
        ins.append(TUBE / f"h9hole_{S_}_native.json")
    m_inp = TJ["pooled"][TUBE_VARIANT]["median_mm"]
    assert len(e_inp) == TJ["pooled"][TUBE_VARIANT]["n"] and round(float(np.median(e_inp)), 3) == m_inp, "distribution != JSON"
    # ---- b: fixed quantiles of the offsets ----
    qv = np.percentile(e_inp, QUANT)
    pins, n = CFG.get("panel_b_pieces") or {}, len(e_inp)
    rank = np.argsort(np.argsort(e_inp, kind="stable"), kind="stable")
    pick = []
    for q, qq in zip(qv, QUANT):
        if pins.get(qq):
            hit = [i for i, c in enumerate(cand) if c[1]["piece"] == pins[qq]]
            assert len(hit) == 1, f"panel_b_pieces {qq}: {pins[qq]} is not a found piece"
            assert abs(rank[hit[0]] - qq / 100 * (n - 1)) <= PIN_RANKS, f"panel_b_pieces {qq}: {pins[qq]} lies beyond {PIN_RANKS} ranks"
            pick.append(hit[0])
        else:
            pick.append(min(range(n), key=lambda i: abs(cand[i][1]["error_mm"] - q)))
    ex = [cand[i] for i in pick]
    # ---- b's blockface: each piece's tube center on its photograph, recomputed and checked against the record ----
    h9 = {S_: {r_["piece"]: r_ for r_ in pieces("h9hole", S_).values()} for S_ in {S_ for S_, _ in ex}}
    bfv = []
    for S_, r in ex:
        rec = h9[S_][r["piece"]]; idx = rec["bf_slice"]
        g = bf_photo(S_, idx); tc = tube_centers(S_, idx, g)
        hit = [t for t in tc if t[0] == rec["mask_px"] and abs(np.hypot(*np.subtract(t[2], t[1])) - rec["center_shift_fixed_px"]) <= SHIFT_TOL_PX]
        assert len(hit) == 1, f"{r['piece']}: the blockface tube center does not reproduce the record ({tc})"
        bfv.append(dict(g=g, c=hit[0][2], fmm=fix_mm(S_), shift=round(float(np.hypot(*np.subtract(hit[0][2], hit[0][1]))), 2),
                        shift_record=rec["center_shift_fixed_px"], idx=idx))
        ins += [bf_data(S_) / f"{S_}_blockface_L2.nii.gz", bf_data(S_) / f"{S_}_holes.nii.gz", bf_data(S_) / "steps" / f"{S_}_03_color.ome.zarr"]
    # ---- figure: 2 rows x 4 columns (the author, 12 Nov 2026) ----
    import glymur
    S.setup()
    W = S.FULL_W_IN * 25.4
    h_px, w_px = pa[0].shape[:2]
    GUT, MG, pt = S.GUTTER_MM, S.MARGIN_MM, 25.4 / 72                # space (the author, 12 Nov 2026): one gutter, the page's margin
    T = text_mm(plt); pad = 2.0 * pt
    xa = MG + S.orient_room("L")                                  # (a)'s column: the L left of it, the R right of it (figstyle)
    ga = S.orient_room("R") + GUT                                 # (a) to (b): the R, then one gutter
    bw = (W - xa - ga - 2 * GUT - MG) / (w_px / h_px + 3)          # a row's height: (b)'s square windows = (a)'s images
    wa, ph = bw * w_px / h_px, bw
    scale_a = wa / (w_px * fmm_a)                                  # (a): print mm per tissue mm
    xb = xa + wa + ga
    heads = ["Blockface photograph", "Registered section outlined", "Registered histology"]   # (a): title, sub-label, bottom label
    TOP = MG + L1_PT * pt + max(T.ink("b", S.LETTER_PT, fontweight="bold"), T.ink("Tubes"),     # the title line's ink, or (b)'s bar
                                (2.0 + 1.2) * pt + T.ink(f"{SCALEBARS_MM[1]:g} mm"))            # length above it (scalebar_title: 2 pt over a 1.2 pt bar)
    Y2 = TOP + ph + GUT + L2_PT * pt + T.ink("Registered histology")                        # the bottom row, its labels a gutter down
    PANELS = CFG.get("panels") or ["a", "b", "c", "d"]
    assert PANELS in (["a", "b"], ["a", "b", "c", "d"]), f"histology panels {PANELS}: [a, b] or all four"
    cd = "c" in PANELS
    CH = 30.0
    H = Y2 + ph + (12.0 + CH + 12.0 if cd else MG)
    fig = plt.figure(figsize=(S.FULL_W_IN, H / 25.4))
    halo = [pe.withStroke(linewidth=2.2, foreground="black")]
    # (a): the photograph on top, the registered histology below; an arrowhead left of each tube, its tip ARROW_GAP_MM short of the hole
    a_marks = []
    for t_ in a_tubes:
        r_, c_ = t_[2][0] - cr[0].start, (cr[1].stop - cr[1].start) - 1 - (t_[2][1] - cr[1].start)
        if 0 <= r_ < h_px and 0 <= c_ < w_px:
            a_marks.append((r_, c_, np.sqrt(t_[4] / np.pi) * fmm_a))     # (row, column, the dark hole's radius mm) in the panel
    ax_ab = []
    for k, img in enumerate(pa):
        ax = S.axes_mm(fig, xa, TOP if k == 0 else Y2, wa, ph); ax_ab.append(ax)
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        if k == 0:
            S.outline(ax, outl, lw=0.6)
        ax.set_xticks([]); ax.set_yticks([]); [s_.set_visible(False) for s_ in ax.spines.values()]
        for r_, c_, rad in a_marks:
            dx_px = (rad + ARROW_GAP_MM / scale_a + ARROW_PT / 2 * pt / scale_a) / fmm_a
            ax.plot([c_ - dx_px], [r_], marker=">", ms=ARROW_PT, mew=0, ls="none", color=S.COL["pred"], zorder=5)
        if k == 0:
            S.letter(ax, "a", dx=(MG - xa) / pt, dy=L1_PT); ta = S.header(ax, heads[0], dx=(MG - xa) / pt + 11, dy=L1_PT); ax_a = ax
            S.header(ax, heads[1], dx=0, dy=L2_PT)                # on (b)'s two lines: its title's, its sub-labels'
        else:
            S.header(ax, heads[2], dx=0, dy=L2_PT)
    S.orient_letters(fig, [(ax_, ax_, "coronal") for ax_ in ax_ab])   # L and R beside each of (a)'s images, each at its tissue's middle
    # (b): per quantile, the blockface window on top, the histology window below; one window of WIN_MM tissue
    half = int(WIN_MM / 2 / PX_MM); f = JP2_LEVEL; hm = WIN_MM / 2
    for k, ((S_, r), q, qq, v) in enumerate(zip(ex, qv, QUANT, bfv)):
        x_k = xb + k * (bw + GUT)
        # top: the blockface, in tissue mm about its tube center (+), flipped left-right as (a)
        ax = S.axes_mm(fig, x_k, TOP, bw, bw)
        n_ = int(np.ceil(hm / v["fmm"])) + 2; rc, cc = (int(round(c)) for c in v["c"])
        r0, r1, c0, c1 = rc - n_, rc + n_ + 1, cc - n_, cc + n_ + 1
        crop = v["g"][max(r0, 0):r1, max(c0, 0):c1]; r0, c0 = max(r0, 0), max(c0, 0); r1, c1 = r0 + crop.shape[0], c0 + crop.shape[1]
        ext = (-(c1 - 0.5 - v["c"][1]) * v["fmm"], -(c0 - 0.5 - v["c"][1]) * v["fmm"], (r1 - 0.5 - v["c"][0]) * v["fmm"], (r0 - 0.5 - v["c"][0]) * v["fmm"])
        ax.imshow(crop[:, ::-1], cmap="gray", vmin=0, vmax=1, extent=ext, interpolation="nearest")   # the registration's contrast, as (a)
        ax.set_xlim(-hm, hm); ax.set_ylim(hm, -hm)
        ax.plot(0, 0, "+", color=S.ON_IMG, ms=10, mew=1.4, path_effects=halo)
        ax.set_xticks([]); ax.set_yticks([]); [s_.set_visible(False) for s_ in ax.spines.values()]
        S.header(ax, f"{S.label(S_)}, {qq}th percentile", dx=0, dy=L2_PT)
        if k == 0:
            S.letter(ax, "b", dx=0, dy=L1_PT); tb = S.header(ax, "Tubes hidden from the registration", dx=11, dy=L1_PT)
        if k == len(QUANT) - 1:                                   # zoomed: the bar in its standard place, on (b)'s title line
            S.scalebar_title(ax, 1.0, SCALEBARS_MM[1], tb)        # (tissue-mm axes)
        # bottom: the histology about the carried center (+), o = the hole found
        jp = Path(a.jp2_root) / JP2.format(S=S_, sid=r["sid"])
        j2k = glymur.Jp2k(str(jp)); ins.append(jp)
        y0, x0_ = r["bf_tube_l0"]; hy, hx = r["hole_l0"]
        r0, c0 = int(max(0, y0 - half)), int(max(0, x0_ - half))
        w = j2k[r0:int(y0 + half):f, c0:int(x0_ + half):f][:, :, :3].astype(np.float32)
        m = w.max(-1) > 0
        ax = S.axes_mm(fig, x_k, Y2, bw, bw)
        ax.imshow((stretch(w, m) if m.any() else w / 255)[:, ::-1])          # same left-right flip as (a)
        wd = w.shape[1]
        ax.plot(wd - 1 - (x0_ - c0) / f, (y0 - r0) / f, "+", color=S.ON_IMG, ms=10, mew=1.4, path_effects=halo)
        rad = np.sqrt(max(r.get("hole_area_mm2", 0.0), 1e-6) / np.pi) / (PX_MM * f)     # the detected hole: its recorded area
        ax.add_patch(Circle((wd - 1 - (hx - c0) / f, (hy - r0) / f), rad, fill=False, ec=S.ON_IMG, lw=1.1, path_effects=halo))
        ax.set_xticks([]); ax.set_yticks([]); [s_.set_visible(False) for s_ in ax.spines.values()]
        S.header(ax, f"Offset {r['error_mm']:.2f} mm", dx=0, dy=L2_PT)
    S.scalebar_title(ax_a, 1 / fmm_a, SCALEBARS_MM[0], ta)         # (a)'s bar as (b)'s, on its title line, top right of its column
    ps = TJ["per_subject"]
    if cd:
        draw_cd(fig, S, e_inp, ps, MG, Y2 + ph + 12.0, W - 2 * MG, CH)
    S.save(fig, str(a.out), ins + [TJP], cmd=" ".join(["python"] + sys.argv),
           extra=dict(panel_a_piece=ra["tag"], panel_a_dice=ra["metrics"]["dice_nonlinear"], panel_a_subject_median_dice=med,
                      panel_b=[dict(subject=S_, piece=r["piece"], quantile=qq, quantile_mm=round(float(q), 3), offset_mm=r["error_mm"],
                                    rank_from_quantile=round(float(rank[i] - qq / 100 * (n - 1)), 1), named=bool(pins.get(qq)))
                               for (S_, r), i, q, qq in zip(ex, pick, qv, QUANT)],
                      per_brain_median_mm={S_: ps[S_][TUBE_VARIANT]["median_mm"] for S_ in MAIN},
                      panels=PANELS, median_tubes_hidden_mm=m_inp, n_pieces=len(e_inp), scale_a=round(scale_a, 3), scalebars_mm=list(SCALEBARS_MM),
                      layout="2 rows x 4 columns: (a) the first column, (b) one column per quantile (the author, 12 Nov 2026)",
                      fixed_px_mm={A_SUBJ: fmm_a, **{S_: v["fmm"] for (S_, _), v in zip(ex, bfv)}},
                      panel_a_tubes_fixed_px=[[round(float(t[2][0]), 1), round(float(t[2][1]), 1)] for t in a_tubes], panel_a_slice=a_idx,
                      panel_a_tube_components_on_slice=len(a_all),
                      panel_b_blockface=[dict(piece=r["piece"], slice=v["idx"], center_fixed_px=[round(float(c), 1) for c in v["c"]],
                                              center_shift_px=v["shift"], center_shift_px_record=v["shift_record"])
                                         for (S_, r), v in zip(ex, bfv)]))
    plt.close(fig)
    print("->", a.out, "a:", ra["tag"], "b:", [(S_, r["piece"], r["error_mm"]) for S_, r in ex])


if __name__ == "__main__":
    main()
