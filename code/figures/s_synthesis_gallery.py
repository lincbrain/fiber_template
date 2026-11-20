#!/usr/bin/env python3
"""Supplementary Section S1, "The blockface pipeline": the completed blockface volume of every one of the 21 sectioned brains in one axial
plane of the template frame
(the author's choice, 11 Nov 2026), in this paper's house style, with this paper's names.

    python code/figures/s_synthesis_gallery.py [--out figures/s_synthesis_gallery.pdf]

Per brain, from the completed volumes of record (products `completed`) and the prepared folders (`completion_inputs`: canon_d3 for the
13 brains with MRI, canon_d3n_prod for the 8 without), on the brain's canonical fine grid (the run-113 world, its rigid placement; RAS,
0.25 mm, one lattice for all 21), with the display parameters of settings/figures.yaml (synthesis_gallery):
  plane    ONE world axial plane for all 21 (settings z_mm, checked against $LINC_NUMBERS/gallery_plane_2026-11.json): among the
           planes with at least 0.8 of the largest median synthesized area, the fewest mask gaps, one-sided missing pieces and
           cerebellum at the junction over the 21 tiles (code/figures/paper_numbers/gallery_plane_2026-11.py); anterior up, the
           subject's left on the left. Without z_mm: z*, the plane with the cohort's largest median synthesized area
  crop     from the brain's caudal end to 22 mm rostral of the junction line's median in that plane, the rows aligned on it
  junction dashed where the product's seam_depth = 0 (caudal of it the tissue is synthesized), oblique in this frame
  scale    one physical scale for all brains (the widest brain fills its tile); one 10 mm bar, its length above it, in the first
           tile's empty bottom-right corner (it adds no room: the author, 12 Nov 2026)
  color    photographed tissue in its own color, cross-faded into the synthesis over 2 mm (the product's seam_ramp_real); synthesized
           tissue colored with the brain's own luminance -> color lookup fitted on its photographs (code/blockface/synthesis/views.py)
  header   name, and the synthesized share of the brain ($LINC_NUMBERS/d3_predicted_share.json; gray: the 8 brains synthesized from
           the stand-in b0, canon_d3n_prod)
  letters  L and R beside every row (figstyle.orient_letters: 1 mm from the brains, at the middle of the row; the author, 12 Nov
           2026), none above or below: the caption states anterior up
  gaps     the brain mask's gaps (holes) are not filled; counted per brain in the provenance
Writes the PDF and a PNG beside it.
The earlier versions, as options:
  --frame flat --view axial --no-letters   the 30 Oct 2026 figure (the synthesis network's paper's Fig. 2, Gonzales et al. 2026, ISBI,
                        extended to all 21; ported from the analysis project's paper/scripts/figS_synthesis_gallery.py): on the brain's
                        section-aligned grid (blockface.synthesis.views.Flat: every plane one photographed section, the junction one
                        plane), the axial plane with the most synthesized tissue caudal of the junction, from the brain's caudal end
                        to 22 mm rostral of it, the junctions aligned on one line per row; anterior up, the subject's left on the left
  --plane fewest_gaps   (with --frame flat) the 6 Nov 2026 rule: among the axial planes with at least 0.9 of the most synthesized area,
                        the one with the fewest gaps of the brain mask in the shown part rostral of the junction
  --view sagittal       (with --frame canonical) the whole sagittal plane at x = --x-mm (4.5 mm, as Fig. 2), uncropped, the same
                        world plane for all 21; letters P/A beside the first row
  --ncol, --letters     tiles per row; orientation letters beside every row (axial L/R, sagittal P/A), in side gutters as wide as
                        the page margin and the letters need
  --z-mm, --planes-json, --plane-label   (canonical axial) one given world plane for all 21, or one plane per brain from a JSON
                        {subject: z mm}; each tile's plane height on its header line
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO / "code"))
import figstyle as FS  # noqa: E402
from shared import paths  # noqa: E402
from blockface.synthesis.data import y_dep_volume  # noqa: E402
from blockface.synthesis.views import Flat, brain_colors, on_black  # noqa: E402

ROSTRAL_MM = 22.0
VOX = 0.25
FRAC = 0.9                                                               # --plane fewest_gaps: candidate planes' share of the most synthesized area
ROOTS = ["canon_d3", "canon_d3n_prod"]                                 # prepared brains: 13 with MRI, 8 without (stand-in b0)


def load(f):
    return np.asanyarray(nib.load(f).dataobj).astype(np.float32)


def holes(B):
    return ndi.binary_fill_holes(B) & ~B


def tile(S, root, OUT, plane="record"):
    """one brain's tile, on its own prepared folder (its flat grid)"""
    F = Flat(root / S)
    D, ob = root / S / "fine", OUT / S
    ps = dict(brain=f"{D}/brain.nii.gz", real=f"{D}/real_mask.nii.gz", g_real=f"{D}/bf.nii.gz", rgb=f"{D}/bf_rgb.nii.gz",
              comp=f"{ob}/{S}_bf_completed.nii.gz", w_real=f"{ob}/{S}_seam_ramp_real.nii.gz")
    for k in ("comp", "w_real"):
        im = nib.load(ps[k])
        assert tuple(im.shape[:3]) == tuple(F.rec["fine_shape"]) and np.allclose(im.affine, F.Afi, atol=1e-5), f"{S}: {k} off grid"
    brain, real = load(ps["brain"]) > 0, load(ps["real"]) > 0
    g_real, rgb = load(ps["g_real"]), load(ps["rgb"])
    zf = F.z_f
    ydep = y_dep_volume(F.rec, brain.shape)
    kept = real & brain & (np.rint(ydep) >= zf)
    disp, photo, pred = brain_colors(g_real, rgb, kept)
    fb = F.sample(brain.astype(np.float32), (slice(None), slice(None), slice(None)), order=0) > 0.5
    jj = np.arange(F.shape[1])[None, :, None]
    syn = (fb & (jj < zf)).sum((0, 1)); z_rec = int(np.argmax(syn))          # the record's plane: the most synthesized tissue
    z = z_rec
    if plane == "fewest_gaps":                                                 # the 6 Nov rule
        yy = np.arange(F.shape[1]); win = (yy >= zf) & (yy < zf + int(round(ROSTRAL_MM / VOX)))
        g = {int(z_): int(holes(fb[:, :, z_])[:, win].sum()) for z_ in np.nonzero(syn >= FRAC * syn[z_rec])[0]}
        z = min(g, key=lambda z_: (g[z_], abs(z_ - z_rec)))
    pl = (slice(None), slice(None), z)
    B = fb[:, :, z]
    K = F.sample(kept.astype(np.float32), pl, order=0) > 0.5
    w = np.where(K, np.clip(F.sample(load(ps["w_real"]), pl, order=1), 0, 1), 0.0)[..., None]
    col = w * photo(F.sample(g_real, pl, order=1), F.sample(rgb, pl, order=1)) + (1 - w) * pred(F.sample(load(ps["comp"]), pl, order=1))
    ys = np.where(B.any(0))[0]; xs = np.where(B.any(1))[0]
    y_lo, y_hi = ys.min(), zf + int(round(ROSTRAL_MM / VOX))
    x_lo, x_hi = xs.min(), xs.max() + 1
    sl = (slice(x_lo, x_hi), slice(y_lo, y_hi))
    img = on_black(col[sl], B[sl])
    img = np.transpose(img, (1, 0, 2))[::-1]                        # rows = anterior (top) -> posterior; columns = x (left -> right)
    return dict(img=img, cut_row=(y_hi - zf), z=z, z_record=z_rec, gaps_shown=int(holes(B[sl]).sum()), junction=None,
                inputs=list(ps.values()) + [f"{root}/{S}/seam.json"])


def canon_planes(brains, OUT):
    """the one canonical axial plane z* (world mm on the shared lattice) maximizing the cohort's median synthesized area"""
    area = {}
    for S, root in brains.items():
        im = nib.load(f"{root}/{S}/fine/brain.nii.gz"); A = im.affine
        syn = (np.asanyarray(im.dataobj) > 0) & (load(f"{OUT}/{S}/{S}_seam_depth.nii.gz") < 0)
        a_ = syn.sum((0, 1))
        for k, n in enumerate(a_):
            area.setdefault(round(float(A[2, 3] + A[2, 2] * k), 2), {})[S] = int(n)
    med = {z: float(np.median([d.get(S, 0) for S in brains])) for z, d in area.items()}
    return max(med, key=med.get), med


def tile_canon(S, root, OUT, view, coord):
    """one brain's tile on its canonical fine grid: the axial plane at world z = coord (crop: caudal end to 22 mm rostral of the
    junction line's median) or the sagittal plane at world x = coord (the whole plane)"""
    F = Flat(root / S)
    D, ob = root / S / "fine", OUT / S
    ps = dict(brain=f"{D}/brain.nii.gz", real=f"{D}/real_mask.nii.gz", g_real=f"{D}/bf.nii.gz", rgb=f"{D}/bf_rgb.nii.gz",
              comp=f"{ob}/{S}_bf_completed.nii.gz", w_real=f"{ob}/{S}_seam_ramp_real.nii.gz", depth=f"{ob}/{S}_seam_depth.nii.gz")
    for k in ("comp", "w_real", "depth"):
        im = nib.load(ps[k])
        assert tuple(im.shape[:3]) == tuple(F.rec["fine_shape"]) and np.allclose(im.affine, F.Afi, atol=1e-5), f"{S}: {k} off grid"
    A = F.Afi
    assert nib.aff2axcodes(A) == ("R", "A", "S") and np.allclose(A[:3, :3], np.diag([VOX] * 3)), f"{S}: canonical grid not RAS 0.25 mm"
    brain, real = load(ps["brain"]) > 0, load(ps["real"]) > 0
    g_real, rgb = load(ps["g_real"]), load(ps["rgb"])
    kept = real & brain & (np.rint(y_dep_volume(F.rec, brain.shape)) >= F.z_f)
    disp, photo, pred = brain_colors(g_real, rgb, kept)
    ax_ = 2 if view == "axial" else 0
    k = (coord - A[ax_, 3]) / VOX; assert abs(k - round(k)) < 1e-3, f"{S}: plane {coord} off the lattice"; k = int(round(k))
    take = (lambda v: v[:, :, k]) if view == "axial" else (lambda v: v[k])                       # noqa: E731
    B, K = take(brain), take(kept)
    w = np.where(K, np.clip(take(load(ps["w_real"])), 0, 1), 0.0)[..., None]
    col = w * photo(take(g_real), take(rgb)) + (1 - w) * pred(take(load(ps["comp"])))
    dep = np.where(B, take(load(ps["depth"])), np.nan)
    a0 = np.where(B.any(1))[0]; a1 = np.where(B.any(0))[0]                      # axial: (x, y); sagittal: (y, z)
    if view == "axial":
        jy = np.nonzero(B & (np.abs(dep) <= VOX / 2))[1]
        y_j = int(np.round(np.median(jy))) if len(jy) else int(a1.min())
        sl = (slice(a0.min(), a0.max() + 1), slice(a1.min(), y_j + int(round(ROSTRAL_MM / VOX))))
        cut_row = sl[1].stop - y_j
    else:
        sl = (slice(max(a0.min() - 2, 0), a0.max() + 3), slice(max(a1.min() - 2, 0), a1.max() + 3)); cut_row = None
    img = on_black(col[sl], B[sl])
    T = lambda v: np.transpose(v, (1, 0) + tuple(range(2, v.ndim)))[::-1]       # noqa: E731   rows: anterior / superior first
    mid = float(np.median(A[0, 3] + VOX * np.nonzero(brain)[0]))
    return dict(img=T(img), junction=T(dep[sl]), cut_row=cut_row, z=k, gaps_shown=int(holes(B[sl]).sum()), midline_x_mm=round(mid, 2),
                inputs=list(ps.values()) + [f"{root}/{S}/seam.json"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    import yaml
    cfg = yaml.safe_load((REPO / "settings" / "figures.yaml").read_text())["synthesis_gallery"]
    ap.add_argument("--out", default="figures/s_synthesis_gallery.pdf")
    ap.add_argument("--plane", choices=["record", "fewest_gaps"], default=cfg["plane"])
    ap.add_argument("--frame", choices=["flat", "canonical"], default=cfg["frame"])
    ap.add_argument("--view", choices=["axial", "sagittal"], default=cfg["view"])
    ap.add_argument("--x-mm", type=float, default=float(cfg["x_mm"]))
    ap.add_argument("--ncol", type=int, default=int(cfg["ncol"]))
    ap.add_argument("--letters", action=argparse.BooleanOptionalAction, default=bool(cfg["letters"]))
    ap.add_argument("--z-mm", type=float, default=cfg.get("z_mm"), help="canonical axial: this world plane for all 21 (none: z*)")
    ap.add_argument("--planes-json", default=None, help="canonical axial: {subject: world z mm} per brain")
    ap.add_argument("--plane-label", action="store_true", help="each tile's plane height in its lower-right corner")
    a = ap.parse_args()
    assert a.frame == "canonical" or a.view == "axial", "the sagittal view is drawn on the canonical grid"
    assert a.frame == "flat" or a.plane == "record", "the canonical frame takes one plane for all brains"
    sag = a.view == "sagittal"
    P = paths.products("products_completion")
    OUT = P["completed"]
    FS.setup()
    lp = REPO / "settings" / "specimen_labels.json"; lab = json.load(open(lp))["labels"]
    sp = paths.numbers_root() / "d3_predicted_share.json"; share = json.load(open(sp))["data"]["per_brain"]
    brains = {}
    for r in ROOTS:
        root = P["completion_inputs"] / r
        for S in json.load(open(root / "canon_root.json"))["brains"]:
            brains[S] = root
    order = sorted(brains, key=lambda s: (lab[s][:2], int(lab[s][2:])))
    assert len(order) == 21
    T, extra, extra_inputs = {}, {}, []
    if a.frame == "canonical":
        coord = a.x_mm if sag else (a.z_mm if a.z_mm is not None else canon_planes({S: brains[S] for S in order}, OUT)[0])
        rec = paths.numbers_root() / "gallery_plane_2026-11.json"          # the plane's rule: the setting must be its plane
        if not sag and a.z_mm is not None and a.z_mm == cfg.get("z_mm") and not a.planes_json:
            z_rec = (json.load(open(rec)).get("data") or {})["plane_z_mm"]
            assert abs(z_rec - a.z_mm) < 1e-6, f"settings z_mm {a.z_mm} is not the plane of {rec.name} ({z_rec})"
            extra_inputs.append(str(rec))
        per = json.load(open(a.planes_json)) if a.planes_json else {}
        coords = {S: float(per.get(S, coord)) for S in order}
        extra = dict(frame="canonical (run-113 world)", view=a.view, plane_world_mm=coords if per else coord)
    for S in order:
        T[S] = tile(S, brains[S], OUT, a.plane) if a.frame == "flat" else tile_canon(S, brains[S], OUT, a.view, coords[S])
        print(lab[S], S, "plane", T[S]["z"], "gaps", T[S]["gaps_shown"], flush=True)
    ncol, gap, hdr = a.ncol, 1.2, 4.0                                  # mm
    G = FS.MARGIN_MM + FS.orient_room("sagittal" if sag else "axial") if a.letters else 0.0    # side gutters for the letters
    nrow = int(np.ceil(len(order) / ncol))
    tw = (FS.FULL_W_IN * 25.4 - 2 * G - gap * (ncol - 1)) / ncol
    mm_per_vox = tw / max(t["img"].shape[1] for t in T.values())
    if sag:                                                            # whole planes, centered in their cells
        above = below = max(t["img"].shape[0] for t in T.values()) * mm_per_vox / 2
    else:
        above = max(t["cut_row"] for t in T.values()) * mm_per_vox
        below = max(t["img"].shape[0] - t["cut_row"] for t in T.values()) * mm_per_vox
    th = above + below
    sb_h = 0.0                                                         # no room for the scale bar: it sits in a tile's empty corner
    H = nrow * (hdr + th) + (nrow - 1) * gap + sb_h
    fig = plt.figure(figsize=(FS.FULL_W_IN, H / 25.4))
    AXS, TILE_IMG = [], []
    for k, S in enumerate(order):
        r, c = divmod(k, ncol)
        t = T[S]; hh, ww = t["img"].shape[:2]
        x0 = G + c * (tw + gap); y0 = r * (hdr + th + gap) + hdr + (sb_h if r > 0 else 0)
        ax = FS.axes_mm(fig, x0, y0, tw, th)
        ax.set_xlim(0, tw); ax.set_ylim(-below, above); ax.axis("off")
        w_, h_ = ww * mm_per_vox, hh * mm_per_vox
        xi = (tw - w_) / 2; yi = -h_ / 2 if sag else -(hh - t["cut_row"]) * mm_per_vox
        ax.imshow(t["img"], extent=[xi, xi + w_, yi, yi + h_], interpolation="antialiased", zorder=2)
        if t["junction"] is None:                                      # the flat grid: the junction is one plane, a straight line
            ax.plot([0.3, tw - 0.3], [0, 0], color=FS.FG, lw=0.6, ls=(0, (2.0, 1.3)), zorder=4)
        else:                                                          # the canonical grid: where seam_depth = 0, as it lies
            yy_, xx_ = np.mgrid[0:hh, 0:ww] + 0.5
            ax.contour(xi + xx_ * mm_per_vox, yi + h_ - yy_ * mm_per_vox, t["junction"], levels=[0.0], colors=[FS.FG], linewidths=0.6,
                       linestyles=[(0, (2.0, 1.3))], zorder=4)
        AXS.append(ax); TILE_IMG.append((t["img"], xi, yi, w_, h_))
        nomri = brains[S].name == "canon_d3n_prod"
        FS.header(ax, lab[S], dx=0, fontweight="bold")
        pc = ax.annotate(f"{100 * share[S]['predicted_share']:.0f}%", xy=(1, 1), xycoords="axes fraction", xytext=(0, 3),
                         textcoords="offset points", ha="right", va="bottom", fontsize=FS.MIN_PT,
                         color=FS.GRAY if nomri else FS.FG, annotation_clip=False)
        if a.plane_label and a.frame == "canonical":                   # the tile's plane height, on the header line left of the share
            e_ = pc.get_window_extent(fig.canvas.get_renderer()); w_pc = e_.width / fig.dpi * 72.0
            ax.annotate(f"{'x' if sag else 'z'} {coords[S]:.2f}", xy=(1, 1), xycoords="axes fraction", xytext=(-w_pc - 4.0, 3),
                        textcoords="offset points", ha="right", va="bottom", fontsize=FS.MIN_PT, color=FS.GRAY, annotation_clip=False)
    # the 10 mm bar in an empty bottom corner of a tile (no room added: the author, 12 Nov 2026): the first tile, in reading order,
    # whose bottom-right (else bottom-left) corner holds no tissue under the bar and its length (7.0 x 3.4 mm, inset 0.8 mm)
    def corner_free(img, xi, yi, w_, h_, corner):
        x0, x1 = (tw - 0.8 - 7.0, tw - 0.8) if corner == "bottom right" else (0.8, 0.8 + 7.0)
        y0, y1 = -below + 0.8, -below + 0.8 + 3.4
        c0, c1 = int(np.floor((x0 - xi) / mm_per_vox)), int(np.ceil((x1 - xi) / mm_per_vox))
        r0, r1 = int(np.floor((yi + h_ - y1) / mm_per_vox)), int(np.ceil((yi + h_ - y0) / mm_per_vox))
        c0, c1, r0, r1 = max(c0, 0), min(c1, img.shape[1]), max(r0, 0), min(r1, img.shape[0])
        return c1 <= c0 or r1 <= r0 or not (np.asarray(img[r0:r1, c0:c1], float).reshape(r1 - r0, c1 - c0, -1).max(-1) > 0.03).any()
    bar_at = next(((k, c) for k, ti in enumerate(TILE_IMG) for c in ("bottom right", "bottom left") if corner_free(*ti, c)), None)
    assert bar_at is not None, "no tile has an empty bottom corner for the scale bar"
    FS.scalebar_in(AXS[bar_at[0]], mm_per_vox / VOX, 10.0, corner=bar_at[1], lw=1.0)
    extra["scale_bar_at"] = dict(tile=lab[order[bar_at[0]]], corner=bar_at[1])
    if a.letters:                                                      # beside every row, on one line at its brains' middle
        FS.orient_letters(fig, [(AXS[r * ncol], AXS[min((r + 1) * ncol, len(AXS)) - 1], a.view) for r in range(nrow)])
    ins = sorted({str(p) for t in T.values() for p in t["inputs"]} | {str(lp), str(sp)} | set(extra_inputs))
    if a.plane != "record" or a.frame != "flat" or a.letters or a.ncol != 7:
        extra.update(options=dict(plane=a.plane, frame=a.frame, view=a.view, ncol=a.ncol, letters=a.letters),
                     gaps_shown={lab[S]: T[S]["gaps_shown"] for S in order})
        if a.frame == "flat":
            extra["planes_record"] = {lab[S]: T[S]["z_record"] for S in order}
        else:
            extra["midline_x_mm"] = {lab[S]: T[S]["midline_x_mm"] for S in order}
    cmd = " ".join(["code/figures/s_synthesis_gallery.py"] + sys.argv[1:])
    FS.save(fig, a.out, ins, cmd, extra=dict(planes={lab[S]: T[S]["z"] for S in order}, scale_bar_mm=10, rostral_mm=ROSTRAL_MM, **extra))
    fig.savefig(str(Path(a.out).with_suffix(".png")), dpi=FS.DPI, facecolor=fig.get_facecolor())
    print("->", a.out, f"{FS.FULL_W_IN * 25.4:.0f} x {H:.1f} mm")


if __name__ == "__main__":
    main()
