#!/usr/bin/env python3
"""The paper's figure "Building the template and aligning histology to it": the pipeline at a glance.

    python reproduce/figures/pipeline.py [--out figures/pipeline.png]

Three stages, read top to bottom. (1) Two constructions of one style: ex vivo MRI -> the MRI template; blockface -> the blockface
template. (2) Their registration (vertical arrow). (3) Histology: a section registered to its blockface photograph, then transformed
to the template by the blockface-based alignment (cyan, through the blockface template) or the MRI-based alignment (brown, through
its own MRI).

Inputs: the products listed in configs/products_figures.yaml (under $LINC_PRODUCTS) and two counts from the numbers records
($LINC_NUMBERS: bf_cohort, histo_tube_landmarks). Display parameters: configs/figures.yaml (pipeline).

Planes (the sagittal plane of the template figure, x = 2.0 mm):
- template-space thumbnails (the MRI channels on the template grid) at x = 2.0 mm;
- the blockface template in its own space, at the point (2.0, 19.5, 15.5) carried by the registration's pull map (as in the blockface template figure);
- the MRI input brain (MF1) at x = 2.0 mm of its canonical frame (its rigid placement on the template grid);
- the blockface volume: MM2's completed volume (photographs + synthesized tissue), the construction's input, at x = 2.0 mm of its
  rigid placement (whose x is the template's x); synthesized voxels as orange-tinted brightness (the network outputs brightness).
One print scale for every image (the largest that fits the 178 mm width); a 10 mm scale bar. Writes the PNG, a PDF beside it, and
provenance.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import figstyle as FS                                          # noqa: E402
from linc.paths import numbers_root, products                 # noqa: E402
from linc.registration import xfmtools as X                    # noqa: E402


# ---- small helpers ---------------------------------------------------------------------------------------------------------
def crop_to(a, thr=0.0, pad=4):
    """Crop an image (2D, or 2D x channels) to the pixels above `thr`, plus `pad` pixels."""
    m = (a.max(-1) if a.ndim == 3 else a) > thr
    r, c = np.nonzero(m)
    return a[max(r.min() - pad, 0):r.max() + pad + 1, max(c.min() - pad, 0):c.max() + pad + 1]


def soft(img, sigma, round_px=0.0):
    """Display only: the tissue edge antialiased on black (the RGB extended outward by its nearest tissue pixel, times the
    support rounded at `round_px` and smoothed with a Gaussian of `sigma` pixels)."""
    from scipy import ndimage as ndi
    m = img.sum(-1) > 0
    if not m.any() or m.all():
        return img
    idx = ndi.distance_transform_edt(~m, return_distances=False, return_indices=True)
    if round_px > 0:
        m = ndi.gaussian_filter(m.astype(np.float32), round_px) > 0.5
    return img[tuple(idx)] * np.clip(ndi.gaussian_filter(m.astype(np.float32), sigma), 0, 1)[..., None]


def read_level(path, level):
    """One pyramid level of a blockface OME-Zarr store as (array (x, sections, z, c), RAS affine in mm)."""
    import zarr
    g = zarr.open_group(str(path), mode="r")
    a = np.asarray(g[str(level)])                              # (c, rows, slices, cols)
    ct = g.attrs["multiscales"][0]["datasets"][level]["coordinateTransformations"]
    tr, sc = ct[0]["translation"], ct[1]["scale"]
    arr = np.transpose(a, (3, 2, 1, 0)) if a.shape[0] > 1 else np.transpose(a[0], (2, 1, 0))
    A = np.eye(4)
    A[0, 0], A[1, 1], A[2, 2] = sc[3] / 1000.0, sc[2] / 1000.0, sc[1] / 1000.0
    A[0, 3], A[1, 3], A[2, 3] = sc[3] * tr[3] / 1000.0, sc[2] * tr[2] / 1000.0, sc[1] * tr[1] / 1000.0
    return arr, A


def extent_coronal(sl, A):
    dx, dz = A[0, 0], A[2, 2]; nx, nz = sl.shape[0], sl.shape[1]
    return (A[0, 3] - dx / 2, A[0, 3] + (nx - 0.5) * dx, A[2, 3] - dz / 2, A[2, 3] + (nz - 0.5) * dz)


def bbox_mm(img2d, ext):
    rr = np.where((img2d > 0).any(1))[0]; cc = np.where((img2d > 0).any(0))[0]
    H, W = img2d.shape; hx = (ext[1] - ext[0]) / W; vz = (ext[3] - ext[2]) / H
    h0, h1 = ext[0] + cc[0] * hx, ext[0] + (cc[-1] + 1) * hx; v1, v0 = ext[3] - rr[0] * vz, ext[3] - (rr[-1] + 1) * vz
    return (h0 + h1) / 2, (v0 + v1) / 2, abs(h1 - h0), abs(v1 - v0)


def _resized_crop(img, m):
    from PIL import Image
    ii = np.where(m.any(1))[0]; jj = np.where(m.any(0))[0]
    x = (img * m)[ii[0]:ii[-1] + 1, jj[0]:jj[-1] + 1]
    return np.asarray(Image.fromarray(x.astype(np.float32)).resize((160, 160), Image.BILINEAR))


def photo_mirrored(photo, mask2d, section_disp):
    """True if the photograph matches the neurologically displayed reconstructed section better after a left-right flip."""
    a = _resized_crop(photo, mask2d); b = _resized_crop(section_disp, section_disp > 0)
    r_same = float(np.corrcoef(a.ravel(), b.ravel())[0, 1]); r_flip = float(np.corrcoef(a[:, ::-1].ravel(), b.ravel())[0, 1])
    assert abs(r_flip - r_same) >= 0.05, f"photo side undecided (r as is {r_same:.3f}, mirrored {r_flip:.3f})"
    return r_flip > r_same


def histo_side(sl, vol, step=4):
    """True if the histology's blockface slice must be mirrored to match the volume's coronal sections shown neurologically:
    the best correlation over every `step`-th section, as is and mirrored; asserts a clear margin."""
    def crop(a):
        m = a > 0.02 * a.max()
        return _resized_crop(a, m)
    s0 = crop(sl); best = [-1.0, -1.0]
    for k in range(0, vol.shape[1], step):
        sec = np.rot90(vol[:, k].astype(np.float32).sum(-1))
        if (sec > 0).sum() < 0.2 * sec.size * 0.1:
            continue
        v = crop(sec)
        for n, q in enumerate((s0, s0[:, ::-1])):
            best[n] = max(best[n], float(np.corrcoef(q.ravel(), v.ravel())[0, 1]))
    assert abs(best[1] - best[0]) >= 0.03, f"histology side undecided {best}"
    return best[1] > best[0]


def count(name, key):
    d = json.loads((numbers_root() / f"{name}.json").read_text()); d = d.get("data", d)
    return str(len(d[key]))


# ---- the figure -------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figures/pipeline.png")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    FS.setup()
    import matplotlib.pyplot as plt
    import nibabel as nib
    import yaml
    from matplotlib.colors import to_rgb
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    from matplotlib.path import Path as MPath
    from PIL import Image
    from scipy.ndimage import binary_erosion, binary_fill_holes

    P = products("products_figures")
    cfg = yaml.safe_load((ROOT / "configs" / "figures.yaml").read_text())["pipeline"]
    XS = float(cfg["sagittal_x_mm"])
    W_LUM = np.asarray(cfg["luminance_weights"], np.float32)
    N = {"n_bf": count("bf_cohort", "rows"), "n_histo": count("histo_tube_landmarks", "per_subject")}
    FG, GRAY, BFA, MRIA = FS.FG, FS.GRAY, FS.COL["direct"], FS.COL["mri"]
    SYNC = FS.COL["pred"]                                       # synthesized tissue = orange (house style; the blockface template figure, c)

    def sagx(path, x):
        """The sagittal section at world x (mm) of a RAS volume on a diagonal grid: rows superior -> inferior, columns
        posterior -> anterior (anterior right); returns (array, row mm, column mm, the x actually sampled)."""
        im = nib.load(str(path)); A = im.affine
        assert nib.aff2axcodes(A) == ("R", "A", "S") and np.allclose(A[:3, :3], np.diag(np.diag(A[:3, :3]))), path
        i = int(round((x - A[0, 3]) / A[0, 0]))
        d = np.asarray(im.dataobj[i], np.float32)
        return np.swapaxes(d, 0, 1)[::-1].copy(), abs(A[2, 2]), abs(A[1, 1]), float(A[0, 3] + A[0, 0] * i)

    def norm_in(a, m):
        a = a / np.percentile(a[m], 99.5); a[~m] = 0
        return np.clip(a, 0, 1)

    def fracx(path, mpath, x, inner=False):
        f, dr, dc, _ = sagx(path, x); m = sagx(mpath, x)[0] > 0
        if inner:                                               # trim the speckled outer rim only; internal mask holes keep
            m = binary_erosion(binary_fill_holes(m), iterations=2) & m   # their true size
        f = crop_to(np.clip(f, 0, 1) * m * (f > 0), 0.005)
        return np.ma.masked_less_equal(f, 0), dr, dc

    def decx(path, mpath, x):
        t, dr, dc, _ = sagx(path, x); m0 = sagx(mpath, x)[0] > 0; m = binary_erosion(binary_fill_holes(m0), iterations=2) & m0
        D = np.stack([t[..., [0, 1, 2]], t[..., [1, 3, 4]], t[..., [2, 4, 5]]], -2)
        w, v = np.linalg.eigh(D); md = w.mean(-1, keepdims=True)
        fa = np.sqrt(1.5 * ((w - md) ** 2).sum(-1) / np.maximum((w ** 2).sum(-1), 1e-20))
        return crop_to(np.clip(np.abs(v[..., :, -1]) * np.clip(fa, 0, 1)[..., None] * 1.6, 0, 1) * m[..., None], 0.02), dr, dc

    # ---- images ------------------------------------------------------------------------------------------------------------
    SPI = P["mri_input"]
    INP = {k: SPI / f"{k}.nii.gz" for k in ("t2", "brain_mask", "tensor", "frac1", "frac2")}
    I = {}
    t2, dr, dc, _ = sagx(INP["t2"], XS); m = sagx(INP["brain_mask"], XS)[0] > 0
    I["i_t2"] = (crop_to(norm_in(t2, m), 0.02), dr, dc, "gray")
    I["i_dec"] = (*decx(INP["tensor"], INP["brain_mask"], XS), None)
    I["i_f1"] = (*fracx(INP["frac1"], INP["brain_mask"], XS, inner=True), "turbo")
    I["i_f2"] = (*fracx(INP["frac2"], INP["brain_mask"], XS, inner=True), "turbo")
    tm, dr, dc, _ = sagx(P["template_t2"], XS); mt = sagx(P["template_mask"], XS)[0] > 0
    I["t_t2"] = (crop_to(norm_in(tm, mt), 0.02), dr, dc, "gray")
    dt, dr, dc, _ = sagx(P["template_dec"], XS)
    I["t_dec"] = (crop_to(np.clip(dt / 255.0 * 1.3, 0, 1), 0.02), dr, dc, None)
    I["t_f1"] = (*fracx(P["template_frac1"], P["template_mask"], XS), "turbo")
    I["t_f2"] = (*fracx(P["template_frac2"], P["template_mask"], XS), "turbo")
    # the template figure's section point carried into the blockface template by the registration's pull map (MRI template -> blockface template)
    xb = float(X.load_map(str(P["bft_pull_map"]), "full")(np.array([cfg["section_point_mm"]], float))[0][0])
    rgb, dr, dc, xb_s = sagx(P["bft_rgb"], xb); mr = sagx(P["bft_mask"], xb)[0] > 0
    I["bft"] = (crop_to(np.clip(rgb, 0, 1) * mr[..., None], 0.02), dr, dc, None)
    # the blockface volume: MM2's completed volume (photographs + synthesized tissue), the construction's input, at x = 2.0 mm of
    # its rigid placement; synthesized voxels shown as orange-tinted brightness (the network outputs brightness only)
    rgb, dr, dc, xv_s = sagx(P["blockface_completed_rgb"], XS); sm = sagx(P["blockface_synth_mask"], XS)[0] > 0
    mv = rgb.sum(-1) > 0
    lum = rgb @ W_LUM; tint = np.clip(lum / np.percentile(lum[mv], 99.5), 0, 1)[..., None] * np.asarray(to_rgb(SYNC))
    vol = np.where((sm & mv)[..., None], tint, np.clip(rgb, 0, 1))
    soft_px = tuple(v * float(cfg["soft_edge_grid_mm"]) / dr for v in cfg["soft_edge_px"])   # the reconstruction figure's edge
    I["bfvol"] = (crop_to(soft(vol, *soft_px), 0.0, pad=2), dr, dc, None)
    n_synth_px = int((sm & mv).sum())
    # one photograph (the reconstruction figure's section), neurological, cropped to its brain box
    fin2, A2 = read_level(P["blockface_store"], int(cfg["blockface_level"]))
    k0 = int(cfg["photo_section"])
    im = np.asarray(Image.open(P["photographs"] / f"SLICE_{k0:04d}_0000.png").convert("RGB"))
    pm = np.asarray(Image.open(P["photo_masks"] / f"SLICE_{k0:04d}.png")) > 0
    sec = np.rot90(fin2[:, k0].astype(np.float32) @ W_LUM)
    mirrored = photo_mirrored(im.astype(np.float32) @ W_LUM, pm, sec)
    _, _, sw, sh = bbox_mm(sec, extent_coronal(fin2[:, k0], A2))
    ii, jj = np.where(pm.any(1))[0], np.where(pm.any(0))[0]
    mm_px = float(np.mean([sw / (jj[-1] - jj[0] + 1), sh / (ii[-1] - ii[0] + 1)]))
    if mirrored:
        im, jj = im[:, ::-1], im.shape[1] - 1 - jj[::-1]
    bx = float(cfg["photo_box"]) / 2
    hw, hh = bx * (jj[-1] - jj[0]), bx * (ii[-1] - ii[0])
    rc, cc = (ii[0] + ii[-1]) / 2, (jj[0] + jj[-1]) / 2
    ph = im[max(0, int(rc - hh)):int(rc + hh), max(0, int(cc - hw)):int(cc + hw)]
    th = Image.fromarray(np.ascontiguousarray(ph)); f_ = max(1, int(np.ceil(max(th.size) / 900))); th = th.reduce(f_)
    I["photo"] = (np.asarray(th, np.float32) / 255.0, mm_px * f_, mm_px * f_, None)
    # one histology section and its blockface photograph (TIRL), neurological
    hi = np.asarray(Image.open(P["histology_piece"] / "moving4_nonlinear_rgb.png").convert("RGB"), np.float32) / 255.0
    sl = np.asarray(Image.open(P["histology_piece"] / "fixed.png").convert("L"), np.float32) / 255.0
    flip = histo_side(sl, fin2)
    if flip:
        hi, sl = hi[:, ::-1], sl[:, ::-1]
    px = float(cfg["histology_mm_px"])
    I["histo"] = (crop_to(np.clip(hi / np.percentile(hi, 99.5), 0, 1), 0.03, pad=6), px, px, None)
    I["slice"] = (crop_to(np.clip(sl / np.percentile(sl[sl > 0], 99.5), 0, 1), 0.03, pad=6), px, px, "gray")

    # ---- the largest print scale that fits ------------------------------------------------------------------------------------
    W, PAD, GAP, MIDW, AGAP = 178.0, 1.8, 2.6, 22.0, 8.0      # box padding, image gap, middle column, arrow gap in a box (mm)
    wd = lambda k, s: I[k][0].shape[1] * I[k][2] * s
    ht = lambda k, s: I[k][0].shape[0] * I[k][1] * s

    def widths(s):
        mri = 2 * max(wd(k, s) for k in ("i_t2", "i_dec", "i_f1", "i_f2")) + GAP + 2 * PAD
        bf = wd("photo", s) + AGAP + wd("bfvol", s) + 2 * PAD
        hist = wd("histo", s) + AGAP + wd("slice", s) + 2 * PAD
        tpl = 2 * max(wd(k, s) for k in ("t_t2", "t_dec", "t_f1", "t_f2")) + GAP + 2 * PAD
        return max(mri, bf, hist), tpl
    s_hi, s_lo = cfg["print_scale_range"]
    for SC in np.arange(s_hi, s_lo, -0.01):
        lw, tw = widths(SC)
        if W - 4.0 - lw - tw >= MIDW:
            break
    SC = float(round(SC, 2)); lw, tw = widths(SC)

    # ---- layout (mm) --------------------------------------------------------------------------------------------------------
    HEAD, LAB, G12, G23 = 6.6, 3.4, 9.0, 4.0                  # box header; label line; the registration's gap; rows 2-3
    LX0, LX1, TX0, TX1 = 2.0, 2.0 + lw, W - 2.0 - tw, W - 2.0
    h1 = max(ht(k, SC) for k in ("i_t2", "i_dec", "t_t2", "t_dec")); h2 = max(ht(k, SC) for k in ("i_f1", "i_f2", "t_f1", "t_f2"))
    h3 = max(ht(k, SC) for k in ("photo", "bfvol", "bft")); h4 = max(ht("histo", SC), ht("slice", SC))
    top = 2.0
    R1 = top + HEAD + h1 / 2; R2 = R1 + h1 / 2 + LAB + 0.8 + h2 / 2; my1 = R2 + h2 / 2 + LAB + 0.6
    by0 = my1 + G12; R3 = by0 + HEAD + h3 / 2; by1 = R3 + h3 / 2 + 2 * LAB + 0.6
    hy0 = by1 + G23; R4 = hy0 + HEAD + h4 / 2; hy1 = R4 + h4 / 2 + LAB + 0.6; H = hy1 + 2.0
    wm = max(wd(k, SC) for k in ("i_t2", "i_dec", "i_f1", "i_f2")); wt = max(wd(k, SC) for k in ("t_t2", "t_dec", "t_f1", "t_f2"))
    C1 = LX0 + (lw - (2 * wm + GAP)) / 2 + wm / 2; C2 = C1 + wm + GAP
    T1c = TX0 + PAD + wt / 2; T2c = T1c + wt + GAP
    xp = LX0 + PAD + wd("photo", SC) / 2; xv = LX1 - PAD - wd("bfvol", SC) / 2
    bx0, bx1 = T1c - wd("bft", SC) / 2 - PAD, T1c + wd("bft", SC) / 2 + PAD     # the blockface template's box (under T1c)
    assert bx1 < T2c - 4.0, "the MRI-based alignment's arrow must pass right of the blockface template"

    fig = plt.figure(figsize=(W * FS.MM, H * FS.MM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    B = {}

    def image(k, xc, yc):
        arr, dr_, dc_, cmap = I[k]
        h, w = arr.shape[0] * dr_ * SC, arr.shape[1] * dc_ * SC
        a_ = FS.axes_mm(fig, xc - w / 2, yc - h / 2, w, h)
        a_.imshow(arr, cmap=cmap, aspect="auto", vmin=0, vmax=1, extent=(0, arr.shape[1] * dc_, arr.shape[0] * dr_, 0))
        a_.axis("off"); B[k] = (xc - w / 2, yc - h / 2, w, h)

    def text(x, y, s, **kw):
        kw.setdefault("fontsize", FS.MIN_PT); kw.setdefault("ha", "center"); kw.setdefault("va", "top"); kw.setdefault("color", FG)
        ax.text(x, y, s, linespacing=1.15, **kw)

    def arrow(p, q, col=FG, lw_=1.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, lw=lw_, color=col, shrinkA=0, shrinkB=0))

    def elbow(p, corner, q, col, lw_=1.3):
        """An arrow along two straight segments, p -> corner -> q."""
        ax.add_patch(FancyArrowPatch(path=MPath([p, corner, q], [MPath.MOVETO, MPath.LINETO, MPath.LINETO]), arrowstyle="-|>",
                                     mutation_scale=6, lw=lw_, color=col, shrinkA=0, shrinkB=0, joinstyle="miter"))

    def box(x0, y0, x1, y1, ec=GRAY, lw_=0.6):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.5", fc="none", ec=ec, lw=lw_))

    def header(x0, x1, y, title, sub=None):
        """Title (bold) and subtitle (gray) on one baseline, centered as a pair over the box."""
        t1 = ax.text(0, y + 4.4, title, fontsize=FS.LETTER_PT, weight="bold", ha="left", va="baseline", color=FG)
        t2 = ax.text(0, y + 4.4, sub or "", fontsize=FS.MIN_PT, ha="left", va="baseline", color=GRAY)
        r = fig.canvas.get_renderer()
        w1, w2 = (t.get_window_extent(r).width / fig.dpi / FS.MM for t in (t1, t2))
        x = (x0 + x1) / 2 - (w1 + (2.0 + w2 if sub else 0.0)) / 2
        t1.set_x(x); t2.set_x(x + w1 + 2.0)

    bot = lambda k: B[k][1] + B[k][3]
    left = lambda k: B[k][0]
    rgt = lambda k: B[k][0] + B[k][2]
    cx = lambda k: B[k][0] + B[k][2] / 2

    # (1) two constructions of one style
    for k, x, y, s in (("i_t2", C1, R1, "T2-weighted"), ("i_dec", C2, R1, "tensor (direction)"), ("i_f1", C1, R2, "soma fraction"),
                       ("i_f2", C2, R2, "neurite fraction"), ("t_t2", T1c, R1, "T2-weighted"), ("t_dec", T2c, R1, "principal direction"),
                       ("t_f1", T1c, R2, "soma fraction"), ("t_f2", T2c, R2, "neurite fraction"), ("photo", xp, R3, "photographs"),
                       ("bfvol", xv, R3, "blockface volume")):
        image(k, x, y); text(cx(k), bot(k) + 0.5, s)
    text(cx("bfvol"), bot("bfvol") + 0.5 + LAB, "synthesized tissue", color=SYNC)
    image("bft", T1c, R3)
    box(LX0, top, LX1, my1); header(LX0, LX1, top, "Ex vivo MRI", "4 brains + their mirrors")
    box(LX0, by0, LX1, by1); header(LX0, LX1, by0, "Blockface", f"{N['n_bf']} brains + their mirrors")
    box(TX0, top, TX1, my1, ec=FG, lw_=0.9); header(TX0, TX1, top, "MRI template")
    box(bx0, by0, bx1, by1, ec=FG, lw_=0.9); header(bx0, bx1, by0, "Blockface template")
    bar = float(cfg["scale_bar_mm"])
    ax.plot([bx1 - 2.5 - bar * SC, bx1 - 2.5], [by1 - 1.6, by1 - 1.6], color=FG, lw=1.2, solid_capstyle="butt")
    arrow((rgt("photo") + 1.2, R3), (left("bfvol") - 1.2, R3), lw_=0.8)
    yA = (R1 + R2) / 2
    arrow((LX1 + 1.5, yA), (TX0 - 1.0, yA)); text((LX1 + TX0) / 2, yA - 1.4, "groupwise\nconstruction", va="bottom")
    arrow((LX1 + 1.5, R3), (bx0 - 1.0, R3)); text((LX1 + bx0) / 2, R3 - 1.4, "groupwise\nconstruction", va="bottom")
    # (2) their registration
    arrow((T1c, by0 - 0.4), (T1c, my1 + 0.4)); text(T1c + 1.5, (my1 + by0) / 2, "registration", ha="left", va="center")
    # (3) histology: a section, its blockface photograph, and the two alignments to the template
    image("histo", xp, R4); image("slice", xv, R4)
    text(cx("histo"), bot("histo") + 0.5, "histology section"); text(cx("slice"), bot("slice") + 0.5, "its blockface photograph")
    arrow((rgt("histo") + 1.2, R4), (left("slice") - 1.2, R4), lw_=0.8)
    box(LX0, hy0, LX1, hy1); header(LX0, LX1, hy0, "Histology", f"{N['n_histo']} brains")
    x0a, yb, ym = rgt("slice") + 1.2, R4 - 2.2, R4 + 2.2
    elbow((x0a, yb), (T1c, yb), (T1c, by1 + 0.4), BFA)
    text((LX1 + T1c) / 2, yb - 0.8, "blockface-based alignment", color=BFA, va="bottom")
    elbow((x0a, ym), (T2c, ym), (T2c, my1 + 0.4), MRIA)
    text((LX1 + T2c) / 2, ym + 0.8, "MRI-based alignment (through its own MRI)", color=MRIA)

    inputs = [P[k] for k in ("template_t2", "template_mask", "template_dec", "template_frac1", "template_frac2", "bft_rgb", "bft_mask",
                             "bft_pull_map", "blockface_completed_rgb", "blockface_synth_mask", "blockface_store", "photographs",
                             "photo_masks", "histology_piece")]
    inputs += list(INP.values()) + [numbers_root() / "bf_cohort.json", numbers_root() / "histo_tube_landmarks.json"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(Path(a.out).with_suffix(".pdf")), facecolor=fig.get_facecolor(), metadata={"CreationDate": None})
    FS.save(fig, a.out, [str(p_) for p_ in inputs], " ".join(["reproduce/figures/pipeline.py"] + sys.argv[1:]),
            extra={"print_scale_mm_per_mm": SC, "planes": {"template_x_mm": XS, "blockface_template_x_mm": xb_s,
                                                            "blockface_volume_x_mm": xv_s, "synthesized_px": n_synth_px},
                   "blockface": {"coronal_section": k0, "photo_mirrored": bool(mirrored), "photo_mm_px": mm_px,
                                 "histology_flipped": bool(flip)}})
    print("wrote", a.out, "scale", SC)

if __name__ == "__main__":
    main()
