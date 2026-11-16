#!/usr/bin/env python3
"""The paper's figure "Building the template and aligning histology to it": the pipeline at a glance.

    python code/figures/pipeline.py [--out figures/pipeline.png]

Three stages, read top to bottom. (1) Two constructions of one style: ex vivo MRI -> the MRI template; blockface -> the blockface
template. (2) Their registration (vertical arrow). (3) Histology: a section registered to its blockface photograph (the pair equally
spaced between the block's L and R), then transformed to the template by the blockface-based alignment (cyan, through the blockface template) or the
MRI-based alignment (brown, through its own MRI), both starting right after the histology block. The blockface template's box starts
at the MRI template's left edge; its image is centered in it below the header.

Inputs: the products listed in settings/internal/products_figures.yaml (under $LINC_PRODUCTS), two counts from the numbers records
($LINC_NUMBERS: bf_cohort, histo_tube_landmarks) and the number of brains in the blockface template's final step
(settings/blockface_template.yaml step9.members). Display parameters: settings/figures.yaml (pipeline).

Planes (the sagittal plane of the template figure, x = 2.0 mm):
- template-space thumbnails (the MRI channels on the template grid) at x = 2.0 mm;
- the blockface template in its own space, at the point (2.0, 19.5, 15.5) carried by the registration's pull map (as in the blockface template figure);
- the MRI input brain (MF1) at x = 2.0 mm of its canonical frame (its rigid placement on the template grid);
- the blockface volume: MM6's completed volume, the construction's input, at x = 4.5 mm of its rigid placement (blockface_x_mm; nearer
  the midline the plane cuts the medial occipital cortex at a tangent). Its photographed part is read from the reconstruction's store
  (level 2: 0.16 mm in plane, every 45 um section) through the brain's registration of record, y -> y + d(y) (warp) -> the linear
  start -> store, with one linear interpolation, on the record's color scale; its synthesized part is the completed RGB, linear, at
  its own 0.25 mm. A dashed line marks where the synthesized tissue begins: the synthesized mask's border on the junction plane.
  Nothing is smoothed, cut or filled.
- one photograph of the same brain, at MM2's place along its stack; the histology row is MM2's: its section, registered (TIRL), and
  the photograph it was registered to, in color from the store (histology_photo_slice; its gray is TIRL's fixed image, checked),
  on the fixed image's pixels and crop, displayed as the blockface volume is (the record's color scale: the record_color_pct
  percentile of the brightest channel over its tissue to 1, the whole photograph alike); the raw photograph above stays raw.
Orientation letters (figstyle.orient_letters, the rule of every figure): L and R on the coronal views (neurological), P and A on
the sagittal views (anterior right), each 1 mm from its image's tissue, the two of a row on one line at the middle of the row's
images, and at least 0.5 mm inside the block's box. One pair per block where one convention holds (Ex vivo MRI, MRI template,
Blockface template, Histology): the first letter left of the block's leftmost image column, the second right of its rightmost. The
Blockface block (photograph and volume, two views) has a pair per view, the four letters on one line; its arrow runs between the
photograph's R and the volume's P. Each block's images and letters are centered in its box. The blockface stores are +x = right:
the tubes, which mark the left hemisphere, lie at lower x (checked on MM6 and MM2, 11 Nov 2026).
One print scale for every image (the largest that fits the 178 mm width, lowered in 0.01 steps while the figure is taller than
max_height_mm); one 10 mm scale bar, outside every block (the bottom-right corner), its length centered above it. Writes the
PNG, a PDF beside it, and provenance.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "code"))
import figstyle as FS                                          # noqa: E402
from shared.paths import numbers_root, products                 # noqa: E402
from blockface.registration import xfmtools as X                    # noqa: E402


# ---- small helpers ---------------------------------------------------------------------------------------------------------
def crop_to(a, thr=0.0, pad=4):
    """Crop an image (2D, or 2D x channels) to the pixels above `thr`, plus `pad` pixels."""
    m = (a.max(-1) if a.ndim == 3 else a) > thr
    r, c = np.nonzero(m)
    return a[max(r.min() - pad, 0):r.max() + pad + 1, max(c.min() - pad, 0):c.max() + pad + 1]


def crop_box(a, thr=0.0, pad=2):
    """(r0, r1, c0, c1): the box of the pixels above `thr`, plus `pad` pixels."""
    m = (a.sum(-1) if a.ndim == 3 else a) > thr
    r = np.where(m.any(1))[0]; c = np.where(m.any(0))[0]
    return max(r[0] - pad, 0), min(r[-1] + pad + 1, a.shape[0]), max(c[0] - pad, 0), min(c[-1] + pad + 1, a.shape[1])


def dashed_segments(syn, photo, depth, dr, dc, depth_mm):
    """The polylines (image mm) of the synthesized mask's border that separate synthesized from photographed voxels and lie on the
    junction plane (|seam_depth| <= depth_mm on both voxels across it), and a record of what is drawn and what is not."""
    from matplotlib import pyplot as plt
    from scipy import ndimage as ndi
    fig_, ax_ = plt.subplots()
    x = (np.arange(syn.shape[1]) + 0.5) * dc; y = (np.arange(syn.shape[0]) + 0.5) * dr
    field = np.where(syn, 1.0, 0.0)
    field[~(syn | photo)] = np.nan                                  # background: no contour against it
    cs = ax_.contour(x, y, field, levels=[0.5])
    segs, n_v, off = [], 0, []
    for path in cs.get_paths():
        for piece in path.to_polygons(closed_only=False):
            cur = []
            for vx, vy in piece:                                    # a contour vertex lies between two voxel centers: that pair decides
                rf, cf = vy / dr - 0.5, vx / dc - 0.5
                if abs(rf - round(rf)) > 1e-6:
                    pair = ((int(np.floor(rf)), int(round(cf))), (int(np.ceil(rf)), int(round(cf))))
                else:
                    pair = ((int(round(rf)), int(np.floor(cf))), (int(round(rf)), int(np.ceil(cf))))
                both = all(0 <= r_ < syn.shape[0] and 0 <= c_ < syn.shape[1] for r_, c_ in pair)
                keep = both and {bool(syn[pair[0]]), bool(syn[pair[1]])} == {True, False} and all(photo[q] or syn[q] for q in pair) \
                    and max(abs(depth[q]) for q in pair) <= depth_mm
                if keep:
                    cur.append((vx, vy)); n_v += 1; off.append(float(max(abs(depth[q]) for q in pair)))
                else:
                    if len(cur) >= 2:
                        segs.append(np.asarray(cur))
                    cur = []
            if len(cur) >= 2:
                segs.append(np.asarray(cur))
    plt.close(fig_)
    bad = 0                                                         # every kept vertex has both kinds of voxel within one voxel
    for s_ in segs:
        for vx, vy in s_:
            r, c = int(vy / dr - 0.5), int(vx / dc - 0.5)
            w = (slice(max(r - 1, 0), r + 3), slice(max(c - 1, 0), c + 3))
            bad += not (syn[w].any() and photo[w].any())
    syn_rostral = syn & (depth >= 0)
    th = ndi.distance_transform_edt(syn_rostral) * 0.25
    rec = dict(vertices=n_v, segments=len(segs), length_mm=round(float(sum(np.hypot(*np.diff(s_, axis=0).T).sum() for s_ in segs)), 1),
               vertices_without_both_sides=int(bad), seam_depth_abs_at_vertices_mm=[round(float(v), 3) for v in np.percentile(off, [50, 100])],
               mask_voxels_in_plane=int(syn.sum()), mask_caudal_of_line=int((syn & (depth < 0)).sum()), mask_rostral_of_line=int(syn_rostral.sum()),
               mask_rostral_share=round(float(syn_rostral.sum() / syn.sum()), 4),
               mask_rostral_half_thickness_mm=[round(float(v), 2) for v in np.percentile(th[syn_rostral], [50, 99, 100])] if syn_rostral.any() else None,
               photographed_caudal_of_line=int((photo & (depth < -depth_mm)).sum()))
    return segs, rec


def hires_plane(P, cfg, x):
    """The blockface volume's sagittal plane at world x, rows superior -> inferior, columns posterior -> anterior, at
    hires_points_per_voxel display points per voxel: rostral of the junction plane the store's photographs (level
    blockface_hires_level) pulled through the registration of record (one linear interpolation) on the record's color scale, caudal
    of it the completed RGB, linearly interpolated. Returns the composite (0-1) and its record."""
    import nibabel as nib
    import zarr
    from scipy import ndimage as ndi
    SUB, LEVEL = int(cfg["hires_points_per_voxel"]), int(cfg["blockface_hires_level"])
    im = nib.load(str(P["blockface_completed_rgb"])); A = im.affine
    assert nib.aff2axcodes(A) == ("R", "A", "S") and np.allclose(A[:3, :3], np.diag(np.diag(A[:3, :3])))
    i = int(round((x - A[0, 3]) / A[0, 0]))
    rgb = np.asarray(im.dataobj[i], np.float32)                     # (J, K, 3)
    dp = np.asarray(nib.load(str(P["blockface_seam_depth"])).dataobj[i], np.float32)
    tissue = rgb.sum(-1) > 0
    jt, kt = np.nonzero(tissue)                                     # seam_depth is affine in the plane (it reads 0 off the tissue)
    Mf = np.c_[jt, kt, np.ones(len(jt))]
    co, *_ = np.linalg.lstsq(Mf, dp[jt, kt], rcond=None)
    nj, nk = rgb.shape[:2]
    off = (np.arange(SUB) + 0.5) / SUB - 0.5
    jj = (np.arange(nj)[:, None] + off[None]).ravel()
    kk = (np.arange(nk)[:, None] + off[None]).ravel()[::-1]          # rows run superior -> inferior
    K, J = np.meshgrid(kk, jj, indexing="ij")
    W = np.stack([np.full(K.shape, A[0, 3] + A[0, 0] * i, float), A[1, 3] + A[1, 1] * J, A[2, 3] + A[2, 2] * K], -1)
    # the registration of record: frame point y -> y + d(y) -> the linear start -> blockface-native world
    D = Path(P["blockface_registration"])
    meta = json.loads((D / "meta.json").read_text())
    kinds = [(c["kind"], c["file"], c["convention"]) for c in meta["components"]]
    assert [k[0] for k in kinds] == ["affine", "warp", "affine"] and kinds[1][2] == "itk_lps_disp", kinds
    d, da = X.read_fireants_field(str(D / kinds[1][1]))
    M = X.Map(L=np.loadtxt(D / "placed_to_bf_pull.txt"), d=d.astype(np.float32), d_affine=da, name="placed -> bf-native")
    pts = M(W.reshape(-1, 3)).reshape(W.shape)
    # the store, at its level: linear interpolation at those points
    store = Path(P["blockface_example_store"])
    z = json.loads((store / "zarr.json").read_text())["attributes"]
    ms = z["multiscales"] if "multiscales" in z else z["ome"]["multiscales"]
    ct = {t["type"]: t for t in ms[0]["datasets"][LEVEL]["coordinateTransformations"]}
    tr, sc = ct["translation"]["translation"], ct["scale"]["scale"]   # [c, rows, slices, cols]
    scale = np.array([sc[3], sc[2], sc[1]]) / 1000.0; origin = scale * np.array([tr[3], tr[2], tr[1]])
    vox = (pts - origin) / scale                                      # (cols, slices, rows)
    arr = zarr.open(str(store), mode="r")[str(LEVEL)]
    lo = np.floor(vox.reshape(-1, 3).min(0)).astype(int) - 1; hi = np.ceil(vox.reshape(-1, 3).max(0)).astype(int) + 2
    n = (arr.shape[3], arr.shape[2], arr.shape[1])
    lo = np.clip(lo, 0, n); hi = np.clip(hi, 0, n)
    blk = np.asarray(arr[:, lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]]).astype(np.float32)
    crd = [(vox[..., 2] - lo[2]).ravel(), (vox[..., 1] - lo[1]).ravel(), (vox[..., 0] - lo[0]).ravel()]
    raw = np.stack([ndi.map_coordinates(blk[c], crd, order=1, mode="constant", cval=0.0).reshape(vox.shape[:2]) for c in range(blk.shape[0])], -1)
    # the record's color scale: the percentile of the brightest channel of the registered colors over the photographed tissue
    cr = np.asanyarray(nib.load(str(P["blockface_registered_rgb"])).dataobj).astype(np.float32) / 255.0
    valid = ndi.binary_fill_holes(np.asanyarray(nib.load(str(P["blockface_registered_grey"])).dataobj).astype(np.float32) > 0)
    lab, nl = ndi.label(valid)
    if nl > 1:
        valid = lab == (1 + int(np.argmax(ndi.sum(np.ones_like(lab), lab, index=range(1, nl + 1)))))
    rscale = float(np.percentile(cr.max(-1)[valid & (cr.sum(-1) > 0)], float(cfg["record_color_pct"])))
    del cr, valid, lab
    # the composite, voxel by voxel as the mask's border runs: caudal of the junction plane the completed RGB, else the photographs
    gj, gk = np.meshgrid(np.arange(nj), np.arange(nk), indexing="ij")
    caudal_vox = co[0] * gj + co[1] * gk + co[2] < 0
    Jn = np.clip(np.rint(J).astype(int), 0, nj - 1); Kn = np.clip(np.rint(K).astype(int), 0, nk - 1)
    caudal = caudal_vox[Jn, Kn]
    up = np.stack([ndi.map_coordinates(rgb[..., c], [J.ravel(), K.ravel()], order=1, mode="constant").reshape(J.shape) for c in range(3)], -1)
    comp = np.where(caudal[..., None], up, np.clip(raw / 255.0 / rscale, 0, 1))
    rec = dict(store_level=LEVEL, store_scale_mm=scale.tolist(), block_shape=list(blk.shape), display_mm=0.25 / SUB, record_scale=rscale,
               chain="y -> y + d(y) (warp) -> placed_to_bf_pull -> store, one linear interpolation")
    return comp, rec


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
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    from matplotlib.path import Path as MPath
    from PIL import Image
    from scipy.ndimage import binary_erosion, binary_fill_holes

    P = products("products_figures")
    cfg = yaml.safe_load((ROOT / "settings" / "figures.yaml").read_text())["pipeline"]
    XS = float(cfg["sagittal_x_mm"])
    W_LUM = np.asarray(cfg["luminance_weights"], np.float32)
    N = {"n_bf": count("bf_cohort", "rows"), "n_histo": count("histo_tube_landmarks", "per_subject"),
         "n_bf_final": len(yaml.safe_load((ROOT / "settings" / "blockface_template.yaml").read_text())["step9"]["members"])}  # step 9's brains
    FG, GRAY, BFA, MRIA = FS.FG, FS.GRAY, FS.COL["direct"], FS.COL["mri"]

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
    # the blockface volume (the docstring): the photographed part from the store through the registration of record, the
    # synthesized part as stored, and the dashed line where the synthesized tissue begins
    XB = float(cfg["blockface_x_mm"]); SUB = int(cfg["hires_points_per_voxel"])
    rgb, dr, dc, xv_s = sagx(P["blockface_completed_rgb"], XB); sm = sagx(P["blockface_synth_mask"], XB)[0] > 0
    depth = sagx(P["blockface_seam_depth"], XB)[0]
    tissue = rgb.sum(-1) > 0; syn = sm & tissue; photo_t = tissue & ~sm
    comp, hires_rec = hires_plane(P, cfg, XB)
    rh0, rh1, ch0, ch1 = crop_box(comp, pad=2 * SUB)                 # on whole voxels, so the line's mm hold
    r0, r1, c0, c1 = rh0 // SUB, int(np.ceil(rh1 / SUB)), ch0 // SUB, int(np.ceil(ch1 / SUB))
    vol = comp[r0 * SUB:r1 * SUB, c0 * SUB:c1 * SUB]
    segs, line_rec = dashed_segments(syn[r0:r1, c0:c1], photo_t[r0:r1, c0:c1], depth[r0:r1, c0:c1], dr, dc, float(cfg["junction_depth_mm"]))
    assert line_rec["vertices_without_both_sides"] == 0 and line_rec["segments"] >= 1, line_rec
    I["bfvol"] = (vol, dr / SUB, dc / SUB, None)
    # one photograph of the same brain (photo_section), neurological, cropped to its brain box
    fin2, A2 = read_level(P["blockface_example_store"], int(cfg["blockface_level"]))
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
    # one histology section and its blockface photograph (TIRL), neurological; the photograph in color: the store's level 0 slice
    # whose gray is TIRL's fixed image (slide orientation = the store's rows and columns reversed), on the fixed image's crop
    import zarr
    hi = np.asarray(Image.open(P["histology_piece"] / "moving4_nonlinear_rgb.png").convert("RGB"), np.float32) / 255.0
    sl = np.asarray(Image.open(P["histology_piece"] / "fixed.png").convert("L"), np.float32) / 255.0
    kh = int(cfg["histology_photo_slice"])
    col = np.moveaxis(np.asarray(zarr.open_group(str(P["blockface_store"]), mode="r")["0"][:, :, kh, :]), 0, -1)[::-1, ::-1]
    col = col.astype(np.float32) / 255.0
    assert col.shape[:2] == sl.shape, (col.shape, sl.shape)
    photo_check = dict(store_slice=kh, store_level=0,
                       r_gray_vs_fixed=round(float(np.corrcoef((col @ W_LUM).ravel(), sl.ravel())[0, 1]), 5),
                       support_agreement=round(float(((col.sum(-1) > 0) == (sl > 0)).mean()), 5))
    assert photo_check["r_gray_vs_fixed"] > 0.99 and photo_check["support_agreement"] > 0.999, photo_check
    flip = histo_side(sl, read_level(P["blockface_store"], int(cfg["blockface_level"]))[0])     # the histology example's own stack
    if flip:
        hi, sl, col = hi[:, ::-1], sl[:, ::-1], col[:, ::-1]
    px = float(cfg["histology_mm_px"])
    I["histo"] = (crop_to(np.clip(hi / np.percentile(hi, 99.5), 0, 1), 0.03, pad=6), px, px, None)
    q0, q1, q2, q3 = crop_box(np.clip(sl / np.percentile(sl[sl > 0], 99.5), 0, 1), 0.03, pad=6)   # the gray's crop, as before
    # its display as the blockface volume's (the author, 12 Nov 2026: the same brightness): the record's color scale, the
    # record_color_pct percentile of the brightest channel over its own tissue mapped to 1, applied to the whole photograph
    hscale = float(np.percentile(col.max(-1)[col.sum(-1) > 0], float(cfg["record_color_pct"])))
    photo_check["display_scale"] = round(hscale, 4)
    I["slice"] = (np.clip(col[q0:q1, q2:q3] / hscale, 0, 1), px, px, None)

    # ---- the largest print scale that fits ------------------------------------------------------------------------------------
    W, PAD, GAP, MIDW, AGAP = 178.0, 1.8, 2.6, 17.0, 8.0      # box padding, image gap, middle column (its 14.2 mm labels + 1.4 mm each
                                                              # side: the room the side letters take, 12 Nov 2026), arrow gap in a box (mm)
    wd = lambda k, s: I[k][0].shape[1] * I[k][2] * s
    ht = lambda k, s: I[k][0].shape[0] * I[k][1] * s
    LCLR, ARROW = 0.5, 4.0                                     # a side letter's distance from the box; the shortest arrow between letters (mm)
    room = {s: FS.orient_room(s) + LCLR for s in FS.DIRS["sagittal"][:2] + FS.DIRS["coronal"][:2]}   # a letter's side room (mm)
    SIDE = max(PAD, *(room[s] for s in FS.DIRS["sagittal"][:2]))   # the sagittal blocks' side margin
    BGAP = max(AGAP, room["R"] + room["P"] - 2 * LCLR + ARROW + 2.0)    # the Blockface block: R, the arrow (1 mm clear each end), P

    def widths(s):
        mri = 2 * max(wd(k, s) for k in ("i_t2", "i_dec", "i_f1", "i_f2")) + GAP + 2 * SIDE
        bf = room["L"] + wd("photo", s) + BGAP + wd("bfvol", s) + room["A"]
        hist = room["L"] + wd("histo", s) + AGAP + wd("slice", s) + room["R"]
        tpl = 2 * max(wd(k, s) for k in ("t_t2", "t_dec", "t_f1", "t_f2")) + GAP + 2 * SIDE
        return max(mri, bf, hist), tpl
    s_hi, s_lo = cfg["print_scale_range"]
    for SC in np.arange(s_hi, s_lo, -0.01):
        lw, tw = widths(SC)
        if W - 4.0 - lw - tw >= MIDW:
            break
    SC = float(round(SC, 2)); SC_width = SC

    # ---- layout (mm), lowering the print scale while the figure is taller than max_height_mm -------------------------------
    HEAD, LAB, G12, G23 = 6.0, 3.4, 4.5, 3.0                  # box header; label line; the registration's gap (its arrow); rows 2-3
    while True:
        lw, tw = widths(SC)
        LX0, LX1, TX0, TX1 = 2.0, 2.0 + lw, W - 2.0 - tw, W - 2.0
        h1 = max(ht(k, SC) for k in ("i_t2", "i_dec", "t_t2", "t_dec")); h2 = max(ht(k, SC) for k in ("i_f1", "i_f2", "t_f1", "t_f2"))
        h3 = max(ht(k, SC) for k in ("photo", "bfvol", "bft")); h4 = max(ht("histo", SC), ht("slice", SC))
        top = 2.0
        R1 = top + HEAD + h1 / 2; R2 = R1 + h1 / 2 + LAB + 0.8 + h2 / 2; my1 = R2 + h2 / 2 + LAB + 0.6
        by0 = my1 + G12; R3 = by0 + HEAD + h3 / 2; by1 = R3 + h3 / 2 + 2 * LAB + 0.6
        hy0 = by1 + G23; R4 = hy0 + HEAD + h4 / 2; hy1 = R4 + h4 / 2 + LAB + 0.6; H = hy1 + 2.0
        if H <= float(cfg["max_height_mm"]) or SC <= s_lo:
            break
        SC = float(round(SC - 0.01, 2))
    wm = max(wd(k, SC) for k in ("i_t2", "i_dec", "i_f1", "i_f2")); wt = max(wd(k, SC) for k in ("t_t2", "t_dec", "t_f1", "t_f2"))
    C1 = LX0 + (lw - (2 * wm + GAP)) / 2 + wm / 2; C2 = C1 + wm + GAP
    T1c = TX0 + SIDE + wt / 2; T2c = T1c + wt + GAP
    x_ = (LX0 + LX1) / 2 - (room["L"] + wd("photo", SC) + BGAP + wd("bfvol", SC) + room["A"]) / 2 + room["L"]   # centered in the box
    xp = x_ + wd("photo", SC) / 2; xv = x_ + wd("photo", SC) + BGAP + wd("bfvol", SC) / 2
    x_ = (LX0 + LX1) / 2 - (room["L"] + wd("histo", SC) + AGAP + wd("slice", SC) + room["R"]) / 2 + room["L"]   # histology, the same
    xh = x_ + wd("histo", SC) / 2; xs = x_ + wd("histo", SC) + AGAP + wd("slice", SC) / 2
    bx0, bx1 = TX0, TX0 + wd("bft", SC) + 2 * SIDE              # the blockface template's box, from the MRI template's left edge
    xbt = (bx0 + bx1) / 2                                         # its image, the registration and the blockface-based arrowhead
    assert bx1 < T2c - 4.0, "the MRI-based alignment's arrow must pass right of the blockface template"

    fig = plt.figure(figsize=(W * FS.MM, H * FS.MM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    B = {}; AX = {}

    def image(k, xc, yc):
        arr, dr_, dc_, cmap = I[k]
        h, w = arr.shape[0] * dr_ * SC, arr.shape[1] * dc_ * SC
        a_ = FS.axes_mm(fig, xc - w / 2, yc - h / 2, w, h)
        a_.imshow(arr, cmap=cmap, aspect="auto", vmin=0, vmax=1, extent=(0, arr.shape[1] * dc_, arr.shape[0] * dr_, 0))
        a_.axis("off"); B[k] = (xc - w / 2, yc - h / 2, w, h); AX[k] = a_

    def text(x, y, s, **kw):
        kw.setdefault("fontsize", FS.MIN_PT); kw.setdefault("ha", "center"); kw.setdefault("va", "top"); kw.setdefault("color", FG)
        return ax.text(x, y, s, linespacing=1.15, **kw)

    def arrow(p, q, col=FG, lw_=1.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, lw=lw_, color=col, shrinkA=0, shrinkB=0))

    def elbow(p, corner, q, col, lw_=1.3):
        """An arrow along two straight segments, p -> corner -> q."""
        ax.add_patch(FancyArrowPatch(path=MPath([p, corner, q], [MPath.MOVETO, MPath.LINETO, MPath.LINETO]), arrowstyle="-|>",
                                     mutation_scale=6, lw=lw_, color=col, shrinkA=0, shrinkB=0, joinstyle="miter"))

    def box(x0, y0, x1, y1, ec=FS.FRAME, lw_=0.6):
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

    LBL = {}                                                        # the label under each image (its text artist)

    def label(k, s):
        LBL[k] = ax.text(cx(k), bot(k) + 0.5, s, fontsize=FS.MIN_PT, ha="center", va="top", color=FG, linespacing=1.15)

    def box_mm(t):                                                  # a text's (x0, x1, y0, y1), figure mm, y down
        e = t.get_window_extent(fig.canvas.get_renderer()); f = lambda v: v / fig.dpi / FS.MM  # noqa: E731
        return f(e.x0), f(e.x1), H - f(e.y1), H - f(e.y0)

    def ink_mm(t):                                                  # an orientation letter's ink (x0, x1, y0, y1), figure mm, y down
        (x, y), e, k = t.get_position(), FS.ink_pt(t.get_text(), t.get_fontsize()), 25.4 / 72
        return x * W + e[0] * k, x * W + e[1] * k, (1 - y) * H - e[3] * k, (1 - y) * H - e[2] * k

    def letters_clear(ts, x0, x1):
        """Asserts a block's letters (their ink) lie LCLR inside its box (x0..x1) and clear of every image and label."""
        for t in ts:
            a0, a1, a2, a3 = ink_mm(t)
            assert a0 >= x0 + LCLR - 0.02 and a1 <= x1 - LCLR + 0.02, f"direction letter {t.get_text()} leaves its block's box"
            others = [(B[k][0], B[k][0] + B[k][2], B[k][1], B[k][1] + B[k][3]) for k in B] + [box_mm(u) for u in LBL.values()]
            for b0, b1, b2, b3 in others:
                assert a1 <= b0 or b1 <= a0 or a3 <= b2 or b3 <= a2, f"direction letter {t.get_text()} touches an image or a label"

    # (1) two constructions of one style
    for k, x, y, s in (("i_t2", C1, R1, "T2-weighted"), ("i_dec", C2, R1, "Tensor"), ("i_f1", C1, R2, "Soma fraction"),
                       ("i_f2", C2, R2, "Neurite fraction"), ("t_t2", T1c, R1, "T2-weighted"), ("t_dec", T2c, R1, "Tensor"),
                       ("t_f1", T1c, R2, "Soma fraction"), ("t_f2", T2c, R2, "Neurite fraction"), ("photo", xp, R3, "Photograph"),
                       ("bfvol", xv, R3, "Blockface volume")):
        image(k, x, y); label(k, s)
    tl = ax.text(cx("bfvol"), bot("bfvol") + 0.5 + LAB, "Dashed: junction plane", fontsize=FS.MIN_PT, ha="center", va="top",
                 color=FG, linespacing=1.15)
    wl = tl.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi / FS.MM
    if cx("bfvol") + wl / 2 > LX1 - 1.0:                           # kept inside the box
        tl.set_x(LX1 - 1.0 - wl / 2)
    for s_ in segs:                                                 # on the volume, in the mm of its extent
        AX["bfvol"].plot(s_[:, 0], s_[:, 1], color="k", lw=0.8, ls=(0, (3, 2)), solid_capstyle="butt", dash_capstyle="butt")
    AX["bfvol"].set_xlim(0, vol.shape[1] * I["bfvol"][2]); AX["bfvol"].set_ylim(vol.shape[0] * I["bfvol"][1], 0)
    image("bft", xbt, (by0 + HEAD + by1) / 2)                       # centered below its header (it has no label line)
    box(LX0, top, LX1, my1); header(LX0, LX1, top, "Ex vivo MRI", "4 brains + their L-R mirrors")
    box(LX0, by0, LX1, by1); header(LX0, LX1, by0, "Blockface", f"{N['n_bf']} brains + their L-R mirrors ({N['n_bf_final']} in the final step)")
    box(TX0, top, TX1, my1, ec=FG, lw_=0.9); header(TX0, TX1, top, "MRI template")
    box(bx0, by0, bx1, by1, ec=FG, lw_=0.9); header(bx0, bx1, by0, "Blockface template")
    yA = (R1 + R2) / 2
    arrow((LX1 + 1.5, yA), (TX0 - 1.0, yA)); g1 = text((LX1 + TX0) / 2, yA - 1.4, "Groupwise\nconstruction", va="bottom")
    arrow((LX1 + 1.5, R3), (bx0 - 1.0, R3)); g2 = text((LX1 + bx0) / 2, R3 - 1.4, "Groupwise\nconstruction", va="bottom")
    for g_ in (g1, g2):                                             # the middle column's labels keep 1 mm from the boxes
        assert box_mm(g_)[0] >= LX1 + 1.0 and box_mm(g_)[1] <= TX0 - 1.0, "a middle-column label is closer than 1 mm to a box"
    # (2) their registration
    arrow((xbt, by0 - 0.4), (xbt, my1 + 0.4)); text(xbt + 1.5, (my1 + by0) / 2, "Registration", ha="left", va="center")
    # (3) histology: a section, its blockface photograph, and the two alignments to the template
    image("histo", xh, R4); image("slice", xs, R4)
    label("histo", "Histology section"); label("slice", "Blockface photograph")
    arrow((rgt("histo") + 1.2, R4), (left("slice") - 1.2, R4), lw_=0.8)
    box(LX0, hy0, LX1, hy1); header(LX0, LX1, hy0, "Histology", f"{N['n_histo']} brains")
    x0a, yb, ym = LX1 + 1.5, R4 - 2.2, R4 + 2.2                    # the two alignments start right after the block
    elbow((x0a, yb), (xbt, yb), (xbt, by1 + 0.4), BFA)
    text((LX1 + T1c) / 2, yb - 0.8, "Blockface-based alignment", color=FS.COL["direct_text"], va="bottom")
    elbow((x0a, ym), (T2c, ym), (T2c, my1 + 0.4), MRIA)
    t_mri = text((LX1 + T2c) / 2, ym + 0.8, "MRI-based alignment", color=FS.COL["mri_text"])
    # one scale bar, outside every block: the figure's bottom-right corner, right of the MRI-based alignment's vertical leg and below
    # the blockface template; its length centered above it, 2 pt over the bar (as figstyle.scalebar), the two ending at the margin
    bar = float(cfg["scale_bar_mm"]); blen, sby = bar * SC, hy1 - 1.6
    t_bar = ax.text(0, sby - (0.6 + 2.0) / 72 * 25.4, f"{bar:g} mm", fontsize=FS.MIN_PT, ha="center", va="baseline", color=FG,
                    gid=FS.SCALEBAR_GID)
    e_ = t_bar.get_window_extent(fig.canvas.get_renderer()); wl_ = (e_.x1 - e_.x0) / fig.dpi / FS.MM
    xc_ = TX1 - max(blen, wl_) / 2; t_bar.set_x(xc_); sb0, sb1 = xc_ - blen / 2, xc_ + blen / 2
    ax.plot([sb0, sb1], [sby, sby], color=FG, lw=1.2, solid_capstyle="butt")
    e_ = t_bar.get_window_extent(fig.canvas.get_renderer()); sl0 = min(sb0, e_.x0 / fig.dpi / FS.MM); sl_top = H - e_.y1 / fig.dpi / FS.MM
    e_ = t_mri.get_window_extent(fig.canvas.get_renderer()); lab_x1 = e_.x1 / fig.dpi / FS.MM
    lab_y1 = H - e_.y0 / fig.dpi / FS.MM                            # the label's bottom (figure mm, y down)
    assert sl0 > T2c + 3.0, "the scale bar and its length must keep clear of the MRI-based alignment's vertical leg"
    assert sl_top > by1 + 3.0 and sby > ym + 3.0, "the scale bar must sit below the blockface template and the MRI-based line"
    assert sl0 > lab_x1 + 3.0 or sl_top > lab_y1 + 2.0, "the scale bar must keep clear of the MRI-based alignment's label"
    assert sl0 > LX1 + 3.0 and sl_top > my1 + 3.0, "the scale bar must lie outside the histology and MRI template blocks"
    # orientation letters (figstyle.orient_letters): one pair per block of one convention, at its sides; the Blockface block's per
    # view, the four on one line; then the Blockface block's arrow, between the photograph's R and the volume's P
    OL = FS.orient_letters(fig, [((AX["i_t2"], AX["i_f1"]), (AX["i_dec"], AX["i_f2"]), "sagittal"),
                                 ((AX["t_t2"], AX["t_f1"]), (AX["t_dec"], AX["t_f2"]), "sagittal"),
                                 (AX["bft"], AX["bft"], "sagittal"),
                                 [(AX["photo"], AX["photo"], "coronal"), (AX["bfvol"], AX["bfvol"], "sagittal")],
                                 (AX["histo"], AX["slice"], "coronal")])
    letters_clear(OL[0:2], LX0, LX1); letters_clear(OL[2:4], TX0, TX1); letters_clear(OL[4:6], bx0, bx1)
    letters_clear(OL[6:10], LX0, LX1); letters_clear(OL[10:12], LX0, LX1)
    yR = H - (OL[7].get_position()[1] * H + FS._cap_pt(FS.MIN_PT) / 72 * 25.4 / 2)       # the letters' middle (mm, y down)
    a0_, a1_ = ink_mm(OL[7])[1] + 1.0, ink_mm(OL[8])[0] - 1.0
    assert a1_ - a0_ >= ARROW - 0.01, f"the Blockface block's arrow, {a1_ - a0_:.2f} mm, is shorter than {ARROW} mm"
    arrow((a0_, yR), (a1_, yR), lw_=0.8)

    inputs = [P[k] for k in ("template_t2", "template_mask", "template_dec", "template_frac1", "template_frac2", "bft_rgb", "bft_mask",
                             "bft_pull_map", "blockface_completed_rgb", "blockface_synth_mask", "blockface_seam_depth",
                             "blockface_registration", "blockface_registered_rgb", "blockface_registered_grey", "blockface_example_store",
                             "blockface_store", "photographs", "photo_masks", "histology_piece")]
    inputs += list(INP.values()) + [numbers_root() / "bf_cohort.json", numbers_root() / "histo_tube_landmarks.json"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    FS.tiles(fig)                                                   # the image panels on black, in the PDF too
    fig.savefig(str(Path(a.out).with_suffix(".pdf")), facecolor=fig.get_facecolor(), metadata={"CreationDate": None})
    FS.save(fig, a.out, [str(p_) for p_ in inputs], " ".join(["code/figures/pipeline.py"] + sys.argv[1:]),
            extra={"print_scale_mm_per_mm": SC, "print_scale_for_width": SC_width, "height_mm": round(H, 1),
                   "planes": {"template_x_mm": XS, "blockface_template_x_mm": xb_s, "blockface_volume_x_mm": xv_s},
                   "blockface": {"coronal_section": k0, "photo_mirrored": bool(mirrored), "photo_mm_px": mm_px,
                                 "histology_flipped": bool(flip), "histology_photo": photo_check},
                   "hires": hires_rec, "dashed_line": line_rec})
    print("wrote", a.out, "scale", SC, "H", round(H, 1), line_rec)

if __name__ == "__main__":
    main()
