#!/usr/bin/env python3
"""The paper's figure "Blockface reconstruction of one brain (MM2)": one panel per step of Methods.

    python reproduce/figures/reconstruction.py [--out figures/reconstruction.png]

  a  a photograph with its 2D brain mask (white), shown neurological (the photograph's side decided by correlation with the
     reconstructed section)
  b  the largest camera jump (the step of the 2D-mask centroid): a sagittal reslice of the photographs as taken | of the aligned stack
  c  every interpolated section of the stack (orange ticks) under a sagittal reslice of the aligned stack
  d  the 3D brain mask on the aligned stack
  e  a brightness step between imaging sessions in a jump-free stretch: reslices of the photographs | of the corrected stack, and each
     section's median brightness before and after correction
  f  marker tubes (an orange arrowhead beside each) before and after filling, on the section with the most tube voxels
  g  the reconstructed volume: coronal, sagittal and axial
Two print scales: whole-brain views (a, c, d, g) and zoomed views (b, e, f); scale bars below the panels, their values in the caption.
A grid of three columns, (a, d, g coronal), (b, e, g sagittal), (c, f, g axial): each cell is a rectangle of its column's width, its
content centered in it and its letter and topic at its left edge; the free width is shared as equal column spaces.
Reads one brain's reconstruction ($LINC_PRODUCTS data/sub-<ID>/blockface: the photographs, 2D masks, the aligned stack, the color
store before tube filling, the final volume, the tubes, the curated background and mask) and its configs/subjects/<ID>.yaml.
Display parameters: configs/figures.yaml `reconstruction`.
Ported from the analysis project's code/fig4_bf_steps.py at the house print scales, with panel (f) as in its copy
code/fig4_bf_steps_r068.py; helpers of code/fig_pipeline_bf.py are included below.
"""
import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import nibabel as nib
import nibabel.processing as nip
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import yaml  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO))
import figstyle as FS  # noqa: E402
from linc import paths  # noqa: E402
from linc.blockface import zarrio  # noqa: E402

CFG = yaml.safe_load((REPO / "configs" / "figures.yaml").read_text())["reconstruction"]
W_LUM = np.asarray(CFG["luminance_weights"], np.float32)              # RGB -> luminance (Rec. 601)
HALF_MM = CFG["jump_half_mm"]                                           # (b): at most this far on either side of the jump
E_HALF = CFG["step_half_mm"]                                            # (e): the symmetric half window around the step (mm)
WS, ZS = CFG["print_scale_whole"], CFG["print_scale_zoom"]              # print mm per tissue mm: whole brain, zoom
PLANE_MM = CFG["plane_window_mm"]                                       # (g): crack-free plane window
SOFT_F, SOFT_G = tuple(CFG["soft_edge_f"]), tuple(CFG["soft_edge_g"])  # (f) 0.08 mm, (g) 0.16 mm px: edge sigma, rounding


# ---- display helpers (code/fig_pipeline_bf.py) ----
def _level(path, lvl):
    arr, A = zarrio.read_level(path, lvl)
    return arr, A


def _rgb01(a):
    a = a.astype(np.float32)
    hi = np.percentile(a[a > 0], 99.5) if (a > 0).any() else 1
    return np.clip(a / hi, 0, 1)


def _photo_mirrored(photo, mask2d, section_disp):
    """True if the photograph shows the subject's LEFT on the viewer's RIGHT, i.e. if it matches the neurologically displayed
    reconstructed section better after a left-right flip. Correlation of the brain crops (gray, 160 x 160)."""
    def crop(img, m):
        ii = np.where(m.any(1))[0]; jj = np.where(m.any(0))[0]
        return (img * m)[ii[0]:ii[-1] + 1, jj[0]:jj[-1] + 1]
    rs = lambda x: np.asarray(Image.fromarray(x.astype(np.float32)).resize((160, 160), Image.BILINEAR))  # noqa: E731
    a = rs(crop(photo, mask2d)); b = rs(crop(section_disp, section_disp > 0))
    r_same = float(np.corrcoef(a.ravel(), b.ravel())[0, 1]); r_flip = float(np.corrcoef(a[:, ::-1].ravel(), b.ravel())[0, 1])
    assert abs(r_flip - r_same) >= 0.05, f"photo side undecided (r as is {r_same:.3f}, mirrored {r_flip:.3f})"
    return r_flip > r_same, r_same, r_flip


def _extent_coronal(sl, A):
    """imshow extent (mm) of np.rot90(sl) for a coronal slice sl = v[:, k] of shape (x, z[, c]): columns = world x ascending,
    rows = z descending."""
    dx, dz = A[0, 0], A[2, 2]; nx, nz = sl.shape[0], sl.shape[1]
    return (A[0, 3] - dx / 2, A[0, 3] + (nx - 0.5) * dx, A[2, 3] - dz / 2, A[2, 3] + (nz - 0.5) * dz)


def _extent_sagittal(sl, A):
    """imshow extent (mm) of np.rot90(sl) for a sagittal slice sl = v[x0] of shape (slices, z[, c]): columns = slice index (world y;
    A[1,1] < 0, so descending), rows = z descending."""
    dy, dz = A[1, 1], A[2, 2]; ny, nz = sl.shape[0], sl.shape[1]
    return (A[1, 3] - dy / 2, A[1, 3] + (ny - 0.5) * dy, A[2, 3] - dz / 2, A[2, 3] + (nz - 0.5) * dz)


def _bbox_mm(img2d, ext):
    """(h_center, v_center, width, height) in mm of the nonzero pixels of a display array drawn with `ext` (origin upper)."""
    rr = np.where((img2d > 0).any(1))[0]; cc = np.where((img2d > 0).any(0))[0]
    H, W = img2d.shape; hx = (ext[1] - ext[0]) / W; vz = (ext[3] - ext[2]) / H
    h0, h1 = ext[0] + cc[0] * hx, ext[0] + (cc[-1] + 1) * hx; v1, v0 = ext[3] - rr[0] * vz, ext[3] - (rr[-1] + 1) * vz
    return (h0 + h1) / 2, (v0 + v1) / 2, abs(h1 - h0), abs(v1 - v0)


P = types.SimpleNamespace(level=_level, rgb01=_rgb01, photo_mirrored=_photo_mirrored, extent_coronal=_extent_coronal,
                          extent_sagittal=_extent_sagittal, bbox_mm=_bbox_mm)


class Subject:
    """One brain's reconstruction folder and its manual inputs (configs/subjects/<ID>.yaml), with the analysis config's file names."""
    def __init__(self, S, root):
        y = yaml.safe_load((REPO / "configs" / "subjects" / f"{S}.yaml").read_text())
        self.subject, self.root = S, str(root)
        self.jumps = tuple(tuple(v) for v in y["jumps"]); self.missing = tuple(tuple(v) for v in y["missing"])
        self.color_store = f"{self.root}/steps/{S}_03_color.ome.zarr"

    def final(self, name):
        return f"{self.root}/{self.subject}_{name}.nii.gz"

    def final_zarr(self, name):
        return f"{self.root}/{self.subject}_{name}.ome.zarr"


def soft(img, sigma, round_px=0.0):
    """Display only (round 10): the tissue edge antialiased on black. The support is first rounded at the scale of the mask
    grid that cut it (Gaussian of `round_px` pixels, thresholded at 0.5: the staircase of a coarser mask), the RGB (0-1)
    extended outward by its nearest tissue pixel, and the result multiplied by the rounded support smoothed with a Gaussian
    of `sigma` pixels (1 inside, 0.5 on the edge, 0 away)."""
    m = img.sum(-1) > 0
    if not m.any() or m.all():
        return img
    idx = ndi.distance_transform_edt(~m, return_distances=False, return_indices=True)
    if round_px > 0:
        m = ndi.gaussian_filter(m.astype(np.float32), round_px) > 0.5
    return img[tuple(idx)] * np.clip(ndi.gaussian_filter(m.astype(np.float32), sigma), 0, 1)[..., None]


def photo(bd, k, reduce=1):
    im = Image.open(f"{bd}/00_slices/SLICE_{k:04d}_0000.png").convert("RGB")
    m = Image.open(f"{bd}/01_brainmask/01_final/SLICE_{k:04d}.png")
    if reduce > 1:
        im, m = im.reduce(reduce), m.reduce(reduce)
    return np.asarray(im), np.asarray(m) > 0


def raw_strip(bd, ks, col, reduce):
    """Sagittal reslice of the photographs as taken, inside their 2D masks: camera column `col` (full-resolution px)."""
    out = []
    for k in ks:
        im, m = photo(bd, k, reduce)
        c = int(round(col / reduce))
        out.append(im[:, c].astype(np.float32) * m[:, c, None])
    return np.stack(out, 1)                                             # (rows, sections, 3)


def section_y(A, k):
    return A[1, 3] + A[1, 1] * np.asarray(k, float)


def cracks(img):
    """Empty pixels inside the brain's outline of a 2-D section (RGB): data gaps, not anatomy."""
    nz = img.sum(-1) > 0
    return int((ndi.binary_fill_holes(nz) & ~nz).sum())


def window(ax, center, w, h):
    ax.set_xlim(center[0] - w / 2, center[0] + w / 2); ax.set_ylim(center[1] - h / 2, center[1] + h / 2)
    ax.set_axis_off()


def strip(ax, img, y_sorted, v_top, v_bot, w_mm, h_mm, top_pad=0.5):
    """A reslice drawn in mm (columns at ascending world y; rows from v_top to v_bot), windowed to w_mm x h_mm: the full span
    of sections, from just above the tissue's dorsal edge downwards."""
    ax.imshow(img, extent=(y_sorted[0], y_sorted[-1], v_bot, v_top), origin="upper", aspect="auto")
    rows = np.where(img.sum(-1).sum(1) > 0)[0]
    v_hi = v_top - (v_top - v_bot) * rows[0] / img.shape[0] + top_pad
    window(ax, (0.5 * (y_sorted[0] + y_sorted[-1]), v_hi - h_mm / 2), w_mm, h_mm)


def arrow_up(ax, x, gap=0.4, size=4.5, color=None):
    """A filled arrowhead (a triangle, no shaft) just below the axes, pointing up at data x (gap in mm of print, size in pt)."""
    import matplotlib.transforms as mtrans
    tr = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
    hgt = ax.get_window_extent().height / ax.figure.dpi * 25.4          # axes height in print mm
    ax.plot([x], [-(gap + size * 25.4 / 72 / 2) / hgt], marker="^", ms=size, mew=0, color=color or FS.COL["annot"],
            transform=tr, clip_on=False, zorder=5)


def head2(ax, top, sub, letter=None, sub_color=None, lift_mm=0.0):
    """Two header lines above a panel: the panel's letter and topic, then its sub-label (either may be empty). lift_mm raises
    both lines by that many print mm (a panel centered lower in its row keeps its headers on the row's top line)."""
    up = lift_mm * 72 / 25.4
    if letter:
        FS.letter(ax, letter, dy=12.5 + up)
    if top:
        FS.header(ax, top, dx=11 if letter else 0, dy=12.5 + up)
    if sub:
        FS.header(ax, sub, dx=0, dy=2.5 + up, color=sub_color or FS.FG)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default=CFG["subject"])
    ap.add_argument("--out", default="figures/reconstruction.png")
    ap.add_argument("--axial-z-mm", type=float, default=CFG["axial_z_mm"],
                    help="(g) axial plane (world mm). The author (2 Oct, item 12): MR252's automatic plane (z 26.44, fewest cracks within "
                         "3 mm of the fullest level) is ragged at the posterior right; z 30.12 (1 mm above the fullest level) has a clean "
                         "outline (chosen by inspection of z 25-32 mm; its dark slits are the lateral ventricles)")
    a = ap.parse_args()
    FS.setup()
    S = a.subject; bd = str(paths.products("products_blockface")["data"] / f"sub-{S}" / "blockface"); c = Subject(S, bd)
    info = dict(l.strip().split("=", 1) for l in open(f"{bd}/{S}.info") if "=" in l)
    fin, Af = P.level(c.final_zarr("blockface"), 1)                    # 0.08 mm reconstructed (x, sections, z, 3)
    st, As = P.level(f"{bd}/02_blockface.ome.zarr", 2)                 # 0.08 mm aligned + corrected, before tube filling
    col, Ac = P.level(c.color_store, 1)                                 # 0.08 mm before tube filling
    for A_ in (Af, As, Ac):
        assert A_[0, 0] > 0 and A_[2, 2] > 0 and A_[1, 1] < 0, "unexpected axis signs"
    nsec = fin.shape[1]; kmid = nsec // 2; dsec = abs(As[1, 1])
    jumps_s = sorted(s0 for s0, _ in c.jumps)
    rec = dict(subject=S, print_scale=dict(whole=WS, zoom=ZS))

    # ---- (a) the photograph: its effective pixel and its side (fig_pipeline_bf's rules) ----
    im_full, pmask = photo(bd, kmid)
    sec_disp = np.rot90(fin[:, kmid].astype(np.float32) @ W_LUM)
    mirrored, r_same, r_flip = P.photo_mirrored(im_full.astype(np.float32) @ W_LUM, pmask, sec_disp)
    ext_d = P.extent_coronal(fin[:, kmid], Af)
    _, _, sw, sh = P.bbox_mm(sec_disp, ext_d)
    jj = np.where(pmask.any(0))[0]; ii = np.where(pmask.any(1))[0]
    mm_px = float(np.mean([sw / (jj[-1] - jj[0] + 1), sh / (ii[-1] - ii[0] + 1)]))
    rec.update(section_a=int(kmid), photo_mirrored=bool(mirrored), photo_corr=dict(as_is=r_same, mirrored=r_flip),
               photo_effective_pixel_um=round(mm_px * 1000, 2), photo_shown="mirrored to neurological" if mirrored else "as taken")

    # ---- (b) the largest camera jump by the step of the 2D-mask centroid ----
    def centroid(k):
        _, m = photo(bd, k, 4)
        r, q = np.nonzero(m)
        return np.array([r.mean(), q.mean()]) * 4
    steps = []
    for s0, s1 in c.jumps:
        try:
            steps.append((float(np.linalg.norm(centroid(s1) - centroid(s0))), s0, s1))
        except Exception:
            pass
    dj, j0, j1 = max(steps)
    nh = int(HALF_MM / dsec)
    lo = max([0, j0 - nh] + [s1 + 5 for s0, s1 in c.jumps if s1 <= j0])
    hi = min([nsec - 1, j1 + nh] + [s0 - 5 for s0, s1 in c.jumps if s0 >= j1 and (s0, s1) != (j0, j1)])
    kb = list(range(lo, hi + 1))
    _, mref = photo(bd, j1 + 20)
    qq = np.where(mref.any(0))[0]
    cam_col = int(qq[0] + 0.3 * (qq[-1] - qq[0]))                     # a camera column 30 % into the brain (camera's left)
    ref = np.rot90(st[:, j1 + 20].astype(np.float32) @ W_LUM); bx, _, bw, _ = P.bbox_mm(ref, P.extent_coronal(st[:, j1 + 20], As))
    x_t = (bx + bw / 2 - 0.3 * bw) if mirrored else (bx - bw / 2 + 0.3 * bw)   # the same place in the brain, in the stack
    x0 = int(np.argmin(np.abs(As[0, 3] + As[0, 0] * np.arange(st.shape[0]) - x_t)))
    rec.update(jump=dict(sections=[int(j0), int(j1)], centroid_step_px=round(dj, 1), centroid_step_mm=round(dj * mm_px, 2),
                         shown=[int(kb[0]), int(kb[-1])], camera_column=cam_col, stack_x_mm=round(float(As[0, 3] + As[0, 0] * x0), 2)))

    # ---- (c) the longest filled run ----
    runs = sorted(((b - a_ - 1, a_, b) for a_, b in c.missing if b > a_ + 1 and a_ >= 0), reverse=True)
    nfill, f0, f1 = runs[0]
    fmid = (f0 + f1) // 2
    rec.update(filled_run=dict(present=[int(f0), int(f1)], filled=[int(f0) + 1, int(f1) - 1], shown=int(fmid),
                               distance_mm=dict(anterior=round((fmid - f0) * dsec, 3), posterior=round((f1 - fmid) * dsec, 3))))

    # ---- (e) a jump-free stretch holding a between-session step ----
    isteps = [int(x) for x in info.get("intensity_steps", "").split(",") if x]
    def stretch(s0):                                                  # up to E_HALF on each side, each side stopping 3 sections
        h_ = int(E_HALF / dsec)                                       # short of the nearest camera jump on that side (the author,
        lo_, hi_ = max(0, s0 - h_), min(nsec - 1, s0 + h_)            # 2 Oct 03:40: (e) must show ONLY the intensity correction;
        for j0, j1 in c.jumps:                                        # MR252's steps lie 3.1-3.6 mm from a jump on ONE side, so
            if j1 <= s0:                                              # the window is asymmetric: 6 mm before, 3.4 mm after)
                lo_ = max(lo_, j1 + 3)
            elif j0 >= s0:
                hi_ = min(hi_, j0 - 3)
            else:
                return s0, s0                                         # a jump at the step itself: unusable
        return lo_, hi_
    jumps_in_e = None
    es = max(isteps, key=lambda s0: stretch(s0)[1] - stretch(s0)[0])
    e0, e1 = stretch(es); ke = list(range(e0, e1 + 1))
    jumps_in_e = [round((j - es) * dsec, 2) for j in jumps_s if e0 <= j <= e1]
    rec.update(intensity=dict(session_steps=isteps, shown_step=es, shown=[e0, e1], window_mm=[round(-(es - e0) * dsec, 2), round((e1 - es) * dsec, 2)],
                              camera_jumps_in_window_mm_from_step=jumps_in_e))

    # ---- (f) the section with the most tube voxels ----
    holes = nib.load(c.final("holes"))
    kt = int(np.argmax((np.asanyarray(holes.dataobj) > 0).sum((0, 2))))
    hvo = np.asanyarray(nip.resample_from_to(holes, (col.shape[:3], Ac), order=0).dataobj) > 0
    rec.update(tubes=dict(section=kt))

    # ---- (g) crack-free planes ----
    fin2, Af2 = P.level(c.final_zarr("blockface"), 2)                  # 0.16 mm
    xw = Af2[0, 3] + Af2[0, 0] * np.arange(fin2.shape[0])
    x_c = As[0, 3] + As[0, 0] * x0
    cand_x = [q for q in range(fin2.shape[0]) if abs(xw[q] - x_c) <= PLANE_MM]
    xg = min(cand_x, key=lambda q: (cracks(np.rot90(fin2[q])), abs(xw[q] - x_c)))
    zfull = int(np.argmax((fin2.sum(-1) > 0).sum((0, 1))))              # the axial level with the most brain
    cand_z = [q for q in range(fin2.shape[2]) if abs(q - zfull) * Af2[2, 2] <= PLANE_MM]
    zg = min(cand_z, key=lambda q: (cracks(np.transpose(fin2[:, :, q], (1, 0, 2))), abs(q - zfull)))
    if a.axial_z_mm is not None:                                       # an explicit plane (item 12), the nearest level
        zg = int(round((a.axial_z_mm - Af2[2, 3]) / Af2[2, 2]))
    rec.update(views=dict(coronal_section=int(kmid), sagittal_x_mm=float(xw[xg]), axial_z_mm=float(Af2[2, 3] + Af2[2, 2] * zg),
                          cracks=dict(sagittal=cracks(np.rot90(fin2[xg])), axial=cracks(np.transpose(fin2[:, :, zg], (1, 0, 2))),
                                      sagittal_at_camera_column=cracks(np.rot90(fin2[int(np.argmin(np.abs(xw - x_c)))])),
                                      axial_at_fullest=cracks(np.transpose(fin2[:, :, zfull], (1, 0, 2))))))

    # =============================== layout (mm) ===============================
    # (a) window: the brain box x 1.12; (c): the whole aligned stack, sagittal, its tissue box + 1 mm; reslices: the shown span
    ztop, zbot = As[2, 3] + As[2, 2] * (st.shape[2] - 1), As[2, 3]
    sgc = np.rot90(st[x0]); yall = section_y(As, np.arange(nsec)); oa = np.argsort(yall)    # (z rows, all sections)
    occ = sgc.sum(-1) > 0                                              # the window: rows / columns with substantial tissue
    rnz = np.where(occ.sum(1) > 0.05 * occ.sum(1).max())[0]
    cl, _ = ndi.label(occ.sum(0) > 0.05 * occ.sum(0).max())           # the contiguous run of tissue columns around the
    cnz = np.where(cl == cl[int(np.argmax(occ.sum(0)))])[0]            # fullest column (drops isolated stray sections)
    zt_c = ztop - (ztop - zbot) * rnz[0] / sgc.shape[0]; zb_c = ztop - (ztop - zbot) * (rnz[-1] + 1) / sgc.shape[0]
    yc0, yc1 = yall[cnz].min(), yall[cnz].max()
    interp = sorted({k for a_, b in c.missing if b > a_ + 1 and a_ >= 0 for k in range(a_ + 1, b)})
    rec["interpolated"] = dict(n_sections=len(interp), n_runs=len([1 for a_, b in c.missing if b > a_ + 1 and a_ >= 0]),
                               fraction_of_stack=round(len(interp) / nsec, 4))
    yb = section_y(As, kb); ob = np.argsort(yb); bspan = yb.max() - yb.min()
    ye = section_y(As, ke); oe = np.argsort(ye); espan = ye.max() - ye.min()
    import matplotlib.transforms as mtrans

    # ---- every panel's data, read once (the drawing runs twice: once to measure the grid, once at its final scale) ----
    # (a) the photograph, neurological, and its 2D mask (white)
    im_show, pm_show = (im_full[:, ::-1], pmask[:, ::-1]) if mirrored else (im_full, pmask)
    jjs = np.where(pm_show.any(0))[0]
    th = Image.fromarray(np.ascontiguousarray(im_show)); th.thumbnail((1400, 1400))
    pm_th = np.asarray(Image.fromarray(np.ascontiguousarray(pm_show)).resize(th.size, Image.NEAREST))
    pext = (0, im_full.shape[1] * mm_px, 0, im_full.shape[0] * mm_px)
    pc = ((jjs[0] + jjs[-1] + 1) / 2 * mm_px, (im_full.shape[0] - (ii[0] + ii[-1] + 1) / 2) * mm_px)
    # (b) the camera jump: photographs as taken | aligned stack; anterior right; dorsal part
    red = 4
    rs = raw_strip(bd, kb, cam_col, red)
    v_top = (rs.shape[0] - 1) * red * mm_px
    sg = np.rot90(st[x0, kb[0]:kb[-1] + 1])                             # rows: z from the top; columns: sections
    yj = float(np.mean(section_y(As, [j0, j1])))
    # (d) the mask's box x (1.1, 1.25)
    bgi = nib.load(f"{bd}/curated/{S}_bg.nii.gz"); bg = np.asanyarray(bgi.dataobj); Ab = bgi.affine
    mki = nib.load(f"{bd}/curated/{S}_mask_edited.nii.gz"); mk = np.asanyarray(mki.dataobj) > 0
    assert np.allclose(mki.affine, Ab) and mk.shape == bg.shape[:3]
    xm = int(np.argmin(np.abs(Ab[0, 3] + Ab[0, 0] * np.arange(bg.shape[0]) - x_c)))
    ext_b = P.extent_sagittal(bg[xm], Ab)
    cxb, czb, wb, hb = P.bbox_mm(np.rot90(mk[xm]).astype(float), ext_b)
    DIM = 0.35                                                         # display only: the background outside the mask dimmed
    rec["mask3d"] = dict(display_outside_dim=DIM)
    # (e) intensity correction: photographs (inside their 2D masks) | corrected stack; the trace under the photographs
    re_ = raw_strip(bd, ke, cam_col, red)
    sge = np.rot90(st[x0, ke[0]:ke[-1] + 1])
    ys0 = float(section_y(As, es))
    # the trace: photographs only where the stack uses them (interpolated sections, never photographed or skipped, are gaps:
    # their frames are black placeholders or rejected photographs); corrected = the stack as built, every section
    used = np.array([k not in set(interp) for k in ke])
    med_raw, med_cor = [], []
    for k in ke:
        imk, mkk = photo(bd, k, 4)
        med_raw.append(float(np.median((imk.astype(np.float32) @ W_LUM)[mkk])))
        sk = st[:, k].astype(np.float32) @ W_LUM
        med_cor.append(float(np.median(sk[sk > 0])))
    med_raw, med_cor = np.array(med_raw), np.array(med_cor)
    rel_raw = np.where(used, med_raw, np.nan) / np.median(med_raw[used]); rel_cor = med_cor / np.median(med_cor[used])
    sd = lambda r: float(np.std(r[used] / np.median(r[used])))                                          # noqa: E731
    # (f) marker tubes before and after filling, each marked by an orange arrowhead beside it in both images (an outline drawn over
    # a tube, 0.25-0.5 mm across, would cover it). A tube = a dark part (< 60 % of the region's median luminance, >= 3 pixels)
    # inside a region of the tube mask (0.32 mm in-plane: its own outline is a box), or, in a region without one, the region's
    # darkest pixel when it is that dark
    ext_f = P.extent_coronal(col[:, kt], Ac)
    hs = np.rot90(hvo[:, kt]).astype(float)
    cxh, czh, wh, hh = P.bbox_mm(hs, ext_f)
    Lb = np.rot90(col[:, kt].astype(np.float32) @ W_LUM); lab_, nlab = ndi.label(hs > 0); dark = np.zeros(hs.shape, bool)
    for q in range(1, nlab + 1):
        comp = (lab_ == q) & (Lb > 0)
        if comp.any():
            dark |= comp & (Lb < 0.6 * np.median(Lb[comp]))
    lb2, n2 = ndi.label(dark); sz2 = ndi.sum(dark, lb2, range(1, n2 + 1))
    keep = 1 + np.where(sz2 >= 3)[0]; dark = np.isin(lb2, keep)
    tubes_rc = [tuple(c_) for c_ in ndi.center_of_mass(dark, lb2, keep)]
    for q in range(1, nlab + 1):
        comp = (lab_ == q) & (Lb > 0)
        if comp.any() and not (dark & comp).any():
            rc_ = np.argwhere(comp)[int(np.argmin(Lb[comp]))]
            if Lb[tuple(rc_)] < 0.6 * np.median(Lb[comp]):
                tubes_rc.append((float(rc_[0]), float(rc_[1])))
    rec["tubes"].update(mask_regions=int(nlab), tubes_marked=len(tubes_rc))
    # (g) coronal / sagittal / axial, each its brain box (tissue mm; x 1.08 at the print scale)
    ext_ax = (Af2[0, 3] - Af2[0, 0] / 2, Af2[0, 3] + (fin2.shape[0] - 0.5) * Af2[0, 0],
              Af2[1, 3] + (fin2.shape[1] - 0.5) * Af2[1, 1], Af2[1, 3] - Af2[1, 1] / 2)
    views = [("coronal", np.rot90(fin2[:, kmid]), P.extent_coronal(fin2[:, kmid], Af2)),
             ("sagittal", np.rot90(fin2[xg]), P.extent_sagittal(fin2[xg], Af2)),
             ("axial", np.transpose(fin2[:, :, zg], (1, 0, 2)), None)]                   # rows: sections, anterior first
    gbase = []
    for nm, img, e_ in views:
        if nm == "axial":                                             # rows = sections (anterior at the top), columns = x
            e_ = (ext_ax[0], ext_ax[1], min(ext_ax[2], ext_ax[3]), max(ext_ax[2], ext_ax[3]))
        cx_, cz_, ww_, hh_ = P.bbox_mm(img.astype(np.float32).sum(-1), e_)
        gbase.append((nm, img, e_, (cx_, cz_), ww_, hh_))

    # =============================== layout (mm) ===============================
    # the author's layout (15 Oct): a grid of three columns, (a, d, g coronal), (b, e, g sagittal), (c, f, g axial). Each cell is a
    # rectangle of its column's width; its content (images, direction letters, the trace's labels) is centered horizontally in
    # it, and its letter and topic start at its left edge, so the letters line up down each column. A column is as wide as its
    # widest cell, and the grid spans the page between the margins with the free width shared as equal column spaces. The print
    # scales are the house ones (the author, 15 Oct: "keep it consistent"); --fit-width instead multiplies both by the factor K
    # that leaves GAP_C between the columns (626c1ea).
    SUB, LEFT, TOP1, TR, GAP_C = 1.2, 5.0, 11.0, 14.0, 6.0             # TR: (e)'s trace height; GAP_C: the column space (--fit-width)
    ESUB = SUB + 4.0                                                    # (e): room for its two column headers
    RIGHT = FS.FULL_W_IN / FS.MM - LEFT
    COLS = (("a", "d", "g0"), ("b", "e", "g1"), ("c", "f", "g2"))

    def sublab(ax, s, color=None):
        """A two-image cell's sub-label above its own image (FS.header at dx 0, dy 2.5); not part of the cell's measured width."""
        t_ = ax.annotate(s, xy=(0, 1), xycoords="axes fraction", xytext=(0, 2.5), textcoords="offset points", annotation_clip=False,
                         fontsize=FS.MIN_PT, ha="left", va="bottom", color=color or FS.FG)
        t_.set_in_layout(False)

    def render(K):
        """The figure at print scales K x (WS, ZS), every cell drawn at x = LEFT (the grid then moves it). Returns the figure, each
        cell's axes and the rows' tops and heights (mm)."""
        ws, zs = WS * K, ZS * K
        aw, ah = sw * 1.12 * ws, sh * 1.12 * ws
        cpw, cph = (yc1 - yc0 + 2.0) * ws, (zt_c - zb_c + 2.0) * ws
        H1 = max(ah, cph); bw_, bh_ = bspan * zs, H1
        dw, dh = wb * 1.1 * ws, hb * 1.25 * ws
        fw, fh = max(wh + 4, 0.75 * (hh + 4)) * zs, (hh + 4) * zs
        ew, eh = espan * zs, max(16.0, max(dh, fh) - TR - 7.0)          # (e): reslice height leaves room for the trace
        H2 = max(dh, fh, eh + 2.0 + TR)
        gbox = [(nm, img, e_, ctr, ww_ * 1.08 * ws, hh_ * 1.08 * ws) for nm, img, e_, ctr, ww_, hh_ in gbase]
        H3 = max(b_[5] for b_ in gbox)
        TOP2 = TOP1 + H1 + 15.0; TOP3 = TOP2 + H2 + 15.0
        fig = plt.figure(figsize=(FS.FULL_W_IN, (TOP3 + H3 + 6.0) * FS.MM))
        cell, x_ = {}, LEFT

        # (a)
        ax = FS.axes_mm(fig, x_, TOP1, aw, ah); cell["a"] = [ax]
        ax.imshow(np.asarray(th), extent=pext, origin="upper", aspect="auto")
        ax.contour(np.linspace(0, pext[1], th.size[0]), np.linspace(pext[3], 0, th.size[1]), pm_th.astype(float), levels=[0.5],
                   colors=[FS.COL["outline"]], linewidths=0.6)
        window(ax, pc, aw / ws, ah / ws); FS.dirs(ax, "coronal"); FS.scalebar(ax, 1.0, 10.0)

        # (b)
        axb = []
        for n, (img, vt, vb, lab) in enumerate(((P.rgb01(rs[:, ob]), v_top, 0.0, "photographs"),
                                                 (P.rgb01(sg[:, ob]), ztop, zbot, "aligned"))):
            ax = FS.axes_mm(fig, x_ + n * (bw_ + SUB), TOP1, bw_, bh_); axb.append(ax)
            strip(ax, img, yb[ob], vt, vb, bspan, bh_ / zs)
            arrow_up(ax, yj)
            sublab(ax, lab)
            if n == 0:
                FS.scalebar(ax, 1.0, 5.0, y_frac=0.16)
        FS.dirs_row(axb[0], axb[1], "sagittal"); cell["b"] = axb

        # (c) every interpolated section of the stack: orange ticks below a sagittal reslice of the aligned stack (camera column)
        ax = FS.axes_mm(fig, x_, TOP1, cpw, cph); cell["c"] = [ax]
        ax.imshow(P.rgb01(sgc[:, oa]), extent=(yall[oa][0], yall[oa][-1], zbot, ztop), origin="upper", aspect="auto")
        window(ax, (0.5 * (yc0 + yc1), 0.5 * (zt_c + zb_c)), cpw / ws, cph / ws)
        ax.vlines(section_y(As, interp), -0.10, -0.015, transform=mtrans.blended_transform_factory(ax.transData, ax.transAxes),
                  colors=[FS.COL["pred"]], lw=0.6, clip_on=False)                # item 12: thicker, taller ticks
        FS.dirs(ax, "sagittal")

        # (d) the 3D mask on the unmasked aligned stack (the 3D model's grid), sagittal
        ax = FS.axes_mm(fig, x_, TOP2, dw, dh); cell["d"] = [ax]
        ax.imshow(P.rgb01(np.rot90(bg[xm])) * np.where(np.rot90(mk[xm]), 1.0, DIM)[..., None], extent=ext_b, origin="upper",
                  aspect="auto")
        ys_ = np.linspace(ext_b[0], ext_b[1], bg.shape[1]); zz_ = np.linspace(ext_b[3], ext_b[2], bg.shape[2])
        ax.contour(ys_, zz_, np.rot90(mk[xm]).astype(float), levels=[0.5], colors=[FS.COL["outline"]], linewidths=0.6)
        window(ax, (cxb, czb), dw / ws, dh / ws); FS.dirs(ax, "sagittal")
        FS.scalebar(ax, 1.0, 10.0)

        # (e)
        axe = []
        for n, (img, vt, vb, lab) in enumerate(((np.clip(re_[:, oe] / np.percentile(re_[re_ > 0], 99.5), 0, 1), v_top, 0.0, "photographs"),
                                                 (P.rgb01(sge[:, oe]), ztop, zbot, "corrected"))):
            ax = FS.axes_mm(fig, x_ + n * (ew + ESUB), TOP2, ew, eh); axe.append(ax)
            strip(ax, img, ye[oe], vt, vb, espan, eh / zs)
            sublab(ax, lab, FS.COL["uncorrected"] if n == 0 else FS.COL["corrected"])        # item 12: the key, as the traces
            arrow_up(ax, ys0, gap=0.2 if n == 0 else 0.4, size=4.0 if n == 0 else 4.5)
        FS.dirs_row(axe[0], axe[1], "sagittal")
        axp = FS.axes_mm(fig, x_, TOP2 + eh + 2.0, ew, TR)
        axp.plot(ye[oe], rel_raw[oe], color=FS.COL["uncorrected"], lw=0.7)          # item 12: distinct lookup colors
        axp.plot(ye[oe], rel_cor[oe], color=FS.COL["corrected"], lw=0.9)
        lo_ = np.floor((np.nanmin(rel_raw) - 0.02) * 20) / 20; hi_ = np.ceil((np.nanmax(rel_raw) + 0.01) * 50) / 50
        axp.set_xlim(*axe[0].get_xlim()); axp.set_ylim(min(lo_, 0.95), max(hi_, 1.04))      # every photographed value inside
        # the key (the author, 2 Oct: color-coded, with a key): two fixed lines right of the plot, a line sample + the name in its
        # color, independent of the traces' values (a value-anchored label can run into the reslices above)
        for q_, (t_, c_) in enumerate((("— photographs", FS.COL["uncorrected"]), ("— corrected", FS.COL["corrected"]))):
            axp.annotate(t_, xy=(1, 0.80 - 0.28 * q_), xycoords="axes fraction", xytext=(3, 0), textcoords="offset points",
                         ha="left", va="center", fontsize=FS.MIN_PT, color=c_, annotation_clip=False)   # right of the plot
        xl = sorted(axp.get_xlim()); dd = [v for v in range(-30, 31, 5) if xl[0] <= ys0 + v <= xl[1]]
        axp.set_xticks([ys0 + v for v in dd]); axp.set_xticklabels([f"{v:d}".replace("-", "−") for v in dd])
        axp.set_xlabel("distance from the step (mm)", labelpad=1.5)
        axp.axvline(ys0, color=FS.GRAY, lw=0.5, ls=(0, (1, 1.5)), zorder=0)
        yl_ = axp.get_ylim(); axp.set_yticks([t for t in np.arange(0.5, 1.21, 0.1) if yl_[0] - 1e-9 <= t <= yl_[1] + 1e-9])
        axp.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f")); axp.tick_params(length=2, pad=1)
        for sp in ("top", "right"):
            axp.spines[sp].set_visible(False)
        axp.annotate("median\nbrightness\n(relative)", xy=(0, 0.5), xycoords="axes fraction", xytext=(-15, 0), textcoords="offset points",
                     ha="right", va="center", multialignment="right", fontsize=FS.MIN_PT, linespacing=1.1, annotation_clip=False)
        cell["e"] = axe + [axp]
        rec["intensity"].update(relative_median_sd=dict(photographs=sd(med_raw), corrected=sd(med_cor), sections="photographed only"),
                                sections_interpolated_in_window=[int(k) for k in ke if k in set(interp)],
                                trace_ylim=[round(float(v), 3) for v in axp.get_ylim()],
                                photographs_min=dict(relative=round(float(np.nanmin(rel_raw)), 3), section=int(ke[int(np.nanargmin(rel_raw))])))

        # (f)
        axf = []
        for n, (vol, A_, lab) in enumerate(((col, Ac, "before"), (fin, Af, "filled"))):
            ax = FS.axes_mm(fig, x_ + n * (fw + SUB), TOP2, fw, fh); axf.append(ax)
            e_ = P.extent_coronal(vol[:, kt], A_)
            ax.imshow(soft(P.rgb01(np.rot90(vol[:, kt])), *SOFT_F), extent=e_, origin="upper", aspect="auto")
            xs_ = np.linspace(ext_f[0], ext_f[1], col.shape[0]); zz2 = np.linspace(ext_f[3], ext_f[2], col.shape[2])
            for r_t, c_t in tubes_rc:                                  # an arrowhead left of each tube, its tip 0.25 mm short of the center
                ax.plot([np.interp(c_t, np.arange(xs_.size), xs_) - 0.7], [np.interp(r_t, np.arange(zz2.size), zz2)], marker=">", ms=3.0,
                        mew=0, ls="none", color=FS.COL["pred"])
            window(ax, (cxh, czh), fw / zs, fh / zs)
            sublab(ax, lab)
            if n == 0:
                FS.scalebar(ax, 1.0, 5.0)
        cell["f"] = axf

        # (g) the reconstructed volume: coronal, sagittal, axial, each centered vertically on the row (the axial is the tallest)
        for q, (nm, img, e_, ctr, ww_, hh_) in enumerate(gbox):
            ax = FS.axes_mm(fig, x_, TOP3 + (H3 - hh_) / 2, ww_, hh_); cell[f"g{q}"] = [ax]
            ax.imshow(soft(P.rgb01(img), *SOFT_G), extent=e_, origin="upper", aspect="auto")
            window(ax, ctr, ww_ / ws, hh_ / ws)
            if q == 0:
                FS.dirs(ax, "coronal"); FS.scalebar(ax, 1.0, 10.0)
            elif nm == "sagittal":
                FS.dirs(ax, "sagittal")
            else:                                                      # axial: anterior up, letters above and below
                for xy, off, t_, va in (((0.5, 1), (0, 2), "A", "bottom"), ((0.5, 0), (0, -2), "P", "top")):
                    ax.annotate(t_, xy=xy, xycoords="axes fraction", xytext=off, textcoords="offset points", ha="center", va=va,
                                fontsize=FS.MIN_PT, color=FS.GRAY, annotation_clip=False)
        return fig, cell, dict(top=(TOP1, TOP2, TOP3), h=(H1, H2, H3))

    def span(fig, axs, frames_only=False):
        """A cell's horizontal extent (print mm): its frames and everything printed beside them (letters, ticks, labels, key);
        the sub-labels above two-image cells are left out (not in layout)."""
        r_ = fig.canvas.get_renderer()
        bb_ = mtrans.Bbox.union([ax.get_window_extent(r_) if frames_only else ax.get_tightbbox(r_) for ax in axs])
        return bb_.x0 / fig.dpi * 25.4, bb_.x1 / fig.dpi * 25.4

    K = 1.0                                                            # the house print scales
    fig, cell, rows = render(K)
    ext = {s: span(fig, v) for s, v in cell.items()}
    Wc = [max(ext[s][1] - ext[s][0] for s in col_) for col_ in COLS]
    G = (RIGHT - LEFT - sum(Wc)) / 2
    assert G >= 3.0, f"columns too wide: spaces {G:.1f} mm (columns {[round(w, 1) for w in Wc]})"
    Xc = [LEFT, LEFT + Wc[0] + G, LEFT + Wc[0] + G + Wc[1] + G]
    Wf = fig.get_size_inches()[0] * 25.4
    for i, col_ in enumerate(COLS):
        for s in col_:
            dx = Xc[i] + Wc[i] / 2 - (ext[s][0] + ext[s][1]) / 2             # the cell's content centered in its rectangle
            for ax in cell[s]:
                p_ = ax.get_position(); ax.set_position([p_.x0 + dx / Wf, p_.y0, p_.width, p_.height])
    # each rectangle's header: the letter and topic at its left edge; a one-image cell's sub-label under them (the two-image cells'
    # sub-labels sit above their own images)
    HEAD = dict(a=("photograph", ""), b=("camera jump", ""), c=("interpolated sections", "aligned stack, sagittal"), d=("3D mask", ""),
                e=("intensity correction", ""), f=("marker tubes, coronal", ""), g0=("reconstructed volume", "coronal"),
                g1=("", "sagittal"), g2=("", "axial"))
    for r_, row in enumerate(zip(*COLS)):
        for i, s in enumerate(row):
            rect = FS.axes_mm(fig, Xc[i], rows["top"][r_], Wc[i], rows["h"][r_]); rect.set_axis_off()
            head2(rect, *HEAD[s], s[0] if s[0] != "g" or s == "g0" else None)
    rec["print_scale"] = dict(whole=round(WS * K, 4), zoom=round(ZS * K, 4), factor=round(K, 4), house=dict(whole=WS, zoom=ZS))
    rec["grid"] = dict(columns_mm=[round(w, 2) for w in Wc], space_mm=round(G, 2), margin_mm=LEFT)
    ins = [f"{bd}/00_slices", f"{bd}/01_brainmask/01_final", f"{bd}/{S}.info", f"{bd}/02_blockface.ome.zarr", c.color_store,
           c.final_zarr("blockface"), c.final("holes"), f"{bd}/curated/{S}_bg.nii.gz", f"{bd}/curated/{S}_mask_edited.nii.gz"]
    FS.save(fig, a.out, ins, " ".join(["reproduce/figures/reconstruction.py"] + sys.argv[1:]), md5=False, extra=rec)
    print(json.dumps(rec, indent=1, default=float))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
