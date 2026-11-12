#!/usr/bin/env python3
"""Fig. 10, the tracer figure: three injections in the template, each charted section's labels where it lands (a) and the bundles at
the right internal capsule (b), with a rotation video of (a) for the supplement.

You need (settings/internal/products_mri.yaml, settings/internal/products_figures.yaml)
  tracer_sections_v2: each charted section's labels carried into the template without interpolation (footprints of the bundles and
  injection sites, the section index, the points file); tracer_template_v2: the bundle label interpolated between sections, as a
  density, read only for a check in the record (not drawn); template_t2, template_mask, template_fa, sarm_level6;
  settings/env/viz3d.yml (pyvista on a headless VTK; PY_VIZ3D).

What it does
  Nothing is interpolated between sections in either panel: both draw each charted section's labels where that section lands.
  (a) one point per template voxel of each section's footprint (carried fraction >= 0.1): fiber bundles (all grades; the "Fibers"
      class left out) and, as larger points with one white edge, injection sites, named outside the image. Nothing is interpolated,
      joined, smoothed, thinned or cut. MF8's case is in the left hemisphere and is mirrored onto the right (x -> -x; the grid is
      centered on x = 0). Parallel projection from the right and front (turned 15 degrees toward the front and 15 up, so the
      coronal plane shows its face without hiding MM2's labels behind it: at 30 degrees it hid 77 % of them, at 15 46 %, in
      the old view, 8 and 10, 18 %; the author, 12 Nov 2026); a turned view has no single scale, so (a) has no scale bar. The T2 on the midline plane and on one coronal plane (right half, y = 18 mm, the middle slab of b), brain cells only,
      on a dimmed linear window; the planes hide what lies behind them. The cases are drawn in layers, MM2 over MM1 over MF8 (each
      case rendered alone with the planes, then stacked), so the smaller cases are not hidden by MF8; within a case, nearer points
      hide farther ones (--by-depth draws all three by depth instead). A thin gray line is the outline of the whole brain as the
      camera sees it (the brain mask's surface rendered with the same camera).
  (b) three 0.5 mm coronal slabs of FA at y = 16, 18 and 20 mm (--alic-y; a's coronal plane follows the middle one), with the same
      charted labels as (a): each case's bundle footprints on the three 0.25 mm template planes that overlap the slab, one square per
      template voxel, in layers MF8 < MM1 < MM2 with each case's outline on top. The sections are oblique to the coronal planes, so
      each one crosses a slab in a band. Dotted, the internal capsule of the SARM atlas (level 6, label 50) on the slab's middle
      plane. The panels' box (x 1.5-12.5, z 13.5-24.5 mm) holds the whole right capsule at each slab.

You get
  <out>.png (180 mm wide) + <out>.provenance.json and <out stem>.pdf, <out stem>_scene.png (the 3D still), with --video
  <out stem>_rotation.mp4 (one turn, 12 s, 30 fps, the outline drawn per frame), and $LINC_NUMBERS/fig_tracer_sections3d_v6.json:
  points and sections per case, the points outside the brain mask, the share of each case hidden by the planes and by the other
  cases, the planes, the projection, the mirror, and per slab and case the charted voxels of (b), how many lie in the capsule (each
  against the capsule on its own plane) and the sections they come from; check_*: how much of the interpolated label (density over
  the slab > 0.5) lies outside the capsule and how much of that the charted labels cover.

    $PY_VIZ3D code/figures/tracer_sections3d.py [--out figures/tracer_sections3d.png] [--video] [--by-depth] [--alic-y 16 18 20]
        [--cases MR243 MF191 MR252]
The cases drawn: settings/figures.yaml tracer_sections3d.cases (or --cases), by default MM1, MF8 and MM2. MM3 (MR256; the author,
12 Nov 2026: Jingjing's three cases are MM1, MM2 and MM3) can be drawn too, in the fourth tracer color (figstyle TRACER); it has no
annotated injection site, so no marker and no tag, and its key says so. The layers go from the largest case to the smallest
(LAYER_ORDER).
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pyvista as pv
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "code"))
import figstyle as FS  # noqa: E402
from shared import paths, provenance  # noqa: E402

NAME = "fig_tracer_sections3d_v6_h9"
P_MRI, P_FIG = paths.products("products_mri"), paths.products("products_figures")
SEC, V2 = Path(P_MRI["tracer_sections_v2"]), Path(P_MRI["tracer_template_v2"])
T2_F, MASK_F, FA_F, SARM = (Path(P_FIG[k]) for k in ("template_t2", "template_mask", "template_fa", "sarm_level6"))
IC_LABEL = 50
CASES = ["MR243", "MF191", "MR252"]                                  # MM1, MF8, MM2 (key order); settings/figures.yaml or --cases
AREA = {"MR243": "area 10", "MF191": "area 13m", "MR252": "SMA"}
KEY_NOTE = {"MR256": "injection site not annotated"}               # a case without an annotated injection site: its key's note
COLOR = {S: FS.TRACER[S] for S in CASES}
MIRROR = {"MF191"}
X_SAG, Y_COR = 0.0, 18.0                                             # mm: the midline plane; the coronal plane (right half) through (b)'s middle slab
LO_PCT, HI_PCT, PLANE_DIM = 5.0, 99.0, 0.55                          # the planes' linear window (brain percentiles), dimmed
ALIC_Y = [16.0, 18.0, 20.0]
ALIC_X, ALIC_Z = (1.5, 12.5), (13.5, 24.5)                           # mm, the capsule panels' box (as the present panel e)
FA_DIM = 0.6
WIN = (4200, 2800)                                                   # the still's render window, px
POINT = 7.0                                                          # px at WIN; injection points x INJ
INJ = 1.7
VIEW_FOC, VIEW_UP = (6.0, 10.0, 14.0), (0.0, 0.0, 1.0)
VIDEO_POS = (150.0, 30.0, 40.0)                                      # the rotation video's start (the still's camera before 12 Nov 2026)
VIEW_AZ_EL = (15.0, 15.0)                                            # the still: from the right, turned toward the front and up (degrees)
_R = float(np.linalg.norm(np.subtract(VIDEO_POS, VIEW_FOC))); _a, _e = np.radians(VIEW_AZ_EL)
VIEW_POS = tuple(round(float(f_ + _R * d), 3) for f_, d in zip(VIEW_FOC, (np.cos(_e) * np.cos(_a), np.cos(_e) * np.sin(_a), np.sin(_e))))
PSCALE = 26.0                                                        # mm, half the window's height (parallel projection)
FILL_A = 0.8
EDGE_UP = 8                                                           # (b) outlines traced on an 8x finer copy of the voxel grid
LAYER_ORDER = ["MF191", "MR243", "MR256", "MR252"]                  # --layered: bottom to top, the largest case first (bundle voxels
                                                                     #   25936, 8621, 5087, 2145): the smaller cases over the larger
LAYERS = [S for S in LAYER_ORDER if S in CASES]
PAL = {"MR243": (255, 0, 0), "MF191": (0, 255, 0), "MR252": (0, 0, 255), "MR256": (255, 0, 255)}
pv.OFF_SCREEN = True


def load(p):
    im = nib.load(str(p)); return np.asanyarray(im.dataobj), im.affine


# --------------------------------------------------------------------------- the scene
def scene_data():
    T2, A = load(T2_F); brain = load(MASK_F)[0] > 0
    vox = float(A[0, 0])
    assert np.allclose(A[:3, :3], np.diag([vox] * 3)) and abs(A[0, 3] + (T2.shape[0] - 1) * vox / 2) < 1e-6, "grid must be centered on x = 0"
    T2 = T2.astype(np.float32)
    lo, hi = (float(v) for v in np.percentile(T2[brain], [LO_PCT, HI_PCT]))
    gray = np.where(brain, np.clip((T2 - lo) / (hi - lo), 0, 1) * PLANE_DIM, 0.0).astype(np.float32)
    ins = [T2_F, MASK_F]
    pts, rec = {}, {}
    for S in CASES:
        pts[S] = {}
        for g in ("Bundles", "Injection_site"):
            f = SEC / S / f"{S}_{g}_mri-template_sections_footprint.nii.gz"
            if g == "Injection_site" and not f.exists() and S in KEY_NOTE:          # none annotated (MM3)
                ijk = np.zeros((0, 3), int)
            else:
                a, Aa = load(f); assert np.allclose(Aa, A), f; ins.append(f)
                ijk = np.argwhere(a > 0)
            xyz = nib.affines.apply_affine(A, ijk.astype(float))
            inb = brain[tuple(ijk.T)]
            if S in MIRROR:
                xyz[:, 0] = -xyz[:, 0]
            pts[S][g] = xyz
            rec.setdefault(S, {})[g] = dict(points=int(len(xyz)), outside_brain_mask=int((~inb).sum()),
                                            share_outside_brain_mask=round(float((~inb).mean()), 5) if len(xyz) else None,
                                            share_left_of_midline_as_drawn=round(float((xyz[:, 0] < 0).mean()), 4) if len(xyz) else None)
        f = SEC / S / f"{S}_sections_points.csv"; ins.append(f)
        rows = list(csv.DictReader(open(f)))
        for g in ("Bundles", "Injection_site"):
            rec[S][g]["sections"] = len({r["section"] for r in rows if r["group"] == g})
            rec[S][g]["points_in_csv"] = sum(r["group"] == g for r in rows)       # the per-section points file, for comparison
        rec[S]["sections_with_bundles_or_injection"] = len({r["section"] for r in rows if r["group"] in ("Bundles", "Injection_site")})
    return dict(A=A, vox=vox, brain=brain, gray=gray, window=(lo, hi), pts=pts, rec=rec, inputs=ins)


def textured_plane(gray, A, axis, pos_mm, keep, brain):
    grid = pv.ImageData(dimensions=np.array(gray.shape) + 1, spacing=np.diag(A)[:3], origin=A[:3, 3] - np.diag(A)[:3] / 2)
    grid.cell_data["rgb"] = np.repeat((np.clip(gray, 0, 1).ravel(order="F") * 255).astype(np.uint8)[:, None], 3, axis=1)
    grid.cell_data["brain"] = brain.ravel(order="F").astype(np.uint8)
    origin = [0.0, 0.0, 0.0]; origin[axis] = pos_mm
    sl = grid.slice(normal=["x", "y", "z"][axis], origin=origin).clip_box(keep, invert=False)
    return sl.threshold(0.5, scalars="brain", preference="cell")


def brain_surface(brain, A):
    g = pv.ImageData(dimensions=brain.shape, spacing=np.diag(A)[:3], origin=A[:3, 3])
    g.point_data["m"] = brain.ravel(order="F").astype(np.float32)
    return g.contour([0.5], scalars="m")


def set_camera(p, size, pos=VIEW_POS, foc=VIEW_FOC, up=VIEW_UP, pscale=PSCALE):
    p.camera.position, p.camera.focal_point, p.camera.up = pos, foc, up
    p.enable_parallel_projection(); p.camera.parallel_scale = pscale; p.camera.clipping_range = (1.0, 600.0)


def project(xyz, size, pos=VIEW_POS, foc=VIEW_FOC, up=VIEW_UP, pscale=PSCALE):
    """World mm -> image pixel (column, row from the top) for the parallel camera; also the depth along the view."""
    pos, foc, up = (np.asarray(v, float) for v in (pos, foc, up))
    d = foc - pos; d /= np.linalg.norm(d)
    u = up - d * up.dot(d); u /= np.linalg.norm(u)
    r = np.cross(d, u)
    W, H = size; k = H / (2.0 * pscale)
    q = np.asarray(xyz, float) - foc
    return np.stack([W / 2.0 + q.dot(r) * k - 0.5, H / 2.0 - q.dot(u) * k - 0.5], -1), q.dot(d)


def build(p, sd, mode, size, ps, only=None, planes=True, alpha=(1.0, 1.0)):
    """mode: 'color' (the figure), 'id' (flat, one pure color per case, planes black), 'inj' (as 'id' but the bundles black: the
    visible injection sites), 'sil' (the brain's surface, flat white)."""
    p.set_background("black")
    acts = {}
    if mode == "sil":
        p.add_mesh(sd["surf"], color="white", lighting=False)
        return acts
    if planes:
        for k, m in (("sag", sd["sag"]), ("cor", sd["cor"])):
            if mode == "color":
                acts[k] = p.add_mesh(m, scalars="rgb", rgb=True, lighting=False, opacity=alpha[0 if k == "sag" else 1])
            else:
                acts[k] = p.add_mesh(m, color="black", lighting=False)
    ids = {S: "#%02x%02x%02x" % PAL[S] for S in PAL}
    for S in CASES:
        if only is not None and S not in only:
            continue
        for g, f in (("Bundles", 1.0), ("Injection_site", INJ)):
            xyz = sd["pts"][S][g]
            if not len(xyz):
                continue
            if mode == "color":
                p.add_mesh(pv.PolyData(xyz), color=COLOR[S], point_size=ps * f, render_points_as_spheres=True, ambient=0.3, specular=0.25)
            else:
                c = "black" if (mode == "inj" and g == "Bundles") else ids[S]
                p.add_mesh(pv.PolyData(xyz), color=c, point_size=ps * f, render_points_as_spheres=True, lighting=False)
    return acts


def plotter(size, ms):
    """An off-screen plotter with ms-sample MSAA (1 = none: the flat id and outline passes must not blend colors); no SSAA."""
    pv.global_theme.multi_samples = ms
    return pv.Plotter(off_screen=True, window_size=size)


def shot(sd, mode, size=WIN, ps=POINT, only=None, planes=True):
    p = plotter(size, 8 if mode == "color" else 1)
    build(p, sd, mode, size, ps, only=only, planes=planes); set_camera(p, size)
    img = p.screenshot(None, return_img=True); p.close()
    return np.asarray(img)[..., :3]


def id_mask(img, S):
    return np.abs(img.astype(int) - np.array(PAL[S])).max(-1) < 60


def id_counts(img):
    """Pixels of each case in an 'id' render (its pure color, within a tolerance)."""
    return {S: int(id_mask(img, S).sum()) for S in PAL}


def layer(imgs, planes_img):
    """--layered: each case rendered alone with the planes (so the planes hide what lies behind them), the cases then stacked
    MF8, MM1, MM2 from the bottom; returns the image and each upper case's drawn pixels."""
    out = imgs[LAYERS[0]].copy(); drawn = {}
    for S in LAYERS[1:]:
        m = np.abs(imgs[S].astype(int) - planes_img.astype(int)).max(-1) > 8
        out[m] = imgs[S][m]; drawn[S] = m
    return out, drawn


# --------------------------------------------------------------------------- panel b data
def capsule(sd):
    """Each slab: the FA (the 0.5 mm box centered on y), the capsule on the middle plane, and each case's charted bundle footprint on
    the three planes that overlap the slab (their union, in the box). Counts in 3D, each voxel against the capsule on its own plane."""
    A, vox = sd["A"], sd["vox"]
    fa = load(FA_F)[0].astype(np.float32); sarm = load(SARM)[0]
    assert fa.shape == sd["brain"].shape and sarm.shape == fa.shape
    ic3 = sarm == IC_LABEL
    near = ndi.binary_dilation(ic3, iterations=2)                                   # within 0.5 mm of the capsule (two voxels)
    FA = np.clip(fa / np.percentile(fa[sd["brain"]], 99.5) * FA_DIM, 0, 1)
    xw = A[0, 3] + vox * np.arange(fa.shape[0]); yw = A[1, 3] + vox * np.arange(fa.shape[1]); zw = A[2, 3] + vox * np.arange(fa.shape[2])
    xr = (xw >= ALIC_X[0] - 1e-6) & (xw <= ALIC_X[1] + 1e-6); zr = (zw >= ALIC_Z[0] - 1e-6) & (zw <= ALIC_Z[1] + 1e-6)
    crop2 = lambda P: P[xr][:, zr]                                                  # noqa: E731
    crop3 = lambda P: P[xr][:, :, zr]                                               # noqa: E731
    slab = lambda V, j: 0.25 * V[:, j - 1, :] + 0.5 * V[:, j, :] + 0.25 * V[:, j + 1, :]   # noqa: E731  the 0.5 mm box centered on y
    F, I, D = {}, {}, {}
    for S in CASES:
        f = load(SEC / S / f"{S}_Bundles_mri-template_sections_footprint.nii.gz")[0] > 0
        i = load(SEC / S / f"{S}_section_index_mri-template.nii.gz")[0]
        d = load(V2 / S / f"{S}_Bundles_mri-template_density.nii.gz")[0].astype(np.float32)   # the check only: not drawn
        F[S], I[S], D[S] = ((np.ascontiguousarray(v[::-1]) if S in MIRROR else v) for v in (f, i, d))
    out, rec = [], {}
    for y in ALIC_Y:
        j = int(np.argmin(np.abs(yw - y))); assert abs(yw[j] - y) < 1e-6
        sl = slice(j - 1, j + 2)
        icb = crop3(ic3[:, sl, :]); ic = crop2(ic3[:, j, :])
        assert icb.sum() == ic3[:, sl, :][xw > 0].sum(), "the box must hold the whole right capsule at the slab"
        lab, r = {}, {}
        for S in CASES:
            Fs = F[S][:, sl, :]; Fb = crop3(Fs)
            lab[S] = Fb.any(1)
            secs = sorted({int(v) for v in I[S][:, sl, :][Fs]} - {0})
            interp = crop2(slab(D[S], j)) > 0.5; outside = interp & ~ic
            r[S] = dict(charted_voxels=int(Fb.sum()), charted_voxels_in_capsule=int((Fb & icb).sum()),
                        charted_voxels_within_0p5mm_of_capsule=int((Fb & crop3(near[:, sl, :])).sum()),
                        charted_voxels_whole_slab=int(Fs.sum()), charted_voxels_whole_slab_in_capsule=int((Fs & ic3[:, sl, :]).sum()),
                        sections=secs, pixels_drawn=int(lab[S].sum()),
                        check_interpolated_pixels=int(interp.sum()), check_interpolated_pixels_outside_capsule=int(outside.sum()),
                        check_of_those_covered_by_charted=int((outside & lab[S]).sum()),
                        check_charted_pixels_outside_capsule=int((lab[S] & ~ic).sum()))
        rec[f"{y:.1f}"] = dict(cases=r, capsule_voxels_three_planes=int(icb.sum()), capsule_pixels_middle_plane=int(ic.sum()))
        out.append(dict(y=y, fa=crop2(slab(FA, j)), ic=ic, lab=lab))
    ins = [FA_F, SARM] + [SEC / S / f"{S}_{k}" for S in CASES for k in
                                                ("Bundles_mri-template_sections_footprint.nii.gz", "section_index_mri-template.nii.gz")] \
        + [V2 / S / f"{S}_Bundles_mri-template_density.nii.gz" for S in CASES]
    return out, rec, ins


LAB_SIDE = {"MR243": "right", "MF191": "right", "MR252": "top"}               # (a)'s injection-site tags: outside the frame, a line to the site
TAG = {S: a_[0].upper() + a_[1:] for S, a_ in AREA.items()}                         # the tags start with a capital (sentence case)
TAG_X, TAG_Y = 0.015, 0.03                                                          # the tags' offsets outside (a)'s frame (axes fraction)
CROP_PAD = 0.0015                                                                   # (a)'s crop around the brain's outline (of the window): the line's width, no more
KEY_X = 0.08                                                                        # the key's texts start here (its width's fraction)
KEY_TEXTS = ["Injection site, white edge (a)", "Outline of the whole brain (a)", "Internal capsule, SARM atlas (b)",
             "One square per charted voxel (b)"]


def text_mm(plt):
    """a function (text, size, **font) -> (width, height) mm as figstyle draws it"""
    fig = plt.figure(figsize=(7, 2), dpi=FS.DPI); r = fig.canvas.get_renderer()   # figstyle's DPI: at 100, hinting widens a text
    def T(s, fs=FS.MIN_PT, **kw):
        t = fig.text(0, 0, s, fontsize=fs, **kw); e = t.get_window_extent(r); t.remove()
        return e.width / fig.dpi * 25.4, e.height / fig.dpi * 25.4
    return T


# --------------------------------------------------------------------------- the figure
def compose(sd, still, sil, cap, rec, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FS.setup()
    disp = lambda P: np.flipud(np.swapaxes(P, 0, 1))                                # noqa: E731  (x, z) -> rows = z down
    H, W = sil.shape
    rows, cols = np.nonzero(sil); m = int(CROP_PAD * W)
    r0, r1, c0, c1 = max(0, rows.min() - m), min(H, rows.max() + m + 1), max(0, cols.min() - m), min(W, cols.max() + m + 1)
    crop, silc = still[r0:r1, c0:c1], sil[r0:r1, c0:c1]
    k = H / (2.0 * PSCALE)                                                          # px per mm in the image plane
    FW, GUT, MG = 180.0, FS.GUTTER_MM, FS.MARGIN_MM
    T = text_mm(plt)
    pad = 2.0 / 72 * 25.4                                                           # figstyle.dirs's pad
    tag_w = max([T(TAG[S])[0] for S in CASES if LAB_SIDE.get(S) == "right"] or [0.0]); tag_h = T("SMA")[1]
    let_h = T("ab", FS.LETTER_PT, fontweight="bold")[1]
    LM = MG + T("P")[0] + pad                                                       # (a): the P left of it, the tags right of it
    wa = (FW - LM - MG - tag_w) / (1 + TAG_X); ha = wa * crop.shape[0] / crop.shape[1]
    TOPA = MG + max(let_h + 4.0 / 72 * 25.4, TAG_Y * ha + tag_h)                       # the panel letter or the top tag, whichever is higher
    keys_k = [f"{FS.label(S)}, {AREA.get(S) or KEY_NOTE[S]}" + (" (mirrored from the left)" if S in MIRROR else "") for S in CASES] + KEY_TEXTS
    KW = max(T(t)[0] for t in keys_k) / (1 - KEY_X)                                  # the key: its texts start at KEY_X of its width
    nb = len(cap); gapb = GUT; LB = MG + T("L")[0] + pad
    wb = (FW - LB - (nb - 1) * GUT - (T("R")[0] + pad) - GUT - KW - MG) / nb       # (b)'s tiles fill the row beside the key
    hb = wb * (ALIC_Z[1] - ALIC_Z[0]) / (ALIC_X[1] - ALIC_X[0])
    TOPB = TOPA + ha + GUT + 12.0 / 72 * 25.4 + max(let_h, (2.0 + 1.2) / 72 * 25.4 + T("2 mm")[1])   # (b)'s title line 12 pt above its
                                                                                    # tiles, or its bar's length above it (scalebar_title)
    FH = TOPB + hb + MG
    fig = plt.figure(figsize=(FW / 25.4, FH / 25.4))
    ax = FS.axes_mm(fig, LM, TOPA, wa, ha)
    ax.imshow(crop, interpolation="lanczos"); ax.set_axis_off()
    ax.contour(silc.astype(float), [0.5], colors=[FS.GRAY], linewidths=0.5)
    for S in CASES:                                                                 # a thin white edge around each visible injection site
        M = sd["inj_px"][S][r0:r1, c0:c1]
        if M.any():
            rr = int(round(POINT * INJ)); yy, xx = np.mgrid[-rr:rr + 1, -rr:rr + 1]
            M = ndi.binary_fill_holes(ndi.binary_closing(np.pad(M, rr), (xx ** 2 + yy ** 2) <= rr * rr))[rr:-rr, rr:-rr]   # one edge around the site's points
            ax.contour(M.astype(float), [0.5], colors=[FS.FG], linewidths=0.4)
    ax.set_xlim(-0.5, crop.shape[1] - 0.5); ax.set_ylim(crop.shape[0] - 0.5, -0.5)
    FS.letter(ax, "a", dx=(MG - LM) * 72 / 25.4, dy=4.0); FS.header(ax, "Every charted section's labels in the template, seen from the right and front", dx=(MG - LM) * 72 / 25.4 + 11, dy=4.0)
    FS.dirs(ax, "sagittal", y=0.25)                                                 # P and A only (no dorsal-ventral letters), low: the area tags
                                                                                    # sit at the sites' heights on the right
    # injection sites: the label outside the image, a thin line to the site's center (its points' mean, projected)
    rec["injection_label_anchor_px"] = {}
    for S in CASES:
        xyz = sd["pts"][S]["Injection_site"]
        if not len(xyz):
            continue
        uv, _ = project(xyz.mean(0)[None], (W, H)); u, v = uv[0][0] - c0, uv[0][1] - r0
        rec["injection_label_anchor_px"][S] = [round(float(u), 1), round(float(v), 1)]
        fu, fv = u / crop.shape[1], 1 - v / crop.shape[0]
        if LAB_SIDE[S] == "right":
            yl = min(max(fv, 0.05), 0.95); xl, xt, ha_, va_ = (1 + TAG_X / 2, yl), (1 + TAG_X, yl), "left", "center"
        else:
            xl0 = min(max(fu, 0.05), 0.95); xl, xt, ha_, va_ = (xl0, 1 + TAG_Y * 2 / 3), (xl0, 1 + TAG_Y), "center", "bottom"
        ax.annotate("", xy=(u, v), xycoords="data", xytext=xl, textcoords="axes fraction", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=FS.FG, shrinkA=0, shrinkB=3))   # the leader line, no text
        ax.text(*xt, TAG[S], transform=ax.transAxes, fontsize=FS.MIN_PT, color=FS.FG, ha=ha_, va=va_, clip_on=False)
    # panel b
    axs = []
    for q, c in enumerate(cap):
        bx = FS.axes_mm(fig, LB + q * (wb + gapb), TOPB, wb, hb); axs.append(bx)
        bg = disp(c["fa"]); bx.imshow(bg, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        for S in LAYERS:                                                          # one square per charted voxel, bottom to top
            M = disp(c["lab"][S])
            if M.any():
                rgba = np.zeros(M.shape + (4,)); rgba[..., :3] = matplotlib.colors.to_rgb(COLOR[S]); rgba[..., 3] = M * FILL_A
                bx.imshow(rgba, interpolation="nearest")
        for S in LAYERS:                                                          # then each case's outline along the voxel edges
            M = disp(c["lab"][S])
            if M.any():
                up = np.kron(M.astype(float), np.ones((EDGE_UP, EDGE_UP)))
                g = lambda n: (np.arange(n) + 0.5) / EDGE_UP - 0.5                 # noqa: E731
                bx.contour(g(up.shape[1]), g(up.shape[0]), up, [0.5], colors=[COLOR[S]], linewidths=0.5)
        bx.contour(disp(c["ic"].astype(float)), [0.5], colors=[FS.FG], linewidths=0.7, linestyles=[(0, (1, 1.2))])
        bx.set_xlim(-0.5, bg.shape[1] - 0.5); bx.set_ylim(bg.shape[0] - 0.5, -0.5); bx.set_axis_off()
        if q == 0:
            FS.letter(bx, "b", dx=(MG - LB) * 72 / 25.4, dy=12.0); tb = FS.header(bx, "The charted labels at the right internal capsule, on FA", dx=(MG - LB) * 72 / 25.4 + 11, dy=12.0)
        FS.header(bx, f"y = {c['y']:g} mm", dx=0, dy=2.5, color=FS.GRAY)
    FS.dirs_row(axs[0], axs[-1], "coronal")
    FS.scalebar_title(axs[-1], 1.0 / sd["vox"], 2.0, tb)                             # zoomed: the bar in its standard place, on (b)'s title line
    # the key, right of (b)
    kx = FW - MG - KW
    axk = FS.axes_mm(fig, kx, TOPB, KW, hb); axk.set_xlim(0, 1); axk.set_ylim(0, 1); axk.set_axis_off()
    rows_k = list(zip(CASES, keys_k))
    yk = 0.92
    for S, t in rows_k:
        axk.scatter([0.03], [yk], s=14, color=COLOR[S], clip_on=False); axk.text(KEY_X, yk, t, va="center", fontsize=FS.MIN_PT); yk -= 0.13
    axk.scatter([0.03], [yk], s=30, facecolor="none", edgecolor=FS.FG, lw=0.6); axk.text(KEY_X, yk, KEY_TEXTS[0], va="center", fontsize=FS.MIN_PT); yk -= 0.13
    axk.plot([0.0, 0.06], [yk, yk], color=FS.GRAY, lw=0.6); axk.text(KEY_X, yk, KEY_TEXTS[1], va="center", fontsize=FS.MIN_PT); yk -= 0.13
    axk.plot([0.0, 0.06], [yk, yk], color=FS.FG, lw=0.7, ls=(0, (1, 1.2))); axk.text(KEY_X, yk, KEY_TEXTS[2], va="center", fontsize=FS.MIN_PT); yk -= 0.13
    axk.text(KEY_X, yk, KEY_TEXTS[3], va="center", fontsize=FS.MIN_PT, color=FS.GRAY)
    rec["layout_mm"] = dict(width=FW, height=round(FH, 1), a=[round(LM, 2), round(TOPA, 2), round(wa, 2), round(ha, 2)], b_panel=[round(wb, 2), round(hb, 2)],
                            a_print_mm_per_tissue_mm=round(wa / crop.shape[1] * k, 4), b_print_mm_per_tissue_mm=round(wb / (ALIC_X[1] - ALIC_X[0]), 4),
                            crop_px=dict(rows=[int(r0), int(r1)], cols=[int(c0), int(c1)]))
    return fig


def main(argv=None):
    global ALIC_Y, Y_COR, CASES, COLOR, LAYERS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figures/tracer_sections3d.png", help="the figure; the still and the video are named after its stem")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--by-depth", action="store_true", help="draw the cases by depth instead of in layers (MF8 < MM1 < MM2)")
    ap.add_argument("--no-numbers", action="store_true", help="draft: do not write paper/numbers/<name>.json")
    ap.add_argument("--alic-y", type=float, nargs=3, default=ALIC_Y, help="(b)'s three slabs, mm; (a)'s coronal plane goes through the middle one")
    ap.add_argument("--cases", nargs="+", help="the cases drawn (default: settings/figures.yaml tracer_sections3d.cases, else MM1, MF8, MM2)")
    a = ap.parse_args(argv); t0 = time.time(); a.layered = not a.by_depth
    import yaml
    cfg = (yaml.safe_load((HERE.parents[1] / "settings" / "figures.yaml").read_text()) or {}).get("tracer_sections3d") or {}
    CASES = list(a.cases or cfg.get("cases") or CASES)
    assert set(CASES) <= set(FS.TRACER) and all(S in AREA or S in KEY_NOTE for S in CASES), f"cases {CASES}: no tracer color or key"
    COLOR = {S: FS.TRACER[S] for S in CASES}; LAYERS = [S for S in LAYER_ORDER if S in CASES]
    ALIC_Y, Y_COR = list(a.alic_y), a.alic_y[1]
    sd = scene_data(); A, brain = sd["A"], sd["brain"]
    bx = brain.any((1, 2)).nonzero()[0]; by = brain.any((0, 2)).nonzero()[0]; bz = brain.any((0, 1)).nonzero()[0]
    Wd = lambda i, k: float(A[k, 3] + i * A[k, k])                                  # noqa: E731
    box = (Wd(bx[0], 0) - 1, Wd(bx[-1], 0) + 1, Wd(by[0], 1) - 1, Wd(by[-1], 1) + 1, Wd(bz[0], 2) - 1, Wd(bz[-1], 2) + 1)
    sd["sag"] = textured_plane(sd["gray"], A, 0, X_SAG, box, brain)
    sd["cor"] = textured_plane(sd["gray"], A, 1, Y_COR, (X_SAG, box[1], box[2], box[3], box[4], box[5]), brain)
    sd["surf"] = brain_surface(brain, A)
    sil = shot(sd, "sil").max(-1) > 127
    alone = {S: shot(sd, "id", only=[S]) for S in CASES}                            # each case alone, with the planes
    inj1 = {S: id_mask(shot(sd, "inj", only=[S]), S) for S in CASES}
    hid = {}
    if a.layered:
        still, drawn = layer({S: shot(sd, "color", only=[S]) for S in CASES}, shot(sd, "color", only=[]))
        cover = {}
        for q, S in enumerate(LAYERS):
            above = np.zeros(sil.shape, bool)
            for T in LAYERS[q + 1:]:
                above |= drawn[T]
            cover[S] = above
        sd["inj_px"] = {S: inj1[S] & ~cover[S] for S in CASES}
        vis = {S: int((id_mask(alone[S], S) & ~cover[S]).sum()) for S in CASES}
    else:
        still = shot(sd, "color"); full = shot(sd, "id")
        sd["inj_px"] = {S: id_mask(shot(sd, "inj"), S) for S in CASES}
        vis = id_counts(full)
    print(f"rendered ({time.time() - t0:.0f} s)", flush=True)
    rec = dict(protocol=__doc__.split("\n\n")[0], cases=sd["rec"], mirrored=sorted(MIRROR), mirror="x -> -x (grid centered on x = 0)",
               projection="parallel", view=dict(position_mm=VIEW_POS, focal_point_mm=VIEW_FOC, up=VIEW_UP, half_height_mm=PSCALE),
               drawing=("layered: each case hidden only by the planes, the cases stacked MF8 < MM1 < MM2" if a.layered else "by depth: nearer points hide farther ones"),
               planes=dict(midline_x_mm=X_SAG, coronal_y_mm=Y_COR, coronal_half="right (x >= 0)", channel="template T2, brain cells only",
                           window=f"linear, brain percentiles {LO_PCT:g}-{HI_PCT:g} ({sd['window'][0]:.0f}-{sd['window'][1]:.0f}), times {PLANE_DIM}"),
               point_px_at_window=POINT, injection_point_factor=INJ, window_px=list(WIN))
    # what the still hides: by the planes (each case alone, with and without planes) and by the other cases
    for S in CASES:
        al = int(id_mask(alone[S], S).sum()); bare = id_counts(shot(sd, "id", only=[S], planes=False))[S]
        hid[S] = dict(pixels_alone_no_planes=bare, pixels_alone_with_planes=al, pixels_as_drawn=vis[S],
                      share_hidden_by_planes=round(1 - al / bare, 4) if bare else None,
                      share_hidden_by_other_cases=round(1 - vis[S] / al, 4) if al else None,
                      injection_pixels_alone=int(inj1[S].sum()), injection_pixels_as_drawn=int(sd["inj_px"][S].sum()))
        P = np.vstack([sd["pts"][S]["Bundles"], sd["pts"][S]["Injection_site"]])
        uv, _ = project(P, WIN); iu = np.clip(np.round(uv).astype(int), 0, [WIN[0] - 1, WIN[1] - 1])
        hid[S]["points_projected_outside_brain_outline"] = int((~sil[iu[:, 1], iu[:, 0]]).sum())
    rec["still_visibility_pixels"] = hid
    print(json.dumps(hid), flush=True)
    cap, crec, cins = capsule(sd); rec["capsule"] = crec; rec["alic_y_mm"] = ALIC_Y; rec["alic_slab_mm"] = 0.5
    tot = lambda S, k: sum(crec[y]["cases"][S][k] for y in crec)                    # noqa: E731
    rec["capsule_totals"] = {S: {k: tot(S, k) for k in ("charted_voxels", "charted_voxels_in_capsule", "charted_voxels_within_0p5mm_of_capsule",
                                                      "check_interpolated_pixels_outside_capsule", "check_of_those_covered_by_charted",
                                                      "check_charted_pixels_outside_capsule")} for S in CASES}
    rec["capsule_box_mm"] = dict(x=list(ALIC_X), z=list(ALIC_Z), slab_mm=0.5); rec["capsule_rule"] = ("charted bundle footprints (carried fraction >= 0.1, all grades) on the three 0.25 mm planes overlapping the 0.5 mm slab, drawn as their union; in the capsule = on the SARM label 50 of the voxel's own plane; counts within the panel's box x 1.5-12.5, z 13.5-24.5 mm, which holds the whole right capsule at each slab; check_* = the earlier panel (interpolated density over the slab > 0.5) against the capsule of the middle plane")
    print(json.dumps(crec), flush=True)
    rec["footprint_min_carried_fraction"] = 0.1                                     # the footprints' rule (the carrier of record, FOOT), for the caption
    out = Path(a.out); stem = out.with_suffix(""); out.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    Image.fromarray(still).save(f"{stem}_scene.png")
    fig = compose(sd, still, sil, cap, rec, a)
    ins = [str(x) for x in sd["inputs"] + cins]
    FS.save(fig, str(out), ins, cmd=" ".join(sys.argv))
    fig.savefig(f"{stem}.pdf", dpi=FS.DPI, facecolor=fig.get_facecolor())          # the same figure as a PDF (the still embedded at 600 ppi)
    print(f"wrote {out} ({time.time() - t0:.0f} s)", flush=True)
    if a.video:
        rec["video"] = video(sd, Path(f"{stem}_rotation.mp4"), layered=a.layered)
    if not a.no_numbers:
        print(provenance.write_numbers(NAME, rec, inputs=ins, md5=False, cmd=" ".join(sys.argv)))
    print(json.dumps(rec["cases"], indent=1))


def video(sd, f, size=(1600, 1072), n=360, layered=False):
    """The scene of (a) turning once about the vertical axis, drawn as the still (by depth, or layered), the brain's outline drawn
    per frame from the same camera; a plane the camera is behind is drawn at 0.3 opacity, so the labels behind it stay visible."""
    import imageio
    ps = POINT * size[1] / WIN[1]
    keys = (["planes"] + CASES) if layered else ["all"]
    pc, acts = {}, {}
    for k in keys:
        pc[k] = plotter(size, 8)
        acts[k] = build(pc[k], sd, "color", size, ps, only=([] if k == "planes" else [k]) if layered else None)
    pz = plotter(size, 1); build(pz, sd, "sil", size, ps)
    foc = np.asarray(VIEW_FOC); r = np.asarray(VIDEO_POS) - foc
    wr = imageio.get_writer(str(f), fps=30, quality=8, macro_block_size=16)
    for t in range(n):
        th = 2 * np.pi * t / n; c, s_ = np.cos(th), np.sin(th)
        pos = foc + np.array([c * r[0] - s_ * r[1], s_ * r[0] + c * r[1], r[2]])
        for k in keys:
            acts[k]["sag"].GetProperty().SetOpacity(1.0 if pos[0] >= X_SAG else 0.3)
            acts[k]["cor"].GetProperty().SetOpacity(1.0 if pos[1] >= Y_COR else 0.3)
        for p in list(pc.values()) + [pz]:
            set_camera(p, size, pos=tuple(pos)); p.render()
        grab = lambda p: np.asarray(p.screenshot(None, return_img=True))[..., :3].copy()   # noqa: E731
        if layered:
            img, _ = layer({S: grab(pc[S]) for S in CASES}, grab(pc["planes"]))
        else:
            img = grab(pc["all"])
        m = grab(pz).max(-1) > 127
        img[m & ~ndi.binary_erosion(m, iterations=2)] = (181, 181, 181)
        wr.append_data(img)
    wr.close()
    for p in list(pc.values()) + [pz]:
        p.close()
    return dict(frames=n, fps=30, size_px=list(size), drawing="layered" if layered else "by depth",
                turn="one full turn about the vertical axis through the view's focal point")


if __name__ == "__main__":
    main()
