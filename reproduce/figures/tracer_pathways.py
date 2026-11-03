#!/usr/bin/env python3
"""The paper's tracer figure (Fig. 10): panels a, b = two stills of the 3D scene (reproduce/figures/tracer_3d.py: the oblique and the
lateral view), c = the dorsal projection, d = the capsule sections (--layout composite, the default); --layout panels draws instead the
earlier four-panel form (a, b = the lateral projections on T2 and on the blockface channel). Panels c, d and the sections view:
the tracer pathways of three cases in the MRI template, drawn from the CONNECTED pathways of record (products_mri `tracer_template_v2c`),
the interpolated densities (`tracer_template_v2`) and the per-section sheets carried without interpolation (`tracer_sections_v2`).

    python reproduce/figures/tracer_pathways.py [--stills figures/tracer_3d_oblique.png figures/tracer_3d_lateral.png] [--out figures/tracer_pathways.png]
    python reproduce/figures/tracer_pathways.py --layout panels [--out figures/tracer_pathways_panels.png]

Three cases: MM1 (MR243, area 10) and MF8 (MF191, area 13m) are the similar pair, in two warm colors; MM2 (MR252, supplementary motor
area) is the different one, in green. MM3 (no injection site charted) is in Supplementary S2 only.
  a  lateral projection over a parasagittal slab of the right hemisphere (x in SLAB_X), on the template's T2 at the slab's middle plane
     (anterior right): the connected pathway filled with a thin dark edge, its inferred bridges under a dotted white edge, the footprint
     of the interpolated density (>= 0.1) as a light fill; the slab's maximum along x; gray: the projected brain outline
  b  the same over the template's blockface channel (the blockface template pulled into the MRI template by the link of record)
  c  dorsal projection of the right hemisphere, on the template's T2 at z = Z_AXIAL (anterior up)
  d  the anterior limb of the right internal capsule: 0.5 mm coronal slabs of the template's FA at ALIC_Y, the connected pathway inside
     the atlas capsule in full color, outside faded; dotted, the internal capsule of the SARM atlas (level 6, label 50)
Injection sites: the projected per-section Injection_site marks, closed over the gaps between sections (binary closing, radius CLOSE_MM),
filled in the case's color with a thin white edge. MF8's left-hemisphere case is mirrored onto the right for display (x -> -x; the
template is symmetric about x = 0); its data are unmirrored. The key gives each case's inferred share and longest bridge.
Each panel is a function (composite = the stills, lateral, dorsal, alic, key) over one loaded scene; main() draws either layout.
Conventions: figstyle (true scale, nothing printed over images, scale bars below). Outputs: <out> (+ .provenance.json); the record
$LINC_NUMBERS/fig_tracer_pathways.json (the layout, the stills' files and md5, the slabs, per case the pathway voxels drawn, the inferred
share and the longest bridge, the sections and their spacing, the capsule counts of (d)); the panels layout writes fig_tracer_pathways_panels.
Reads (products): products_figures template_t2 / template_mask / template_fa / sarm_level6; products_blockface template_link (the blockface
template in the MRI template); products_mri tracer_sections_v2, tracer_template_v2, tracer_template_v2c.
Ported from the analysis project's paper/fig_tracer_v2c_fig9.py over paper/scripts/fig_tracer_r061.py (3 Nov 2026); it renders the
paper's file byte for byte with the same matplotlib.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
import figstyle as FS  # noqa: E402
from linc import paths, provenance  # noqa: E402

NAME = "fig_tracer_pathways"
IC_LABEL = 50
CASES = ["MR243", "MF191", "MR252"]                       # MM1, MF8, MM2
DRAW_ORDER = ["MF191", "MR243", "MR252"]                  # bottom to top
MIRROR = {"MF191"}
COLOR = {"MR243": FS.OKABE["vermilion"], "MF191": FS.OKABE["yellow"], "MR252": FS.OKABE["green"]}   # the similar pair warm
AREA = {"MR243": "area 10", "MF191": "area 13m", "MR252": "SMA"}
SLAB_X = (0.0, 8.0)                                      # mm, right hemisphere, the parasagittal slab of (a)/(b)
Z_AXIAL = 18.0                                           # mm, the background plane of (c)
ALIC_Y = [20.0, 22.0, 24.0]
ALIC_X, ALIC_Z = (1.5, 12.5), (13.5, 24.5)
CLOSE_MM = 1.0
MIN_PX = 4                                               # projected injection components below this are not drawn
INJ_EDGE = 0.3                                           # pt, the thin white edge of the case-colored injection fill
MARK_S, MARK_A = 0.9, 0.85                               # marker area (pt^2) and alpha (the per-section marks of the sections view)
SCALE_AB, SCALE_C, SCALE_D = 1.05, 0.85, 2.2             # print mm per tissue mm
DIM, OUT_A, FILL_A = 0.62, 0.3, 0.9
DARK = "#111111"


def load(p):
    im = nib.load(str(p)); d, A = FS.canonical(im); return np.asarray(d), A


def points(SEC, S):
    rows = list(csv.DictReader(open(SEC / S / f"{S}_sections_points.csv")))
    out = {}
    for g in ("Bundles", "Injection_site"):
        P = np.array([[float(r["x_mm"]), float(r["y_mm"]), float(r["z_mm"]), int(r["section"])] for r in rows if r["group"] == g])
        if len(P) and S in MIRROR:
            P[:, 0] *= -1
        out[g] = P
    return out


def window(img, brain, dim=DIM):
    return np.clip(img / np.percentile(img[brain], 99.5) * dim, 0, 1)


def occupancy(P2, x0, y0, nx, ny, vox):
    """2-D occupancy of projected marks on a panel grid (rows = vertical axis top to bottom), closed over the section gaps."""
    G = np.zeros((ny, nx), bool)
    if not len(P2):
        return G
    i = np.clip(np.round((P2[:, 0] - x0) / vox).astype(int), 0, nx - 1)
    j = np.clip(np.round((y0 - P2[:, 1]) / vox).astype(int), 0, ny - 1)
    G[j, i] = True
    r = int(round(CLOSE_MM / vox)); yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    st = (xx ** 2 + yy ** 2) <= r * r
    O = ndi.binary_fill_holes(ndi.binary_closing(np.pad(G, r), st))[r:-r, r:-r]
    lab, n = ndi.label(O); sz = np.bincount(lab.ravel()); sz[0] = 0
    return np.isin(lab, np.nonzero(sz >= MIN_PX)[0])                            # single-pixel islands would draw as white specks


def draw_proj(ax, Dp, Sp, Ip, color, ext):
    """A projected pathway on a panel: the footprint (density >= 0.1) light, the connected pathway filled with a thin dark edge,
    the inferred bridges filled under a dotted white edge."""
    if (Dp >= 0.1).any():
        ax.contourf((Dp >= 0.1).astype(float), levels=[0.5, 1.5], colors=[color], alpha=0.35, extent=ext, origin="upper", zorder=3)
    if Sp.any():
        ax.contourf(Sp.astype(float), levels=[0.5, 1.5], colors=[color], alpha=0.95, extent=ext, origin="upper", zorder=3)
        ax.contour(Sp.astype(float), [0.5], colors=[DARK], linewidths=0.25, extent=ext, origin="upper", zorder=3)
    if Ip.any():
        ax.contour(Ip.astype(float), [0.5], colors=[FS.FG], linewidths=0.5, linestyles=[(0, (0.8, 0.8))], extent=ext, origin="upper", zorder=4)


# --------------------------------------------------------------------------- the scene: everything the panels read

def scene():
    """Loads the template channels, the atlas capsule, the per-section marks and the pathways of the three cases; returns a dict."""
    P1, P2, P3 = paths.products("products_blockface"), paths.products("products_mri"), paths.products("products_figures")
    SEC, V2, V2C = P2["tracer_sections_v2"], P2["tracer_template_v2"], P2["tracer_template_v2c"]
    BFT = P1["template_link"] / "d2bs8_s5sym_in_mri_template.nii.gz"
    t2, A = load(P3["template_t2"]); brain = load(P3["template_mask"])[0] > 0
    assert nib.aff2axcodes(A) == ("R", "A", "S") and np.allclose(A[:3, :3], np.diag(np.diag(A[:3, :3])))
    vox = float(A[0, 0])
    assert abs(A[0, 3] + (t2.shape[0] - 1) * vox / 2) < 1e-6 and np.array_equal(brain, brain[::-1]), "the template must be centered on x = 0"
    bft, Ab = load(BFT); assert bft.shape == t2.shape and np.allclose(Ab, A, atol=1e-4)
    fa = load(P3["template_fa"])[0].astype(np.float32)
    sarm = load(P3["sarm_level6"])[0]; assert sarm.shape == t2.shape
    xw = A[0, 3] + vox * np.arange(t2.shape[0]); yw = A[1, 3] + vox * np.arange(t2.shape[1]); zw = A[2, 3] + vox * np.arange(t2.shape[2])
    T2 = window(t2.astype(np.float32), brain); BF = window(bft.astype(np.float32), brain); FA = window(fa, brain, 0.6)
    pts = {S: points(SEC, S) for S in CASES}
    vol = {}; conn = {}
    for S in CASES:
        D = load(V2 / S / f"{S}_Bundles_mri-template_density.nii.gz")[0].astype(np.float32)
        C = load(V2C / S / f"{S}_Bundles_mri-template_connected.nii.gz")[0]
        assert D.shape == t2.shape and C.shape == t2.shape
        Sd, Inf = C > 0, C == 2
        if S in MIRROR:
            D, Sd, Inf = np.ascontiguousarray(D[::-1]), np.ascontiguousarray(Sd[::-1]), np.ascontiguousarray(Inf[::-1])
        vol[S] = (D, Sd, Inf)
        conn[S] = json.load(open(V2C / S / f"{S}_connect_v2c.json"))["final"]
    secj = {S: json.load(open(SEC / S / f"{S}_sections.json")) for S in CASES}
    sol = {}
    for S in CASES:
        v = (np.asanyarray(nib.load(str(V2C / S / f"{S}_Bundles_mri-template_solid_v2c.nii.gz")).dataobj) > 0).astype(np.float32)
        sol[S] = np.ascontiguousarray(v[::-1]) if S in MIRROR else v
    # extents: the brain's bounding box (+1 mm), right hemisphere for (c)
    bi = np.argwhere(brain); lo = bi.min(0) * vox + A[:3, 3] - 1.0; hi = bi.max(0) * vox + A[:3, 3] + 1.0
    ext = dict(Y0=lo[1], Y1=hi[1], Z0=lo[2], Z1=hi[2], X0=0.0 - 1.0, X1=hi[0])
    ins = [P3["template_t2"], P3["template_mask"], P3["template_fa"], BFT, P3["sarm_level6"]] + \
          [SEC / S / f"{S}_sections_points.csv" for S in CASES] + [SEC / S / f"{S}_sections.json" for S in CASES] + \
          [V2C / S / f"{S}_Bundles_mri-template_solid_v2c.nii.gz" for S in CASES] + [V2C / S / f"{S}_Bundles_mri-template_connected.nii.gz" for S in CASES] + \
          [V2C / S / f"{S}_connect_v2c.json" for S in CASES] + [V2 / S / f"{S}_Bundles_mri-template_density.nii.gz" for S in CASES]
    return dict(A=A, vox=vox, brain=brain, xw=xw, yw=yw, zw=zw, T2=T2, BF=BF, FA=FA, sarm=sarm, pts=pts, vol=vol, conn=conn, secj=secj, sol=sol,
                ext=ext, inputs=ins)


# --------------------------------------------------------------------------- the panels

def lateral(fig, sc, rec, left, top, wab, hab, gap):
    """(a), (b): the lateral projection over the slab, on T2 and on the blockface channel; returns the two axes."""
    A, vox, brain, xw, yw, zw = sc["A"], sc["vox"], sc["brain"], sc["xw"], sc["yw"], sc["zw"]
    pts, vol = sc["pts"], sc["vol"]; E = sc["ext"]; Y0, Y1, Z0, Z1 = E["Y0"], E["Y1"], E["Z0"], E["Z1"]

    def sag_plane(V, xmid):
        i = int(np.argmin(np.abs(xw - xmid)))
        return FS.section(V, "sagittal", i), float(xw[i])

    xmid = 0.5 * (SLAB_X[0] + SLAB_X[1])
    axs = []
    for k, (V, chan) in enumerate(((sc["T2"], "T2"), (sc["BF"], "blockface"))):
        ax = FS.axes_mm(fig, left + k * (wab + gap), top, wab, hab); axs.append(ax)
        img, xm = sag_plane(V, xmid)
        jy0, jy1 = int(round((Y0 - A[1, 3]) / vox)), int(round((Y1 - A[1, 3]) / vox))
        kz0, kz1 = int(round((Z0 - A[2, 3]) / vox)), int(round((Z1 - A[2, 3]) / vox))
        nz = img.shape[0]
        crop = img[nz - 1 - kz1:nz - kz0, jy0:jy1 + 1]
        ext = [yw[jy0] - vox / 2, yw[jy1] + vox / 2, zw[kz0] - vox / 2, zw[kz1] + vox / 2]
        ax.imshow(crop, cmap="gray", vmin=0, vmax=1, extent=ext, interpolation="nearest")
        sil = np.flipud(brain[(xw >= SLAB_X[0]) & (xw < SLAB_X[1])].any(0).T)[nz - 1 - kz1:nz - kz0, jy0:jy1 + 1]
        ax.contour(sil.astype(float), [0.5], colors=[FS.GRAY], linewidths=0.6, extent=ext, origin="upper", zorder=2)
        for S in DRAW_ORDER:
            B = pts[S]["Bundles"]; sel = (B[:, 0] >= SLAB_X[0]) & (B[:, 0] < SLAB_X[1])
            xs = (xw >= SLAB_X[0]) & (xw < SLAB_X[1]); D, Sd, Inf = vol[S]
            cut = lambda P: np.flipud(P.T)[nz - 1 - kz1:nz - kz0, jy0:jy1 + 1]  # noqa: E731
            draw_proj(ax, cut(D[xs].max(0)), cut(Sd[xs].any(0)), cut(Inf[xs].any(0)), COLOR[S], ext)
            if k == 0:
                rec["cases"].setdefault(S, {})["v2c_pathway_voxels_in_slab"] = int(Sd[xs].sum())
                rec["cases"][S]["v2c_inferred_voxels_in_slab"] = int(Inf[xs].sum())
            I = pts[S]["Injection_site"]
            if len(I):
                Isel = I[(I[:, 0] >= SLAB_X[0]) & (I[:, 0] < SLAB_X[1])]
                nx, ny = int(round((ext[1] - ext[0]) / vox)), int(round((ext[3] - ext[2]) / vox))
                O = occupancy(Isel[:, [1, 2]], ext[0], ext[3], nx, ny, vox)
                ax.contourf(O.astype(float), levels=[0.5, 1.5], colors=[COLOR[S]], alpha=0.95, extent=ext, origin="upper", zorder=4)
                ax.contour(O.astype(float), [0.5], colors=[FS.FG], linewidths=INJ_EDGE, extent=ext, origin="upper", zorder=5)
            if k == 0:
                rec["cases"].setdefault(S, {})["lateral_marks_in_slab"] = int(sel.sum())
                rec["cases"][S]["lateral_share_of_bundle_marks"] = round(float(sel.mean()), 3)
                rec["cases"][S]["injection_marks_in_slab"] = int(len(Isel)) if len(I) else 0
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_axis_off()
        FS.letter(ax, "ab"[k], dy=12.0)
        FS.header(ax, f"lateral projection, x = {SLAB_X[0]:g}–{SLAB_X[1]:g} mm (right), on {chan}", dy=12.0)
        FS.header(ax, f"{chan} plane x = {xm:.1f} mm", dx=0, dy=2.5, color=FS.GRAY)
    FS.dirs(axs[0], "sagittal"); FS.scalebar(axs[0], 1.0, 5.0)
    rec["sagittal_plane_x_mm"] = xm
    return axs


def dorsal(fig, sc, rec, left, top, wc, hc):
    """(c): the dorsal projection of the right hemisphere, on T2 at Z_AXIAL; returns the axis."""
    A, vox, brain, xw, yw, zw = sc["A"], sc["vox"], sc["brain"], sc["xw"], sc["yw"], sc["zw"]
    pts, vol = sc["pts"], sc["vol"]; E = sc["ext"]; Y0, Y1, X0, X1 = E["Y0"], E["Y1"], E["X0"], E["X1"]
    axc = FS.axes_mm(fig, left, top, wc, hc)
    kz = int(np.argmin(np.abs(zw - Z_AXIAL)))
    img = FS.section(sc["T2"], "axial", kz)                   # rows: y high (anterior) at the top; columns: x increasing
    ix0, ix1 = int(round((X0 - A[0, 3]) / vox)), int(round((X1 - A[0, 3]) / vox))
    jy0, jy1 = int(round((Y0 - A[1, 3]) / vox)), int(round((Y1 - A[1, 3]) / vox)); ny = img.shape[0]
    crop = img[ny - 1 - jy1:ny - jy0, ix0:ix1 + 1]
    ext = [xw[ix0] - vox / 2, xw[ix1] + vox / 2, yw[jy0] - vox / 2, yw[jy1] + vox / 2]
    axc.imshow(crop, cmap="gray", vmin=0, vmax=1, extent=ext, interpolation="nearest")
    silc = np.flipud((brain & (xw >= 0)[:, None, None]).any(2).T)[ny - 1 - jy1:ny - jy0, ix0:ix1 + 1]
    axc.contour(silc.astype(float), [0.5], colors=[FS.GRAY], linewidths=0.6, extent=ext, origin="upper", zorder=2)
    for S in DRAW_ORDER:
        B = pts[S]["Bundles"]; sel = B[:, 0] >= 0
        D, Sd, Inf = vol[S]; right = (xw >= 0)[:, None, None]
        cutc = lambda P: np.flipud(P.T)[ny - 1 - jy1:ny - jy0, ix0:ix1 + 1]  # noqa: E731
        draw_proj(axc, cutc(np.where(right, D, 0).max(2)), cutc((Sd & right).any(2)), cutc((Inf & right).any(2)), COLOR[S], ext)
        rec["cases"].setdefault(S, {})["v2c_pathway_voxels_right_hemisphere"] = int((Sd & right).sum())
        I = pts[S]["Injection_site"]
        if len(I):
            nx, ny2 = int(round((ext[1] - ext[0]) / vox)), int(round((ext[3] - ext[2]) / vox))
            O = occupancy(I[:, [0, 1]], ext[0], ext[3], nx, ny2, vox)
            axc.contourf(O.astype(float), levels=[0.5, 1.5], colors=[COLOR[S]], alpha=0.95, extent=ext, origin="upper", zorder=4)
            axc.contour(O.astype(float), [0.5], colors=[FS.FG], linewidths=INJ_EDGE, extent=ext, origin="upper", zorder=5)
        rec["cases"][S]["dorsal_marks_right_hemisphere"] = int(sel.sum())
        rec["cases"][S]["dorsal_share_of_bundle_marks"] = round(float(sel.mean()), 3)
        rec["cases"][S]["contralateral_percent_of_bundle_marks"] = round(100.0 * float((B[:, 0] < 0).mean()), 1)   # in no panel
    axc.set_xlim(ext[0], ext[1]); axc.set_ylim(ext[2], ext[3]); axc.set_axis_off()
    FS.letter(axc, "c", dy=12.0); FS.header(axc, "dorsal projection, on T2", dy=12.0)
    FS.header(axc, f"right hemisphere; plane z = {zw[kz]:.1f} mm", dx=0, dy=2.5, color=FS.GRAY)
    FS.dirs(axc, "axial")
    for t, xy, va in (("A", (1, 1), "top"), ("P", (1, 0), "bottom")):
        axc.annotate(t, xy=xy, xycoords="axes fraction", xytext=(2.0, 0), textcoords="offset points", fontsize=FS.MIN_PT, color=FS.GRAY,
                     ha="left", va=va, annotation_clip=False)
    FS.scalebar(axc, 1.0, 5.0)
    return axc


def alic(fig, sc, rec, left, top, pd, hd):
    """(d): the anterior limb of the right internal capsule, 0.5 mm coronal slabs of FA, the connected pathway; returns the axes."""
    vox, xw, yw, zw, sarm, FA, sol = sc["vox"], sc["xw"], sc["yw"], sc["zw"], sc["sarm"], sc["FA"], sc["sol"]
    xr = (xw >= ALIC_X[0] - 1e-6) & (xw <= ALIC_X[1] + 1e-6); zr = (zw >= ALIC_Z[0] - 1e-6) & (zw <= ALIC_Z[1] + 1e-6)
    crop2 = lambda P: P[xr][:, zr]                                                  # noqa: E731
    disp = lambda P: np.flipud(np.swapaxes(P, 0, 1))                                # noqa: E731
    slab = lambda V, j: 0.25 * V[:, j - 1, :] + 0.5 * V[:, j, :] + 0.25 * V[:, j + 1, :]   # noqa: E731
    ic = ndi.binary_dilation(sarm == IC_LABEL, iterations=2)
    axd = []; rec["capsule"] = {}
    for q, y in enumerate(ALIC_Y):
        ax = FS.axes_mm(fig, left + q * (pd + 3.0), top, pd, hd); axd.append(ax)
        j = int(np.argmin(np.abs(yw - y))); assert abs(yw[j] - y) < 1e-6
        bg = disp(crop2(slab(FA, j))); ax.imshow(bg, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        icm = crop2(ic[:, j - 1:j + 2, :].any(1)); rec["capsule"][f"{y:.1f}"] = {}
        for S in DRAW_ORDER:
            F = crop2(slab(sol[S], j))
            ii = np.nonzero((F > 0.5) & icm)[0]
            rec["capsule"][f"{y:.1f}"][S] = dict(voxels_in_capsule=int(len(ii)), voxels_total=int((F > 0.5).sum()))
            for M, al in ((np.where(icm, 0, F), OUT_A), (np.where(icm, F, 0), FILL_A)):
                if (M > 0.5).any():
                    ax.contourf(disp(M), levels=[0.5, 2.0], colors=[COLOR[S]], alpha=al, antialiased=True)
            if ((F > 0.5) & icm).any():
                ax.contour(disp(np.where(icm, F, 0)), [0.5], colors=[DARK], linewidths=0.35)
        ax.contour(disp(crop2(sarm[:, j, :] == IC_LABEL).astype(float)), [0.5], colors=[FS.FG], linewidths=0.7, linestyles=[(0, (1, 1.2))])
        ax.set_xlim(-0.5, bg.shape[1] - 0.5); ax.set_ylim(bg.shape[0] - 0.5, -0.5); ax.set_axis_off()
        if q == 0:
            FS.letter(ax, "d", dy=12.0); FS.header(ax, "anterior limb, right internal capsule, on FA", dy=12.0)
        FS.header(ax, f"y = {y:.1f} mm", dx=0, dy=2.5, color=FS.GRAY)
    FS.dirs_row(axd[0], axd[-1], "coronal"); FS.scalebar(axd[0], 1.0 / vox, 2.0)
    return axd


def key(fig, sc, rec, left, top, w, h, matplotlib):
    """The key (outside every image): case, area, color, the inferred share; injection site; capsule; the bridge / footprint / outline row."""
    pts, conn, secj = sc["pts"], sc["conn"], sc["secj"]
    axk = FS.axes_mm(fig, left, top, w, h); axk.set_axis_off(); axk.set_xlim(0, 1); axk.set_ylim(0, 1)
    rows = []
    for S in CASES:
        sp = secj[S]["spacing_summary_mm"]
        rec["cases"][S].update(name=FS.label(S), area=AREA[S], mirrored=S in MIRROR, sections=len(secj[S]["sections"]),
                               section_spacing_mm=sp, bundle_marks=int(len(pts[S]["Bundles"])),
                               injection_marks=int(len(pts[S]["Injection_site"])))
        rec["cases"][S].update(v2c_inferred_share=conn[S]["inferred_share"], v2c_longest_bridge_mm=conn[S]["longest_bridge_mm"], v2c_pieces_attached=conn[S]["pieces_attached"])
        rows.append((S, f"{FS.label(S)}{' (mirrored)' if S in MIRROR else ''}, {AREA[S]}: connected pathway; inferred {100 * conn[S]['inferred_share']:.1f} % (longest bridge {conn[S]['longest_bridge_mm']:.1f} mm)"))
    for r, (S, txt) in enumerate(rows):
        yk = 0.92 - r * 0.17
        axk.scatter([0.03, 0.045, 0.06], [yk] * 3, s=6, c=COLOR[S], lw=0)
        axk.text(0.1, yk, txt, va="center", ha="left", fontsize=FS.MIN_PT, color=FS.FG)
    yk = 0.92 - 3 * 0.17
    for q, S in enumerate(CASES):                                               # one swatch per case color, thin white edge
        axk.add_patch(matplotlib.patches.Rectangle((0.02 + q * 0.022, yk - 0.04), 0.018, 0.08, facecolor=COLOR[S], edgecolor=FS.FG,
                                                   lw=INJ_EDGE))
    axk.text(0.1, yk, "injection site (filled)", va="center", ha="left", fontsize=FS.MIN_PT, color=FS.FG)
    yk -= 0.17
    axk.plot([0.02, 0.075], [yk, yk], color=FS.FG, lw=0.7, ls=(0, (1, 1.2)))
    axk.text(0.1, yk, "internal capsule (SARM atlas)", va="center", ha="left", fontsize=FS.MIN_PT, color=FS.FG)
    yk -= 0.17
    axk.plot([0.02, 0.075], [yk, yk], color=FS.FG, lw=0.5, ls=(0, (0.8, 0.8)))
    axk.text(0.1, yk, "inferred bridge (dotted edge); light fill = footprint, density ≥ 0.1; gray = projected brain outline", va="center", ha="left", fontsize=FS.MIN_PT, color=FS.FG)
    return axk


# --------------------------------------------------------------------------- the paper's layout

def composite(fig, sc, rec, stills, left, top, w, h, gap):
    """(a), (b): the two 3D stills of reproduce/figures/tracer_3d.py (the oblique and the lateral view) as images, true to their
    pixels; returns the two axes. The record keeps each still's file, size and md5."""
    import hashlib
    from PIL import Image
    axs = []
    rec["stills"] = {}
    for k, (name, path) in enumerate(stills):
        im = np.asarray(Image.open(path).convert("RGB"))
        ax = FS.axes_mm(fig, left + k * (w + gap), top, w, h); axs.append(ax)
        ax.imshow(im, interpolation="lanczos"); ax.set_axis_off()
        FS.letter(ax, "ab"[k], dy=12.0)
        FS.header(ax, {"oblique": "3D view, from the front, above and the right", "lateral": "3D view, from the right"}[name], dy=12.0)
        FS.header(ax, {"oblique": "template T2: midline plane, coronal backdrop, floor; bridges in a lighter tint", "lateral": "the same scene"}[name], dx=0, dy=2.5, color=FS.GRAY)
        rec["stills"][name] = dict(file=Path(path).name, pixels=[int(im.shape[1]), int(im.shape[0])], md5=hashlib.md5(open(path, "rb").read()).hexdigest())
    return axs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="default figures/tracer_pathways.png (composite) or figures/tracer_pathways_panels.png (panels)")
    ap.add_argument("--layout", choices=["composite", "panels"], default="composite",
                    help="composite (the paper's Fig. 10): a, b = the 3D stills, c = the dorsal projection, d = the capsule sections; "
                         "panels: a, b = the lateral projections on T2 and on the blockface channel, c, d as above")
    ap.add_argument("--stills", nargs=2, metavar=("OBLIQUE_PNG", "LATERAL_PNG"), default=None,
                    help="composite: the two stills (default: figures/tracer_3d_oblique.png and figures/tracer_3d_lateral.png beside --out)")
    a = ap.parse_args(argv)
    out = a.out or ("figures/tracer_pathways.png" if a.layout == "composite" else "figures/tracer_pathways_panels.png")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cmd = " ".join(["reproduce/figures/tracer_pathways.py"] + (argv if argv is not None else sys.argv[1:]))
    name = NAME if a.layout == "composite" else NAME + "_panels"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FS.setup()
    sc = scene(); E = sc["ext"]
    wc, hc = (E["X1"] - E["X0"]) * SCALE_C, (E["Y1"] - E["Y0"]) * SCALE_C
    pd, hd = (ALIC_X[1] - ALIC_X[0]) * SCALE_D, (ALIC_Z[1] - ALIC_Z[0]) * SCALE_D
    LEFT, TOP = 7.0, 13.0
    rec = dict(layout=a.layout, slab_x_mm=list(SLAB_X), z_axial_mm=Z_AXIAL, alic_y_mm=ALIC_Y, alic_slab_mm=0.5, close_mm=CLOSE_MM, cases={})
    ins = list(sc["inputs"])
    if a.layout == "composite":
        od = Path(out).parent
        stills = [("oblique", Path(a.stills[0]) if a.stills else od / "tracer_3d_oblique.png"), ("lateral", Path(a.stills[1]) if a.stills else od / "tracer_3d_lateral.png")]
        for _, p in stills:
            if not p.exists():
                raise SystemExit(f"{p}: the 3D still is missing (reproduce/figures/tracer_3d.py renders it; --layout panels needs none)")
        WS, GAPS = 80.0, 4.0; HS = WS * 0.75                                   # the stills are 4:3
        TOP2 = TOP + HS + 17.0
        H = TOP2 + max(hc, hd + 42.0) + 9.0
        fig = plt.figure(figsize=(FS.FULL_W_IN, H * FS.MM))
        composite(fig, sc, rec, stills, LEFT, TOP, WS, HS, GAPS)
        ins += [p for _, p in stills]
    else:
        wab, hab, GAP = (E["Y1"] - E["Y0"]) * SCALE_AB, (E["Z1"] - E["Z0"]) * SCALE_AB, 7.0
        TOP2 = TOP + hab + 17.0
        H = TOP2 + max(hc, hd + 42.0) + 9.0
        fig = plt.figure(figsize=(FS.FULL_W_IN, H * FS.MM))
        lateral(fig, sc, rec, LEFT, TOP, wab, hab, GAP)
    dorsal(fig, sc, rec, LEFT, TOP2, wc, hc)
    DX = LEFT + wc + 30.0
    alic(fig, sc, rec, DX, TOP2, pd, hd)
    key(fig, sc, rec, DX, TOP2 + hd + 9.0, 3 * pd + 6.0, 33.0, matplotlib)
    FS.save(fig, out, [str(p) for p in ins], cmd=cmd, extra=rec)
    print(provenance.write_numbers(name, rec, inputs=[str(p) for p in ins], md5=False, cmd=cmd))
    print(out, json.dumps(rec["cases"], indent=1))


if __name__ == "__main__":
    main()
