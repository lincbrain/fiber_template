#!/usr/bin/env python3
"""The paper's tracer figure, Fig. 10.

You need (configs/products_*.yaml)
  template_t2 / template_mask / template_fa and sarm_level6 (products_figures); template_link: the blockface template in the MRI
  template (products_blockface); tracer_sections_v2, tracer_template_v2, tracer_template_v2c (products_mri); for the composite, the
  three stills of reproduce/figures/tracer_3d.py --for-composite (oblique, lateral, glass).

What it does
  composite (the paper's figure): a-c = the three 3D stills, cropped to the brain; d = the dorsal projection of the right hemisphere
  on T2; e = the anterior limb of the internal capsule on three coronal slabs of FA; one key;
  panels (the earlier four-panel form): a, b = the lateral projections on T2 and on the blockface channel, c = dorsal, d = capsule.
  The connected pathway is drawn filled, its inferred joins under a dotted edge, the density footprint as a light fill.

You get
  <out>.png + <out>.provenance.json, and $LINC_NUMBERS/fig_tracer_pathways.json (composite) or fig_tracer_pathways_panels.json
  (the layout, the stills' md5, per case the pathway voxels drawn, the inferred fraction and the longest join, the capsule counts)

    python reproduce/figures/tracer_pathways.py [--stills OBLIQUE LATERAL GLASS] [--out figures/tracer_pathways.png]
    python reproduce/figures/tracer_pathways.py --layout panels [--out figures/tracer_pathways_panels.png]
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
COLOR = {S: FS.TRACER[S] for S in CASES}                  # the tracer cases' colors (figstyle.TRACER): the similar pair warm
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
    the inferred joins filled under a dotted white edge."""
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


def dorsal(fig, sc, rec, left, top, wc, hc, letter="c"):
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
    FS.letter(axc, letter, dy=12.0); FS.header(axc, "dorsal projection, on T2", dy=12.0)
    FS.header(axc, f"right hemisphere; plane z = {zw[kz]:.1f} mm", dx=0, dy=2.5, color=FS.GRAY)
    FS.dirs(axc, "axial")
    for t, xy, va in (("A", (1, 1), "top"), ("P", (1, 0), "bottom")):
        axc.annotate(t, xy=xy, xycoords="axes fraction", xytext=(2.0, 0), textcoords="offset points", fontsize=FS.MIN_PT, color=FS.GRAY,
                     ha="left", va=va, annotation_clip=False)
    FS.scalebar(axc, 1.0, 5.0)
    return axc


def alic(fig, sc, rec, left, top, pd, hd, letter="d"):
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
            FS.letter(ax, letter, dy=12.0); FS.header(ax, "anterior limb, right internal capsule, on FA", dy=12.0)
        FS.header(ax, f"y = {y:.1f} mm", dx=0, dy=2.5, color=FS.GRAY)
    FS.dirs_row(axd[0], axd[-1], "coronal"); FS.scalebar(axd[0], 1.0 / vox, 2.0)
    return axd


def key(fig, sc, rec, left, top, w, h, matplotlib, compact=False):
    """The key (outside every image): case, area, color, the inferred share; injection site; capsule; the join / footprint / outline
    row. compact: shorter lines in seven rows (the composite's narrower column)."""
    pts, conn, secj = sc["pts"], sc["conn"], sc["secj"]
    axk = FS.axes_mm(fig, left, top, w, h); axk.set_axis_off(); axk.set_xlim(0, 1); axk.set_ylim(0, 1)
    x0, xt, dy = (0.015, 0.07, 0.14) if compact else (0.03, 0.1, 0.17)         # the marks' first x, the text's x, the row step
    xs = [x0, x0 + 0.015, x0 + 0.03]
    rows = []
    for S in CASES:
        sp = secj[S]["spacing_summary_mm"]
        rec["cases"][S].update(name=FS.label(S), area=AREA[S], mirrored=S in MIRROR, sections=len(secj[S]["sections"]),
                               section_spacing_mm=sp, bundle_marks=int(len(pts[S]["Bundles"])),
                               injection_marks=int(len(pts[S]["Injection_site"])))
        rec["cases"][S].update(v2c_inferred_share=conn[S]["inferred_share"], v2c_longest_bridge_mm=conn[S]["longest_bridge_mm"], v2c_pieces_attached=conn[S]["pieces_attached"])
        rows.append((S, (f"{FS.label(S)}{' (mirrored)' if S in MIRROR else ''}, {AREA[S]}: inferred {100 * conn[S]['inferred_share']:.1f} % (longest join {conn[S]['longest_bridge_mm']:.1f} mm)" if compact else
                         f"{FS.label(S)}{' (mirrored)' if S in MIRROR else ''}, {AREA[S]}: connected pathway; inferred {100 * conn[S]['inferred_share']:.1f} % (longest join {conn[S]['longest_bridge_mm']:.1f} mm)")))
    y1 = 0.93 if compact else 0.92
    for r, (S, txt) in enumerate(rows):
        yk = y1 - r * dy
        axk.scatter(xs, [yk] * 3, s=6, c=COLOR[S], lw=0)
        axk.text(xt, yk, txt, va="center", ha="left", fontsize=FS.MIN_PT, color=FS.FG)
    yk = y1 - 3 * dy
    for q, S in enumerate(CASES):                                               # one swatch per case color, thin white edge
        axk.add_patch(matplotlib.patches.Rectangle((x0 - 0.01 + q * 0.022, yk - 0.04), 0.018, 0.08, facecolor=COLOR[S], edgecolor=FS.FG,
                                                   lw=INJ_EDGE))
    axk.text(xt, yk, "injection site (filled)", va="center", ha="left", fontsize=FS.MIN_PT, color=FS.FG)
    yk -= dy
    axk.plot([x0 - 0.01, x0 + 0.045], [yk, yk], color=FS.FG, lw=0.7, ls=(0, (1, 1.2)))
    axk.text(xt, yk, "internal capsule (SARM atlas)", va="center", ha="left", fontsize=FS.MIN_PT, color=FS.FG)
    yk -= dy
    axk.plot([x0 - 0.01, x0 + 0.045], [yk, yk], color=FS.FG, lw=0.5, ls=(0, (0.8, 0.8)))
    if compact:
        axk.text(xt, yk, "inferred join (dotted edge); the pathways connected, as in a-c", va="center", ha="left", fontsize=FS.MIN_PT, color=FS.FG)
        yk -= dy
        axk.text(xt, yk, "light fill = footprint (density ≥ 0.1); gray = projected brain outline", va="center", ha="left", fontsize=FS.MIN_PT, color=FS.FG)
    else:
        axk.text(xt, yk, "inferred join (dotted edge); light fill = footprint, density ≥ 0.1; gray = projected brain outline", va="center", ha="left", fontsize=FS.MIN_PT, color=FS.FG)
    return axk


# --------------------------------------------------------------------------- the paper's layout

STILL_W, STILL_MARGIN, STILL_DARK = 80.0, 0.03, 40                 # mm, the still's panel width; the crop's margin (of the still's width); background level


def still_crop(im):
    """The still's content box: the smallest box holding every pixel brighter than STILL_DARK (the brain, the planes, the letters and
    the bar; not the black background), widened by STILL_MARGIN of the still's width on every side."""
    on = np.argwhere(im.max(-1) > STILL_DARK)
    m = int(round(STILL_MARGIN * im.shape[1]))
    r0, c0 = np.maximum(on.min(0) - m, 0); r1, c1 = np.minimum(on.max(0) + m + 1, im.shape[:2])
    return int(r0), int(r1), int(c0), int(c1)


def composite(fig, sc, rec, stills, positions, w):
    """(a), (b), (c): the 3D stills of reproduce/figures/tracer_3d.py (the oblique view, the lateral view, the lateral view with the
    brain as a translucent shell), each cropped to its content (still_crop) and drawn at width w at the given (x, y) mm positions, so the
    brain shows at a similar size in the three; returns the axes and the panel heights. The record keeps each still's file, size, md5 and
    crop box."""
    import hashlib
    from PIL import Image
    axs, heights = [], []
    rec["stills"] = {}
    for k, ((name, path), (x, y)) in enumerate(zip(stills, positions)):
        im = np.asarray(Image.open(path).convert("RGB"))
        r0, r1, c0, c1 = still_crop(im); crop = im[r0:r1, c0:c1]
        h = w * crop.shape[0] / crop.shape[1]
        ax = FS.axes_mm(fig, x, y, w, h); axs.append(ax); heights.append(h)
        ax.imshow(crop, interpolation="lanczos"); ax.set_axis_off()
        FS.letter(ax, "abc"[k], dy=12.0)
        FS.header(ax, {"oblique": "3D view, from the front, above and the right", "lateral": "3D view, from the right",
                       "glass": "the same view as b, the brain's surface drawn as a translucent shell"}[name], dy=12.0)
        FS.header(ax, {"oblique": "template T2: midline plane, coronal backdrop, floor; joins in a lighter tint", "lateral": "the same scene",
                       "glass": "the pathways lie inside the brain"}[name], dx=0, dy=2.5, color=FS.GRAY)
        rec["stills"][name] = dict(file=Path(path).name, pixels=[int(im.shape[1]), int(im.shape[0])], md5=hashlib.md5(open(path, "rb").read()).hexdigest(),
                                   crop_px=dict(rows=[r0, r1], cols=[c0, c1]), panel_mm=[round(w, 2), round(h, 2)])
    return axs, heights


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="default figures/tracer_pathways.png (composite) or figures/tracer_pathways_panels.png (panels)")
    ap.add_argument("--layout", choices=["composite", "panels"], default="composite",
                    help="composite (the paper's Fig. 10): a, b, c = the 3D stills (oblique, lateral, lateral with the brain shell), d = the dorsal projection, "
                         "e = the capsule sections; panels: a, b = the lateral projections on T2 and on the blockface channel, c = dorsal, d = capsule")
    ap.add_argument("--stills", nargs=3, metavar=("OBLIQUE_PNG", "LATERAL_PNG", "GLASS_PNG"), default=None,
                    help="composite: the three stills (default: figures/tracer_3d_{oblique,lateral,glass}.png beside --out)")
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
        stills = [(n, Path(a.stills[k]) if a.stills else od / f"tracer_3d_{n}.png") for k, n in enumerate(("oblique", "lateral", "glass"))]
        for _, p in stills:
            if not p.exists():
                raise SystemExit(f"{p}: the 3D still is missing (reproduce/figures/tracer_3d.py renders it; --layout panels needs none)")
        from PIL import Image
        WS, GAPS = STILL_W, 4.0                                                 # rows: a | b, c | d, e | key; the stills cropped to their content
        hs = []
        for _, p in stills:
            im = np.asarray(Image.open(p).convert("RGB")); r0, r1, c0, c1 = still_crop(im); hs.append(WS * (r1 - r0) / (c1 - c0))
        TOP2 = TOP + max(hs[0], hs[1]) + 17.0
        TOP3 = TOP2 + max(hs[2], hc) + 17.0
        KW, KH = FS.FULL_W_IN / FS.MM - (LEFT + 3 * pd + 6.0 + 6.0) - 2.0, 40.0   # the key fills the row to the right edge
        H = TOP3 + max(hd, KH) + 9.0
        fig = plt.figure(figsize=(FS.FULL_W_IN, H * FS.MM))
        composite(fig, sc, rec, stills, [(LEFT, TOP), (LEFT + WS + GAPS, TOP), (LEFT, TOP2)], WS)
        ins += [p for _, p in stills]
        dorsal(fig, sc, rec, LEFT + WS + 10.0, TOP2, wc, hc, letter="d")
        alic(fig, sc, rec, LEFT, TOP3, pd, hd, letter="e")
        key(fig, sc, rec, LEFT + 3 * pd + 6.0 + 6.0, TOP3, KW, KH, matplotlib, compact=True)
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
