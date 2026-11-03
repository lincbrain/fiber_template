#!/usr/bin/env python3
"""The paper's figure "Tracer labels carried into the template" (--figure main) and its supplementary figure "Two rules for drawing a
label" (--figure rules; reproduce/figures/s_tracer_rules.py runs it).

    python reproduce/figures/tracer.py [--out figures/tracer.png]
    python reproduce/figures/tracer.py --figure rules [--out figures/s_tracer_rules.png]

Labels: the final tracer labels in the MRI template (products tracer_template: <S>/<S>_<class>_mri-template_density.nii.gz, the
demons-interpolated labels carried by the chains of record; MM1 / MM2 / MM3 through their own MRI, MF8 through the blockface
template), on the template's T2 and FA channels, with the internal capsule of the SARM atlas (level 6, label 50) carried from NMT v2.
Main figure (the layout the author chose on 30 Oct):
  (a) lateral and dorsal projections of the bundles, dense and moderate classes and MM2's ungraded class, drawn where the carried
      density exceeds 0.5; injection sites as white outlines;
  (b) injection sites in 0.5 mm coronal slabs of the T2 channel;
  (c) bundles in the anterior limb of the internal capsule in 0.5 mm coronal slabs of the FA channel (faded outside the capsule);
  (d) 0.5 mm coronal slabs of the T2 channel, one case per column; dense and moderate terminals as lighter contours.
  MF8 is mirrored onto the right hemisphere for display (exact: the template is symmetric and centered on x = 0). Its bundle voxels
  that come from sections outside every interpolated pair (its interpolation table) are drawn as thin outlines in (a) and (d): a
  template voxel is taken to MF8's blockface grid by its chain (the registration of the blockface template to the MRI template, MF8's
  member warp of the blockface template's construction, the inverse of its rigid placement), and its section is looked up.
  Writes $LINC_NUMBERS/fig_tracer_r058.json (volumes per case and class under both rules, the counts of (c), the injection sites
  of (b), and the share of MF8's bundle voxels drawn as outlines, which the paper prints).
Rules figure: per case, all bundle classes (light bundles included; terminals translucent) drawn where the density is at least 0.1 and
where it exceeds 0.5, in the lateral and dorsal projections.
Reads (configs/products_figures.yaml): tracer_template, template_t2 / template_mask / template_fa, sarm_level6, bft_pull_map,
bft_member_warps, mf8_rigid, mf8_interp, mf8_label_grid; $LINC_NUMBERS/acpc.json is read from numbers_of_record (products_mri).
Ported from the analysis project's paper/fig_tracer_r058.py (paper/fig_tracer_variants_r058.py, Variant A, with the record's changes).
"""
import argparse
import json
import sys
from glob import glob
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.patheffects as pe
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
import figstyle as FS  # noqa: E402
from linc import paths, provenance  # noqa: E402

P = {}                                                                        # products (set by roots())
TT, T2_, MASK_, FA_, SARM6, ACPC = (None,) * 6
IC_LABEL = 50                                                                 # SARM_key_all.txt: 50 ic internal_capsule
ALL_CASES = ["MR243", "MR252", "MR256", "MF191"]
DRAW_ORDER = ["MF191", "MR256", "MR243", "MR252"]                            # bottom to top (v3's order)
MIRROR = {"MF191"}
TRACER = {"MR243": "Lucifer yellow", "MR252": "fluorescein", "MR256": "Lucifer yellow", "MF191": "Fluoro-Ruby"}
BUNDLE_A = ["Fiber_dense_bundle", "Fiber_moderate_bundle", "Bundle"]         # (a): dense + moderate, and MR252's ungraded "Bundle"
TERM_A = ["Terminal_dense", "Terminal_moderate"]                              # (d) only, contours
V3_FIB = ["Fiber_dense_bundle", "Fiber_moderate_bundle", "Fiber_light_bundle", "Bundle"]   # v3's group (no "Fibers")
V3_TERM = ["Terminal_dense", "Terminal_moderate", "Terminal_light"]
SOLID, FOOT = 0.5, 0.1
ALIC_Y = [20.0, 22.0, 24.0]
ALIC_X, ALIC_Z = (1.5, 12.5), (13.5, 24.5)
SLABS_Y = [30.0, 23.0, 17.0, 13.0, 9.0]
SLAB_HALF = 0.25                                                              # (c): 0.5 mm slabs
WS, WD, ZB = 0.6, 0.5, 3.0                                                    # print mm per mm: (a)/(b0); (d); (c)
INJ_FIELD, INJ_LW, INJ_FILL = 12.0, 1.3, 0.7
DIM, OUT_A, FILL_A = 0.6, 0.3, 0.9
DARK = "#111111"


def roots():
    """The products this figure reads (configs/products_figures.yaml under $LINC_PRODUCTS; acpc.json from products_mri's numbers_of_record)."""
    global TT, T2_, MASK_, FA_, SARM6, ACPC
    P.update(paths.products("products_figures"))
    TT, T2_, MASK_, FA_, SARM6 = P["tracer_template"], P["template_t2"], P["template_mask"], P["template_fa"], P["sarm_level6"]
    ACPC = paths.products("products_mri")["numbers_of_record"] / "acpc.json"


def load(p):
    return np.asanyarray(nib.load(str(p)).dataobj)


def color(S):
    import matplotlib.colors as mc
    return tuple(mc.to_rgb(FS.brain_color(S)))


def tint(c, w=0.5):
    return tuple(w * np.asarray(c) + (1 - w) * np.ones(3))


def name(S):
    return f"{FS.label(S)} (mirrored)" if S in MIRROR else FS.label(S)


def disp(P):
    return np.flipud(np.swapaxes(P, 0, 1))


def density(S, cl, mirror=True):
    p = TT / S / f"{S}_{cl}_mri-template_density.nii.gz"
    if not p.exists():
        return None
    d = load(p).astype(np.float32)
    solid = load(TT / S / f"{S}_{cl}_mri-template.nii.gz") > 0                # the record's solid file = density > 0.5
    assert np.array_equal(solid, d > SOLID), (S, cl)
    return np.ascontiguousarray(d[::-1]) if (mirror and S in MIRROR) else d


def group(S, classes, mirror=True):
    ds = [d for d in (density(S, c, mirror) for c in classes) if d is not None]
    return np.max(ds, 0) if ds else None


def slab(V, yw, y):
    """The 0.5 mm slab mean at y: planes y - 0.25, y, y + 0.25 weighted 1/4, 1/2, 1/4."""
    j = int(np.argmin(np.abs(yw - y)))
    assert abs(yw[j] - y) < 1e-6, f"slab level {y} not on a voxel plane"
    return 0.25 * V[:, j - 1, :] + 0.5 * V[:, j, :] + 0.25 * V[:, j + 1, :]


def fill(ax, F, rgb, alpha=FILL_A, lw=0.35, edge=DARK):
    if F is None or not (F > SOLID).any():
        return
    ax.contourf(F, levels=[SOLID, max(float(F.max()), SOLID) + 1.0], colors=[rgb], alpha=alpha, antialiased=True)
    ax.contour(F, [SOLID], colors=[edge], linewidths=lw)


def line(ax, F, rgb, lw=0.5):
    if F is None or not (F > SOLID).any():
        return
    ax.contour(F, [SOLID], colors=[rgb], linewidths=lw)


def inj_outline(ax, F):
    if F is None or not (F > SOLID).any():
        return
    cs = ax.contour(F, [SOLID], colors=[FS.COL["outline"]], linewidths=INJ_LW)
    cs.set_path_effects([pe.withStroke(linewidth=INJ_LW + 1.4, foreground="black")])


def keep_limits(ax, shape):
    ax.set_xlim(-0.5, shape[1] - 0.5); ax.set_ylim(shape[0] - 0.5, -0.5)


def head2(ax, top, sub, letter=None):
    if letter:
        FS.letter(ax, letter, dy=12.5)
    if top:
        FS.header(ax, top, dx=11 if letter else 0, dy=12.5)
    if sub:
        FS.header(ax, sub, dx=0, dy=2.5)



def mf8_chain():
    """MRI-template world points -> MF8's blockface world points: the registration of the blockface template to the MRI template
    (pull map), MF8's member warp of the blockface template's construction (fsl_rel on its canon grid; valid in photographed tissue,
    where the tracer sheets lie), and the inverse of MF8's rigid placement (T_world_bf_to_run113)."""
    from linc import xfm
    from linc.registration import xfmtools as X
    link = X.load_map(str(P["bft_pull_map"]), "full")
    w = json.load(open(P["bft_member_warps"]))
    row = [r for r in w["data"] if r["member"] == "MF191"][0]
    assert "photographed" in row["warp_validity"]["valid_in"], row["warp_validity"]
    W = xfm.Warp(paths.rerooted(row["file"]), "fsl_rel", src_grid=paths.rerooted(row["src_grid"]))
    T = np.array(json.load(open(P["mf8_rigid"]))["T_world_bf_to_run113"], float); Ti = np.linalg.inv(T)

    def f(y):
        p = np.asarray(W.pull(link(y)))
        return p @ Ti[:3, :3].T + Ti[:3, 3]
    return f


def _split_unbridged(G, CASES, A):
    """MF8 (MF191): split the bundle group into bridged (filled) and unbridged (outlined) voxels; returns the record of the split."""
    S = "MF191"
    if S not in CASES or G[S]["fib"] is None:
        return {}
    f = mf8_chain()
    J = json.load(open(P["mf8_interp"]))
    bridged = set()
    for p_ in J["pairs"]:
        if p_["status"] == "interpolated":
            bridged.update(range(p_["a"], p_["b"] + 1))
    sh = nib.load(str(P["mf8_label_grid"]))
    Ainv = np.linalg.inv(sh.affine)
    fib = G[S]["fib"][::-1]                                                  # un-mirror (the display mirrors MF191)
    idx = np.argwhere(fib > 0.02)
    w = idx @ A[:3, :3].T + A[:3, 3]
    bf = np.concatenate([np.asarray(f(w[k:k + 200000])) for k in range(0, len(w), 200000)])
    jj = np.rint(bf @ Ainv[:3, :3].T + Ainv[:3, 3])[:, 1].astype(int)
    U = np.zeros(fib.shape, bool)
    un = np.array([j not in bridged for j in jj])
    U[tuple(idx[un].T)] = True
    U = U[::-1]
    fm = G[S]["fib"]
    G[S]["fib_u"] = np.where(U, fm, 0.0).astype(np.float32); G[S]["fib"] = np.where(U, 0.0, fm).astype(np.float32)
    solid = fm > SOLID
    return dict(bridged_slices=sorted(int(x) for x in bridged), solid_bundle_voxels=int(solid.sum()),
                solid_bundle_voxels_unbridged=int((solid & U).sum()), share_unbridged=round(float((solid & U).sum() / max(solid.sum(), 1)), 4))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure", choices=("main", "rules"), default="main")
    ap.add_argument("--out", default=None, help="default figures/tracer.png (main) or figures/s_tracer_rules.png (rules)")
    a = ap.parse_args(argv)
    out_png = a.out or ("figures/tracer.png" if a.figure == "main" else "figures/s_tracer_rules.png")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    cmd = " ".join(["reproduce/figures/tracer.py"] + (argv if argv is not None else sys.argv[1:]))
    CASES = ALL_CASES
    roots()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    FS.setup()
    ref = nib.load(str(T2_)); A = ref.affine
    assert nib.aff2axcodes(A) == ("R", "A", "S") and np.allclose(A[:3, :3], np.diag(np.diag(A[:3, :3])))
    vox = float(A[0, 0]); vv = vox ** 3
    t2 = load(T2_).astype(np.float32); brain = load(MASK_) > 0
    assert abs(A[0, 3] + (t2.shape[0] - 1) * vox / 2) < 1e-6 and np.array_equal(t2, t2[::-1]) and np.array_equal(brain, brain[::-1])
    t2 = t2 / np.percentile(t2[brain], 99.5)
    fa = load(FA_).astype(np.float32); fa = np.clip(fa / np.percentile(fa[brain], 99.5) * DIM, 0, 1)
    sarm = load(SARM6); assert sarm.shape == t2.shape
    ic = ndi.binary_dilation(sarm == IC_LABEL, iterations=2)
    AC = json.loads(ACPC.read_text())["data"]["definitive"]["AC"]
    assert all(y > AC[1] for y in ALIC_Y)
    xw = A[0, 3] + vox * np.arange(t2.shape[0]); yw = A[1, 3] + vox * np.arange(t2.shape[1]); zw = A[2, 3] + vox * np.arange(t2.shape[2])
    lo, hi = np.argwhere(brain).min(0) - 4, np.argwhere(brain).max(0) + 5
    box = lambda P, k0, k1: P[lo[k0]:hi[k0], lo[k1]:hi[k1]]                 # noqa: E731
    ext = (hi - lo) * vox
    rng = lambda w, r: (w >= r[0] - 1e-6) & (w <= r[1] + 1e-6)              # noqa: E731

    # ------------------------------------------------------------------ labels and the record -----------------------------------
    G = {S: dict(fib=group(S, BUNDLE_A), term=group(S, TERM_A), inj=group(S, ["Injection_site"])) for S in CASES}
    split_rec = _split_unbridged(G, CASES, A) if a.figure == "main" else {}
    inj_cases = [S for S in CASES if G[S]["inj"] is not None and (G[S]["inj"] > SOLID).any()]
    rec = dict(method=__doc__.split("\n")[1:33], cases={}, voxel_mm3=vv, rules=dict(solid=f"density > {SOLID}", footprint=f"density >= {FOOT}"),
               groups=dict(variant_bundles=BUNDLE_A, variant_terminals_d_only=TERM_A, v3_bundles=V3_FIB, v3_terminals=V3_TERM),
               slab=dict(half_mm=SLAB_HALF, weights=[0.25, 0.5, 0.25], rule="slab-mean density > 0.5"),
               colors={S: dict(hex=matplotlib.colors.to_hex(color(S)), figstyle=FS.brain_color(S), proposed=False) for S in CASES},
               mirrored_for_display=sorted(MIRROR & set(CASES)))
    for S in CASES:
        cls = sorted(Path(p).name[len(S) + 1:-len("_mri-template_density.nii.gz")] for p in glob(str(TT / S / f"{S}_*_mri-template_density.nii.gz")))
        crec = dict(label=FS.label(S), tracer=TRACER[S], classes={})
        inj_raw = group(S, ["Injection_site"], mirror=False)
        side = None if inj_raw is None or not (inj_raw > SOLID).any() else float(np.sign(xw[np.where(inj_raw > SOLID)[0]].mean()))
        crec["injection_side"] = None if side is None else ("right (+x)" if side > 0 else "left (-x)")
        for cl in cls:
            d = density(S, cl, mirror=False)
            s, f = d > SOLID, d >= FOOT
            crec["classes"][cl] = dict(solid_mm3=round(float(s.sum() * vv), 3), footprint_mm3=round(float(f.sum() * vv), 3),
                                       footprint_over_solid=round(float(f.sum() / s.sum()), 2) if s.any() else None,
                                       solid_frac_in_brain=round(float((s & brain).sum() / s.sum()), 4) if s.any() else None)
        for gname, cl_ in (("variant_bundles", BUNDLE_A), ("variant_terminals", TERM_A), ("v3_bundles", V3_FIB), ("v3_terminals", V3_TERM)):
            g = group(S, cl_, mirror=False)
            if g is None:
                crec[gname] = None
                continue
            s, f = g > SOLID, g >= FOOT
            ys_s = np.where(s.any((0, 2)))[0]; ys_f = np.where(f.any((0, 2)))[0]
            emp = lambda ys, m: round(float(1 - m.any((0, 2))[ys.min():ys.max() + 1].mean()), 4) if len(ys) else None   # noqa: E731
            lab, n = ndi.label(s, structure=np.ones((3, 3, 3)))
            big = np.bincount(lab.ravel())[1:].max() if n else 0
            crec[gname] = dict(solid_mm3=round(float(s.sum() * vv), 3), footprint_mm3=round(float(f.sum() * vv), 3),
                               solid_frac_in_brain=round(float((s & brain).sum() / s.sum()), 4) if s.any() else None,
                               solid_frac_on_injection_side=(round(float((np.sign(xw[np.where(s)[0]]) == side).mean()), 4)
                                                             if side is not None and s.any() else None),
                               empty_coronal_planes_share_solid=emp(ys_s, s), empty_coronal_planes_share_footprint=emp(ys_f, f),
                               solid_components_26=int(n), solid_largest_component_share=round(float(big / s.sum()), 4) if s.any() else None,
                               y_range_solid_mm=[round(float(yw[ys_s.min()]), 2), round(float(yw[ys_s.max()]), 2)] if len(ys_s) else None)
        if S == "MR252":                                                       # how MM2's ungraded "Bundle" relates to its graded classes
            b = density(S, "Bundle", mirror=False) > SOLID
            o = group(S, ["Fiber_dense_bundle", "Fiber_moderate_bundle", "Fiber_light_bundle"], mirror=False) > SOLID
            crec["bundle_class_overlap"] = dict(bundle_solid_mm3=round(float(b.sum() * vv), 3), graded_solid_mm3=round(float(o.sum() * vv), 3),
                                                share_of_bundle_in_graded=round(float((b & o).sum() / b.sum()), 4),
                                                share_of_graded_in_bundle=round(float((b & o).sum() / o.sum()), 4))
        rec["cases"][S] = crec
        print(S, json.dumps({k: crec[k] for k in ("variant_bundles", "v3_bundles")}), flush=True)

    # ------------------------------------------------------------------ the main-figure variants A and B -------------------------
    def main_figure(out):
        wl, hl = ext[1] * WS, ext[2] * WS
        wd, hd = ext[0] * WS, ext[1] * WS
        TOP, LEFT, GAPL, gb = 11.0, 9.0, 9.0, 2.5
        xb0 = LEFT + wl + GAPL + wd + GAPL
        pi_ = (177.0 - 3.0 - xb0 - (len(inj_cases) - 1) * gb) / max(len(inj_cases), 1)
        H1 = max(hl, hd, pi_)
        TOP2 = TOP + H1 + 14.0
        pb, hb = (ALIC_X[1] - ALIC_X[0]) * ZB, (ALIC_Z[1] - ALIC_Z[0]) * ZB
        xk = LEFT + len(ALIC_Y) * pb + (len(ALIC_Y) - 1) * gb + GAPL
        TOP3 = TOP2 + hb + 9.0
        # (d): rows = levels, columns = cases; each row cropped to its own brain box (+1 mm), the same crop in every column
        rows_ = []
        for y in SLABS_Y:
            j = int(np.argmin(np.abs(yw - y))); bs = brain[:, j - 1:j + 2, :].any(1)
            xi, zi = np.where(bs.any(1))[0], np.where(bs.any(0))[0]
            rows_.append((y, (xi.min() - 4, xi.max() + 5), (zi.min() - 4, zi.max() + 5)))
        cw = max((x1 - x0) for _, (x0, x1), _z in rows_) * vox * WD
        LEFTD, gx, gy = 22.0, 2.0, 2.5
        assert LEFTD + len(CASES) * cw + (len(CASES) - 1) * gx <= 178, "(d) too wide"
        hrow = [(z1 - z0) * vox * WD for _, _x, (z0, z1) in rows_]
        Htot = TOP3 + 8.0 + sum(hrow) + gy * (len(rows_) - 1) + 7.0
        fig = plt.figure(figsize=(FS.FULL_W_IN, Htot * FS.MM))

        # (a)
        axl = FS.axes_mm(fig, LEFT, TOP, wl, hl); axd = FS.axes_mm(fig, LEFT + wl + GAPL, TOP, wd, hd)
        for ax, axis, (k0, k1), plane, head in ((axl, 0, (1, 2), "sagittal", "Lateral"), (axd, 2, (0, 1), "axial", "Dorsal")):
            sil = box(brain.mean(axis), k0, k1)
            ax.imshow(disp(sil / sil.max()), cmap="gray", vmin=0, vmax=1.6)
            for S in DRAW_ORDER:
                if S in CASES and G[S]["fib"] is not None:
                    fill(ax, disp(box(G[S]["fib"].max(axis), k0, k1)), color(S))
                    if G[S].get("fib_u") is not None:
                        line(ax, disp(box(G[S]["fib_u"].max(axis), k0, k1)), color(S), lw=0.35)
            for S in CASES:
                if G[S]["inj"] is not None:
                    inj_outline(ax, disp(box(G[S]["inj"].max(axis), k0, k1)))
            keep_limits(ax, disp(sil).shape)
            ax.set_axis_off(); FS.header(ax, head, dx=0, dy=2.5); FS.dirs(ax, plane)
        FS.letter(axl, "a", dy=21.0)
        FS.header(axl, "Glass-brain projections, bundles", dx=11, dy=21.0)
        FS.scalebar(axl, 1.0 / vox, 10.0)

        # (b) injection sites, 0.5 mm slabs
        axi, inj_info = [], {}
        for q, S in enumerate(inj_cases):
            m3 = G[S]["inj"] > SOLID
            c_ = np.argwhere(m3).mean(0); cx, cy, cz = xw[0] + c_[0] * vox, yw[0] + c_[1] * vox, zw[0] + c_[2] * vox
            ys = float(yw[int(round(c_[1]))])
            xr2, zr2 = rng(xw, (cx - INJ_FIELD / 2, cx + INJ_FIELD / 2)), rng(zw, (cz - INJ_FIELD / 2, cz + INJ_FIELD / 2))
            cr2 = lambda P: P[xr2][:, zr2]                                     # noqa: E731
            ax = FS.axes_mm(fig, xb0 + q * (pi_ + gb), TOP, pi_, pi_); axi.append(ax)
            bg = disp(cr2(slab(t2, yw, ys)))
            ax.imshow(bg, cmap="gray", vmin=0, vmax=1)
            jj = int(round(c_[1])); F = disp(cr2(G[S]["inj"][:, jj - 1:jj + 2, :].max(1)))   # any solid site voxel in the slab
            fill(ax, F, color(S), alpha=INJ_FILL, lw=0.0, edge=color(S)); inj_outline(ax, F)
            keep_limits(ax, bg.shape); ax.set_axis_off()
            if q == 0:
                FS.letter(ax, "b", dy=21.0); FS.header(ax, "Injection sites", dx=11, dy=21.0); FS.scalebar(ax, 1.0 / vox, 2.0)
            FS.header(ax, name(S), dx=0, dy=11.0); FS.header(ax, f"y = {ys:.1f} mm", dx=0, dy=2.5)
            inj_info[S] = dict(centroid_nmt_mm=[round(float(v), 2) for v in (cx, cy, cz)], slab_y_mm=ys, solid_voxels=int(m3.sum()),
                               solid_mm3=round(float(m3.sum() * vv), 3), mirrored=S in MIRROR,
                               drawn="any voxel with density > 0.5 in the planes y - 0.25, y, y + 0.25 (the site is small under the solid rule)")
        if axi:
            FS.dirs_row(axi[0], axi[-1], "coronal")
            no_inj = [S for S in CASES if S not in inj_cases]
            if no_inj:
                axi[min(1, len(axi) - 1)].annotate(", ".join(FS.label(S) for S in no_inj) + ": no injection site annotated", xy=(0, 0),
                                                   xycoords="axes fraction", xytext=(0, -4), textcoords="offset points", ha="left",
                                                   va="top", fontsize=FS.MIN_PT, color=FS.FG)

        # (c) ALIC: 0.5 mm slabs on FA; bundles inside the capsule in full color, outside faded; capsule dotted
        xr, zr = rng(xw, ALIC_X), rng(zw, ALIC_Z)
        crop = lambda P: P[xr][:, zr]                                         # noqa: E731
        axb, counts = [], {}
        for q, y in enumerate(ALIC_Y):
            ax = FS.axes_mm(fig, LEFT + q * (pb + gb), TOP2, pb, hb); axb.append(ax)
            bg = disp(crop(slab(fa, yw, y))); ax.imshow(bg, cmap="gray", vmin=0, vmax=1)
            j = int(np.argmin(np.abs(yw - y))); icm = crop(ic[:, j - 1:j + 2, :].any(1)); counts[f"{y:.1f}"] = {}
            for S in DRAW_ORDER:
                if S not in CASES or G[S]["fib"] is None:
                    continue
                F = crop(slab(G[S]["fib"], yw, y))
                ii, kk = np.nonzero((F > SOLID) & icm)
                counts[f"{y:.1f}"][S] = dict(voxels_in_capsule=int(len(ii)), voxels_total=int((F > SOLID).sum()),
                                             centroid_x_mm=round(float(xw[xr][ii].mean()), 2) if len(ii) else None,
                                             centroid_z_mm=round(float(zw[zr][kk].mean()), 2) if len(ii) else None)
                fill(ax, disp(np.where(icm, 0, F)), color(S), alpha=OUT_A, lw=0.0, edge=color(S))
                fill(ax, disp(np.where(icm, F, 0)), color(S))
            ax.contour(disp(crop(sarm[:, j, :] == IC_LABEL).astype(float)), [0.5], colors=[FS.FG], linewidths=0.7, linestyles=[(0, (1, 1.2))])
            keep_limits(ax, bg.shape); ax.set_axis_off()
            head2(ax, "Internal capsule, anterior limb, bundles" if q == 0 else "", f"y = {y:.1f} mm", "c" if q == 0 else None)
        FS.dirs_row(axb[0], axb[-1], "coronal"); FS.scalebar(axb[0], 1.0 / vox, 2.0)

        # key
        rows_k = [("heads", None, None)] + [("case", S, name(S)) for S in CASES] + [("line", None, "Injection site"),
                                                                                   ("dots", None, "Internal capsule (atlas)")]
        kh = hb; pitch = kh / len(rows_k); kw = 177.0 - xk
        lax = FS.axes_mm(fig, xk, TOP2, kw, kh); lax.set_xlim(0, kw); lax.set_ylim(kh, 0); lax.axis("off")
        for i, (kind, v, text) in enumerate(rows_k):
            yc = (i + 0.5) * pitch
            if kind == "heads":
                for x_h, t_h in ((0.0, "Bundles"), (10.5, "Terminals (d)")):
                    lax.text(x_h, yc, t_h, va="center", ha="left", fontsize=FS.MIN_PT, color=FS.GRAY)
                continue
            if kind == "case":
                lax.add_patch(Rectangle((0, yc - 0.9), 3.0, 1.8, facecolor=color(v), alpha=FILL_A, edgecolor=DARK, lw=0.35))
                if G[v]["term"] is not None and (G[v]["term"] > SOLID).any():
                    lax.add_patch(Rectangle((14.0, yc - 0.9), 3.0, 1.8, facecolor="#808080", edgecolor=tint(color(v)), lw=0.6))
                lax.text(27.0, yc, text, va="center", ha="left", fontsize=FS.MIN_PT)
                continue
            if kind == "line":
                ln, = lax.plot([0, 17.0], [yc, yc], color=FS.COL["outline"], lw=INJ_LW)
                ln.set_path_effects([pe.withStroke(linewidth=INJ_LW + 1.4, foreground="black")])
            else:
                lax.plot([0, 17.0], [yc, yc], color=FS.FG, lw=0.7, ls=(0, (1, 1.2)))
            lax.text(27.0, yc, text, va="center", ha="left", fontsize=FS.MIN_PT)

        # (d) one column per case, rows = levels, 0.5 mm slabs
        y_ = TOP3 + 8.0; first = None
        for r, ((y, (x0, x1), (z0, z1)), h) in enumerate(zip(rows_, hrow)):
            w_ = (x1 - x0) * vox * WD
            for c, S in enumerate(CASES):
                xc = LEFTD + c * (cw + gx) + (cw - w_) / 2
                ax = FS.axes_mm(fig, xc, y_, w_, h)
                cr = lambda P: P[x0:x1, z0:z1]                                # noqa: E731
                bg = disp(cr(slab(t2, yw, y))); ax.imshow(bg, cmap="gray", vmin=0, vmax=1)
                if G[S]["fib"] is not None:
                    fill(ax, disp(cr(slab(G[S]["fib"], yw, y))), color(S))
                    if G[S].get("fib_u") is not None:
                        line(ax, disp(cr(slab(G[S]["fib_u"], yw, y))), color(S), lw=0.35)
                if G[S]["term"] is not None:
                    line(ax, disp(cr(slab(G[S]["term"], yw, y))), tint(color(S)))
                if G[S]["inj"] is not None:
                    inj_outline(ax, disp(cr(slab(G[S]["inj"], yw, y))))
                keep_limits(ax, bg.shape); ax.set_axis_off()
                if r == 0:
                    FS.header(ax, name(S), dx=0, dy=2.5)
                    if c == 0:
                        dx0 = (9.0 - xc) * 72 / 25.4                           # the letter at x = 9 mm, as in (a) and (c)
                        FS.letter(ax, "d", dx=dx0, dy=12.5)
                        FS.header(ax, "Coronal slabs, 0.5 mm; one case per column", dx=dx0 + 11, dy=12.5)
                if c == 0:
                    ax.annotate(f"y = {y:.1f} mm", xy=(0, 0.5), xycoords="axes fraction", xytext=(-(xc - LEFTD) * 72 / 25.4 - 6, 0),
                                textcoords="offset points", ha="right", va="center", fontsize=FS.MIN_PT, color=FS.FG, annotation_clip=False)
                if r == len(rows_) - 1 and c == 0:
                    FS.scalebar(ax, 1.0 / vox, 10.0)
                    ax.annotate("L", xy=(0, -0.12), xycoords="axes fraction", ha="left", va="top", fontsize=FS.MIN_PT, color=FS.GRAY,
                                annotation_clip=False, xytext=(0, -6), textcoords="offset points")
                if r == len(rows_) - 1 and c == len(CASES) - 1:
                    ax.annotate("R", xy=(1, -0.12), xycoords="axes fraction", ha="right", va="top", fontsize=FS.MIN_PT, color=FS.GRAY,
                                annotation_clip=False, xytext=(0, -6), textcoords="offset points")
            y_ += h + gy
        ins = [ACPC, T2_, MASK_, FA_, SARM6] + \
              [Path(f) for S in CASES for f in sorted(glob(str(TT / S / f"{S}_*_mri-template_density.nii.gz")))]
        FS.save(fig, out, [str(p) for p in ins], cmd, md5=False,
                extra=dict(variant="A", cases=CASES, slabs_y_mm=SLABS_Y, alic_y_mm=ALIC_Y, slab_half_mm=SLAB_HALF,
                           print_scale=dict(a=WS, c=ZB, d=WD), colors=rec["colors"]))
        plt.close(fig)
        return counts, inj_info

    if a.figure == "main":
        rec["alic_counts_slab_solid"], rec["injection_sites_b"] = main_figure(out_png); print("->", out_png, flush=True)
        ins = [str(p) for S in CASES for p in sorted(glob(str(TT / S / f"{S}_*_mri-template_density.nii.gz")))] + \
              [str(MASK_), str(SARM6), str(ACPC)]
        rec["outputs"] = [out_png]
        rec["mf8_unbridged_outlined"] = split_rec
        print(provenance.write_numbers("fig_tracer_r058", rec, inputs=ins, md5=False, cmd=cmd))
        return

    # ------------------------------------------------------------------ companion: the two rules side by side (5b) --------------
    S_ = 0.5
    wl, hl = ext[1] * S_, ext[2] * S_; wd, hd = ext[0] * S_, ext[1] * S_
    LEFT, TOP, gx, gy = 26.0, 10.0, 3.0, 5.0
    rowh = max(hl, hd)
    fig = plt.figure(figsize=(FS.FULL_W_IN, (TOP + len(CASES) * (rowh + gy) + 4.0) * FS.MM))
    cols = [("Lateral, density ≥ 0.1", 0, FOOT), ("Lateral, density > 0.5", 0, SOLID),
            ("Dorsal, density ≥ 0.1", 2, FOOT), ("Dorsal, density > 0.5", 2, SOLID)]
    xs = [LEFT, LEFT + wl + gx, LEFT + 2 * wl + 2 * gx + 4.0, LEFT + 2 * wl + wd + 3 * gx + 4.0]
    v3g = {S: dict(fib=group(S, V3_FIB), term=group(S, V3_TERM), inj=group(S, ["Injection_site"])) for S in CASES}
    for r, S in enumerate(CASES):
        yr = TOP + r * (rowh + gy)
        for c, (head, axis, thr) in enumerate(cols):
            k0, k1 = (1, 2) if axis == 0 else (0, 1)
            w_, h_ = (wl, hl) if axis == 0 else (wd, hd)
            ax = FS.axes_mm(fig, xs[c], yr, w_, h_)
            sil = box(brain.mean(axis), k0, k1); ax.imshow(disp(sil / sil.max()), cmap="gray", vmin=0, vmax=1.6)
            for g, al in (("term", 0.45), ("fib", 1.0)):
                if v3g[S][g] is None:
                    continue
                m = disp(box((v3g[S][g] >= thr if thr == FOOT else v3g[S][g] > thr).any(axis), k0, k1))
                rgba = np.zeros(m.shape + (4,)); rgba[m] = (*color(S), al); ax.imshow(rgba, interpolation="nearest")
            if v3g[S]["inj"] is not None:
                inj_outline(ax, disp(box(v3g[S]["inj"].max(axis), k0, k1)))
            keep_limits(ax, sil.T.shape); ax.set_axis_off()
            if r == 0:
                FS.header(ax, head, dx=0, dy=2.5)
            if c == 0:
                ax.annotate(name(S), xy=(0, 0.5), xycoords="axes fraction", xytext=(-4, 0), textcoords="offset points", ha="right",
                            va="center", fontsize=FS.MIN_PT, color=FS.FG, annotation_clip=False)
            if r == len(CASES) - 1 and c == 0:
                FS.scalebar(ax, 1.0 / vox, 10.0)
    FS.save(fig, out_png, [str(p) for S in CASES for p in sorted(glob(str(TT / S / f"{S}_*_mri-template_density.nii.gz")))],
            cmd, md5=False,
            extra=dict(panel="footprints", groups=dict(bundles=V3_FIB, terminals=V3_TERM), rules=[FOOT, SOLID], cases=CASES))
    plt.close(fig); print("->", out_png, flush=True)


if __name__ == "__main__":
    main()
