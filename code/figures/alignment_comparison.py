#!/usr/bin/env python3
"""The paper's figure "Blockface-based and MRI-based alignment": the alignment comparison on design 1c's pair folds.

    python code/figures/alignment_comparison.py [--out figures/alignment_comparison.png] [--pair pair_k]

  a  per pair of brains left out together (the unit), the measures themselves after each alignment, as paired points joined by a line:
     above, the NCC between the two brains' blockface volumes (on the pre-registered evaluation mask); below, the Dice between their
     brain masks (inside the fold average's core); blockface-based and MRI-based in the two alignments' colors
  b  the carried images of one pair (the pair with the median NCC difference), one coronal section of its fold average (header: its
     NMT v2 y through the fold's own link): rows = alignment, columns = the two brains (each in the crop's own 2nd-98th percentile window
     inside its mask) and their absolute difference after normalizing each to zero mean and unit variance where both have tissue (as
     NCC does; one window for both alignments; darker = more alike); true scale, scale bar 2 mm. The crop is centered on the section's
     tissue (settings/figures.yaml alignment_comparison).
Reads $LINC_NUMBERS/route_test_1c.json (code/figures/paper_numbers/route_test_1c.py) and the route test's products (settings/internal/products_blockface.yaml
`route_test*`). Also writes $LINC_NUMBERS/fig_routes_1c_section.json (the section's NMT v2 y, the caption's token).
Ported from the analysis project's paper/scripts/fig9_routes_1c_r061.py: its predecessor, fig9_routes_1c_r057.py, with panel (a)
showing the two alignments' values per pair instead of their differences (panel (b) and every number unchanged).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO / "code"))
import figstyle as FS  # noqa: E402
from shared import paths, provenance  # noqa: E402

CFG = yaml.safe_load((REPO / "settings" / "figures.yaml").read_text())["alignment_comparison"]
FOV, ZOOM, ZPCT = CFG["fov_mm"], CFG["zoom"], CFG["crop_height_pct"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figures/alignment_comparison.png")
    ap.add_argument("--pair", default=None, help="the example pair of (b); default = the pair with the median NCC difference")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401
    from scipy import ndimage as ndi
    from matplotlib.colors import ListedColormap
    FS.setup()
    NUM = paths.numbers_root() / "route_test_1c.json"
    prod = paths.products("products_blockface")
    OUTD, FOLDS, DIRECT = prod["route_test"], prod["route_test_folds"], prod["route_test_direct"]
    CA, CB, REF = FS.COL["direct"], FS.COL["mri"], FS.GRAY
    d = json.loads(NUM.read_text())["data"]
    pairs = d["pairs"]; keys = sorted(pairs)
    assert all(pairs[k]["gates_passed"] for k in keys), "a plotted pair failed a gate"
    dn = np.array([pairs[k]["A_minus_B"]["ncc"] for k in keys]); dd = np.array([pairs[k]["A_minus_B"]["dice"] for k in keys])
    ex = a.pair or keys[int(np.argsort(dn)[len(dn) // 2])]
    L = FS.label

    TOP, H, BG = 9.0, 40.0, 5.0; BH = (H - BG) / 2
    side = FOV * ZOOM
    bottom = TOP + max(H + 10.0, 2 * side + 4.0 + 2.5)                       # (a): two lines of tick labels; (b): the scale bar
    fig = plt.figure(figsize=(FS.FULL_W_IN, (bottom + (TOP - 4.5)) * FS.MM))    # the blank below = the blank above (~4.5 mm)

    # a: the values themselves per pair, blockface-based and MRI-based as paired points; NCC above, mask Dice below
    AX, AW = 16.0, 52.0; xs = np.arange(len(keys)); OFF = 0.13
    axa = [FS.axes_mm(fig, AX, TOP, AW, BH), FS.axes_mm(fig, AX, TOP + BH + BG, AW, BH)]
    va = {m: np.array([pairs[k]["A"][m] for k in keys]) for m in ("ncc", "dice")}
    vb = {m: np.array([pairs[k]["B"][m] for k in keys]) for m in ("ncc", "dice")}
    assert np.allclose(va["ncc"] - vb["ncc"], dn) and np.allclose(va["dice"] - vb["dice"], dd)    # the record's own differences
    for ax, m, ylab in ((axa[0], "ncc", "blockface NCC"), (axa[1], "dice", "brain-mask Dice")):
        for i in range(len(keys)):
            ax.plot([xs[i] - OFF, xs[i] + OFF], [va[m][i], vb[m][i]], color=REF, lw=0.6, zorder=2)
        ax.scatter(xs - OFF, va[m], s=11, color=CA, lw=0, zorder=3, label="blockface-based")
        ax.scatter(xs + OFF, vb[m], s=11, color=CB, lw=0, zorder=3, label="MRI-based")
        lo, hi = min(va[m].min(), vb[m].min()), max(va[m].max(), vb[m].max()); pad = 0.15 * max(hi - lo, 1e-6)
        ax.set_xlim(-0.6, len(keys) - 0.4); ax.set_ylim(lo - pad, hi + pad); ax.set_ylabel(ylab)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axa[0].set_xticks([]); axa[0].spines["bottom"].set_visible(False)
    axa[1].set_xticks(xs); axa[1].set_xticklabels([f"{L(pairs[k]['brains'][0])}\n{L(pairs[k]['brains'][1])}" for k in keys])
    axa[1].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.4f"))
    axa[0].legend(loc="lower left", frameon=False, handletextpad=0.2, borderaxespad=0.1)
    FS.letter(axa[0], "a", dx=-12); FS.header(axa[0], "the two alignments, per pair of brains", dx=-3)

    # b: the example pair's carried images, coronal section through the center of the pair's evaluation support
    tname = f"d1cs2_{ex}"
    gp = FOLDS / f"{ex}/pub/{tname}/final_t1/{tname}_grey.nii.gz"
    bp = FOLDS / f"{ex}/pub/{tname}/final_t1/{tname}_brainmask.nii.gz"
    g = nib.load(str(gp)); A = g.affine; G = np.asanyarray(g.dataobj).astype(np.float32)
    assert np.allclose(A[:3, :3], np.diag(np.diag(A[:3, :3]))) and (np.diag(A)[:3] > 0).all(), "expects a RAS-diagonal grid"
    vox = float(A[0, 0]); half = int(round(FOV / 2 / vox))
    BM = np.asanyarray(nib.load(str(bp)).dataobj) > 0
    S1, S2 = pairs[ex]["brains"]
    imgs, masks, ins = {}, {}, [str(NUM), str(gp), str(bp)]
    for S in (S1, S2):
        pa = DIRECT / f"route1c_{ex}_A/{S}/{S}_bf_in_template.nii.gz"; ma = DIRECT / f"route1c_{ex}_A/{S}/{S}_mask_in_template.nii.gz"
        pb = OUTD / ex / f"{S}_routeB_grey.nii.gz"; mb = OUTD / ex / f"{S}_routeB_mask.nii.gz"
        for R, p, m in (("A", pa, ma), ("B", pb, mb)):
            im = nib.load(str(p)); assert im.shape[:3] == G.shape and np.allclose(im.affine, A, atol=1e-4), f"{S} {R}: grid"
            imgs[S, R] = np.asanyarray(im.dataobj).astype(np.float32); masks[S, R] = np.asanyarray(nib.load(str(m)).dataobj) > 0
            ins += [str(p), str(m)]
    U = np.logical_and.reduce([masks[k] for k in masks]) & BM
    ci = np.round(np.argwhere(U).mean(0)).astype(int)
    ci[2] = int(np.percentile(np.argwhere(U[:, ci[1], :])[:, 1], ZPCT))       # the crop's height: centered on the section's tissue
    sl = lambda V: FS.section(V[ci[0] - half:ci[0] + half + 1, :, ci[2] - half:ci[2] + half + 1], "coronal", ci[1])   # noqa: E731
    nmt_y = float(FS.to_nmt(str(OUTD / ex / "link"), [A[:3, :3] @ ci + A[:3, 3]])[0][1])   # through the fold's own link (MRI template = NMT v2 mm)
    BX = AX + AW + 18.0; cols = [(S1, L(S1)), (S2, L(S2)), (None, "|difference|")]
    DIFF_MAX = CFG["difference_window_z"]                                     # z units, one window for both routes
    for r, (R, rname, col) in enumerate((("A", "blockface-based", CA), ("B", "MRI-based", CB))):
        Mc = sl((masks[S1, R] & masks[S2, R]).astype(np.float32)) > 0.5       # the crop where both brains have tissue
        z = {}
        for S in (S1, S2):                                                    # zero mean, unit variance inside Mc (as NCC normalizes)
            v = sl(imgs[S, R]); z[S] = np.where(Mc, (v - v[Mc].mean()) / max(float(v[Mc].std()), 1e-6), 0.0)
        for c, (S, cname) in enumerate(cols):
            ax = FS.axes_mm(fig, BX + c * (side + 3.0), TOP + r * (side + 4.0), side, side)
            if S is None:                                                     # the two brains' absolute difference, darker = more alike
                ax.imshow(np.where(Mc, np.clip(np.abs(z[S1] - z[S2]) / DIFF_MAX, 0, 1), 0.0), cmap="gray", vmin=0, vmax=1,
                          interpolation="bilinear")
            else:
                v = sl(imgs[S, R]); m = sl(masks[S, R].astype(np.float32)) > 0.5
                lo, hi = np.percentile(v[m], [2, 98])                         # the crop's own window (whole-brain windows saturate here)
                ax.imshow(np.where(m, np.clip((v - lo) / (hi - lo), 0, 1), 0.0), cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                FS.header(ax, cname, dx=0, dy=2)
            if c == 0:
                ax.text(-0.06, 0.5, rname, transform=ax.transAxes, rotation=90, ha="right", va="center", color=col, fontsize=FS.MIN_PT)
            if r == 1 and c == 2:
                FS.scalebar(ax, 1.0 / vox, 2.0)
            if r == 0 and c == 0:
                FS.letter(ax, "b", dx=-10, dy=2)
            if r == 0 and c == 2:
                ax.text(1.0, -0.02, f"coronal, y = {nmt_y:.1f} mm".replace("-", "−"), transform=ax.transAxes, ha="right", va="top",
                        color=REF, fontsize=FS.MIN_PT)
    FS.save(fig, a.out, ins, " ".join(["code/figures/alignment_comparison.py"] + sys.argv[1:]),
            extra=dict(example_pair=ex, brains=[S1, S2], section_index_ijk=[int(v) for v in ci], section_nmt_y_mm=nmt_y, fov_mm=FOV, zoom=ZOOM,
                       difference_window_z=DIFF_MAX,
                       pairs_A_minus_B=dict(ncc={k: float(v) for k, v in zip(keys, dn)}, dice={k: float(v) for k, v in zip(keys, dd)})))
    provenance.write_numbers("fig_routes_1c_section", dict(example_pair=ex, brains=[S1, S2], section_index_ijk=[int(v) for v in ci],
                                                        section_nmt_y_mm=nmt_y, crop_height_pct=ZPCT, fov_mm=FOV, figure=a.out),
                         inputs=ins, cmd=" ".join(["code/figures/alignment_comparison.py"] + sys.argv[1:]), md5=False)   # the caption's y
    print(a.out, "example", ex, S1, S2, "coronal index", ci.tolist())


if __name__ == "__main__":
    main()
