#!/usr/bin/env python3
"""Supplementary S1, the completed blockface volume of every one of the 21 sectioned brains: the synthesis
network's paper (Gonzales et al. 2026, ISBI) Fig. 2, re-rendered in this paper's house style, extended from its 12 brains to all 21,
with this paper's names.

    python reproduce/figures/s_synthesis_gallery.py [--out figures/s_synthesis_gallery.pdf]

Per brain, from the completed volumes of record (products `completed`) and the prepared folders (`completion_inputs`: canon_d3 for the
13 brains with MRI, canon_d3n_prod for the 8 without), on the brain's section-aligned grid (linc.completion.views.Flat: every plane is
one photographed section, the junction plane is one plane):
  plane    the axial plane with the most synthesized tissue caudal of the junction plane; anterior up, the subject's left on the left
  crop     from the brain's caudal end to 22 mm rostral of the junction plane; the junction planes aligned on one line per row
  scale    one physical scale for all brains (the widest brain fills its tile)
  color    photographed tissue in its own color, cross-faded into the synthesis over 2 mm (the product's seam_ramp_real); synthesized
           tissue colored with the brain's own luminance -> color lookup fitted on its photographs (linc/completion/views.py)
  header   name, and the synthesized share of the brain ($LINC_NUMBERS/d3_predicted_share.json; gray: synthesized from the stand-in b0)
Writes the PDF and a PNG beside it. Ported from the analysis project's paper/scripts/figS_synthesis_gallery.py (the figure unchanged).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO))
import figstyle as FS  # noqa: E402
from linc import paths  # noqa: E402
from linc.completion.data import y_dep_volume  # noqa: E402
from linc.completion.views import Flat, brain_colors, on_black  # noqa: E402

ROSTRAL_MM = 22.0
VOX = 0.25
ROOTS = ["canon_d3", "canon_d3n_prod"]                                 # prepared brains: 13 with MRI, 8 without (stand-in b0)


def load(f):
    return np.asanyarray(nib.load(f).dataobj).astype(np.float32)


def tile(S, root, OUT):
    """one brain's tile, on its own prepared folder"""
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
    z = int(np.argmax((fb & (jj < zf)).sum((0, 1))))
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
    return dict(img=img, cut_row=(y_hi - zf), z=z, inputs=list(ps.values()) + [f"{root}/{S}/seam.json"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figures/s_synthesis_gallery.pdf")
    a = ap.parse_args()
    P = paths.products("products_completion")
    OUT = P["completed"]
    FS.setup()
    lp = REPO / "docs" / "specimen_labels.json"; lab = json.load(open(lp))["labels"]
    sp = paths.numbers_root() / "d3_predicted_share.json"; share = json.load(open(sp))["data"]["per_brain"]
    brains = {}
    for r in ROOTS:
        root = P["completion_inputs"] / r
        for S in json.load(open(root / "canon_root.json"))["brains"]:
            brains[S] = root
    order = sorted(brains, key=lambda s: (lab[s][:2], int(lab[s][2:])))
    assert len(order) == 21
    T = {}
    for S in order:
        T[S] = tile(S, brains[S], OUT); print(lab[S], S, "plane", T[S]["z"], flush=True)
    ncol, gap, hdr = 7, 1.2, 4.0                                       # mm
    nrow = int(np.ceil(len(order) / ncol))
    tw = (FS.FULL_W_IN * 25.4 - gap * (ncol - 1)) / ncol
    mm_per_vox = tw / max(t["img"].shape[1] for t in T.values())
    above = max(t["cut_row"] for t in T.values()) * mm_per_vox
    below = max(t["img"].shape[0] - t["cut_row"] for t in T.values()) * mm_per_vox
    th = above + below
    sb_h = 3.0                                                         # room under the first row for the scale bar
    H = nrow * (hdr + th) + (nrow - 1) * gap + sb_h
    fig = plt.figure(figsize=(FS.FULL_W_IN, H / 25.4))
    for k, S in enumerate(order):
        r, c = divmod(k, ncol)
        t = T[S]; hh, ww = t["img"].shape[:2]
        x0 = c * (tw + gap); y0 = r * (hdr + th + gap) + hdr + (sb_h if r > 0 else 0)
        ax = FS.axes_mm(fig, x0, y0, tw, th)
        ax.set_xlim(0, tw); ax.set_ylim(-below, above); ax.axis("off")
        w_, h_ = ww * mm_per_vox, hh * mm_per_vox
        xi = (tw - w_) / 2; yi = -(hh - t["cut_row"]) * mm_per_vox
        ax.imshow(t["img"], extent=[xi, xi + w_, yi, yi + h_], interpolation="antialiased", zorder=2)
        ax.plot([0.3, tw - 0.3], [0, 0], color=FS.FG, lw=0.6, ls=(0, (2.0, 1.3)), zorder=4)
        nomri = brains[S].name == "canon_d3n_prod"
        FS.header(ax, lab[S], dx=0, fontweight="bold")
        ax.annotate(f"{100 * share[S]['predicted_share']:.0f}%", xy=(1, 1), xycoords="axes fraction", xytext=(0, 3),
                    textcoords="offset points", ha="right", va="bottom", fontsize=FS.MIN_PT,
                    color=FS.GRAY if nomri else FS.FG, annotation_clip=False)
        if k == 0:
            ax.plot([tw - 10.0 / VOX * mm_per_vox - 0.3, tw - 0.3], [-below - 1.2] * 2, color=FS.FG, lw=1.0, clip_on=False,
                    solid_capstyle="butt")
    ins = sorted({str(p) for t in T.values() for p in t["inputs"]} | {str(lp), str(sp)})
    cmd = " ".join(["reproduce/figures/s_synthesis_gallery.py"] + sys.argv[1:])
    FS.save(fig, a.out, ins, cmd, extra=dict(planes={lab[S]: T[S]["z"] for S in order}, scale_bar_mm=10, rostral_mm=ROSTRAL_MM))
    fig.savefig(str(Path(a.out).with_suffix(".png")), dpi=FS.DPI, facecolor=fig.get_facecolor())
    print("->", a.out, f"{FS.FULL_W_IN * 25.4:.0f} x {H:.1f} mm")


if __name__ == "__main__":
    main()
