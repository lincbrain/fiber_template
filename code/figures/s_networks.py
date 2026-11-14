#!/usr/bin/env python3
"""Supplementary figure "The trained networks": each of the paper's six trained networks drawn as its input, the network and its
output, with the facts of each network beside it.

You need
  the records under $LINC_NUMBERS: segmodels_{mri_brainmask,bf_mask2d,bf_mask3d,bf_holes}_2026-11 (Supplementary S5, Table S8) and
  d3_methods (Supplementary S1); the products of record under $LINC_PRODUCTS: the segmentation networks' tasks, training cases and
  held-out masks (settings/internal/products_segmentation.yaml `segmodels`), and the synthesis networks' inputs and outputs
  (settings/internal/products_completion.yaml `completed`, `completion_inputs`). settings/env/linc.yml ($PY).

What it does
  One row per network, the example brain chosen by one rule: MM3, the brain whose held-out masks the training runs kept for every
  segmentation network; the stand-in network's row, settings/figures.yaml networks.standin (MM12: of the eight brains without MRI,
  one whose stand-in b0 looks natural at the back; the author, 12 Nov 2026, chose it from the candidates sheet). The MRI row shows
  MM6's b0 (MRI_CASE): of the cases with a kept held-out mask, the one with the most signal outside the brain after MM3's (the
  fraction of the image's energy outside the manual mask), and one that lies upright as scanned, so it needs no rotation for display.
    input    the image the network reads, as it reads it: the segmentation networks' training case of the brain (the photograph at
             /4, the b0 image, the color volume with its background, the masked color volume), the synthesis networks' b0 input
    output   what the network returned for that input: for the segmentation networks the held-out mask of the cross-validation fold
             that left the brain out (val_pred, written by the trainer); for the synthesis networks the predicted blockface
             brightness of record over the whole brain (the completed volume uses it caudal of the junction plane only)
    plane    the photograph itself (2D); for the 3D networks one sagittal plane of the left hemisphere: in the blockface volumes
             the plane through the most voxels of the tube network's output; in the synthesis rows' b0 images the plane at the same
             distance left of the center of the brain's mask; in the MRI row the plane 4 to 12 mm left of that center that keeps the
             network's mask in one piece (whole_plane)
    scale    one physical scale for every panel, the largest that leaves the widest text line room in its box and keeps the
             figure within networks.max_height_mm; the dashed frame on each input is the network's patch (the whole photograph
             for the 2D network) at that scale; one 10 mm bar, its length above it, in the empty place nearest the bottom right of the last output (figstyle.scalebar_free)
    box      one template for all six (the author, 12 Nov 2026): architecture, levels and parameters; Input; Grid (voxel or pixel
             size, patch or image size); Output; Loss; Training (iterations of a batch); Data (N volumes or photographs of M
             brains), every value read from the records above
  Display: the photograph and color volumes in their own colors with one gain per panel (99.5th percentile of the tissue), gray
  images from 0 to the 99.5th percentile of the tissue; masks white; in the tube row the brain's outline in gray (ON_IMG2).

You get
  --out (default figures/s_networks.pdf), a PNG beside it, and the provenance (planes, brains, every value drawn).

    $PY code/figures/s_networks.py [--out figures/s_networks.pdf]
"""
import argparse
import glob
import json
import re
import os
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO / "code"))
import figstyle as FS  # noqa: E402
from shared import paths  # noqa: E402

EXAMPLE = "MR256"                       # MM3: held-out masks kept for all four segmentation networks (fold 0, 0, 0 and 2)
MRI_CASE = "MR251_b0"                   # the MRI row: MM6's b0, as scanned (the author, 11 Nov 2026)
FOV_MM = (80.0, 64.0)                   # each panel's window (tissue mm, width x height), centered on the brain
SCALE_BAR_MM = 10
BOX_TEXT_X, BOX_RIGHT, LINE = 23.5, 1.5, 3.0          # a network box: its text's x and right inset, its line pitch (mm)
RECORDS = dict(mri="segmodels_mri_brainmask_2026-11", photo="segmodels_bf_mask2d_2026-11", bf="segmodels_bf_mask3d_2026-11",
               tubes="segmodels_bf_holes_2026-11", synth="d3_methods")
TASKS = dict(mri="mri_brainmask", photo="bf_mask2d", bf="bf_mask3d", tubes="bf_holes")


# ---- inputs -----------------------------------------------------------------------------------------------------------------
def record(name, used):
    p = paths.numbers_root() / f"{name}.json"
    used.append(str(p))
    return json.load(open(p))["data"]


def fold_of(task, S):
    return next(k for k, v in task["folds"].items() if S in v)


def canon(p, used):
    used.append(str(p))
    return FS.canonical(nib.load(p))


def case_3d(seg, key, S, used, case_id=None):
    """A 3D segmentation network's training case of brain S (or the case `case_id`) and its held-out mask, both in canonical RAS
    order."""
    t = json.load(open(seg / "tasks" / f"{TASKS[key]}.json")); used.append(str(seg / "tasks" / f"{TASKS[key]}.json"))
    c = next(c for c in t["cases"] if (c["id"] == case_id if case_id else
                                       c["subject"] == S and c.get("counted", True) and c.get("modality", "b0") == "b0"))
    S = c["subject"]
    chans = [canon(paths.rerooted(p), used) for p in c["image"]]
    img, A = (np.stack([d for d, _ in chans], -1), chans[0][1]) if len(chans) > 1 else chans[0]
    k = fold_of(t, S)
    pred, Ap = canon(seg / "runs" / TASKS[key] / f"fold_{k}" / "val_pred" / f"{c['id']}.nii.gz", used)
    assert pred.shape == img.shape[:3] and np.allclose(Ap, A, atol=1e-4), f"{key}: the held-out mask is not on its input's grid"
    return dict(img=img.astype(np.float32), A=A, pred=pred > 0, fold=k, case=c["id"], subject=S)


def case_photo(seg, S, used):
    """The 2D network's photograph of brain S with the largest training mask, and the held-out mask of the fold that left S out."""
    tp = seg / "tasks" / "bf_mask2d.json"; t = json.load(open(tp)); used.append(str(tp))
    raw = Path(paths.rerooted(t["nnunet_raw"]))
    labs = sorted(glob.glob(str(raw / "labelsTr" / f"{S}_*.png")))
    area = [int((np.asarray(Image.open(p)) > 0).sum()) for p in labs]
    lab = labs[int(np.argmax(area))]; cid = Path(lab).stem
    k = fold_of(t, S)
    pi, pp = raw / "imagesTr" / f"{cid}_0000.png", seg / "runs" / "bf_mask2d" / f"fold_{k}" / "val_pred" / f"{cid}.png"
    used += [str(pi), str(pp)]
    img, pred = np.asarray(Image.open(pi)).astype(np.float32), np.asarray(Image.open(pp)) > 0
    assert pred.shape == img.shape[:2]
    return dict(img=img, pred=pred, px=float(t["spacing_by_subject"][S][0]), fold=k, case=cid)


def synth(done, S, used):
    i, A = canon(done / S / f"{S}_b0_input_0p5.nii.gz", used)
    o, Ao = canon(done / S / f"{S}_bf_synth.nii.gz", used)
    return dict(img=i, A=A, out=o, Ao=Ao)


# ---- planes -----------------------------------------------------------------------------------------------------------------
def centroid_x(mask, A):
    """World x (mm) of a canonical mask's center of mass."""
    return float(A[0, 0] * (np.arange(mask.shape[0]) * mask.sum((1, 2))).sum() / mask.sum() + A[0, 3])


def sag(vol, A, x_mm):
    i = FS.world_index(A, "sagittal", x_mm)
    return FS.section(vol, "sagittal", i), i


def whole_plane(mask, A, d_ref, lo=4.0, hi=12.0):
    """The MRI row's sagittal plane: of the planes of the mask's own grid lying lo to hi mm left of its center of mass, the one where
    the mask's largest connected piece (8-connected in the plane) holds the largest share of the plane's mask pixels; ties go to the
    offset nearest d_ref. Returns (world x, offset left of the center, share)."""
    from scipy import ndimage as ndi
    xc, best = centroid_x(mask, A), None
    for i in range(mask.shape[0]):
        x = float(A[0, 0] * i + A[0, 3]); off = xc - x
        if not lo <= off <= hi or not mask[i].any():
            continue
        lab, n = ndi.label(mask[i], structure=np.ones((3, 3)))
        share = float(ndi.sum(mask[i], lab, range(1, n + 1)).max() / mask[i].sum())
        if best is None or (-share, abs(off - d_ref)) < (-best[2], abs(best[1] - d_ref)):
            best = (x, off, share)
    return best


def extent_mm(A, shape2d):
    """imshow extent (mm, y = anterior to the right, z = superior up) of a sagittal display array of a canonical volume."""
    ny, nz = shape2d[1], shape2d[0]
    y0, z0 = A[1, 3] - A[1, 1] / 2, A[2, 3] - A[2, 2] / 2
    return [y0, y0 + ny * A[1, 1], z0, z0 + nz * A[2, 2]]


def gain(rgb, tissue):
    v = rgb[tissue].max(-1) if rgb.ndim == 3 else rgb[tissue]
    return np.clip(rgb / max(np.percentile(v, 99.5), 1e-6), 0, 1)


# ---- drawing ----------------------------------------------------------------------------------------------------------------
def panel(fig, x, y, w, h, img, ext, center, cmap=None, vmax=1.0):
    ax = FS.axes_mm(fig, x, y, w, h)
    ax.imshow(img, extent=ext, cmap=cmap, vmin=0, vmax=vmax, origin="upper", zorder=1)
    cx, cy = center
    ax.set_xlim(cx - FOV_MM[0] / 2, cx + FOV_MM[0] / 2); ax.set_ylim(cy - FOV_MM[1] / 2, cy + FOV_MM[1] / 2)
    ax.set_facecolor(FS.BG); ax.axis("off")
    return ax


def patch_frame(ax, center, w_mm, h_mm):
    cx, cy = center
    ax.add_patch(Rectangle((cx - w_mm / 2, cy - h_mm / 2), w_mm, h_mm, fill=False, ec=FS.FG, lw=0.6, ls=(0, (2.2, 1.6)), zorder=3))


def glyph(ax, x, y, w, h, levels, dim3, head=False):
    """A plain U-Net glyph in axes mm (x, y: lower left): one bar per level down the encoder and back up the decoder, each level
    lower than the one above and its bar half as tall (the grid halves); skip connections as gray lines between the two bars of a
    level. 3D networks: each bar a box with depth. `head`: the 2x upsampling stage after the decoder, one bar twice as tall as the
    first level's, at its height."""
    n = 2 * levels - 1 + (1 if head else 0)
    pitch = w / n; bw = pitch * 0.58
    d = bw * 0.6 if dim3 else 0.0
    h0 = h * (0.30 if head else 0.42)
    hb = [max(h0 / 2 ** l, 0.7) for l in range(levels)]
    top = y + h - (2 * h0 if head else h0) / 2 - d * 0.6
    step = (top - (y + hb[-1] / 2)) / max(levels - 1, 1)
    yc = [top - l * step for l in range(levels)]
    lev = list(range(levels)) + list(range(levels - 2, -1, -1))
    bars = [(x + k * pitch, yc[l], hb[l]) for k, l in enumerate(lev)]
    if head:
        bars.append((x + (n - 1) * pitch, yc[0], 2 * h0))
    for l in range(levels - 1):                                         # skip connections
        ax.plot([bars[l][0] + bw, bars[n - 1 - l - (1 if head else 0)][0]], [yc[l]] * 2, color=FS.GRAY, lw=0.4, zorder=2)
    for xi, ycen, hi in bars:
        y0 = ycen - hi / 2
        ax.add_patch(Rectangle((xi, y0), bw, hi, fc="#3a3a3a", ec=FS.FG, lw=0.4, zorder=3))
        if dim3:
            ax.add_patch(Polygon([(xi, y0 + hi), (xi + d, y0 + hi + d * 0.6), (xi + bw + d, y0 + hi + d * 0.6), (xi + bw, y0 + hi)],
                                 closed=True, fc="#5a5a5a", ec=FS.FG, lw=0.4, zorder=3))
            ax.add_patch(Polygon([(xi + bw, y0), (xi + bw + d, y0 + d * 0.6), (xi + bw + d, y0 + hi + d * 0.6), (xi + bw, y0 + hi)],
                                 closed=True, fc="#242424", ec=FS.FG, lw=0.4, zorder=3))


def box(fig, x, y, w, h, title, lines, levels, dim3, head=False):
    ax = FS.axes_mm(fig, x, y, w, h); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.3, 0.3), w - 0.6, h - 0.6, boxstyle="round,pad=0,rounding_size=1.2", fc="#141414", ec=FS.GRAY,
                                lw=0.5, zorder=1))
    gw = BOX_TEXT_X - 4.5
    glyph(ax, 2.5, 2.0, gw, h - 4.0, levels, dim3, head)
    tx = BOX_TEXT_X
    ax.text(tx, h - 2.2, title, fontsize=FS.MIN_PT, fontweight="bold", color=FS.FG, ha="left", va="top")
    for k, s in enumerate(lines):
        ax.text(tx, h - 2.2 - LINE * (k + 1), s, fontsize=FS.MIN_PT, color=FS.FG if k < len(lines) - 1 else FS.GRAY, ha="left",
                va="top")
    return ax


def arrow(fig, x0, x1, y):
    ax = FS.axes_mm(fig, x0, y - 1.5, x1 - x0, 3.0); ax.set_xlim(0, x1 - x0); ax.set_ylim(-1.5, 1.5); ax.axis("off")
    ax.annotate("", xy=(x1 - x0 - 0.3, 0), xytext=(0.3, 0), arrowprops=dict(arrowstyle="-|>", color=FS.FG, lw=0.7,
                                                                            mutation_scale=6))


# ---- the facts of each network, from the records ------------------------------------------------------------------------------
def millions(n):
    return f"{n / 1e6:.1f} M parameters"


def seg_lines(r, inp, grid, data, out):
    net = r["recipe"]["network"]
    title = f"{net['dim']}D U-Net, {len(net['features'])} levels, {millions(r['recipe']['parameters'])}"
    rc = r["recipe"]; it = rc["epochs"] * rc["iterations_per_epoch"]
    unit = "photographs" if net["dim"] == 2 else "patches"
    loss = re.sub(r"\s*\([^)]*\)", "", rc["loss"].split(";")[0]).strip()            # "Dice + cross-entropy"
    return title, len(net["features"]), net["dim"] == 3, [inp, grid, out, f"Loss: {loss}", f"Training: {it:,} iterations of {rc['batch']} {unit}",
                                                         data]


CLASS_OUT = {"brain": "brain mask", "tube": "marker-tube mask"}


def facts(R):
    """Each row's text, one template for all six (the author, 12 Nov 2026: the same fields in the same order and words): the
    architecture, levels and parameters; the input; its grid; the output; the loss; the training's length; the training data.
    Everything from the records."""
    out = {}
    for key in ("mri", "photo", "bf", "tubes"):
        r = R[key]; i = r["input"]; o = r["output"]
        what = CLASS_OUT[o["classes"]["1"]]
        outl = f"Output: {what}, on the input's {'pixels' if i['whole_image'] else 'grid'}"
        if key == "photo":
            pt = i["patch"]; assert i["whole_image"]
            out[key] = seg_lines(r, f"Input: {i['channels']} color channels",
                                 f"Grid: {pt[0]} × {pt[1]} pixels, the whole photograph at 1/{i['native_downsample']}",
                                 f"Data: {r['data']['images_counted']:,} photographs of {r['data']['brains']} brains", outl)
            continue
        ts, pt = i["target_spacing"], i["patch"]; assert len(set(pt)) == 1
        if key == "mri":
            pm = r["data"]["per_modality"]; n_b0, n_t2 = pm["b0"] + pm.get("b0m", 0), pm["t2"]
            assert n_b0 + n_t2 == r["data"]["images_counted"], pm                # t2d: denoised copies, not counted
            assert len(set(ts)) == 1
            out[key] = seg_lines(r, f"Input: {i['channels']} channel, b0 or T2-weighted",
                                 f"Grid: {ts[0]:g} mm voxels, {pt[0]}³ patches ({pt[0] * ts[0]:g} mm)",
                                 f"Data: {r['data']['images_counted']} volumes of {r['data']['brains']} brains ({n_b0} b0, {n_t2} T2-weighted)", outl)
            continue
        assert ts[0] == ts[2] and r["data"]["images_counted"] == r["data"]["brains"]   # in plane x, z; between sections y; one per brain
        inp = "background kept" if key == "bf" else "inside the brain mask"
        out[key] = seg_lines(r, f"Input: {i['channels']} color channels, {inp}",
                             f"Grid: {ts[0]:.2f} × {ts[1]:.2f} mm voxels, {pt[0]}³ patches ({pt[0] * ts[0]:.0f} × {pt[0] * ts[1]:.0f} mm)",
                             f"Data: {r['data']['images_counted']} volumes of {r['data']['brains']} brains", outl)
    s = R["synth"]["models"]
    NAMES = dict(l1="L1", gradient_l1="gradient L1", one_minus_ms_ssim="(1 − MS-SSIM)")
    for key, k, inp, data in (("own", "mri", "the brain's own b0", ""), ("standin", "no_mri", "a stand-in b0", ", from their stand-in b0")):
        g, a, t = s[k]["grids"], s[k]["architecture"], s[k]["training"]
        levels = len(a["features"]) - 1                                # MONAI BasicUNet: five encoder levels, the last number the output stage
        loss = " + ".join((f"{w:g} " if w != 1 else "") + NAMES[n] for n, w in t["loss"].items())
        it = t["epochs"] * t["patches_per_epoch"] // t["batch"]
        assert len(t["brains"]) == t["n_brains"]                       # one b0 and one blockface volume per brain
        out[key] = (f"3D U-Net + 2× upsampling, {levels} levels, {millions(s[k]['parameters'])}", levels, True,
                    [f"Input: 1 channel, {inp}",
                     f"Grid: {g['input_mm']:g} mm voxels, {g['patch_input_vox']}³ cubes ({g['patch_mm']:g} mm)",
                     f"Output: blockface brightness, {g['output_mm']:g} mm voxels",
                     f"Loss: {loss}",
                     f"Training: {it:,} iterations of {t['batch']} cubes",
                     f"Data: {t['n_brains']} volumes of {t['n_brains']} brains{data}"])
    return out


# ---- main -------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figures/s_networks.pdf")
    a = ap.parse_args()
    FS.setup()
    used = []
    R = {k: record(v, used) for k, v in RECORDS.items()}
    F = facts(R)
    seg = paths.products("products_segmentation")["segmodels"]
    PC = paths.products("products_completion")
    lab = json.load(open(REPO / "settings" / "specimen_labels.json"))["labels"]; used.append(str(REPO / "settings" / "specimen_labels.json"))
    order = json.load(open(REPO / "settings" / "specimen_labels.json"))["order"]
    canon_root = PC["completion_inputs"] / "canon_d3n_prod" / "canon_root.json"; used.append(str(canon_root))
    NCFG = yaml.safe_load(open(REPO / "settings" / "figures.yaml"))["networks"]
    standin = NCFG["standin"]                                           # row (f)'s brain (the author, 12 Nov 2026: MM12, a natural back)
    assert standin in json.load(open(canon_root))["brains"], f"{standin} is not a brain without MRI"
    S = EXAMPLE

    photo = case_photo(seg, S, used)
    mri = case_3d(seg, "mri", S, used, MRI_CASE)
    bf, tubes = (case_3d(seg, k, S, used) for k in ("bf", "tubes"))
    own, sti = synth(PC["completed"], S, used), synth(PC["completed"], standin, used)

    # the sagittal plane: through the most tube voxels; elsewhere at the same distance left of the brain's center
    xt = int(np.argmax(tubes["pred"].sum((1, 2))))
    x_tube = float(tubes["A"][0, 0] * xt + tubes["A"][0, 3])
    d_left = centroid_x(bf["pred"], bf["A"]) - x_tube
    assert d_left > 0, "the tubes lie in the left hemisphere"
    planes = {}
    rows = []
    # (key, title, brain, input display, input extent, center, output display, cmap_in, cmap_out, patch (w, h) mm)
    ph = photo; H, W = ph["pred"].shape
    ext = [0, W * ph["px"], 0, H * ph["px"]]
    cen = (ext[1] / 2, ext[3] / 2)                                      # the whole photograph is the network's tile: center on it
    pt = R["photo"]["input"]["patch"]
    rows.append(dict(key="photo", title="Brain mask, blockface photograph", brain=S, inp=gain(ph["img"], ph["pred"]), ext=ext,
                     cen=cen, out=ph["pred"].astype(float), cin=None, patch=(pt[1] * ph["px"], pt[0] * ph["px"]), out_ext=ext))
    planes["photo"] = ph["case"]

    def vol_row(key, title, brain, img, out, A, Aout, brain_mask, patch, cin="gray", outline=None, x_mm=None):
        """One 3D row: the sagittal plane x_mm (by default d_left left of the brain mask's center); the window centered on the
        mask in that plane."""
        x_mm = centroid_x(brain_mask, A) - d_left if x_mm is None else x_mm
        si, i = sag(img, A, x_mm); so, io = sag(out, Aout, x_mm)
        sm = sag(brain_mask.astype(np.float32), A, x_mm)[0] > 0.5
        e, eo = extent_mm(A, si.shape[:2]), extent_mm(Aout, so.shape[:2])
        zz, yy = np.nonzero(sm)
        cen = (e[0] + (yy.min() + yy.max() + 1) / 2 * A[1, 1], e[3] - (zz.min() + zz.max() + 1) / 2 * A[2, 2])
        disp = gain(si, sm)
        planes[key] = dict(brain=brain, x_mm=round(x_mm, 2), index=i, index_out=io)
        ol = None if outline is None else sag(outline.astype(np.float32), A, x_mm)[0] > 0.5
        rows.append(dict(key=key, title=title, brain=brain, inp=disp, ext=e, cen=cen, out=so.astype(float), cin=cin, patch=patch,
                         out_ext=eo, outline=ol))

    ts = R["mri"]["input"]["target_spacing"]; pt = R["mri"]["input"]["patch"]
    x_b, off_b, share_b = whole_plane(mri["pred"], mri["A"], d_left)
    vol_row("mri", "Brain mask, MRI", mri["subject"], mri["img"], mri["pred"].astype(np.float32), mri["A"], mri["A"], mri["pred"],
            (pt[1] * ts[1], pt[2] * ts[2]), x_mm=x_b)
    planes["mri"].update(left_of_center_mm=round(off_b, 2), largest_piece_share=round(share_b, 4))
    for key, title, v in (("bf", "Brain mask, blockface volume", bf), ("tubes", "Marker tubes, blockface volume", tubes)):
        ts = R[key]["input"]["target_spacing"]; pt = R[key]["input"]["patch"]
        tissue = v["img"].max(-1) > 0                                   # the tube network's input is masked to the brain
        vol_row(key, title, S, v["img"], v["pred"].astype(np.float32), v["A"], v["A"], v["pred"] if key == "bf" else tissue,
                (pt[1] * ts[1], pt[2] * ts[2]), cin=None, outline=tissue if key == "tubes" else None, x_mm=x_tube)
    g = R["synth"]["models"]["mri"]["grids"]
    for key, title, brain, v in (("own", "Synthesis from the brain's own b0", S, own),
                                 ("standin", "Synthesis from a stand-in b0", standin, sti)):
        vol_row(key, title, brain, v["img"], v["out"], v["A"], v["Ao"], v["img"] > 0, (g["patch_mm"], g["patch_mm"]))

    # ---- layout (mm): the space (the author, 12 Nov 2026): one print scale for every panel, the largest that both leaves the
    # widest text line room in its box and keeps the figure within its max_height_mm; the page margins outside ----
    M_, PAGE = FS.MARGIN_MM, FS.FULL_W_IN * 25.4
    ar, head, title_h, gap = 5.5, 5.0, 4.6, 1.6                       # ar: the room of an arrow between panel and box
    f_ = plt.figure(); r_ = f_.canvas.get_renderer()
    textw = max(max(f_.text(0, 0, t, fontsize=FS.MIN_PT, fontweight="bold").get_window_extent(r_).width,
                    *(f_.text(0, 0, l_, fontsize=FS.MIN_PT).get_window_extent(r_).width for l_ in lines)) for t, _, _, lines in F.values())
    textw = textw / f_.dpi * 25.4; plt.close(f_)
    bw_min = BOX_TEXT_X + textw + BOX_RIGHT
    ps_w = (PAGE - 2 * M_ - 2 * ar - bw_min) / 2 / FOV_MM[0]
    ps_h = (float(NCFG["max_height_mm"]) - head - M_ - 6 * (title_h + gap) + gap) / (6 * FOV_MM[1])
    PS = min(ps_w, ps_h)
    pw, phh = FOV_MM[0] * PS, FOV_MM[1] * PS
    x_in = M_; x_box = x_in + pw + ar; x_out = PAGE - M_ - pw; bw = x_out - ar - x_box
    row_h = title_h + phh + gap
    Htot = head + 6 * row_h - gap + M_
    fig = plt.figure(figsize=(FS.FULL_W_IN, Htot / 25.4))
    ys = [head + k * row_h for k in range(len(rows))]
    for k, r in enumerate(rows):
        y0 = ys[k] + title_h
        ai = panel(fig, x_in, y0, pw, phh, r["inp"], r["ext"], r["cen"], cmap=r["cin"])
        patch_frame(ai, r["cen"], *r["patch"])
        ao = panel(fig, x_out, y0, pw, phh, r["out"], r["out_ext"], r["cen"], cmap="gray")
        if r.get("outline") is not None:
            ao.contour(r["outline"].astype(float), [0.5], colors=[FS.ON_IMG2], linewidths=0.4, extent=r["ext"], origin="upper", zorder=2)
        FS.letter(ai, "abcdef"[k], dx=0.0, dy=2.0)
        FS.header(ai, f"{r['title']}  ({FS.label(r['brain'])})", dx=9.0, dy=2.0)
        t, levels, dim3, lines = F[r["key"]]
        bx = box(fig, x_box, y0 + 0.6, bw, phh - 1.2, t, lines, levels, dim3, head=r["key"] in ("own", "standin"))
        arrow(fig, x_in + pw + 0.8, x_box - 0.4, y0 + phh / 2); arrow(fig, x_box + bw + 0.4, x_out - 0.8, y0 + phh / 2)
        if k == 0:
            for xx_, s_ in ((x_in, "Input"), (x_box, "Network"), (x_out, "Output")):
                hx = FS.axes_mm(fig, xx_, 0.4, 30, 3.5); hx.axis("off")
                hx.text(0, 0.5, s_, fontsize=FS.MIN_PT, color=FS.GRAY, ha="left", va="center")

    # ONE scale bar, bottom right (the author, 12 Nov 2026): the empty place nearest the figure's bottom-right corner in the last row's
    # output column, 1.5 mm clear of the tissue and of every text (figstyle.scalebar_free, measured)
    FS.scalebar_free(fig, SCALE_BAR_MM, PS, (x_out, PAGE - M_, ys[-1] + title_h, Htot - M_), corner="bottom right")
    used = sorted(set(used))
    cmd = " ".join(["code/figures/s_networks.py"] + sys.argv[1:])
    drawn = {k: dict(title=t, lines=l) for k, (t, _, _, l) in F.items()}
    FS.save(fig, a.out, used, cmd, extra=dict(example=FS.label(S), stand_in_example=FS.label(standin), planes=planes,
                                              tube_plane_left_of_center_mm=round(d_left, 2), print_scale=round(PS, 4), max_height_mm=float(NCFG['max_height_mm']), text_width_mm=round(textw, 2),
                                              mri_case=mri["case"],
                                              fov_mm=FOV_MM, scale_bar_mm=SCALE_BAR_MM, drawn=drawn,
                                              folds={k: v["fold"] for k, v in dict(photo=photo, mri=mri, bf=bf, tubes=tubes).items()}))
    if Path(a.out).suffix != ".png":
        fig.savefig(str(Path(a.out).with_suffix(".png")), dpi=FS.DPI, facecolor=fig.get_facecolor())
    print("->", a.out, f"{FS.FULL_W_IN * 25.4:.0f} x {Htot:.1f} mm")


if __name__ == "__main__":
    main()
