#!/usr/bin/env python3
"""The animation at the top of the repository's README: the template's six channels in a 2 x 3 grid, coronal planes sweeping
front to back through the brain.

    python code/figures/readme_animation.py [--tree $LINC_DATA_TREE] [--out docs/media/template_coronal.gif] [--step-mm 0.75]
    make -C code/figures template-animation

Panels: T2-weighted, b0, Tensor (the principal diffusion direction, FA-weighted color, template_DEC, display gamma 1/1.6 as in the
template figure); Soma fraction, Neurite fraction (0-1, turbo), Blockface (the color channel, inside its own mask). The MRI channels
are shown inside the template's brain mask; T2-weighted and b0 are normalized to their 99.5th percentile in the brain.
Frames: coronal planes 0.75 mm apart (3 voxels of the 0.25 mm grid) through the brain mask's front-to-back extent, front to back,
then a clean restart (84 frames). 10 frames per second, the first and last held 0.6 s. Each panel is the union of the brain and
blockface masks' coronal extent (+ 0.5 mm), the same in every frame, at one print scale (960 px wide: about 295 px per panel);
neurological orientation (L and R once, beside the first row). Each title centered over its panel. Below the grid, centered under
it, one 10 mm bar, 6 mm of space and a running "y = ... mm" counter (NMT v2), in a strip CLEAR_MM (1.5 mm) under the tiles, so
every frame's tissue keeps that from them. (0.5 mm steps make a 10.4 MB GIF, 1 mm steps 5.2 MB; 0.75 mm: 6.9 MB.)
Reads <tree>/template/outputs/template_{T2w,b0,DEC,soma_fraction,neurite_fraction,blockface_rgb,blockface_mask,brain_mask}.nii.gz.
Writes the GIF: one 256-color palette for every frame, 128 of its colors for the tensor panel and 48 for the blockface (palette()),
no dithering, no comment or other metadata block. No provenance file is written next to it (the repository keeps no paths or host
names).
"""
import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import nibabel as nib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "code"))
import figstyle as S                                           # noqa: E402

CH = [("T2w", "T2-weighted"), ("b0", "b0"), ("DEC", "Tensor"),                       # the grid, row by row
      ("soma_fraction", "Soma fraction"), ("neurite_fraction", "Neurite fraction"), ("blockface_rgb", "Blockface")]
NCOL = 3
FPS, HOLD_S, PAD_MM, DEC_GAMMA, BAR_MM, BAR_GAP_MM = 10, 0.6, 0.5, 1.6, 10, 6.0
STEP_MM = 0.75                                                   # the planes' spacing (mm)
COLOR = ("DEC", "blockface_rgb")                                 # the panels shown in RGB


def render(tree, width=960, step_mm=STEP_MM):
    """The frames (H x W x 3 uint8, front to back), their durations (ms), every tile's pixel box {key: (r0, r1, c0, c1)}, the planes'
    y (mm) and the print scale."""
    T = Path(tree) / "template" / "outputs"
    import matplotlib
    matplotlib.use("Agg")
    S.setup()
    import matplotlib.pyplot as plt

    load = lambda n: S.canonical(nib.load(str(T / f"template_{n}.nii.gz")))   # noqa: E731
    brain, A = load("brain_mask"); brain = brain > 0
    bfm = load("blockface_mask")[0] > 0
    vx = abs(A[0, 0])
    vols = {}
    for key, _ in CH:
        v = load(key)[0].astype(np.float32)
        if key in ("T2w", "b0"):
            v = np.where(brain, v / np.percentile(v[brain], 99.5), np.nan)
        elif key == "DEC":
            v = np.clip(v / 255.0, 0, 1) ** (1 / DEC_GAMMA) * brain[..., None]
        elif key == "blockface_rgb":
            v = np.clip(v, 0, 1) * bfm[..., None]
        else:
            v = np.where(brain, v, np.nan)
        vols[key] = v
    # the frames: coronal planes step_mm apart (whole voxels) through the brain's front-to-back extent, front (anterior) first
    step = int(round(step_mm / vx)); assert abs(step * vx - step_mm) < 1e-6, f"step {step_mm} mm is not a whole number of voxels"
    js = np.where(brain.any((0, 2)))[0]
    planes = list(range(int(js.max()), int(js.min()) - 1, -step))
    ys = [float(A[1, 3] + A[1, 1] * j) for j in planes]
    # the panel box: the union of the masks over every plane (+ PAD_MM), in the coronal display (rows z top-down, columns x)
    union = brain | bfm
    proj = S.section(union.any(1, keepdims=True).astype(np.uint8), "coronal", 0) > 0
    rr, cc = np.where(proj.any(1))[0], np.where(proj.any(0))[0]; p = int(round(PAD_MM / vx))
    r0, r1, c0, c1 = max(rr[0] - p, 0), min(rr[-1] + 1 + p, proj.shape[0]), max(cc[0] - p, 0), min(cc[-1] + 1 + p, proj.shape[1])
    w_mm, h_mm = (c1 - c0) * vx, (r1 - r0) * vx
    ctext = lambda y: f"y = {y:.1f} mm".replace("-", "\u2212")     # noqa: E731  the counter, with a true minus sign
    # layout (mm): NCOL columns, the letters outside the first row, one gutter between the panels and between the rows (below a
    # row's lowest panel edge to the next row's titles), the titles 2 pt above each panel; below the grid, a strip CLEAR_MM under the
    # tiles (so the tissue, inside each tile, keeps at least that) with the bar, BAR_GAP_MM of space and the counter, centered
    W = S.FULL_W_IN * 25.4; pt = 25.4 / 72
    f_ = plt.figure(); r_ = f_.canvas.get_renderer()
    ext = lambda s_: f_.text(0, 0, s_, fontsize=S.MIN_PT).get_window_extent(r_)   # noqa: E731
    mm = lambda e: (e.width / f_.dpi * 25.4, e.height / f_.dpi * 25.4)            # noqa: E731
    wl = max(mm(ext(s_))[0] for s_ in "LR"); th = mm(ext("Tg"))[1]
    bw_t, bh_t = mm(ext(f"{BAR_MM} mm")); cw = max(mm(ext(ctext(y)))[0] for y in ys)
    plt.close(f_)
    nrow = -(-len(CH) // NCOL)
    side = S.MARGIN_MM + wl + 2.0 * pt
    pw = (W - 2 * side - (NCOL - 1) * S.GUTTER_MM) / NCOL; scale = pw / w_mm; ph = h_mm * scale
    head = th + 2.0 * pt                                                            # a title's room above its panel
    tops = [S.MARGIN_MM + head + r * (ph + S.GUTTER_MM + head) for r in range(nrow)]
    pair_w = max(BAR_MM * scale, bw_t); pair_h = bh_t + (2.0 + 1.2) * pt          # the bar's length, its 2 pt gap and the bar
    strip = tops[-1] + ph + S.CLEAR_MM                                              # the strip's top (figure mm from the top)
    H = strip + pair_h + S.MARGIN_MM
    dpi = width / S.FULL_W_IN
    fig = plt.figure(figsize=(S.FULL_W_IN, H / 25.4), dpi=dpi)
    axes, ims = {}, {}
    for k, (key, title) in enumerate(CH):
        ax = S.axes_mm(fig, side + (k % NCOL) * (pw + S.GUTTER_MM), tops[k // NCOL], pw, ph); ax.set_axis_off(); axes[key] = ax
        if key in COLOR:
            ims[key] = ax.imshow(np.zeros((r1 - r0, c1 - c0, 3)), interpolation="antialiased")
        else:
            cm = plt.get_cmap("gray" if key in ("T2w", "b0") else "turbo").copy(); cm.set_bad(S.BG)
            ims[key] = ax.imshow(np.full((r1 - r0, c1 - c0), np.nan), cmap=cm, vmin=0, vmax=1, interpolation="antialiased")
        ax.annotate(title, xy=(0.5, 1), xycoords="axes fraction", xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                    fontsize=S.MIN_PT, color=S.FG, annotation_clip=False)              # each title centered over its panel
    S.dirs_row(axes[CH[0][0]], axes[CH[NCOL - 1][0]], "coronal")                    # L and R once, beside the first row
    x0 = W / 2 - (pair_w + BAR_GAP_MM + cw) / 2                                     # bar, space, counter: centered under the grid
    S.scalebar_mm(fig, x0, strip + pair_h - 0.6 * pt, BAR_MM, scale, panels=False)
    counter = fig.text((x0 + pair_w + BAR_GAP_MM) / W, 1 - (strip + pair_h / 2) / H, "", ha="left", va="center", fontsize=S.MIN_PT,
                       color=S.GRAY)

    def show(j, y):
        for key, _ in CH:
            v = vols[key]
            sl = S.section(v, "coronal", j) if v.ndim == 3 else np.stack([S.section(v[..., c], "coronal", j) for c in range(3)], -1)
            ims[key].set_data(sl[r0:r1, c0:c1])
        counter.set_text(ctext(y))

    k = int(np.argmax([len(ctext(y)) for y in ys]))
    show(planes[k], ys[k])
    bad = S.check(fig)                                                  # the house rules, the 1.5 mm clearance of the bar included
    fig.set_dpi(dpi)
    if bad:
        raise SystemExit("readme_animation: house rules broken:\n  " + "\n  ".join(bad))
    frames = []
    for j, y in zip(planes, ys):
        show(j, y); fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    r_ = fig.canvas.get_renderer(); Hp = frames[0].shape[0]; boxes = {}
    for key, ax in axes.items():
        e = ax.get_window_extent(r_); boxes[key] = (int(Hp - e.y1), int(np.ceil(Hp - e.y0)), int(e.x0), int(np.ceil(e.x1)))
    plt.close(fig)
    dur = [int(round(1000 / FPS))] * len(frames); dur[0] = dur[-1] = int(HOLD_S * 1000)
    return frames, dur, boxes, ys, scale


SHARE = dict(DEC=128, blockface_rgb=48)                          # of the GIF's 256 colors, the color panels' own; the rest for all else


def palette(frames, boxes, share=SHARE):
    """One 256-color palette for every frame, from 12 evenly spaced frames: share[key] colors for each color panel's tissue (the
    tensor's many hues, the blockface's tans), the rest for every other pixel (the gray and turbo panels, the text, the black). Each
    group: median cut, then 12 k-means iterations on up to 300,000 of its pixels (a fixed seed), which lowers the color error."""
    from PIL import Image
    from scipy.cluster.vq import kmeans2
    pick = [frames[i] for i in np.linspace(0, len(frames) - 1, 12).round().astype(int)]

    def colors(px, n):
        init = np.array(Image.fromarray(px[:, None, :]).quantize(colors=n, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
                        .getpalette()[:3 * n], np.float64).reshape(n, 3)
        s = px[np.random.default_rng(0).choice(len(px), min(len(px), 300000), replace=False)].astype(np.float64)
        with warnings.catch_warnings():                                # a cluster that empties keeps its median-cut color
            warnings.simplefilter("ignore", UserWarning)
            c, _ = kmeans2(s, init, iter=12, minit="matrix", missing="warn")
        return np.clip(np.rint(c), 0, 255).astype(np.uint8)
    taken = np.zeros(pick[0].shape[:2], bool); parts = []
    for key, n in share.items():
        r0, r1, c0, c1 = boxes[key]
        inside = np.zeros_like(taken); inside[r0:r1, c0:c1] = True; taken |= inside
        px = np.concatenate([f[inside] for f in pick]); parts.append(colors(px[px.max(1) > 8], n))   # the tile's tissue
    parts.append(colors(np.concatenate([f[~taken] for f in pick]), 256 - sum(share.values())))
    img = Image.new("P", (1, 1)); img.putpalette(np.concatenate(parts).ravel().tolist())
    return img


def encode(frames, dur, out, boxes, share=SHARE):
    """The GIF: every frame mapped to one palette (palette()), no dithering (steady colors, small frame deltas). No comment or other
    metadata block: only the loop block and the frames' delays."""
    from PIL import Image
    pal = palette(frames, boxes, share)
    q = [Image.fromarray(f).quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    q[0].save(out, save_all=True, append_images=q[1:], duration=dur, loop=0, optimize=False, disposal=1)
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", default=os.environ.get("LINC_DATA_TREE"), help="the data tree (default $LINC_DATA_TREE)")
    ap.add_argument("--out", default=str(HERE.parents[1] / "docs" / "media" / "template_coronal.gif"))
    ap.add_argument("--width", type=int, default=960, help="pixels")
    ap.add_argument("--step-mm", type=float, default=STEP_MM, help="the planes' spacing (a whole number of 0.25 mm voxels)")
    a = ap.parse_args()
    if not a.tree:
        raise SystemExit("set --tree or LINC_DATA_TREE (settings/roots.sh)")
    frames, dur, boxes, ys, scale = render(a.tree, a.width, a.step_mm)
    encode(frames, dur, a.out, boxes)
    out = Path(a.out)
    print(f"{out}: {len(frames)} frames, {frames[0].shape[1]} x {frames[0].shape[0]} px, {out.stat().st_size / 1e6:.2f} MB; "
          f"y {ys[0]:.1f} to {ys[-1]:.1f} mm; print scale {scale:.3f}")


if __name__ == "__main__":
    main()
