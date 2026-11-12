#!/usr/bin/env python3
"""The animation at the top of the repository's README: the template's five channels in one row, coronal planes sweeping front to
back through the brain.

    python code/figures/readme_animation.py [--tree $LINC_DATA_TREE] [--out docs/media/template_coronal.gif] [--webp]
    make -C code/figures template_coronal

Panels (as the template blocks of the pipeline figure): T2-weighted; Tensor (the principal diffusion direction, FA-weighted color,
template_DEC, display gamma 1/1.6 as in the template figure); Soma fraction and Neurite fraction (0-1, turbo); Blockface (the color
channel, inside its own mask). The MRI channels are shown inside the template's brain mask. T2-weighted is normalized to its 99.5th
percentile in the brain.
Frames: every second coronal plane of the 0.25 mm grid (0.5 mm apart) through the brain mask's front-to-back extent, front to back,
then a clean restart. 10 frames per second, the first and last held 0.6 s. Each panel is the union of the brain and blockface masks'
coronal extent (+ 0.5 mm), the same in every frame, at one print scale; neurological orientation (L and R once, outside the row).
A running "y = ... mm" counter (NMT v2) and one 10 mm bar, placed in an empty corner of every frame (figstyle.scalebar_free on the
masks' projection along y, where every frame's tissue lies).
Reads <tree>/template/outputs/template_{T2w,DEC,soma_fraction,neurite_fraction,blockface_rgb,blockface_mask,brain_mask}.nii.gz.
Writes the GIF (one global 256-color palette, no dithering, no comment or other metadata block) and, with --webp, an animated WebP
beside it. No provenance file is written next to it (the repository keeps no paths or host names).
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "code"))
import figstyle as S                                           # noqa: E402

CH = [("T2w", "T2-weighted"), ("DEC", "Tensor"), ("soma_fraction", "Soma fraction"), ("neurite_fraction", "Neurite fraction"),
      ("blockface_rgb", "Blockface")]
STEP_VOX, FPS, HOLD_S, PAD_MM, DEC_GAMMA, BAR_MM = 2, 10, 0.6, 0.5, 1.6, 10


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", default=os.environ.get("LINC_DATA_TREE"), help="the data tree (default $LINC_DATA_TREE)")
    ap.add_argument("--out", default=str(HERE.parents[1] / "docs" / "media" / "template_coronal.gif"))
    ap.add_argument("--width", type=int, default=960, help="pixels")
    ap.add_argument("--webp", action="store_true", help="also write an animated WebP beside the GIF")
    a = ap.parse_args()
    if not a.tree:
        raise SystemExit("set --tree or LINC_DATA_TREE (settings/roots.sh)")
    T = Path(a.tree) / "template" / "outputs"
    import matplotlib
    matplotlib.use("Agg")
    S.setup()
    import matplotlib.pyplot as plt
    from PIL import Image

    load = lambda n: S.canonical(nib.load(str(T / f"template_{n}.nii.gz")))   # noqa: E731
    brain, A = load("brain_mask"); brain = brain > 0
    bfm = load("blockface_mask")[0] > 0
    vx = abs(A[0, 0])
    vols = {}
    for key, _ in CH:
        v = load(key)[0].astype(np.float32)
        if key == "T2w":
            v = np.where(brain, v / np.percentile(v[brain], 99.5), np.nan)
        elif key == "DEC":
            v = np.clip(v / 255.0, 0, 1) ** (1 / DEC_GAMMA) * brain[..., None]
        elif key == "blockface_rgb":
            v = np.clip(v, 0, 1) * bfm[..., None]
        else:
            v = np.where(brain, v, np.nan)
        vols[key] = v
    # the frames: every STEP_VOX-th coronal plane through the brain's front-to-back extent, front (anterior) first
    js = np.where(brain.any((0, 2)))[0]
    planes = list(range(int(js.max()), int(js.min()) - 1, -STEP_VOX))
    # the panel box: the union of the masks over every plane (+ PAD_MM), in the coronal display (rows z top-down, columns x)
    union = brain | bfm
    proj = S.section(union.any(1, keepdims=True).astype(np.uint8), "coronal", 0) > 0
    rr, cc = np.where(proj.any(1))[0], np.where(proj.any(0))[0]; p = int(round(PAD_MM / vx))
    r0, r1, c0, c1 = max(rr[0] - p, 0), min(rr[-1] + 1 + p, proj.shape[0]), max(cc[0] - p, 0), min(cc[-1] + 1 + p, proj.shape[1])
    w_mm, h_mm = (c1 - c0) * vx, (r1 - r0) * vx
    # layout (mm): the letters outside the row, one gutter between the panels, the titles and counter above
    W = S.FULL_W_IN * 25.4
    f_ = plt.figure(); r_ = f_.canvas.get_renderer()
    txt = lambda s_: f_.text(0, 0, s_, fontsize=S.MIN_PT).get_window_extent(r_)   # noqa: E731
    wl = max(txt(s_).width for s_ in "LR") / f_.dpi * 25.4; th = txt("Tg").height / f_.dpi * 25.4
    plt.close(f_)
    side = S.MARGIN_MM + wl + 2.0 / 72 * 25.4
    pw = (W - 2 * side - 4 * S.GUTTER_MM) / len(CH); scale = pw / w_mm; ph = h_mm * scale
    top = S.MARGIN_MM + th + 2.0 / 72 * 25.4
    H = top + ph + S.MARGIN_MM
    dpi = a.width / S.FULL_W_IN
    fig = plt.figure(figsize=(S.FULL_W_IN, H / 25.4), dpi=dpi)
    axes, ims = [], []
    for k, (key, title) in enumerate(CH):
        ax = S.axes_mm(fig, side + k * (pw + S.GUTTER_MM), top, pw, ph); ax.set_axis_off(); axes.append(ax)
        if key in ("DEC", "blockface_rgb"):
            im = ax.imshow(np.zeros((r1 - r0, c1 - c0, 3)), interpolation="antialiased")
        else:
            cm = plt.get_cmap("gray" if key == "T2w" else "turbo").copy(); cm.set_bad(S.BG)
            im = ax.imshow(np.full((r1 - r0, c1 - c0), np.nan), cmap=cm, vmin=0, vmax=1, interpolation="antialiased")
        ims.append(im)
        S.header(ax, title, dx=0, dy=2)
    S.dirs_row(axes[0], axes[-1], "coronal")
    counter = axes[-1].annotate("", xy=(1, 1), xycoords="axes fraction", xytext=(0, 2), textcoords="offset points", ha="right",
                                va="bottom", fontsize=S.MIN_PT, color=S.GRAY, annotation_clip=False)

    def show(j):
        for (key, _), im in zip(CH, ims):
            v = vols[key]
            sl = S.section(v, "coronal", j) if v.ndim == 3 else np.stack([S.section(v[..., c], "coronal", j) for c in range(3)], -1)
            im.set_data(sl[r0:r1, c0:c1])
        counter.set_text(f"y = {A[1, 3] + A[1, 1] * j:.1f} mm")

    # the bar: an empty corner of every frame, found on the masks' projection (every frame's tissue lies inside it)
    for (key, _), im in zip(CH, ims):
        m = (brain if key != "blockface_rgb" else bfm).any(1, keepdims=True)
        pr = S.section(m.astype(np.float32), "coronal", 0)[r0:r1, c0:c1]
        im.set_data(np.repeat(pr[..., None], 3, -1) if key in ("DEC", "blockface_rgb") else pr)
    counter.set_text("y = -20.8 mm")                                  # the widest counter text
    S.scalebar_free(fig, BAR_MM, scale, (S.MARGIN_MM, W - S.MARGIN_MM, S.MARGIN_MM, H - S.MARGIN_MM), corner="bottom left")
    show(planes[len(planes) // 2])
    bad = S.check(fig)
    fig.set_dpi(dpi)
    if bad:
        raise SystemExit("readme_animation: house rules broken:\n  " + "\n  ".join(bad))
    frames = []
    for j in planes:
        show(j); fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    dur = [int(round(1000 / FPS))] * len(frames); dur[0] = dur[-1] = int(HOLD_S * 1000)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    # one palette for every frame (from a mosaic of evenly spaced frames), no dithering: steady colors and small frame deltas
    pick = [frames[i] for i in np.linspace(0, len(frames) - 1, 12).round().astype(int)]
    pal = Image.fromarray(np.concatenate(pick, 0)).quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    q = [Image.fromarray(f).quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(out, save_all=True, append_images=q[1:], duration=dur, loop=0, optimize=False, disposal=1)
    print(f"{out}: {len(frames)} frames, {frames[0].shape[1]} x {frames[0].shape[0]} px, {out.stat().st_size / 1e6:.2f} MB; "
          f"y {A[1, 3] + A[1, 1] * planes[0]:.1f} to {A[1, 3] + A[1, 1] * planes[-1]:.1f} mm; print scale {scale:.3f}")
    if a.webp:
        w = out.with_suffix(".webp")
        Image.fromarray(frames[0]).save(w, save_all=True, append_images=[Image.fromarray(f) for f in frames[1:]], duration=dur,
                                        loop=0, quality=80, method=6)
        print(f"{w}: {w.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
