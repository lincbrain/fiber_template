"""MF8 and MF9: the photographs aligned to the brain's external, already aligned series of the same sections (Methods, "Blockface
reconstruction": "MF8 and MF9 were aligned instead by registering each photograph to its counterpart in an already aligned series
of the same sections").

You need
  the brain's photographs and accepted 2D masks (00_slices, 01_brainmask/01_final: photos.py) and the record of the alignment
  (settings/brains/<ID>.yaml `series_alignment`: sub-<ID>/blockface/wvig/align.json, one transform per section).
What it does
  The camera of these two old stacks moved on nearly every frame, so each photograph was registered to its counterpart in the
  external series (a translation per section, plus a scale and rotation per camera setup). That fit was made once; what is here
  replays its record: apply() warps the photographs and their masks into the series' frame. Such a brain lists no camera jumps.
You get
  00_slices and 01_brainmask/01_final in the series' frame; the originals kept as 00_slices_prealign and 01_final_prealign.

Ported from the analysis project's code/wvig_align.py `apply_` (the warp unchanged).
"""
import glob
import json
import os
import shutil

import numpy as np
from scipy import ndimage as ndi
from PIL import Image

from blockface import zarrio


def _rot(r, J):
    """A row's rotation, falling back to the subject's (an interpolated row can carry an explicit null)."""
    v = r.get("rot")
    return J["rotation"] if v is None else v


def _linear(scale, rot):
    R = np.deg2rad(rot)
    c, s = np.cos(R), np.sin(R)
    return np.array([[c, -s], [s, c]]) * scale


def apply(cfg, align_json, frame_shape):
    """Warp the photographs (00_slices) and their accepted 2D masks (01_final) into the external series' frame (frame_shape =
    (rows, cols) at our resolution) by each section's recorded transform; the originals are backed up once and always read from
    the backup."""
    J = json.load(open(align_json))
    if not os.path.isdir(f"{cfg.root}/00_slices_prealign"):
        shutil.copytree(cfg.slices, f"{cfg.root}/00_slices_prealign")
        shutil.copytree(cfg.brainmask("01_final"), f"{cfg.root}/01_final_prealign")
        print(f"backed up the unaligned photos -> {cfg.root}/00_slices_prealign")
    src_p = sorted(glob.glob(f"{cfg.root}/00_slices_prealign/*_0000.png"))
    src_m = sorted(glob.glob(f"{cfg.root}/01_final_prealign/*.png"))
    dst_p = sorted(glob.glob(os.path.join(cfg.slices, "*_0000.png")))
    dst_m = sorted(glob.glob(os.path.join(cfg.brainmask("01_final"), "*.png")))
    SH = tuple(int(x) for x in frame_shape)                      # the reference frame, at OUR resolution
    for r in J["rows"]:
        k = r["slice"]
        if not r.get("ok"):
            print(f"  slice {k}: flagged, left unwarped"); continue
        Ai = np.linalg.inv(_linear(r.get("scale", J["scale"]), _rot(r, J)))
        b = -Ai @ (np.array(r["offset"], float) * J["ds"])
        img = np.asarray(Image.open(src_p[k]).convert("RGB")).astype(np.float32)
        w = np.stack([ndi.affine_transform(img[..., c], Ai, offset=b, output_shape=SH, order=1)
                      for c in range(3)], -1)
        if (w > 5).mean() < 0.005:
            raise SystemExit(f"slice {k}: the warp left almost nothing in the frame - refusing to write it")
        Image.fromarray(np.clip(w, 0, 255).astype(np.uint8)).save(dst_p[k])
        m = zarrio.load_mask(src_m[k]) > 0
        wm = ndi.affine_transform(m.astype(np.float32), Ai, offset=b, output_shape=SH, order=1) > 0.5
        zarrio.save_mask(wm.astype(np.uint8), dst_m[k])
        print(f"  slice {k:3d}: warped -> {SH}", flush=True)
    print(f"{cfg.subject}: in the external series' frame; no camera-jump correction downstream")
