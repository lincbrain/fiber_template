#!/usr/bin/env python3
"""The camera-jump registration that the reconstruction figure's panel (b) shows, recomputed as the reconstruction's step 2 made it
(code/blockface/jumps.py; Methods, "Blockface reconstruction", second step): the reconstruction keeps the corrected stack, not the
step's working files (_work/, its TIRL registrations), so the one registration the figure needs is run again here.

The brain and the jump: settings/figures.yaml reconstruction (subject, jump_shown: the jump with the largest step of the 2D-mask
centroid, which code/figures/reconstruction.py checks). The two frames are cut as cut_jumps cuts them: the raw stack's level 0 (the
photograph as taken, padded by the stack's margin, zarrio.STACK_PAD px per side, for a brain whose stack_pad is 0), its background
(2D mask 0) zeroed, upright; the frame before the jump is TIRL's "histology", the frame after it the "block". tirl linc.h2b with
settings/tirl/camera_jumps.yml exactly as tirl_jumps fills it in (rotation search, rigid, affine; the pixel size, settings/brains/<ID>.yaml
vx), on the CPU. The registration is read back as register_jumps reads it (fixed3_affine and moving: the pullback from a pixel of the
frames after the jump to the frame before it) and written as its 2 x 3 matrix on (row, col) pixel coordinates of the padded frame
(the pullback is affine; the fit's largest residual is recorded).
The check: the 2D masks of the two sections, before and after the correction, their Dice overlap and the step between their
centroids (mm), beside the same two numbers for the stack of record (02_blockface_mask, level 2: the reconstruction's own correction).
-> $LINC_NUMBERS/bf_jump_pullback.json; TIRL's files in $LINC_WORK/bf_jump_pullback/
    python code/figures/paper_numbers/bf_jump_pullback.py          (settings/env/tirl.yml, PY_TIRL; a few minutes on a CPU)"""
import copy
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
from shared import paths, provenance  # noqa: E402
from blockface import external, zarrio  # noqa: E402
from blockface.jumps import TIRLPARAM  # noqa: E402

CFG = yaml.safe_load((REPO / "settings" / "figures.yaml").read_text())["reconstruction"]


def main():
    S = CFG["subject"]; j0, j1 = (int(v) for v in CFG["jump_shown"])
    B = yaml.safe_load((REPO / "settings" / "brains" / f"{S}.yaml").read_text())
    assert [j0, j1] in [list(v) for v in B["jumps"]], f"{[j0, j1]} is not one of {S}'s camera jumps"
    assert j1 == j0 + 1, "a jump over dropped sections is cut from its nearest present frames (cut_jumps' snap): not handled here"
    miss = {z for b0, b1 in B.get("missing") or [] if b0 >= 0 for z in range(b0 + 1, b1)}
    assert j0 not in miss and j1 not in miss, "a frame marked for interpolation: cut_jumps would snap to a neighbor"
    pad, vx = int(B.get("stack_pad") or 0) or zarrio.STACK_PAD, float(B["vx"])
    bd = paths.products("products_blockface")["data"] / f"sub-{S}" / "blockface"
    out = paths.work_root() / "bf_jump_pullback"; (out / "camera_jumps").mkdir(parents=True, exist_ok=True)
    ins = [bd / "00_slices" / f"SLICE_{k:04d}_0000.png" for k in (j0, j1)] + [bd / "01_brainmask" / "01_final" / f"SLICE_{k:04d}.png"
                                                                               for k in (j0, j1)]

    def frame(k):                                                   # cut_jumps' frame of section k
        im = np.asarray(Image.open(bd / "00_slices" / f"SLICE_{k:04d}_0000.png").convert("RGB")).copy()
        m = zarrio.load_mask(str(bd / "01_brainmask" / "01_final" / f"SLICE_{k:04d}.png"))
        im[m == 0] = 0
        return np.pad(im, ((pad, pad), (pad, pad), (0, 0))), np.pad(m > 0, pad)
    (f0, m0), (f1, m1) = frame(j0), frame(j1)
    for k, f in ((j0, f0), (j1, f1)):
        Image.fromarray(f).save(out / "camera_jumps" / f"SLICE_{k:04d}_0000.png")
    d = out / "tirl_jumps" / f"SLICE_{j0:04d}"; d.mkdir(parents=True, exist_ok=True)
    p = copy.deepcopy(yaml.full_load(open(TIRLPARAM)))              # tirl_jumps, verbatim
    p["histology"]["file"] = str(out / "camera_jumps" / f"SLICE_{j0:04d}_0000.png"); p["histology"]["resolution"] = vx / 1000
    p["block"]["file"] = str(out / "camera_jumps" / f"SLICE_{j1:04d}_0000.png"); p["block"]["resolution"] = vx / 1000
    p["general"].update(outputdir=str(d), logfile=str(d / "logfile.log"), paramlogfile=str(d / "paramlogfile.log"), system="linux")
    yaml.dump(p, open(d / "tirlparam.yml", "w"))
    external.tirl_h2b(str(d / "tirlparam.yml"))
    external.check_tirl_import()
    from tirl import load
    P = load(str(d / "fixed3_affine.timg")).domain.all_tx().reduce() + load(str(d / "moving.timg")).domain.all_tx().reduce().inverse()
    H, W = f0.shape[:2]
    g = np.stack([a.ravel() for a in np.indices((H, W))], axis=1).astype(float)
    src = P.reduce(merge="linear").map(g, merge="linear")
    X = np.c_[g, np.ones(len(g))]
    A = np.linalg.lstsq(X, src, rcond=None)[0].T                    # (2, 3): (row, col, 1) after -> (row, col) before
    fit = float(np.abs(X @ A.T - src).max()); del g, src, X

    # the check: the frame before the jump pulled into the frame after it (nearest, as register_jumps does for the masks)
    yy, xx = np.indices((H, W)).astype(np.float32)
    my = (A[0, 0] * yy + A[0, 1] * xx + A[0, 2]).astype(np.float32); mx = (A[1, 0] * yy + A[1, 1] * xx + A[1, 2]).astype(np.float32)
    m0a = cv2.remap(m0.astype(np.uint8), mx, my, cv2.INTER_NEAREST, borderValue=0) > 0
    dice = lambda a, b: float(2 * (a & b).sum() / (a.sum() + b.sum()))                                     # noqa: E731
    step = lambda a, b, px: float(np.linalg.norm(np.argwhere(a).mean(0) - np.argwhere(b).mean(0)) * px)     # noqa: E731
    grp = zarrio.read_group(str(bd / "02_blockface_mask.ome.zarr"))
    r0, r1 = (np.asarray(grp["2"][0, :, k, :]) == 1 for k in (j0, j1))                     # the stack of record, level 2 (x 4)
    lin = A[:, :2]; ang = float(np.degrees(np.arctan2(lin[1, 0] - lin[0, 1], lin[0, 0] + lin[1, 1])))
    data = dict(subject=S, jump=[j0, j1], frame_shape_px=[H, W], pad_px=pad, pixel_mm=vx / 1000,
                pullback_rowcol_affine=A.tolist(), affine_fit_max_residual_px=round(fit, 6),
                rotation_deg=round(ang, 3), scale=[round(float(v), 5) for v in np.linalg.svd(lin, compute_uv=False)],
                check=dict(masks_dice=dict(as_taken=round(dice(m0, m1), 4), corrected=round(dice(m0a, m1), 4),
                                           stack_of_record=round(dice(r0, r1), 4)),
                           centroid_step_mm=dict(as_taken=round(step(m0, m1, vx / 1000), 3), corrected=round(step(m0a, m1, vx / 1000), 3),
                                                 stack_of_record=round(step(r0, r1, 4 * vx / 1000), 3))),
                note=("pullback_rowcol_affine maps a pixel (row, col) of the padded upright frame of the sections after the jump to the "
                      "frame of the sections up to it: every section up to the jump is sampled there (register_jumps' band, without the "
                      "later jumps the reconstruction composes on every section of this stretch alike)"))
    provenance.write_numbers("bf_jump_pullback", data, ins + [TIRLPARAM, REPO / "settings" / "brains" / f"{S}.yaml",
                                                              bd / "02_blockface_mask.ome.zarr"],
                             cmd="python code/figures/paper_numbers/bf_jump_pullback.py")
    print(data)


if __name__ == "__main__":
    main()
