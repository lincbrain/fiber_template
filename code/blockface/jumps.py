"""Step 2: the camera jumps. The camera moved between some photographs; each such jump is corrected so the whole stack lines up
(Methods, "Blockface reconstruction", second step).

You need
  the raw and mask stacks (_work/blockface_raw.ome.zarr, _work/brainmask.ome.zarr, photos.py), the jumps listed manually
  (settings/brains/<ID>.yaml `jumps`: before, after) and TIRL (external.py).
What it does
  Each jump is corrected by a 2D TIRL registration of the masked frames on either side of it (rotation, rigid, affine;
  settings/tirl/camera_jumps.yml), and every section up to the jump is resampled through the composed transforms, so the sub-stack
  lines up with the rest.
    cut_jumps        the raw stack     -> _work/camera_jumps/*.png (masked, upright frames: TIRL's input)
    tirl_jumps       camera_jumps      -> _work/tirl_jumps/SLICE_*/ (one registration per jump)
    register_jumps   the composed transforms applied to the raw (linear) and mask (nearest) stacks
You get
  the two stacks corrected in place. Not idempotent: re-stack (photos.stack_blockface, stack_masks) to start again.

Ported unchanged from the analysis project's code/recon.py: the affine mode, the one every listed jump of the paper's brains used.
"""
import copy
import os

import cv2
import numpy as np
import yaml
import zarr
from PIL import Image

from blockface import external
from blockface.subject import REPO

TIRLPARAM = str(REPO / "settings" / "tirl" / "camera_jumps.yml")


def _validated_jumps(cfg):
    """Camera jumps as a sorted [(start, end), ...] list. A jump is a discontinuity between slice
    `start` (last present before it) and slice `end` (first present after it): the correction bands on
    `start` and registers start->end. Usually end==start+1, but end>start+1 is allowed and normal - it
    means the slices in between were dropped and are filled by interpolate_missing (the camera often
    jumped precisely because sections were lost); e.g. MR252 jump (1062,1070) with missing (1062,1070).
    Requires end>start and ASCENDING, DISTINCT starts so `reversed(app)` composes reference-ward."""
    jumps = sorted(cfg.jumps)
    starts = [s for s, _ in jumps]
    if len(set(starts)) != len(starts):
        raise SystemExit(f"duplicate camera-jump start in {cfg.jumps}")
    for s, e in jumps:
        if e <= s:
            raise SystemExit(f"camera jump {(s, e)} must have end > start")
    return jumps


def cut_jumps(cfg):
    """Save each jump's (before, after) frame as MASKED (brain-only), UPRIGHT PNGs for TIRL.
    Background is zeroed so the affine locks onto the brain; the v4 row/col reversal is undone
    so TIRL registers in the same upright space as the stack (not 180-degrees rotated), keeping
    the transform composable with the later histo/MRI TIRL steps."""
    bf = zarr.open_group(cfg.raw_stack, mode="r")["0"]       # (c, rows, slices, cols), stored reversed
    mk = zarr.open_group(cfg.mask_stack, mode="r")["0"]      # (1, rows, slices, cols)
    mkc = zarr.open_group(cfg.mask_stack, mode="r")["2"]     # coarse mask -> fast present-check
    out = cfg.jumps_dir
    os.makedirs(out, exist_ok=True)
    jumps = _validated_jumps(cfg)
    present = set(s for s in range(mkc.shape[2]) if np.asarray(mkc[0, :, s, :]).any())
    # A slice the curator marked for interpolation is NOT a valid registration source, even though its photo is
    # still in the stack: measuring a jump against a frame we are about to discard measures the defect that got
    # it discarded (MR261 2026-09-19: the step at the deleted slice 887 reads 0.77 mm against 887 but 0.82 mm
    # between its surviving neighbours 886 and 888, and at a black-filled numbering gap the reading is spurious
    # altogether - 0.08 mm at (1029,1030) where 1029<->1031 is 3.63 mm).
    present -= set(z for b0, b1 in cfg.missing if b0 >= 0 for z in range(b0 + 1, b1))

    def snap(s, before):        # the slice itself if it has brain, else the nearest present slice on
        if s in present or not present:   # the given side (so a jump whose start/end landed on a
            return s                       # dropped/missing frame still registers real tissue)
        side = [p for p in present if (p < s if before else p > s)]
        return min(side or present, key=lambda p: abs(p - s))

    for start, end in jumps:
        for idx, src in [(start, snap(start, True)), (start + 1, snap(end, False))]:
            img = np.asarray(bf[:, :, src, :]).transpose(1, 2, 0)     # (rows, cols, c)
            img[np.asarray(mk[0, :, src, :]) == 0] = 0                # remove background
            img = img[::-1, ::-1, :]                                  # undo v4 reversal -> upright
            Image.fromarray(img).save(os.path.join(out, f"SLICE_{idx:04d}_0000.png"))
    print(f"cut {len(jumps)} masked upright jump pair(s)")


def tirl_jumps(cfg):
    """Run `tirl linc.h2b` on each jump pair (slice start -> start+1)."""
    inp = cfg.jumps_dir
    out = cfg.tirl_jumps_dir
    os.makedirs(out, exist_ok=True)
    template = yaml.full_load(open(TIRLPARAM))
    for start, _ in _validated_jumps(cfg):
        d = os.path.join(out, f"SLICE_{start:04d}")
        os.makedirs(d, exist_ok=True)
        p = copy.deepcopy(template)
        p["histology"]["file"] = os.path.join(inp, f"SLICE_{start:04d}_0000.png")
        p["histology"]["resolution"] = cfg.vx / 1000
        p["block"]["file"] = os.path.join(inp, f"SLICE_{start + 1:04d}_0000.png")
        p["block"]["resolution"] = cfg.vx / 1000
        p["general"]["outputdir"] = d
        p["general"]["logfile"] = os.path.join(d, "logfile.log")
        p["general"]["paramlogfile"] = os.path.join(d, "paramlogfile.log")
        p["general"]["system"] = "linux"
        fname = os.path.join(d, "tirlparam.yml")
        yaml.dump(p, open(fname, "w"))
        external.tirl_h2b(fname)


def register_jumps(cfg):
    """Apply the composed jump transforms to the raw blockface (linear) and mask (nearest) stacks.
    The affine mode (TIRL), as for every listed jump of the paper's brains."""
    if cfg.jumps and getattr(cfg, "jump_mode", "affine") != "affine":
        raise SystemExit(f"jump_mode {cfg.jump_mode!r}: the release implements the affine mode only")
    external.check_tirl_import()
    from tirl import load
    starts = [s for s, _ in _validated_jumps(cfg)]      # sorted ascending -> reversed(app) composes correctly
    if not starts:
        print("no jumps")
        return

    def pullback(start):                 # fixed voxel -> moving voxel (as the old pipeline)
        d = os.path.join(cfg.tirl_jumps_dir, f"SLICE_{start:04d}")
        moving = load(os.path.join(d, "moving.timg"))
        fixed = load(os.path.join(d, "fixed3_affine.timg"))
        return fixed.domain.all_tx().reduce() + moving.domain.all_tx().reduce().inverse()

    P = {j: pullback(j) for j in starts}

    def composed(app):                   # compose every jump that applies to a slice
        c = None
        for j in reversed(app):
            c = P[j].copy() if c is None else c + P[j]
        return c.reduce(merge="linear")

    _apply(cfg.raw_stack, starts, composed, order=1)
    _apply(cfg.mask_stack, starts, composed, order=0)


def _apply(path, starts, composed, order):
    omz = zarr.open_group(path, mode="r+")
    a0 = omz["0"]                                    # (c, rows, slices, cols)
    C, H, nz, W = a0.shape                           # H=rows, W=cols
    grid = np.stack([a.ravel() for a in np.indices((H, W))], axis=1).astype(float)

    bands = {}                                       # slices grouped by which jumps apply
    for idx in range(nz):
        app = tuple(j for j in starts if idx <= j)
        if app:
            bands.setdefault(app, []).append(idx)

    interp = cv2.INTER_LINEAR if order else cv2.INTER_NEAREST
    for app, idxs in bands.items():
        src = composed(app).map(grid, merge="linear")           # transform lives in upright space
        mapy = src[:, 0].reshape(H, W).astype(np.float32)       # source row per dst pixel
        mapx = src[:, 1].reshape(H, W).astype(np.float32)       # source col per dst pixel
        for idx in idxs:                                        # cv2.remap: all channels at once, fast
            up = np.asarray(a0[:, :, idx, :]).transpose(1, 2, 0)[::-1, ::-1]    # stored -> upright (H,W,C)
            out = cv2.remap(up, mapx, mapy, interp, borderValue=0)
            a0[:, :, idx, :] = out[::-1, ::-1].reshape(H, W, C).transpose(2, 0, 1)   # upright -> stored
    # Edit L0 only, do NOT rebuild the pyramid here. Nothing reads these stale L1/L2 before they are
    # refreshed or dropped: remove_background rebuilds the mask L2 (interpolate's present-check reads it),
    # the raw L1/L2 are never read again (crop_and_mask reads raw L0; the final blockface builds its own
    # fresh pyramid). Saves a full rebuild of both stacks per jump step.
    print(f"registered {os.path.basename(path)} ({len(bands)} band(s))")
