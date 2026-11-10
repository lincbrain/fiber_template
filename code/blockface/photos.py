"""Step 1 and the stacks: the photographs, their 2D brain masks, and both stacked (Methods, "Blockface reconstruction", first and
second steps).

You need
  the brain's photographs ($LINC_RAW/<raw>/..., settings/brains/<ID>.yaml raw, raw_glob, raw_regex, idx_range, rotate180) and the
  masks the photo network predicted (external.segment).
What it does
    ingest              raw photographs  -> 00_slices/SLICE_<k>_0000.png (a black frame for a missing or unreadable photograph,
                                            so the order of the sections is kept; turned 180 degrees when rotate180)
    normalize_masks     the 2D masks     -> 01_brainmask/00_predict (x50 PNGs, interior holes filled: mask_rules.fill_photo_mask)
    finalize_brainmask  00_predict       -> 01_brainmask/01_final (existing files are kept, so a hand correction survives)
    stack_blockface     00_slices        -> _work/blockface_raw.ome.zarr
    stack_masks         01_final         -> _work/brainmask.ome.zarr
                        (both stacks put slice 0 at the frame count's front end, or at the brain's stack_origin_frames')
    masks_to_png        an edited stack  -> 01_final (a hand-edited stack written back)
You get
  00_slices/, 01_brainmask/{00_predict, 01_final}/ and the two working stacks under _work/.

Ported unchanged from the analysis project's code/tasks.py (a debugging switch removed).
"""
import os
import re
import shutil
from glob import glob

import numpy as np
from PIL import Image

from blockface import zarrio


def ingest(cfg):
    """Rename raw photos to SLICE_%04d_0000.png (the networks' case names), inserting a black frame
    for any missing index so physical slice order is preserved."""
    os.makedirs(cfg.slices, exist_ok=True)

    nums = {}
    for f in sorted(glob(os.path.join(cfg.raw_dir, cfg.raw_glob))):
        m = re.search(cfg.raw_regex, os.path.basename(f))
        if m:
            nums[int(m.group(1))] = f
    lo, hi = cfg.idx_range or (min(nums), max(nums))
    # The re-run is idempotent only while the INPUTS are the same. Counting the PNGs is not enough: setting
    # MF196's rotate180=True (2026-09-21) changed nothing at all, because 79 photos were already present, and
    # the masks that came out were byte-identical to the upside-down ones. So the parameters that decide what
    # the photos look like are recorded next to them and compared.
    import json as _json
    sig = {"raw_dir": cfg.raw_dir, "raw_glob": cfg.raw_glob, "raw_regex": cfg.raw_regex,
           "idx_range": [int(lo), int(hi)], "rotate180": bool(cfg.rotate180),
           "drop": sorted(int(i) for b, a in cfg.missing if b < 0 for i in range(max(0, b + 1), a))}
    sig_path = os.path.join(cfg.slices, ".ingest.json")
    have = len(glob(os.path.join(cfg.slices, "SLICE_*_0000.png"))) == hi - lo + 1
    if have:
        old_sig = _json.load(open(sig_path)) if os.path.exists(sig_path) else None
        if old_sig == sig:
            print(f"ingest: {hi - lo + 1} slices already present, same parameters -> skip")
            return                                        # reuse a completed ingest (slow full-res JPG->PNG)
        changed = "no record of the parameters" if old_sig is None else \
            ", ".join(f"{k}: {old_sig.get(k)!r} -> {sig[k]!r}" for k in sig if old_sig.get(k) != sig[k])
        print(f"ingest: re-reading the photos - {changed}")
        for d in (cfg.brainmask("00_predict"), cfg.brainmask("01_brainmask"), cfg.brainmask("01_final")):
            if os.path.isdir(d):                          # masks derived from the OLD photos are now wrong
                shutil.rmtree(d); print(f"  dropped {os.path.basename(d)} (derived from the old photos)")
    rows, cols = np.array(Image.open(nums[lo])).shape[:2]

    # Leading non-brain frames to DROP (subject-name card / calibration grid at the stack start): the xlsx
    # marks them as a `missing` entry (before, after) with before < 0 -> black-fill positions before+1..after-1
    # so they become leading empties (cropped away), and are never used as an interpolation source.
    drop = set(idx for b, a in cfg.missing if b < 0 for idx in range(max(0, b + 1), a))
    missing, corrupt = [], []
    for i in range(lo, hi + 1):
        idx = i - lo
        dst = os.path.join(cfg.slices, f"SLICE_{idx:04d}_0000.png")
        if idx in drop:                                          # non-brain leading frame -> exclude (black)
            Image.new("RGB", (cols, rows), (0, 0, 0)).save(dst)
        elif i in nums:
            try:                                                  # some raw JPGs are truncated/corrupt:
                img = Image.open(nums[i]); img.load()             # force-decode so it fails HERE, not later
                if cfg.rotate180:                                 # lossless 180deg (upside-down scans)
                    img = img.transpose(Image.ROTATE_180)
                img.save(dst)
            except Exception as e:                                # treat an unreadable photo as a missing
                Image.new("RGB", (cols, rows), (0, 0, 0)).save(dst)   # slice (black-fill); interpolate_missing
                corrupt.append(idx)                              # fills it if the bracket is in cfg.missing
                print(f"  ! corrupt {os.path.basename(nums[i])} (slice {idx}) -> black-filled: {type(e).__name__}")
        else:
            Image.new("RGB", (cols, rows), (0, 0, 0)).save(dst)   # PIL size is (W, H)
            missing.append(idx)
    _json.dump(sig, open(sig_path, "w"), indent=1)
    print(f"ingested {hi - lo + 1} slices; missing (black-filled): {missing}; corrupt (black-filled): {corrupt}")


def normalize_masks(cfg):
    """Re-save the photo masks (01_brainmask/00_predict) as *50-scaled PNGs (project-wide
    standard) and fill any interior holes (the rule of step 1: code/blockface/mask_rules.fill_photo_mask)."""
    from blockface.mask_rules import fill_photo_mask
    n = zarrio.normalize_mask_dir(cfg.brainmask("00_predict"))
    for p in glob(os.path.join(cfg.brainmask("00_predict"), "*.png")):
        zarrio.save_mask(fill_photo_mask(zarrio.load_mask(p)), p)
    print(f"normalized + hole-filled {n} masks to *{zarrio.MASK_SCALE} scaling")


def finalize_brainmask(cfg):
    """Accept the brain masks as final -> 01_brainmask/01_final (what reconstruction stacks).
    Copy of the automatic 00_predict while the fsleyes edit is skipped; existing 01_final files are
    kept, so a hand-corrected slice survives a re-run and 00_predict is never overwritten."""
    src, dst = cfg.brainmask("00_predict"), cfg.brainmask("01_final")
    os.makedirs(dst, exist_ok=True)
    n = 0
    for p in sorted(glob(os.path.join(src, "*.png"))):
        d = os.path.join(dst, os.path.basename(p))
        if not os.path.exists(d):
            shutil.copyfile(p, d)
            n += 1
    print(f"accepted {n} brain masks as final -> {dst} (kept existing manual fixes)")


def _y_anterior0(cfg):
    """World y (mm) of a stack's slice 0 when the brain pins the frame count its origin counts (settings/brains/<ID>.yaml
    stack_origin_frames: the stack of record held frames that are not this brain's photographs); None = the stack's own count.
    The same expression as zarrio.write_stack's, so the pinned origin equals the one of record exactly."""
    n = cfg.stack_origin_frames
    return None if n is None else cfg.vz * (n - 1 + 0.5) / 1000.0


def stack_blockface(cfg):
    """Stack the raw (background-kept) blockface photos into the working OME-Zarr."""
    photos = sorted(glob(os.path.join(cfg.slices, "*_0000.png")))
    rows, cols = zarrio.load_image(photos[0]).shape[:2]

    def get_slice(s):
        return zarrio.load_image(photos[s])

    zarrio.write_stack(cfg.raw_stack, get_slice, len(photos), rows, cols, 3,
                       cfg.vx, cfg.vz, cfg.reduce, pad=cfg.stack_pad or zarrio.STACK_PAD, y_anterior0=_y_anterior0(cfg))


def stack_masks(cfg):
    """Stack the accepted brain masks (01_brainmask/01_final) into the working OME-Zarr in the
    same geometry as the raw blockface, for export to NIfTI and QC/editing in fsleyes."""
    masks = sorted(glob(os.path.join(cfg.brainmask("01_final"), "*.png")))
    rows, cols = zarrio.load_mask(masks[0]).shape[:2]

    def get_slice(s):
        return zarrio.load_mask(masks[s])                # integer labels (/50)

    zarrio.write_stack(cfg.mask_stack, get_slice, len(masks), rows, cols, 1,
                       cfg.vx, cfg.vz, "stride", pad=cfg.stack_pad or zarrio.STACK_PAD,   # masks: no mean-blur
                       y_anterior0=_y_anterior0(cfg))


def masks_to_png(cfg):
    """Convert the fsleyes-edited brain-mask zarr (_work/brainmask_edit) back to per-slice *50
    PNGs in slices orientation -> 01_brainmask/01_final."""
    vol = np.asarray(zarrio.read_group(cfg.mask_edit_zarr)["0"])
    # Name the outputs the way 01_final is ALREADY named (the network drops the _0000 channel suffix, and
    # finalize_brainmask copies 00_predict verbatim). Naming them from 00_slices instead wrote a second,
    # differently-named copy of every mask beside the first, and stack_masks globs *.png - which silently
    # doubled the mask stack's z axis against the raw stack.
    names = sorted(glob(os.path.join(cfg.brainmask("00_predict"), "*.png"))) \
        or sorted(glob(os.path.join(cfg.brainmask("01_final"), "*.png"))) \
        or [os.path.basename(f).replace("_0000.png", ".png")
            for f in sorted(glob(os.path.join(cfg.slices, "*_0000.png")))]
    out = cfg.brainmask("01_final")
    os.makedirs(out, exist_ok=True)

    axis = _slice_axis(vol.shape, len(names))
    p = cfg.stack_pad or zarrio.STACK_PAD
    for s in range(len(names)):
        sl = np.squeeze(np.take(vol, s, axis=axis))      # (rows, cols) padded, reversed
        sl = sl[p:-p, p:-p][::-1, ::-1]                  # unpad + un-reverse -> native
        zarrio.save_mask(sl, os.path.join(out, os.path.basename(names[s])))
    print(f"wrote {len(names)} fixed masks to {out}")


def _slice_axis(shape, nz):
    """Index of the axis whose length equals the number of slices."""
    cand = [i for i, n in enumerate(shape) if n == nz]
    assert len(cand) == 1, f"cannot identify slice axis in {shape} for nz={nz}"
    return cand[0]


def export_for_editing(cfg):
    """The stage-2 exports for a hand correction of the 2D masks (code/run.py stage 2): the raw stack at L2
    (nii/00_blockface_raw_L2) and the full-resolution mask stack (nii/01_brainmask_L0), to edit in fsleyes."""
    import nibabel as nib
    from blockface import external
    from blockface.prepare import _squeeze_export
    os.makedirs(cfg.nii_dir, exist_ok=True)
    external.zarr2nii(2, cfg.raw_stack, cfg.snap("00_blockface_raw_L2"))
    _squeeze_export(cfg.snap("00_blockface_raw_L2"))
    tmp = cfg.snap("01_brainmask_L0")[:-7] + "_5d.nii.gz"
    external.zarr2nii(0, cfg.mask_stack, tmp)
    nib.save(nib.funcs.squeeze_image(nib.load(tmp)), cfg.snap("01_brainmask_L0"))
    os.remove(tmp)


def import_edited_masks(cfg):
    """After editing nii/01_brainmask_L0 (code/run.py stage 2b): back to x50 PNGs in 01_final, and the mask stack rebuilt."""
    from blockface import external
    external.nii2zarr(cfg.mask_edit_nii, cfg.mask_edit_zarr)
    masks_to_png(cfg)
    stack_masks(cfg)
    shutil.rmtree(cfg.mask_edit_zarr)
