"""The training cases of the two blockface 3D networks, staged from the data tree, so the networks can be trained again from the
reconstructions (models/segmentation.md; the task files of record: configs/segment/bf_mask3d.json and bf_holes.json).

You need
  a task file (configs/segment/bf_mask3d.json or bf_holes.json: its brains, folds and rules) and, for each of its brains, the data
  tree $LINC_DATA/sub-<ID>/blockface (configs/subjects/<ID>.yaml for the brain's section factor f): for bf_mask3d curated/<ID>_bg and
  curated/<ID>_mask_edited; for bf_holes steps/<ID>_03_color.ome.zarr, <ID>.info, curated/<ID>_holes_edited and the brain mask the
  reconstruction used.
What it does
  bf_mask3d  image = curated/<ID>_bg (the RGB volume with its background, L4 in-plane, sections block-averaged by f: the network's
             input in the pipeline), label = curated/<ID>_mask_edited (the mask of record). Where the two files' grids differ, the
             label keeps its own grid, cropped to the part the image covers, and the image is sampled linearly at the exact position
             of every label voxel (the fraction of the offset between the two headers kept).
  bf_holes   image = steps/<ID>_03_color.ome.zarr at its L4 level, block-averaged by f (masked, shading- and color-corrected, before
             the tubes are filled; the tight-crop grid of <ID>.info), label = curated/<ID>_holes_edited cropped to that grid, and the
             brain mask on the same grid (the tube rule needs it).
  Each brain keeps its own grid (no resampling).
You get
  in the output folder: imagesTr/<ID>_000{0,1,2}.nii.gz, labelsTr/<ID>.nii.gz (and masksTr/<ID>.nii.gz for bf_holes), uint8, every
  file of a case with the label's affine; <task>.json, the given task file with each case's paths and label_md5 those of the staged
  files (folds and rules unchanged), for `cli/linc-segment train --task`; provenance.json.

Ported from the analysis project's models_wrapup/data_bf/stage_bf3d.py (t3, t4) and stage_bf3d_v2.py (the exact offset, which the
record applied to the two brains whose grids differ, MF275 and MR248): the staging of record, one rule per task.
"""
import hashlib
import json
import os

import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates

from linc.blockface import curate, segment3d, zarrio
from linc.blockface.subject import Subject
from linc.provenance import write_provenance

TASKS = ("bf_mask3d", "bf_holes")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _write_case(out, S, img, lab, aff, mask=None):
    """The case's files: three image channels, the label (and the brain mask), all with the label's affine."""
    paths = {"image": [os.path.join(out, "imagesTr", f"{S}_{c:04d}.nii.gz") for c in range(3)],
             "label": os.path.join(out, "labelsTr", f"{S}.nii.gz")}
    for c, p in enumerate(paths["image"]):
        nib.save(nib.Nifti1Image(np.ascontiguousarray(img[..., c]), aff), p)
    nib.save(nib.Nifti1Image(lab.astype(np.uint8), aff), paths["label"])
    if mask is not None:
        paths["mask"] = os.path.join(out, "masksTr", f"{S}.nii.gz")
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), aff), paths["mask"])
    return paths


def mask3d_case(cfg, out):
    """bf_mask3d: curated/<S>_bg -> image, curated/<S>_mask_edited -> label (stage_bf3d.t3, with stage_bf3d_v2's exact offset)."""
    S = cfg.subject
    bi, mi = nib.load(cfg.curated_bg), nib.load(cfg.mask_edited)
    assert bi.shape[3] == 3, (S, bi.shape)
    lab = np.asanyarray(mi.dataobj) > 0
    if bi.shape[:3] == mi.shape and np.abs(bi.affine - mi.affine).max() <= 1e-3:
        return _write_case(out, S, np.asanyarray(bi.dataobj).astype(np.uint8), lab, mi.affine), {}
    # the grids differ: the label keeps its own grid, cropped to the image's extent; the image is sampled at the exact offset
    dz = float(np.dot(bi.affine[:3, 3] - mi.affine[:3, 3], mi.affine[:3, 1]) / np.dot(mi.affine[:3, 1], mi.affine[:3, 1]))
    assert abs(dz) < 1e-3, (S, "the image's and the label's section origins differ", dz)
    assert np.allclose(np.abs(mi.affine[:3, :3]), np.abs(bi.affine[:3, :3]), atol=1e-6), (S, "the voxel axes differ")
    img = np.asanyarray(bi.dataobj).astype(np.float32)
    off, crop = np.zeros(3), []
    for a in range(3):
        col = bi.affine[:3, a]
        off[a] = float(np.dot(mi.affine[:3, 3] - bi.affine[:3, 3], col) / np.dot(col, col)) if a != 1 else 0.0
        o = int(round(off[a]))
        crop.append(slice(max(0, -o), min(lab.shape[a], img.shape[a] - o)))
    outside = int(lab.sum() - lab[tuple(crop)].sum())
    assert outside == 0, (S, "label voxels outside the image's extent", outside)
    lab = lab[tuple(crop)]
    aff = mi.affine.copy()
    aff[:3, 3] = mi.affine[:3, :3] @ np.array([s.start for s in crop]) + mi.affine[:3, 3]
    u = np.meshgrid(*[np.arange(s.start, s.stop, dtype=np.float64) for s in crop], indexing="ij")
    coords = np.stack([u[a] + off[a] for a in range(3)])                     # the image coordinate of every label voxel
    im2 = np.stack([map_coordinates(img[..., c], coords, order=1, mode="nearest") for c in range(3)], -1)
    im2 = np.clip(np.rint(im2), 0, 255).astype(np.uint8)
    return _write_case(out, S, im2, lab, aff), dict(offset_vox=[round(float(v), 4) for v in off],
                                                    label_crop=[[s.start, s.stop] for s in crop])


def holes_case(cfg, out):
    """bf_holes: steps/<S>_03_color at L4, block-averaged by f -> image; <S>_holes_edited and the brain mask on its grid
    (stage_bf3d.t4)."""
    S = cfg.subject
    f = segment3d._f(cfg)
    _, _, i0, i1, j0, j1, rl = curate._info(cfg)
    arr, aff = zarrio.read_level(cfg.color_store, 4 - rl)                    # L4 in-plane, native sections, tight crop
    img = np.clip(np.rint(segment3d._zblock(arr.astype(np.float32), f, "mean")), 0, 255).astype(np.uint8)
    hi = nib.load(cfg.holes_edited)
    H = np.asanyarray(hi.dataobj) > 0
    M = np.asanyarray(nib.load(cfg.curated_mask).dataobj) > 0                # the mask the reconstruction used (edited wins)
    assert (i1 - i0, j1 - j0) == (img.shape[0], img.shape[2]), (S, img.shape, (i0, i1, j0, j1))
    zc = min(img.shape[1], H.shape[1])
    img, lab, brain = img[:, :zc], H[i0:i1, :zc, j0:j1], M[i0:i1, :zc, j0:j1]
    Ac = hi.affine.copy()
    Ac[:3, 3] = hi.affine[:3, :3] @ np.array([i0, 0, j0]) + hi.affine[:3, 3]  # the label crop's own affine
    A = segment3d._z_affine(aff, f)
    return _write_case(out, S, img, lab, Ac, mask=brain), dict(
        tube_voxels_outside_crop=int(H.sum() - lab.sum()), tube_voxels_outside_mask=int((lab & ~brain).sum()),
        image_vs_label_affine_maxdiff_mm=round(float(np.abs(A[:3] - Ac[:3]).max()), 4))


def stage(task_file, out):
    """Stage every case of `task_file` into `out` and write <out>/<task>.json. Refuses a non-empty `out`."""
    t = json.load(open(task_file))
    task = t["task"]
    if task not in TASKS:
        raise SystemExit(f"{task_file}: task {task!r}; this stages {', '.join(TASKS)}")
    if os.path.isdir(out) and os.listdir(out):
        raise SystemExit(f"{out} is not empty: stage into a new folder")
    for sub in ("imagesTr", "labelsTr") + (("masksTr",) if task == "bf_holes" else ()):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    one = mask3d_case if task == "bf_mask3d" else holes_case
    checks, inputs = {}, []
    for c in t["cases"]:
        cfg = Subject(c["id"])
        paths, checks[c["id"]] = one(cfg, out)
        inputs += [cfg.curated_bg, cfg.mask_edited] if task == "bf_mask3d" else [cfg.holes_edited, cfg.curated_mask, cfg.info_file]
        c.update(paths, label_md5=md5(paths["label"]))
        print(task, c["id"], "label", c["label_md5"], json.dumps(checks[c["id"]]), flush=True)
    p = os.path.join(out, f"{task}.json")
    json.dump(t, open(p, "w"), indent=1)
    write_provenance(out, inputs=[task_file, *inputs], outputs=[p], extra=dict(task=task, checks=checks))
    return p
