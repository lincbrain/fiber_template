"""The training cases of the three blockface networks, staged from the data tree, so the networks can be trained again from the
reconstructions (settings/models/README.md; the task files of record: settings/segment/bf_mask2d.json, bf_mask3d.json, bf_holes.json).

You need
  a task file (settings/segment/<task>.json: its brains, folds and rules) and, for each of its brains, the data tree
  $LINC_DATA/sub-<ID>/blockface (settings/brains/<ID>.yaml for the brain's spacing, jumps, missing sections and section factor f):
  for bf_mask2d 00_slices/, 01_brainmask/01_final/, 02_blockface_mask.ome.zarr, curated/<ID>_bg and curated/<ID>_mask_edited; for
  bf_mask3d curated/<ID>_bg and curated/<ID>_mask_edited; for bf_holes steps/<ID>_03_color.ome.zarr, <ID>.info,
  curated/<ID>_holes_edited and the brain mask the reconstruction used.
What it does
  bf_mask2d  each photograph at /4 (a 4 x 4 block mean of the native RGB), with its label carried back from the 3D mask of record:
             per inter-jump segment, a scale and translation measured between the accepted photograph masks and the stage-3 mask
             store (02_blockface_mask, level 2); the mask's signed distance interpolated linearly between the two nearest curated
             sections and in-plane, sampled at the /4 pixel centres (the curated grid's voxel centre 7.5 native px from its origin).
             Photographs are kept as the record kept them: frames beyond the caudal cutoff or the curated volume, and blank frames
             (annotated `missing`, or tiny and dark PNGs) dropped; masks whose area departs by more than 60 % from the local median
             dropped; empty labels kept as negatives, at most 8 % of a brain's photographs.
  bf_mask3d  image = curated/<ID>_bg (the RGB volume with its background, L4 in-plane, sections block-averaged by f: the network's
             input in the pipeline), label = curated/<ID>_mask_edited (the mask of record). Where the two files' grids differ, the
             label keeps its own grid, cropped to the part the image covers, and the image is sampled linearly at the exact position
             of every label voxel (the fraction of the offset between the two headers kept).
  bf_holes   image = steps/<ID>_03_color.ome.zarr at its L4 level, block-averaged by f (masked, shading- and color-corrected, before
             the tubes are filled; the tight-crop grid of <ID>.info), label = curated/<ID>_holes_edited cropped to that grid, and the
             brain mask on the same grid (the tube rule needs it).
  Each brain keeps its own grid (no resampling).
You get
  in the output folder: imagesTr/ and labelsTr/ (bf_mask2d: <ID>_<k>_0000.png and <ID>_<k>.png, uint8, plus dataset.json and index/
  <ID>.tsv, every photograph kept or why not; bf_mask3d / bf_holes: <ID>_000{0,1,2}.nii.gz and <ID>.nii.gz, and masksTr/<ID>.nii.gz
  for bf_holes, every file of a case with the label's affine); <task>.json, the given task file pointed at the staged files (folds
  and rules unchanged), for `code/shared/linc-segment train --task`; provenance.json.

Ported from the analysis project's models_wrapup/data_bf/stage_bf2d.py (v2), stage_bf3d.py (t3, t4) and stage_bf3d_v2.py (the exact
offset, which the record applied to the two brains whose grids differ, MF275 and MR248), with the helpers they used from
code/migrate_masks.py (the segment fit, the signed distance) and code/build_trainset.py (the screening): the staging of record.
"""
import hashlib
import json
import os
from glob import glob
from multiprocessing import Pool

import numpy as np
import nibabel as nib
import zarr
from PIL import Image
from scipy.ndimage import affine_transform, distance_transform_edt, map_coordinates

from blockface import curate, segment3d, zarrio
from blockface.subject import Subject, params
from shared.provenance import write_provenance

TASKS = ("bf_mask2d", "bf_mask3d", "bf_holes")
Image.MAX_IMAGE_PIXELS = None


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


# ------------------------------------------------------------------------------------------------ bf_mask2d
Q = 4                               # the /4 working level of the photographs
FIT_LEVEL = 2                       # the stage-3 mask store's level the segment fit runs at
SCALES = np.arange(0.84, 1.021, 0.0015)
L4_CENTRE = 7.5                     # native px between the curated grid's L4 origin (16 j) and its voxel centre (16 j + 7.5)
MIN_FG_FRAC = 0.001                 # below this a label counts as a negative (no brain)
NEG_FRAC_CAP = 0.08                 # negatives: at most this fraction of a brain's kept photographs
BLANK_MEAN = 8.0                    # a photograph darker than this (0-255) is a blank frame
BLANK_MAX_BYTES = 500_000           # a black PNG compresses to ~29 KB, a real one is 3-9 MB
AREA_REL_TOL = 0.6                  # a label whose area departs from the local median by more than this is dropped
AREA_MIN_MED = 1e-4                 # ... except where the local median is ~0 (the poles)


def _segments(cfg, nz):
    """Inter-jump segments (lo, hi, njumps): the transform is constant within a segment."""
    starts = sorted(j for j, _ in cfg.jumps)
    segs, lo = [], 0
    for i, j in enumerate(starts):
        segs.append((lo, min(j, nz - 1), len(starts) - i))
        lo = j + 1
    if lo <= nz - 1:
        segs.append((lo, nz - 1, 0))
    return [(a, b, n) for a, b, n in segs if a <= b]


def _last_kept_slice(cfg, nz):
    """The last section the reconstruction keeps (a trailing (n, -1) `missing` entry is the caudal cutoff)."""
    cuts = [b for b, a in cfg.missing if a < 0]
    return min(cuts) if cuts else nz - 1


def _native_padded(png, level):
    """A photograph mask in raw-stack space: reversed, STACK_PAD-padded, decimated to `level`."""
    a = (np.array(Image.open(png)) > 0)[::-1, ::-1]
    P = zarrio.STACK_PAD
    H, W = a.shape
    p = np.zeros((H + 2 * P, W + 2 * P), bool)
    p[P:P + H, P:P + W] = a
    return p[::2 ** level, ::2 ** level]


def _dice(a, b):
    t = a.sum() + b.sum()
    return 0.0 if t == 0 else 2.0 * (a & b).sum() / t


def _fit_segments(cfg, nsamp=5, level=FIT_LEVEL):
    """Per segment, the (scale, dy, dx) mapping a native-padded photograph to the cropped jump-corrected stack, in L0 px: the best
    Dice between five of its accepted photograph masks and the stage-3 mask store."""
    pngs = sorted(glob(os.path.join(cfg.brainmask("01_final"), "*.png")))
    A = zarr.open_group(cfg.blockface_mask, mode="r")[str(level)]
    us = 2 ** level
    nz = A.shape[2]
    out = []
    for lo, hi, nj in _segments(cfg, len(pngs)):
        pr = []
        for k in [int(k) for k in np.linspace(lo, min(hi, nz - 1), nsamp)]:
            n_ = _native_padded(pngs[k], level)
            c_ = np.asarray(A[0, :, k, :]) > 0
            if n_.sum() > 20 and c_.sum() > 20:
                ny, nx = np.nonzero(n_)
                cy, cx = np.nonzero(c_)
                pr.append((n_, c_, ny.mean(), nx.mean(), cy.mean(), cx.mean()))
        if not pr:
            out.append(dict(lo=lo, hi=hi, nj=nj, scale=np.nan, dy=np.nan, dx=np.nan, dice=0.0, n=0))
            continue
        best, bd = (np.nan, np.nan, np.nan), -1.0
        for sc in SCALES:
            dy = np.median([cy - ny * sc for _, _, ny, _, cy, _ in pr])
            dx = np.median([cx - nx * sc for _, _, _, nx, _, cx in pr])
            d = np.mean([_dice(affine_transform(n_.astype(np.float32), np.array([[1 / sc, 0], [0, 1 / sc]]),
                                                offset=(-dy / sc, -dx / sc), output_shape=c_.shape, order=0) > 0.5, c_)
                         for n_, c_, *_ in pr])
            if d > bd:
                best, bd = (sc, dy * us, dx * us), d
        sc, dy, dx = best
        out.append(dict(lo=lo, hi=hi, nj=nj, scale=sc, dy=dy, dx=dx, dice=bd, n=len(pr)))
    return out


def _signed_distance(M, sampling):
    """Signed distance of a binary volume, inside positive, in physical units (interpolated instead of the labels, so no staircase)."""
    M = M.astype(bool)
    if not M.any():
        return np.full(M.shape, -1e3, np.float32)
    return (distance_transform_edt(M, sampling=sampling) - distance_transform_edt(~M, sampling=sampling)).astype(np.float32)


def _labels_q(cfg, segs):
    """Each photograph's label at /4: (k, png, s_float, label or None, status)."""
    pngs = sorted(glob(os.path.join(cfg.brainmask("01_final"), "*.png")))
    nz = len(pngs)
    mi, bi = nib.load(cfg.mask_edited), nib.load(cfg.curated_bg)
    M = np.asanyarray(mi.dataobj) > 0
    frac = np.zeros(3)                                                   # the offset's fraction left after the integer placement
    if mi.shape != bi.shape[:3] or np.abs(mi.affine - bi.affine).max() > 1e-3:
        Mb = np.zeros(bi.shape[:3], bool)                                # the mask on the bg grid, by the integer world offset
        sl_m, sl_b = [], []
        for a in range(3):
            col = bi.affine[:3, a]
            oe = float(np.dot(mi.affine[:3, 3] - bi.affine[:3, 3], col) / np.dot(col, col)) if a != 1 else 0.0
            o = int(round(oe)); frac[a] = oe - o
            lo, hi = max(0, -o), min(M.shape[a], bi.shape[a] - o)
            sl_m.append(slice(lo, hi)); sl_b.append(slice(lo + o, hi + o))
        Mb[tuple(sl_b)] = M[tuple(sl_m)]
        assert Mb.sum() == M.sum(), (cfg.subject, "label voxels outside the bg canvas", int(M.sum() - Mb.sum()))
        M = Mb
    W4, nzc, H4 = M.shape
    f = cfg.curation_f(path=cfg.mask_edited)
    iso = cfg.vx_header * 16 / 1000.0
    D = _signed_distance(M, sampling=(iso, float(params()["reconstruction"]["curation_z_target_um"]) / 1000.0, iso))
    PAD = zarrio.STACK_PAD
    H, W = np.array(Image.open(pngs[0])).shape[:2]
    assert H % Q == 0 and W % Q == 0, (H, W)
    seg_of = {k: s for s in segs for k in range(s["lo"], s["hi"] + 1)}
    last = _last_kept_slice(cfg, nz)
    for k, png in enumerate(pngs):
        s = seg_of.get(k)
        sf = (k - (f - 1) / 2.0) / f
        why = ""
        if s is None or not np.isfinite(s.get("scale", np.nan)):
            why = "no-transform"
        elif k > last:
            why = "beyond-caudal-cutoff"
        elif sf > nzc - 1 + 0.5:
            why = "beyond-curated-volume"
        if why:
            yield k, png, sf, None, why
            continue
        s0 = int(np.clip(np.floor(sf), 0, nzc - 1))
        s1 = int(np.clip(s0 + 1, 0, nzc - 1))
        w = float(np.clip(sf - s0, 0.0, 1.0))
        D2 = ((1 - w) * D[:, s0, :] + w * D[:, s1, :]).T
        sc, dy, dx = s["scale"], s["dy"], s["dx"]
        m = affine_transform(D2, np.array([[sc * Q / 16.0, 0], [0, sc * Q / 16.0]]),
                             offset=((sc * (PAD + (Q - 1) / 2.0) + dy - L4_CENTRE) / 16.0 - frac[2],
                                     (sc * (PAD + (Q - 1) / 2.0) + dx - L4_CENTRE) / 16.0 - frac[0]),
                             output_shape=(H // Q, W // Q), order=1, mode="constant", cval=-1e3)
        yield k, png, sf, (m[::-1, ::-1] > 0).astype(np.uint8), "ok"


def _annotated_blank(cfg):
    """Sections the brain's `missing` marks as not photographed (a trailing (n, -1), the caudal cutoff, excepted)."""
    out = set()
    for b, a in cfg.missing:
        if a >= 0:
            out.update(range(b + 1, a))
    return out


def _size_blank(rows):
    """Blank frames the annotation missed: a small PNG (< BLANK_MAX_BYTES), confirmed dark by decoding."""
    out = set()
    for r in rows:
        try:
            if os.path.getsize(r["image"]) >= BLANK_MAX_BYTES:
                continue
        except OSError:
            continue
        if np.asarray(Image.open(r["image"]).convert("L").resize((256, 192))).mean() < BLANK_MEAN:
            out.add(r["k"])
    return out


def _screen(rows, blanks=frozenset()):
    """Drop blank frames and implausible labels; keep negatives, capped. Returns (kept, [(name, why)])."""
    keep, drop = [], []
    if rows:
        with Image.open(rows[0]["image"]) as im:
            npx = float(im.size[0] * im.size[1])
    else:
        npx = 1.0
    areas = []
    for r in rows:
        r["fg"] = r["area"] / npx
        areas.append(r["fg"])
    areas = np.array(areas)
    W = 15
    med = np.array([np.median(areas[max(0, i - W):i + W + 1]) for i in range(len(areas))])
    negatives = []
    for i, r in enumerate(rows):
        why = ""
        blank = r["k"] in blanks
        if r["fg"] < MIN_FG_FRAC:
            if not blank:
                blank = np.asarray(Image.open(r["image"]).convert("L").resize((256, 192))).mean() < BLANK_MEAN
            if blank:
                drop.append((r, "blank-photo"))
            else:
                r["negative"] = True
                negatives.append(r)
            continue
        if blank:
            why = "blank-photo"
        elif med[i] > AREA_MIN_MED and abs(areas[i] / med[i] - 1.0) > AREA_REL_TOL:
            why = "area-outlier"
        (drop if why else keep).append((r, why))
    kept = [r for r, _ in keep]
    cap = int(NEG_FRAC_CAP * max(len(kept), 1))
    if len(negatives) > cap:
        idx = np.linspace(0, len(negatives) - 1, cap).astype(int) if cap else []
        chosen = {id(negatives[i]) for i in idx}
        for r in negatives:
            if id(r) not in chosen:
                drop.append((r, "negative-over-cap"))
        negatives = [negatives[i] for i in idx]
    for r in kept:
        r["negative"] = False
    return kept + negatives, [(r["name"], w) for r, w in drop]


def _save_png(a, path):
    tmp = path + ".tmp.png"
    Image.fromarray(a).save(tmp, compress_level=1)
    os.replace(tmp, path)


def mask2d_brain(args):
    """bf_mask2d, one brain: its kept photographs at /4 with their labels, and index/<ID>.tsv (every photograph, kept or why not)."""
    S, out = args
    cfg = Subject(S)
    segs = _fit_segments(cfg)
    labs, rows, info = {}, [], {}
    for k, png, sf, lab, why in _labels_q(cfg, segs):
        name = os.path.basename(png)
        info[name] = dict(k=k, s_float=round(sf, 3), status=why)
        if lab is None:
            continue
        labs[name] = lab
        rows.append(dict(subject=S, image=os.path.join(cfg.slices, name.replace(".png", "_0000.png")), name=name, k=k,
                         area=int(lab.sum()) * Q * Q))
    kept, dropped = _screen(rows, _annotated_blank(cfg) | _size_blank(rows))
    for name, why in dropped:
        info[name]["status"] = why
    for r in sorted(kept, key=lambda r: r["k"]):
        cid = f"{S}_{r['k']:04d}"
        rgb = np.asarray(Image.open(r["image"]).convert("RGB")).astype(np.float32)
        H, W = rgb.shape[:2]
        q = rgb.reshape(H // Q, Q, W // Q, Q, 3).mean(axis=(1, 3))
        _save_png(np.clip(np.rint(q), 0, 255).astype(np.uint8), os.path.join(out, "imagesTr", f"{cid}_0000.png"))
        _save_png(labs[r["name"]], os.path.join(out, "labelsTr", f"{cid}.png"))
        info[r["name"]].update(status="kept-negative" if r.get("negative") else "kept", case=cid, area_q=int(labs[r["name"]].sum()))
    with open(os.path.join(out, "index", f"{S}.tsv"), "w") as fh:
        fh.write("k\tname\ts_float\tstatus\tcase\tarea_q\n")
        for name, v in sorted(info.items(), key=lambda kv: kv[1]["k"]):
            fh.write(f"{v['k']}\t{name}\t{v['s_float']}\t{v['status']}\t{v.get('case', '')}\t{v.get('area_q', '')}\n")
    print("bf_mask2d", S, "photographs", len(info), "kept", len(kept), "fit Dice min %.4f" % min(s["dice"] for s in segs), flush=True)
    return S, len(kept), [dict(lo=s["lo"], hi=s["hi"], scale=s["scale"], dy=s["dy"], dx=s["dx"], dice=s["dice"]) for s in segs]


def stage(task_file, out, jobs=8):
    """Stage every case of `task_file` into `out` and write <out>/<task>.json. Refuses a non-empty `out`. jobs: brains staged at a
    time (bf_mask2d)."""
    t = json.load(open(task_file))
    task = t["task"]
    if task not in TASKS:
        raise SystemExit(f"{task_file}: task {task!r}; this stages {', '.join(TASKS)}")
    if os.path.isdir(out) and os.listdir(out):
        raise SystemExit(f"{out} is not empty: stage into a new folder")
    for sub in ("imagesTr", "labelsTr") + (("masksTr",) if task == "bf_holes" else ()) + (("index",) if task == "bf_mask2d" else ()):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    if task == "bf_mask2d":                             # one folder of photographs (the task's nnunet_raw), its brains as listed
        brains = sorted(t["spacing_by_subject"])
        with Pool(jobs) as pool:
            res = pool.map(mask2d_brain, [(S, out) for S in brains])
        n = len(glob(os.path.join(out, "labelsTr", "*.png")))
        json.dump({"channel_names": {"0": "RGB"}, "labels": {"background": 0, "brain": 1}, "numTraining": n, "file_ending": ".png",
                   "description": "Blockface photographs at /4 (4 x 4 block mean of the native RGB) with brain labels from the 3D "
                                  "masks of record (code/blockface/training.py)"}, open(os.path.join(out, "dataset.json"), "w"), indent=1)
        t["nnunet_raw"] = out
        p = os.path.join(out, f"{task}.json")
        json.dump(t, open(p, "w"), indent=1)
        write_provenance(out, inputs=[task_file] + [Subject(S).mask_edited for S in brains], outputs=[p],
                         extra=dict(task=task, kept={S: k for S, k, _ in res}, segments={S: g for S, _, g in res}))
        return p
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
