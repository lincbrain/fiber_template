"""Annotations on histology pushed onto the blockface (bf-native), through each piece's histology -> blockface registration (TIRL).

Paper: Methods, "Histology registration and label propagation". Parameters: configs/labels.yaml (push, layout).

Tracer annotations (`push_tracer`). For every section with any non-empty annotation class and every registered piece of it:
  1. the class masks are read at level 6, cropped to the section exactly as `sections.Section` (OME-Zarr annotation stacks aligned
     with the histology image stack);
  2. all classes of a section travel as ONE small-integer image (each pixel's class combination -> a code < 255), so the chain is
     evaluated once per piece and 255 keeps meaning "outside the moving domain"; classes are not exclusive;
  3. the code image is placed on the piece's moving grid with the registration's own geometry (crop to the piece mask's bounding box,
     scaled isotropically into the blockface piece's box: `place`);
  4. it goes through the piece's TIRL chain (moving.timg -> fixed4_nonlinear.timg, nearest neighbor), is turned back to the store's
     orientation (the blockface piece is the store slice turned 180 degrees), max-pooled `pool` x `pool` onto the target grid and
     placed at the piece's blockface slice (photograph number from the match table - the first photograph number).
Output: one NIfTI per class, <S>_<class>_bfnative_L2.nii.gz (uint8 0/1, the target grid's affine = bf-native world), and
push_tracer_<S>.json (per piece and class: pixels on the section, on the moving grid, in the blockface piece, on the target grid).

LGN/STN labels (`push_structures`), two stages: (A) each labeled frame of the annotators' histology stack goes onto its piece's moving
grid through the frame -> piece transform (SimpleITK; composed with the moving-grid similarity when the transform was fitted on another
rendering of the same piece); (B) the same TIRL chain and placement as above, at the whole-section blockface slice.

Ported from the analysis project's labels/push_tracer.py, labels/push_labels.py and labels/bf_push_grid.py (final path only).
"""
import json
import os
import re
from glob import glob
from pathlib import Path

import numpy as np

from .. import paths

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "labels.yaml"


# --------------------------------------------------------------------------- configuration and layout

def config():
    import yaml
    return yaml.safe_load(CONFIG.read_text())


def layout(subject, key, cfg=None):
    """$LINC_DATA/sub-<S>/<layout[key]> (configs/labels.yaml)."""
    cfg = cfg or config()
    return paths.subject_dir(subject) / cfg["layout"][key].format(S=subject)


def subject_config(subject):
    """configs/subjects/<S>.yaml (the reconstruction's inputs: idx_range, vx, vz, ...)."""
    import yaml
    return yaml.safe_load((CONFIG.parent / "subjects" / f"{subject}.yaml").read_text())


def first_photo(subject):
    """Photograph number of the first blockface slice (slice index = photograph number - this)."""
    return int(subject_config(subject)["idx_range"][0])


# --------------------------------------------------------------------------- shared pieces

def match_table(path):
    """Parse a match table into a lookup: section number -> [whole, left, right, cerebellum] photograph numbers (-1 = no match)."""
    rows = []
    with open(path) as f:
        next(f)                                        # header
        for line in f:
            v = re.split(r"[,;.:]+", line.strip())
            if len(v) < 10:
                continue
            lo = int(v[0])
            hi = np.inf if "inf" in v[1] else int(v[1])
            rows.append((lo, hi, [int(x) for x in v[2:10]]))   # (multiplier, adder) x (whole, left, right, cerebellum)

    def lookup(num):
        for lo, hi, c in rows:
            if lo <= num <= hi:
                return [num * c[2 * i] + c[2 * i + 1] for i in range(4)]
        return [-1, -1, -1, -1]

    return lookup


def _first_nonzero_row(img):
    for i in range(img.shape[0]):
        if np.sum(img[i]) > 0:
            return i
    return img.shape[0]


def borders(img):
    """Distances (top, bottom, left, right) from each edge to the tissue, less a 2 px margin (the registration's moving geometry)."""
    t = _first_nonzero_row(img)
    b = _first_nonzero_row(img[::-1])
    l = _first_nonzero_row(np.rot90(img, 3))
    r = _first_nonzero_row(np.rot90(img, 1))
    return [max(0, k - 2) for k in (t, b, l, r)]


def place(pm, content, photo):
    """The registration's moving-image geometry for an integer label image (nearest): crop to the piece mask's bounding box and
    scale it isotropically into the blockface piece's box."""
    from PIL import Image
    H, W = photo.shape[:2]
    tt, tb, ll, lr = borders(photo)
    t, b_, l, rr_ = borders(np.where(pm, 255, 0).astype(np.uint8))
    h, w = pm.shape[0] - t - b_, pm.shape[1] - l - rr_
    box_h, box_w = H - tt - tb, W - ll - lr
    while h < box_h and w < box_w:
        h, w = 1.2 * h, 1.2 * w
    while h > box_h or w > box_w:
        h, w = 0.9 * h, 0.9 * w
    h, w = max(1, int(h)), max(1, int(w))
    crop = np.where(pm, content, 0)[t:pm.shape[0] - b_, l:pm.shape[1] - rr_].astype(np.uint8)
    canvas = np.zeros((H, W), np.uint8)
    canvas[tt:tt + h, ll:ll + w] = np.asarray(Image.fromarray(crop).resize((w, h), Image.NEAREST))
    return canvas


def tirl_to_blockface(moving_label, reg_dir):
    """A uint8 image on the piece's moving grid -> the blockface piece grid, through the piece's TIRL chain (nearest neighbor).
    Pixels outside the moving domain come back as `push.outside_code` (255)."""
    from tirl import load
    from tirl.timage import TImage
    from tirl.interpolation.scipyinterpolator import ScipyInterpolator
    mv, fx = load(os.path.join(reg_dir, "moving.timg")), load(os.path.join(reg_dir, "fixed4_nonlinear.timg"))
    L = TImage(moving_label)
    L.domain = mv.domain.copy()
    L.interpolator = ScipyInterpolator(order=0)
    return np.array(L.evaluate(fx.domain)).squeeze()


def target_grid(grid_path):
    """-> (affine, (rows, slices, cols)) of the target grid (a NIfTI stored as (cols, slices, rows))."""
    import nibabel as nib
    im = nib.load(str(grid_path))
    X, Y, Z = im.shape[:3]
    return im.affine, (Z, Y, X)


def has_registration(reg_dir, photo):
    return os.path.exists(os.path.join(reg_dir, "fixed4_nonlinear.timg")) and os.path.exists(photo)


# --------------------------------------------------------------------------- tracer annotations

def annotation_classes(annotations_dir, exclude=()):
    cl = sorted(os.path.basename(p)[:-len(".ome.zarr")] for p in glob(os.path.join(str(annotations_dir), "*.ome.zarr")))
    return [c for c in cl if c not in set(exclude)]


def push_tracer(subject, *, annotations_dir, section_map, match_table_path, piece_masks, blockface_pieces, registrations, grid,
                first_photo_number, out_dir, cfg=None, log=print):
    """Push every annotation class of `subject` onto the target grid (see the module docstring). Returns the summary dict."""
    import nibabel as nib
    from PIL import Image
    from .sections import Section, load_map
    Image.MAX_IMAGE_PIXELS = None
    P = (cfg or config())["push"]
    level, K, outside = int(P["annotation_level"]), int(P["pool"]), int(P["outside_code"])
    look = match_table(match_table_path)
    rows = [r for r in load_map(section_map) if r["section"] != "UNMATCHED"]
    classes = annotation_classes(annotations_dir, P["exclude_classes"])
    aff, (rows2, n_sl, cols2) = target_grid(grid)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sheets = {c: {} for c in classes}                                 # class -> {blockface slice: 2D array on the target grid}
    plog = []
    n_masks = n_ready = n_sections = n_no_masks = 0
    for r in rows:
        ms = {}
        for c in classes:
            a = Section(os.path.join(str(annotations_dir), c + ".ome.zarr"), r["z"], r["h0"], r["w0"]).level(level)
            a = (a[..., 0] if a.ndim == 3 else a) > 0
            if a.any():
                ms[c] = a
        if not ms:
            continue
        sid = r["section"]
        bits = sum((ms[c].astype(np.int64) << i) for i, c in enumerate(ms))    # class combination -> code (1..n), 0 = none
        combos, inv = np.unique(bits, return_inverse=True)
        assert len(combos) < outside
        code = inv.reshape(bits.shape).astype(np.uint8)                # combos[0] == 0 when background is present
        zero = int(np.nonzero(combos == 0)[0][0]) if (combos == 0).any() else None
        if zero not in (None, 0):
            raise RuntimeError("unexpected code order")
        masks = sorted(glob(os.path.join(str(piece_masks), sid + "_*.png")))
        n_sections += 1
        n_no_masks += not masks
        for mpath in masks:
            piece = os.path.basename(mpath)[:-4]
            tdir = os.path.join(str(registrations), piece)
            photo_p = os.path.join(str(blockface_pieces), piece + ".png")
            n_masks += 1
            if not has_registration(tdir, photo_p):
                continue
            n_ready += 1
            pm = np.asarray(Image.open(mpath)) > 0
            if not (pm & (code > 0)).any():
                continue
            canvas = place(pm, code, np.asarray(Image.open(photo_p)))
            o = tirl_to_blockface(canvas, tdir)
            o = np.where(o == outside, 0, o).astype(np.int64)[::-1, ::-1]      # store orientation
            num = int(sid.split("_s")[1][:2])
            idx = look(num)[int(piece[-4:])] - first_photo_number
            if not (0 <= idx < n_sl):
                continue
            h, w = o.shape[0] // K * K, o.shape[1] // K * K
            rec = dict(section=sid, piece=piece, bf_idx=int(idx), classes={})
            for i, c in enumerate(ms):
                sel = np.isin(o, np.nonzero((combos >> i) & 1)[0])
                pooled = sel[:h, :w].reshape(h // K, K, w // K, K).max((1, 3))[:rows2, :cols2]
                if not pooled.any():
                    continue
                sh = sheets[c].setdefault(int(idx), np.zeros((rows2, cols2), bool))
                sh[:pooled.shape[0], :pooled.shape[1]] |= pooled
                rec["classes"][c] = dict(px_section=int((ms[c] & pm).sum()),
                                         px_moving=int(np.isin(canvas, np.nonzero((combos >> i) & 1)[0]).sum()),
                                         px_blockface=int(sel.sum()), px_L2=int(pooled.sum()))
            plog.append(rec)
            log(f"{subject} {piece} -> slice {idx}: " + ", ".join(f"{c} {v['px_L2']}" for c, v in rec["classes"].items()))
    if not plog:
        raise SystemExit(f"{subject}: nothing pushed, nothing written. Found {len(classes)} annotation classes in {annotations_dir}; {n_masks} "
                         f"component masks in {piece_masks}, {n_ready} of them with a photograph in {blockface_pieces} and a registration in "
                         f"{registrations}. linc-histology makes the masks and the photographs.")
    log(f"{subject}: {n_sections} annotated sections, {n_no_masks} with no component mask; {n_masks} component masks, {n_masks - n_ready} with no "
        f"photograph or registration. What these skip is missing from the labels. {len(plog)} components pushed.")
    summary, outputs = {}, []
    for c, sl in sheets.items():
        if not sl:
            continue
        v = np.zeros((cols2, n_sl, rows2), np.uint8)                     # NIfTI (cols, slices, rows)
        for idx, a in sl.items():
            v[:, idx, :] = a.T
        f = out_dir / f"{subject}_{c}_bfnative_L2.nii.gz"
        nib.save(nib.Nifti1Image(v, aff), str(f))
        outputs.append(f)
        summary[c] = dict(slices=len(sl), voxels=int(v.sum()))
        del v
    rec = dict(subject=subject, grid=os.path.basename(str(grid)), pool=K, registrations=str(registrations), classes=classes,
               summary=summary, pieces=plog)
    (out_dir / f"push_tracer_{subject}.json").write_text(json.dumps(rec, indent=1))
    log(f"{subject} {summary}")
    return rec, outputs


# --------------------------------------------------------------------------- LGN / STN labels (two stages)

def moving_similarity(old_png, new_png):
    """Two renderings of one piece's moving image (the same histology object cropped to its own bounding box and scaled
    isotropically into the blockface piece's box) differ by a scale + translation read off their content boxes. Returns the
    SimpleITK transform new-physical -> old-physical (physical = pixel index - 1.5) and the check (correlation of the old image
    carried onto the new grid with the new image)."""
    import SimpleITK as sitk
    from PIL import Image
    a = [np.asarray(Image.open(p).convert("L")).astype(np.float32) for p in (old_png, new_png)]
    box = []
    for im in a:
        t, b, l, r = borders(im)
        box.append((t, l, im.shape[0] - t - b, im.shape[1] - l - r))
    (t0, l0, h0, w0), (t1, l1, h1, w1) = box
    sx, sy = w1 / w0, h1 / h0
    sim = sitk.AffineTransform(2)
    sim.SetMatrix((1 / sx, 0, 0, 1 / sy))
    sim.SetTranslation(((l0 - 1.5) - (l1 - 1.5) / sx, (t0 - 1.5) - (t1 - 1.5) / sy))
    o = sitk.GetImageFromArray(a[0]); o.SetOrigin((-1.5, -1.5))
    ref = sitk.Image(a[1].shape[1], a[1].shape[0], sitk.sitkFloat32); ref.SetOrigin((-1.5, -1.5))
    on_new = sitk.GetArrayFromImage(sitk.Resample(o, ref, sim, sitk.sitkLinear, 0.0))
    r = float(np.corrcoef(on_new.ravel(), a[1].ravel())[0, 1])
    return sim, dict(scale_x=round(sx, 4), scale_y=round(sy, 4), corr_old_on_new=round(r, 4))


def frames_from_stack(stack_nii, label_nii):
    """A label volume resampled onto its histology stack's grid by WORLD coordinates (labels may sit on shifted or padded grids):
    returns (volume on the stack grid, the frames z that carry the label)."""
    import nibabel as nib
    stack = nib.load(str(stack_nii))
    L = nib.load(str(label_nii))
    ijk = np.argwhere(np.asarray(L.dataobj) > 0)
    t = np.round(nib.affines.apply_affine(np.linalg.inv(stack.affine), nib.affines.apply_affine(L.affine, ijk))).astype(int)
    vol = np.zeros(stack.shape[:3], np.uint8)
    ok = np.all((t >= 0) & (t < stack.shape[:3]), 1)
    vol[tuple(t[ok].T)] = 1
    return vol, sorted(set(t[ok][:, 2].tolist()))


def structure_to_moving(frame_label, transform, moving_png, fitted_png=None):
    """Stage A for one frame: the upright frame label (rot270 of the stack frame) -> the piece's moving grid through the frame ->
    piece transform (nearest). `fitted_png` = the moving rendering the transform was fitted on, when it is not `moving_png`."""
    import SimpleITK as sitk
    from PIL import Image
    tx = sitk.ReadTransform(str(transform))
    chk = {}
    if fitted_png is not None and os.path.realpath(fitted_png) != os.path.realpath(moving_png):
        sim, chk = moving_similarity(fitted_png, moving_png)
        ct = sitk.CompositeTransform(2)
        ct.AddTransform(tx)
        ct.AddTransform(sim)                                      # applied FIRST (SimpleITK: last added, first applied)
        tx = ct
    W, H = Image.open(moving_png).size
    lab = np.ascontiguousarray(np.rot90(frame_label, 3)).astype(np.float32)
    mi = sitk.GetImageFromArray(lab); mi.SetOrigin((-1.5, -1.5))
    ref = sitk.Image(W, H, sitk.sitkFloat32); ref.SetOrigin((-1.5, -1.5))
    out = sitk.GetArrayFromImage(sitk.Resample(mi, ref, tx, sitk.sitkNearestNeighbor, 0.0)).astype(np.uint8)
    return out, int(lab.sum()), chk


def push_structures(subject, items, *, match_table_path, registrations, grid, first_photo_number, out_dir, structs, cfg=None,
                    log=print):
    """Stage B: `items` = [{struct, frame, piece, moving: uint8 label on the piece's moving grid}] (stage A) -> one NIfTI per
    structure, <S>_<struct>_bfnative_L2.nii.gz, at the whole-section blockface slice of each piece."""
    import nibabel as nib
    P = (cfg or config())["push"]
    K, outside = int(P["pool"]), int(P["outside_code"])
    look = match_table(match_table_path)
    aff, (rows2, n_sl, cols2) = target_grid(grid)
    vols = {st: np.zeros((rows2, n_sl, cols2), np.uint8) for st in structs}
    plog = []
    for e in items:
        st, piece, lab = e["struct"], e["piece"], e["moving"]
        out = tirl_to_blockface(lab, os.path.join(str(registrations), piece))
        out = np.where(out == outside, 0, out).astype(np.uint8)
        store = out[::-1, ::-1]                                               # undo the blockface piece's 180-degree turn
        h, w = store.shape[0] // K * K, store.shape[1] // K * K
        pooled = store[:h, :w].reshape(h // K, K, w // K, K).max((1, 3))
        num = int(piece.split("_s")[1][:2])
        idx = look(num)[0] - first_photo_number                               # whole-section column
        vols[st][:pooled.shape[0], idx, :pooled.shape[1]] |= pooled[:rows2, :cols2]
        plog.append(dict(struct=st, frame=e.get("frame"), section=num, piece=piece, bf_idx=int(idx),
                         px_moving=int(lab.sum()), px_fixed=int(out.sum()), px_L2=int(pooled.sum())))
        log(f"{subject} {st} {piece}: moving {int(lab.sum())} px -> blockface piece {int(out.sum())} px -> slice {idx}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for st, v in vols.items():
        f = out_dir / f"{subject}_{st}_bfnative_L2.nii.gz"
        nib.save(nib.Nifti1Image(v.transpose(2, 1, 0), aff), str(f))
        outputs.append(f)
    rec = dict(grid=os.path.basename(str(grid)), pool=K, registrations=str(registrations), pieces=plog)
    (out_dir / f"push_structures_{subject}.json").write_text(json.dumps(rec, indent=1))
    return rec, outputs


def structures_stage_a(subject, structures_dir, moving_dir, structs, log=print):
    """Stage A over every labeled frame: -> (items for push_structures, log records). Layout of `structures_dir`: histo_L6.nii.gz,
    histo_L6_<struct>.nii.gz, frame_map.json {frame: section number}, transforms/f<frame>_<piece>.tfm, fitted_moving/<piece>.png."""
    sd = Path(structures_dir)
    fmap = {int(k): int(v) for k, v in json.loads((sd / "frame_map.json").read_text()).items()}
    items, rlog = [], []
    for st in structs:
        vol, frames = frames_from_stack(sd / "histo_L6.nii.gz", sd / f"histo_L6_{st}.nii.gz")
        for z in frames:
            if z not in fmap:
                rlog.append(dict(struct=st, frame=z, status="frame outside the registered range")); continue
            n = fmap[z]
            cand = sorted(glob(os.path.join(str(moving_dir), f"*_s{n:02d}nDF_0000.png")))
            if not cand:
                rlog.append(dict(struct=st, frame=z, section=n, status="piece not registered")); continue
            piece = os.path.basename(cand[0])[:-4]
            tfm = sd / "transforms" / f"f{z:02d}_{piece}.tfm"
            if not tfm.exists():
                rlog.append(dict(struct=st, frame=z, section=n, status="no transform")); continue
            fitted = sd / "fitted_moving" / f"{piece}.png"
            lab, px_frame, chk = structure_to_moving(vol[..., z], tfm, cand[0], str(fitted) if fitted.exists() else None)
            items.append(dict(struct=st, frame=z, piece=piece, moving=lab))
            rlog.append(dict(struct=st, frame=z, section=n, piece=piece, px_frame=px_frame, px_moving=int(lab.sum()), status="ok", **chk))
            log(f"{subject} {st} frame {z} -> {piece}: {px_frame} px -> {int(lab.sum())} px on moving")
    return items, rlog
