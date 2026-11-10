"""The blockface photographs split into the components the histology sections were cut into, so each component is registered on its
own (Methods, "Histology registration and label propagation").

You need
  the matched photographs (match.match_photos) and the brain's regions volume on the blockface: a 3D label volume (1 left
  hemisphere, 2 right hemisphere, 3 cerebellum + brainstem, 4 marker tubes) transformed from the atlas through the template to this
  brain's blockface; the corpus-callosum section range [ccs, cce] (settings/brains/<ID>.yaml).
What it does
  partition()   for every matched photograph, the regions are sampled at its world coordinates (the world affine of the store's
                level 1) and restricted to its tissue; unlabeled tissue gets the nearest label, and sections whose tissue is two
                separate lobes side by side take the geometric split. Anatomical left and right become the component classes
                (1 = the image-left component of the upright photograph, the order of the histology tissue masks) once per brain,
                from the data.
  cut_pieces()  each matched photograph split by its partition into components: 0001 left, 0002 right, 0003 cerebellum. Inside
                [ccs, cce] the hemispheres are joined and a whole-section photograph is kept whole (0000); a photograph matched for
                one hemisphere or the cerebellum is never kept whole.
You get
  <seg_dir>/<photo>.png (class x 50) and <seg_dir>/partition.json; <section>_<tag>.png per component.

Ported from the analysis project's labels/seg_from_regions.py (with labels/propose_reg.split_separate_hemispheres) and
code/histo.py segment_bf, the code of the products of record.
"""
import json
import os
from glob import glob

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, label

from . import match as M

Image.MAX_IMAGE_PIXELS = None


def store_affine(store, level):
    """World (RAS mm) affine of a blockface store's pyramid level, voxel order (cols, slices, rows): world = scale * (voxel +
    translation), micrometers -> mm (the OME-Zarr -> NIfTI convention of the reconstruction; equals its <ID>_blockface_L2 nifti
    for level 1)."""
    import zarr
    g = zarr.open_group(str(store), mode="r")
    ct = g.attrs["multiscales"][0]["datasets"][level]["coordinateTransformations"]
    tr, sc = ct[0]["translation"], ct[1]["scale"]                  # [c, z=rows, y=slices, x=cols]
    A = np.eye(4)
    A[0, 0], A[1, 1], A[2, 2] = sc[3] / 1000.0, sc[2] / 1000.0, sc[1] / 1000.0
    A[0, 3], A[1, 3], A[2, 3] = sc[3] * tr[3] / 1000.0, sc[2] * tr[2] / 1000.0, sc[1] * tr[1] / 1000.0
    return A


def save_mask(lab, fname, scale):
    Image.fromarray((np.asarray(lab).astype(np.uint8) * scale)).save(fname)


def load_mask(fname, scale):
    return (np.asarray(Image.open(fname)).astype(np.int16) // scale).astype(np.uint8)


def split_separate_hemispheres(lab, tissue, cfg=None):
    """Where a slice's tissue is exactly two large side-by-side components (the frontal pole, where the hemispheres are
    physically apart), the image-left one is the left piece (1) and the other the right piece (2), each whole. Components that
    are mostly cerebellum (3) are left alone; a slice with one connected tissue component keeps the regions' boundary."""
    cfg = cfg or M.config()
    g = cfg["geometric_split"]
    if not g["enabled"]:
        return lab
    comp, n = label(tissue)
    if n < 2:
        return lab
    sizes = np.bincount(comp.ravel())[1:]
    o = np.argsort(sizes)[::-1]
    if sizes[o[1]] < g["second_component_min_fraction"] * sizes[o[0]]:
        return lab
    a, b = o[0] + 1, o[1] + 1
    for c in (a, b):
        v = np.bincount(lab[comp == c], minlength=4)
        if v[3] > g["cerebellum_majority"] * max(v.sum(), 1):
            return lab
    L, R = (a, b) if np.nonzero(comp == a)[1].mean() < np.nonzero(comp == b)[1].mean() else (b, a)
    out = lab.copy()
    out[comp == L] = 1
    out[comp == R] = 2
    return out


def regions_scale(subject_cfg, regions_spacing_um=None):
    """World scale (x, y, z) that takes a regions volume into the store's current, calibrated frame: (vx / vx_r, vz / vz_r, vx / vx_r),
    x and z in plane, y = the section axis (anterior-posterior), about the world origin, as the header-only spacing calibration did
    (code/blockface/spacing.py). vx, vz: the brain's calibrated pixel size and section step (settings/brains/<ID>.yaml); vx_r, vz_r:
    the spacing the regions volume was made at (settings/histology.yaml regions.made_at_spacing_um, the nominal spacing before the
    2 Oct calibration for the histology brains of the paper). 1 when the regions were made on the current frame (every new brain)."""
    vx, vz = float(subject_cfg["vx"]), float(subject_cfg["vz"])
    r = regions_spacing_um or {"vx": vx, "vz": vz}
    sx, sz = vx / float(r["vx"]), vz / float(r["vz"])
    return (sx, sz, sx)


def partition(matched_dir, regions, store, table, idx0, seg_dir, cfg=None, log=print, scale_xyz=None):
    """Pieces of every matched photo from the atlas regions -> seg_dir/<photo>.png (class * 50) + seg_dir/partition.json.
    regions: the regions NIfTI; idx0: raw index of the store's first slice. scale_xyz: world scale that takes the regions into the
    store's frame (regions_scale(); None = already in the store's frame)."""
    import nibabel as nib
    cfg = cfg or M.config()
    rc = cfg["regions"]
    scale = rc["mask_scale"]
    A_d = store_affine(store, rc["sample_level"])
    rim = nib.load(str(regions))
    R3 = np.asarray(rim.dataobj).astype(np.uint8)
    R3[R3 == rc["punch_hole_label"]] = rc["punch_hole_as"]
    A_r = rim.affine if scale_xyz is None else np.diag(list(scale_xyz) + [1.0]) @ rim.affine     # about the world origin
    Rinv = np.linalg.inv(A_r)
    look = M.match_table(table, cfg)
    col = M.piece_column(cfg)
    suffixes = M.piece_suffixes(cfg)
    samples = {}
    for p in sorted(glob(os.path.join(str(matched_dir), "*_0000.png"))):
        name = os.path.basename(p)[:-9]
        suf = next((s for s in suffixes if name.endswith(s)), "")
        hb = name[: -len(suf)] if suf else name
        idx = look(M.section_number(hb, cfg))[col[suf]] - idx0
        photo = np.asarray(Image.open(p))
        Hd, Wd = photo.shape[:2]
        U, V = np.meshgrid(np.arange(Hd), np.arange(Wd), indexing="ij")
        R, C = Hd - 1 - U, Wd - 1 - V                                      # store level 0 (upright -> store)
        vox = np.stack([(C - 0.5) / 2, np.full(C.shape, float(idx)), (R - 0.5) / 2, np.ones(C.shape)], -1)   # -> level 1
        w = vox @ A_d.T
        v = np.round(w @ Rinv.T)[..., :3].astype(int)
        ok = np.all((v >= 0) & (v < np.array(R3.shape)), -1)
        lab = np.zeros((Hd, Wd), np.uint8)
        lab[ok] = R3[v[ok, 0], v[ok, 1], v[ok, 2]]
        tissue = photo.max(-1) > 0
        samples[name] = (lab, tissue, idx)
    # anatomical -> image side: the anatomical label whose pixels sit further image-left is piece class 1
    an = rc["anatomical"]
    cl = {a: np.mean(np.concatenate([np.nonzero((lab == a) & t)[1] for lab, t, _ in samples.values()] or [[np.nan]]))
          for a in (an["left"], an["right"])}
    img_left = min(cl, key=cl.get)
    trans = np.array([0, 0, 0, 3], np.uint8)
    trans[img_left] = 1
    trans[3 - img_left] = 2
    os.makedirs(seg_dir, exist_ok=True)
    rows = {}
    for name, (lab, tissue, idx) in samples.items():
        piece = trans[lab]
        piece[~tissue] = 0
        cov = float((piece > 0).sum() / max(tissue.sum(), 1))
        miss = tissue & (piece == 0)
        if miss.any() and (piece > 0).any():
            _, (a, b) = distance_transform_edt(piece == 0, return_indices=True)
            piece[miss] = piece[a[miss], b[miss]]
        piece = split_separate_hemispheres(piece, tissue, cfg)
        save_mask(piece, os.path.join(seg_dir, name + ".png"), scale)
        rows[name] = dict(bf_slice=int(idx), regions_cover=round(cov, 4), px={int(k): int((piece == k).sum()) for k in (1, 2, 3)})
    summ = dict(regions=str(regions), anatomical_label_on_image_left=int(img_left), slices=len(rows),
                regions_cover_median=round(float(np.median([r["regions_cover"] for r in rows.values()])), 4) if rows else None)
    with open(os.path.join(seg_dir, "partition.json"), "w") as f:
        json.dump(dict(summary=summ, slices=rows), f, indent=1)
    log(f"partitioned {len(rows)} matched photos from the atlas regions -> {seg_dir} {summ}")
    return summ


def cut_pieces(matched_dir, seg_dir, table, ccs, cce, out_dir, whole_from_table=None, cfg=None, log=print):
    """Split each matched photo into its pieces -> out_dir/<section>_<tag>.png (empty pieces are not written).
    ccs, cce: the histology section range where the corpus callosum joins the hemispheres; whole_from_table: a whole-section
    row itself means "keep whole" (default: the brain has no corpus-callosum range, ccs = cce = 0)."""
    cfg = cfg or M.config()
    scale = cfg["regions"]["mask_scale"]
    if whole_from_table is None:
        whole_from_table = not (ccs or cce)
    pieces = M.piece_suffixes(cfg)
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for p in sorted(glob(os.path.join(str(matched_dir), "*_0000.png"))):
        name = os.path.basename(p)[:-9]
        hbase = name
        for suf in pieces:
            if hbase.endswith(suf):
                hbase = hbase[:-len(suf)]
        num = M.section_number(hbase, cfg)
        photo = np.array(Image.open(p))
        mask = load_mask(os.path.join(str(seg_dir), name + ".png"), scale)

        def save(arr, tag):
            if not (arr > 0).any():
                return 0
            Image.fromarray(arr).save(os.path.join(out_dir, f"{hbase}_{tag}.png"))
            return 1

        is_piece_photo = any(s in name for s in pieces)
        whole_row = whole_from_table and not is_piece_photo
        if (ccs <= num <= cce and not is_piece_photo) or whole_row:
            n += save(photo, cfg["pieces"]["whole"]["tag"])
        elif not is_piece_photo:                                     # whole photo -> all three pieces
            for lab, tag in pieces.values():
                n += save(np.where(mask[..., None] == lab, photo, 0), tag)
        else:                                                        # piece photo -> that piece only
            for suf, (lab, tag) in pieces.items():
                if suf in name:
                    n += save(np.where(mask[..., None] == lab, photo, 0), tag)
    log(f"wrote {n} blockface pieces -> {out_dir}")
    return n
