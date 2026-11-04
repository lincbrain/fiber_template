"""A prepared brain folder (layout: linc/completion/__init__.py) from one brain's own products, the input of cli/linc-complete
(paper: Methods, "Registration to the brain's own MRI and caudal completion"). One process, nothing written but the folder (and, when
asked, the intermediate grids). CPU, numpy / scipy / nibabel; the OME-Zarr store is read for its JSON metadata only.

Inputs (path templates, {ID}, {mri} = $LINC_DATA/sub-<ID>/mri, {bf} = $LINC_DATA/sub-<ID>/blockface, {nonlin} = the products folder):
  nonlin        the registration's products on its 0.25 mm grid: {ID}_mri (the grid), {ID}_bf_nonlin (grey), {ID}_bf_nonlin_rgb (color),
                {ID}_warp (the deformation; brains with MRI), {ID}_mri_full (the b0 on its frame, when present)
  matrix        the registration's linear start, a FLIRT matrix blockface iso grey -> frame
  frame         {mri}/{ID}_mri, the b0 on its 0.5 mm frame (the grid of the brain mask and the register)
  brain mask    the completion brain on the frame (the mask of record minus the register's label 1)
  register      optional: the MRI-side register on the frame; label 1 (tissue the MRI shows and the photographs lack) is trimmed
                from the brain before the cut face is found
  stand-in b0   a brain without MRI: the MRI template's b0 carried into the brain (its frame and its b0; no warp)
  blockface     {bf}/{ID}_blockface.ome.zarr (level 1: shape and affine), {ID}.info (interpolated sections), {ID}_holes (filled marker
                tubes), {ID}_blockface_iso_grey, {ID}_mask_iso

Steps (each the step of record it ports; the products of record were made by exactly this arithmetic):
  unrotated     the deploy grid D: the frame cropped to the completion brain with a 3 mm margin, at 0.25 mm (nibabel resample_to_output's
                grid: the frame's center-aligned 2x lattice), zero-padded by the same whole voxels on both sides so the brain keeps
                >= 4 mm. Brain = the completion mask mapped nearest. The photographed tissue `valid` on the products grid: bf > 0, holes
                filled, largest component, inside the brain; grey normalized by its 1st / 99th percentiles; pulled onto D support-
                normalized: pull(I * valid) / pull(valid) where pull(valid) >= 0.5, else 0; real_mask = that support inside the brain.
                b0 normalization of record: the 0.5 / 99.5 percentiles over the completion brain (synthesis/preprocess.py `process`,
                SYN_BRAIN_MASK, SYN_PAD_MM = 4, masknorm on)
  photo_fraction  1 on photographed tissue and on the surface shell rostral of each column's seam, 0 caudal of it, minus the partial-
                volume fractions of interpolated sections and filled tubes: indicators on the blockface's level-1 grid, pulled (linear)
                onto the iso grey grid, placed by the linear start, pulled onto the products grid, warped, pulled onto D, clipped
                (synthesis/code/provenance.py `photo_fraction`, `fractions_to_deploy`; isbi2027/code/cache_bf2mri.py: the matrix of the
                products' registration, and for a brain without MRI the stand-in frame and no warp)
  color         the products' RGB / 255, ONE scale for the three channels (the 99th percentile of the brightest channel over valid,
                no floor), pulled onto D support-normalized, inside real_mask (synthesis/e1a/build_rgb.py --rule max)
  flat          the section normal n (inv(T)^T times the blockface section axis, pointing rostrally), the rotation R carrying n onto the
                grid's anterior-posterior axis (Rodrigues), the pads that keep the rotated brain >= 4 mm (+1 voxel; sizes a multiple of
                16), and the brain (nearest) and the photographed support (cubic, >= 0.5, inside the brain) rotated about the padded
                grid's center (synthesis/code/flatten_seam.py `flat`, SYN_FLAT_PAD_MM = 4, SYN_FLAT_MULT = 16, masknorm on). The rotation
                of the seam record is rebuilt from n rounded to 4 decimals (flatten_seam's seam.json -> isbi2027/code/reblend_prep.py
                rotations.json): it maps the register onto the flat grid and defines y_dep; the arrays were rotated with the unrounded R
  trim          the register's label 1 on the flat grid, nearest through world coordinates (isbi2027/code/mri_trim.py, register mode)
  clean face    on the flat planes (axis 1, caudal low): brain = flat brain minus the trim; a plane is complete when the photographs
                cover >= tau of the brain eroded by erode_mm; z_c the most caudal plane from which every plane over window_mm is
                complete; z_f = z_c + margin_mm; d* the planes from z_f whose photographed volume reaches the brain volume caudal of z_f
                (isbi2027/code/clean_face.py; configs/completion.yaml clean_face)
  folder        the input box C: the frame voxels that hold D (0.5 mm), its 2x refinement F (fine voxel 2k on frame voxel k);
                coarse/in = clip((b0 - lo) / (hi - lo), 0, 1) inside the completion brain, (lo, hi) = the b0 percentiles, or lo = 0
                for the stand-in network's deployed input (lo0); every fine array copied from D by an integer index shift; valid =
                real_mask minus each anterior-posterior column's caudal-most 0.25 mm; y_dep(k) = the flat plane coordinate of fine voxel
                k, affine in k (isbi2027/code/canon_cache.py `build`, the frozen isbi2027/code/frozen/trim_valid.py; canon_relo.py --lo 0)

Grids that the chain of record wrote to a NIfTI file and read back carry that file's float32 affine here too (`_stored`), and the
arrays keep the dtypes of the record's files, so every step sees the numbers its step of record saw.

    python -m linc.completion.prepare --subject MF278 --out <dir>          (= cli/linc-complete prepare ...; --help for the inputs)
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import nibabel as nib
import nibabel.processing as nip
import numpy as np
from scipy import ndimage as ndi

from . import CFG

HELP = """Build a brain's prepared folder, the input of `linc-complete`.

You need
  the environment env/deep.yml, on the CPU (about a minute and 2.3 GB per brain). The brain's products, at the paths configs/completion.yaml
  lists under prepare.inputs. The paper's 21 brains each have their row. Another brain needs no paths: it uses what `linc-register mri` wrote when that
  exists, else what `linc-register direct-placement` wrote in mri/stand_in/ (a brain without MRI):
    from the registration to the brain's own b0: the b0 on its frame, the registered photographs (grey and color), the deformation, and the
    linear start; a brain without MRI: the stand-in b0, the MRI template's b0 transformed into the brain, in place of its own and no deformation
    the brain mask on the frame. Tissue the MRI shows and the photographs lack, such as a longer brainstem: a label volume (label 1) for the paper's
    brains, which holds a detector's result and hand trims; a new brain with MRI gets the detector's alone, written beside its folder as <ID>_mri_only_tissue
    from the reconstruction: $LINC_DATA/sub-<ID>/blockface/<ID>_blockface.ome.zarr, <ID>.info, <ID>_holes, <ID>_blockface_iso_grey, <ID>_mask_iso

What it does
  1. crops the b0 frame to the brain and refines it to 0.25 mm, the network's output grid
  2. transforms the photographs, their color and each voxel's photographed fraction onto that grid
  3. rotates the grid so that the sections lie flat, because the junction plane is one plane of a section
  4. removes the tissue only the MRI shows, and finds the junction plane: the caudal end of the sections the photographs fully cover
     (configs/completion.yaml clean_face holds its parameters)
  5. normalizes the b0 to [0, 1] inside the brain and writes the network's input

You get
  <out>/<ID>/seam.json (the junction plane and the normalization), coarse/{in, brain}.nii.gz at 0.5 mm, and fine/{bf, bf_rgb, real_mask,
  photo_fraction, brain, valid}.nii.gz at 0.25 mm. An existing folder is never written over.
"""
from ..registration.fsl import fsl_to_world

# --- parameters of record ---
ISO = CFG["output_mm"]                        # 0.25 mm: the deploy grid and the output grid F
FRAME_MM = CFG["input_mm"]                    # 0.5 mm: the b0 frame and the input box C
B0_PCT = tuple(CFG["input_percentiles"])      # (0.5, 99.5): the b0 over the completion brain (preprocess.py robust_lohi)
STAND_IN_LO = CFG["stand_in_lo"]              # 0.0: the stand-in network's input lower limit (canon_relo.py --lo 0)
P = CFG["prepare"]
BF_PCT = tuple(P["bf_percentiles"])           # preprocess.py: the photographs' grey over the photographed tissue
RGB_PCT = P["rgb_percentile"]                 # build_rgb.py --rule max: one scale, the 99th percentile of the brightest channel
CROP_MM = P["crop_mm"]                        # preprocess.py crop_bbox: margin around the completion brain, round(3 / 0.5) frame voxels
PAD_MM = P["pad_mm"]                          # reblend_1c.sh (go): PAD_MM = 4 -> preprocess SYN_PAD_MM
FLAT_PAD_MM, FLAT_MULT = P["flat_pad_mm"], P["flat_multiple"]   # reblend_1c.sh (go): flatten_seam SYN_FLAT_PAD_MM = 4, SYN_FLAT_MULT = 16
SUPPORT_THR = P["support_threshold"]          # code/masknorm.py THR (synthesis/v2/v2b_env.sh: LT_MASKNORM = 1)
BF_LEVEL = P["bf_level"]                      # provenance.py bf_space_labels: store level 2 - RECON_LEVEL (code/config.py, 1)
INTERP, INPAINT = 2, 3                        # provenance.py tiers: interpolated section, filled marker tube
LPS_SIGN = np.array([-1.0, -1.0, 1.0], np.float32)   # provenance.py apply_warp: the saved field's x/y are LPS-signed (a pull, mm)
NORMAL_DECIMALS = P["normal_decimals"]        # flatten_seam.py: the section normal as its seam.json records it
VALID_TRIM_MM = CFG["training"]["valid_trim_mm"]     # 0.25: canon_cache.py K_MM (the frozen trim_valid, as every batch)
CLEAN = CFG["clean_face"]                     # tau 0.95, erode 1 mm, window 40 mm, margin 1.5 mm (clean_face.py)
FRAME_NAME = "run113"                         # the name the folders of record give the b0 frame (every {ID}_mri lives on it)

FRAME = "{mri}/{ID}_mri.nii.gz"
B0 = "{nonlin}/{ID}_mri_full.nii.gz"         # the b0 when present, else the frame (preprocess.py, canon_cache.py)


def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _data(p):
    return np.asanyarray(nib.load(p).dataobj)


def _stored(A):
    """the affine that a NIfTI file written with A reads back (sform rows in float32)"""
    h = nib.Nifti1Header(); h.set_sform(np.asarray(A, float), code="aligned")
    return h.get_sform()


def _pull(arr, A, shape, affine, order):
    """nibabel resample_from_to of the array arr (affine A) onto (shape, affine): the resample of every step of record"""
    return np.asanyarray(nip.resample_from_to(nib.Nifti1Image(arr, A), (tuple(int(n) for n in shape), affine), order=order).dataobj)


def _divide(num, den, thr=SUPPORT_THR):
    """support normalization (code/masknorm.py divide): num / den where den >= thr, else 0; (float32, support)"""
    sup = den >= thr
    d = np.where(sup, den, 1.0).astype(np.float32)
    out = np.where(sup[..., None], num / d[..., None], 0.0) if num.ndim == den.ndim + 1 else np.where(sup, num / d, 0.0)
    return out.astype(np.float32), sup


def ap_axis(A):
    """the array axis most aligned with world A-P and whether anterior is at its high index (common_ms.ap_axis)"""
    ax = int(np.argmax(np.abs(A[:3, :3][1])))
    return ax, bool(A[1, ax] > 0)


# flat-grid geometry (isbi2027/code/common.py): a flat grid is the grid rotated about its own voxel center, x_flat = c + R (x - c)
def _center(A, shape):
    return A[:3, :3] @ (0.5 * (np.array(shape[:3], float) - 1)) + A[:3, 3]


def _vox_world(idx, A):
    return idx @ A[:3, :3].T + A[:3, 3]


def _world_vox(X, A):
    Ai = np.linalg.inv(A)
    return X @ Ai[:3, :3].T + Ai[:3, 3]


def _to_unrot(X, A, shape, R):
    c = _center(A, shape)
    return c + (X - c) @ R


def _from_unrot(X, A, shape, R):
    c = _center(A, shape)
    return c + (X - c) @ R.T


def rotation_to(n, target):
    """the smallest rotation carrying unit vector n onto unit vector target (Rodrigues; flatten_seam.py)"""
    n = n / np.linalg.norm(n); target = target / np.linalg.norm(target)
    v = np.cross(n, target); c = float(np.dot(n, target))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + c))


def apply_R(vol, R, affine, order):
    """vol rotated by the world rotation R about its grid's center, on the same grid (flatten_seam.py apply_R)"""
    A = affine[:3, :3]
    Mvox = np.linalg.inv(A) @ R.T @ A
    c = 0.5 * (np.array(vol.shape[:3]) - 1)
    return ndi.affine_transform(vol.astype(np.float32), Mvox, offset=c - Mvox @ c, order=order, mode="constant", cval=0.0,
                                prefilter=order > 1)


def box_copy(src, lo, n, fill=0):
    """src[lo : lo + n] per axis (any lo, n), `fill` outside src (canon_cache.py)"""
    out = np.full(tuple(n) + src.shape[3:], fill, dtype=src.dtype)
    s_lo = np.maximum(lo, 0); s_hi = np.minimum(lo + n, src.shape[:3])
    if np.any(s_hi <= s_lo):
        return out
    d_lo = s_lo - lo
    out[d_lo[0]:d_lo[0] + s_hi[0] - s_lo[0], d_lo[1]:d_lo[1] + s_hi[1] - s_lo[1], d_lo[2]:d_lo[2] + s_hi[2] - s_lo[2]] = \
        src[s_lo[0]:s_hi[0], s_lo[1]:s_hi[1], s_lo[2]:s_hi[2]]
    return out


def colwise_depth_mm(mask, affine):
    """mm from each anterior-posterior column's own caudal end of `mask` (nan in empty columns; the frozen trim_valid.py)"""
    ax, _ = ap_axis(affine)
    iso = float(np.abs(affine[:3, ax]).max())
    ant_low = affine[1, ax] > 0
    rm = np.moveaxis(mask, ax, 0); shp = rm.shape; n = shp[0]
    flat = rm.reshape(n, -1).astype(bool); ap = np.arange(n, dtype=np.float32)[:, None]; BIG = n + 10.0
    has = flat.any(0)
    if ant_low:
        seam = np.where(flat, ap, BIG).min(0); s = ap - seam[None, :]
    else:
        seam = np.where(flat, ap, -BIG).max(0); s = seam[None, :] - ap
    s = np.where(has[None, :], s * iso, np.nan)
    return np.moveaxis(s.reshape(shp), 0, ax)


def zarr_level_geometry(store, level=BF_LEVEL):
    """(shape, affine) of level `level` of an OME-Zarr blockface store from its JSON metadata alone (Zarr v3 zarr.json or v2
    .zattrs / .zarray), as code/zarrio.py read_level gives them: the (c, rows, slices, cols) array as (cols, slices, rows), world =
    scale * (voxel + translation), micrometers -> mm"""
    store = Path(store)
    if (store / "zarr.json").exists():
        attrs = json.loads((store / "zarr.json").read_text()).get("attributes", {})
        shape = json.loads((store / str(level) / "zarr.json").read_text())["shape"]
        meta = [store / "zarr.json", store / str(level) / "zarr.json"]
    else:
        attrs = json.loads((store / ".zattrs").read_text())
        shape = json.loads((store / str(level) / ".zarray").read_text())["shape"]
        meta = [store / ".zattrs", store / str(level) / ".zarray"]
    ms = attrs["multiscales"] if "multiscales" in attrs else attrs["ome"]["multiscales"]
    ds = ms[0]["datasets"][level]
    assert str(ds["path"]) == str(level), f"{store}: dataset {level} is at path {ds['path']}"
    ct = {t["type"]: t for t in ds["coordinateTransformations"]}
    tr, sc = ct["translation"]["translation"], ct["scale"]["scale"]           # [c, z = rows, y = slices, x = cols]
    A = np.eye(4)
    A[0, 0], A[1, 1], A[2, 2] = sc[3] / 1000.0, sc[2] / 1000.0, sc[1] / 1000.0
    A[0, 3], A[1, 3], A[2, 3] = sc[3] * tr[3] / 1000.0, sc[2] * tr[2] / 1000.0, sc[1] * tr[1] / 1000.0
    return (int(shape[3]), int(shape[2]), int(shape[1])), A, meta


# --------------------------------------------------------------------------- the steps

def unrotated(fine_ref, bf, b0, brain_mask, pad_mm=PAD_MM):
    """the photographs on the deploy grid D and the normalization (preprocess.py `process`); arrays on D padded"""
    fine = nib.load(fine_ref); fbi = nib.load(bf)
    assert np.allclose(fine.affine, fbi.affine) and fine.shape == fbi.shape, f"{bf}: not on the grid of {fine_ref}"
    full = nib.load(b0); mimg = nib.load(brain_mask)
    assert mimg.shape == full.shape and np.allclose(mimg.affine, full.affine, atol=1e-4), f"{brain_mask} is not on the frame of {b0}"
    assert np.allclose(np.sqrt((full.affine[:3, :3] ** 2).sum(0)), FRAME_MM), f"{b0}: not a {FRAME_MM} mm frame"
    fb = np.asanyarray(fbi.dataobj).astype(np.float32)
    Fm = np.asanyarray(full.dataobj).astype(np.float32)
    brain_full = np.asanyarray(mimg.dataobj) > 0
    bimg = brain_full.astype(np.float32)
    brain_fine = _pull(bimg, full.affine, fine.shape[:3], fine.affine, 0) > 0.5
    mlo, mhi = (float(np.percentile(Fm[brain_full], q)) for q in B0_PCT)
    valid = fb > 0                                                   # the photographed tissue: filled, largest component, in the brain
    if valid.any():
        valid = ndi.binary_fill_holes(valid)
        lab, n = ndi.label(valid)
        if n > 1:
            valid = lab == (1 + int(np.argmax(ndi.sum(np.ones_like(lab), lab, index=range(1, n + 1)))))
    valid = valid & brain_fine
    blo, bhi = (float(np.percentile(fb[valid], q)) for q in BF_PCT)
    # the deploy grid: the frame cropped to the brain (3 mm), at 0.25 mm on nibabel's output grid (resample_to_output)
    idx = np.argwhere(brain_full); m = int(round(CROP_MM / FRAME_MM))
    lo = np.maximum(0, idx.min(0) - m); hi = np.minimum(np.array(full.shape[:3]) - 1, idx.max(0) + m)
    A0 = full.affine.copy(); A0[:3, 3] += A0[:3, :3] @ lo.astype(float)
    dshape, daff = nip.vox2out_vox((tuple(int(b - a + 1) for a, b in zip(lo, hi)), A0), np.asarray([ISO] * 3))
    dshape = tuple(int(n) for n in dshape[:3])
    dep_brain = _pull(bimg, full.affine, dshape, daff, 0) > 0.5
    bfn = np.clip((fb - blo) / max(1e-6, bhi - blo), 0.0, 1.0) * valid
    num = _pull(bfn.astype(np.float32), fine.affine, dshape, daff, 1).astype(np.float32)
    den = _pull(valid.astype(np.float32), fine.affine, dshape, daff, 1).astype(np.float32)
    real_bf, sup = _divide(num, den)
    real_mask = sup & dep_brain
    real_bf = real_bf * real_mask
    # zero pad: the same whole voxels on both sides of each axis, so the brain keeps >= pad_mm
    pv = np.zeros(3, int)
    if pad_mm > 0:
        ib = np.argwhere(dep_brain)
        lo_m, hi_m = ib.min(0), np.array(dshape) - 1 - ib.max(0)
        pv = np.maximum(0, int(np.ceil(pad_mm / ISO - 1e-9)) - np.minimum(lo_m, hi_m)).astype(int)
    paff = daff.copy(); paff[:3, 3] -= daff[:3, :3] @ pv.astype(float)
    P = lambda a: np.pad(a, [(int(q), int(q)) for q in pv])  # noqa: E731
    return dict(fine_shape=tuple(fine.shape[:3]), fine_affine=_stored(fine.affine), valid=valid,
                mlo=mlo, mhi=mhi, blo=blo, bhi=bhi, core_shape=dshape, pad=pv.tolist(),
                shape=tuple(int(n + 2 * q) for n, q in zip(dshape, pv)), affine=_stored(paff),
                brain=P(dep_brain), real_bf=P(real_bf).astype(np.float32), real_mask=P(real_mask), synth=P(dep_brain & ~real_mask))


def apply_warp(vol, warp, ref_affine, order=1):
    """vol pulled through a saved displacement field (mm, x/y LPS-signed; provenance.py apply_warp)"""
    W = np.squeeze(np.asanyarray(nib.load(warp).dataobj)).astype(np.float32)
    vox = np.abs(np.diag(ref_affine)[:3]).astype(np.float32)        # float32 coordinates, as numpy 1 computed them (numpy 2: also)
    g = np.indices(vol.shape[:3]).astype(np.float32)
    crd = [g[i] + LPS_SIGN[i] * W[..., i] / vox[i] for i in range(3)]
    return ndi.map_coordinates(vol.astype(np.float32), crd, order=order, mode="constant", cval=0)


def interpolated_sections(info):
    """the native section indices the reconstruction interpolated (<ID>.info interp_native)"""
    kv = dict(line.strip().split("=", 1) for line in open(info) if "=" in line)
    return [int(z) for z in kv.get("interp_native", "").split(",") if z.strip()]


def photo_fraction(U, *, store, info, holes, iso_grey, frame, matrix, warp):
    """each deploy voxel's photographed share (provenance.py photo_fraction / fractions_to_deploy); float32 on D"""
    A, shape = U["affine"], U["shape"]
    brain, real, synth = U["brain"], U["real_mask"], U["synth"]
    ax, ant_hi = ap_axis(A)                                  # the per-column seam (cond.py seamdist_channel; only its sign is used)
    obs = np.moveaxis(real, ax, 0); n = obs.shape[0]; has = obs.any(axis=0)
    first = (np.argmax(obs, axis=0) if ant_hi else (n - 1 - np.argmax(obs[::-1], axis=0))).astype(np.float32)
    first[~has] = 0.0
    d = (np.arange(n, dtype=np.float32)[:, None, None] - first[None]) * (1.0 if ant_hi else -1.0)
    d[:, ~has] = -1.0
    sd = np.moveaxis(d, 0, ax); del d, obs
    w = np.zeros(shape, np.float32)
    w[brain & real] = 1.0
    w[brain & synth & (sd >= 0)] = 1.0                       # the surface shell rostral of the seam counts as photographed
    # the interpolated-section and filled-tube fractions, from the blockface's level-1 grid through the registration
    lshape, Al, _ = zarr_level_geometry(store, BF_LEVEL)
    lab = np.zeros(lshape, np.uint8)
    for z in interpolated_sections(info):                   # slice axis 1
        if 0 <= z < lshape[1]:
            lab[:, z, :] = INTERP
    if holes is not None:
        h = nib.load(holes)
        lab[_pull((np.asanyarray(h.dataobj) > 0).astype(np.uint8), h.affine, lshape, Al, 0) > 0] = INPAINT
    iso = nib.load(iso_grey); fine_shape, fine_affine = U["fine_shape"], U["fine_affine"]
    T = fsl_to_world(np.loadtxt(matrix), iso, nib.load(frame))
    for tier in (INTERP, INPAINT):
        ind = (lab == tier).astype(np.float32)
        if ind.max() == 0:
            continue
        f_iso = _pull(ind, Al, iso.shape[:3], iso.affine, 1).astype(np.float32); del ind
        f_fine = _pull(f_iso, T @ iso.affine, fine_shape, fine_affine, 1); del f_iso
        f_fine = apply_warp(f_fine, warp, fine_affine, 1) if warp is not None else f_fine.astype(np.float32)
        w = w - np.clip(_pull(f_fine.astype(np.float32), fine_affine, shape, A, 1).astype(np.float32), 0, 1)
    del lab
    w = np.clip(w, 0.0, 1.0)
    w[~brain] = 0.0
    w[brain & synth & (sd < 0)] = 0.0                        # caudal of the seam: not photographed
    return w.astype(np.float32)


def color(U, rgb):
    """the photographs' RGB on D, one scale for the three channels (build_rgb.py --rule max, place_rgb); float32 (..., 3)"""
    c = np.asanyarray(nib.load(rgb).dataobj).astype(np.float32) / 255.0
    valid, Af = U["valid"], U["fine_affine"]
    assert c.shape[:3] == valid.shape, f"{rgb}: not on the products grid"
    hi = float(np.percentile(c.max(-1)[valid & (c.sum(-1) > 0)], RGB_PCT))
    c = np.clip(c / hi, 0, 1) * valid[..., None]
    dep = np.stack([_pull(c[..., k].astype(np.float32), Af, U["shape"], U["affine"], 1).astype(np.float32) for k in range(3)], -1)
    dep, _ = _divide(dep, _pull(valid.astype(np.float32), Af, U["shape"], U["affine"], 1).astype(np.float32))
    return np.clip(dep * U["real_mask"][..., None], 0, 1)


def section_normal(iso_grey, mask_iso, frame, matrix):
    """the photographed sections' normal in frame world, pointing rostrally (flatten_seam.py seam_normal_world)"""
    bf_grey, bfm, mri = nib.load(iso_grey), nib.load(mask_iso), nib.load(frame)
    T = fsl_to_world(np.loadtxt(matrix), bf_grey, mri)
    A_bf = bfm.affine
    n = np.linalg.inv(T[:3, :3]).T @ A_bf[:3, 1]
    n /= np.linalg.norm(n)
    M = np.asanyarray(bfm.dataobj) > 0
    ys = np.where(M.any(axis=(0, 2)))[0]
    ic, kc = M.shape[0] / 2.0, M.shape[2] / 2.0
    p = (T @ (A_bf @ np.array([ic, int(ys.max()), kc, 1.0])))[:3]
    pp = (T @ (A_bf @ np.array([ic, int(ys.min()), kc, 1.0])))[:3]
    return -n if np.dot(n, pp - p) > 0 else n


def flat(U, n, pad_mm=FLAT_PAD_MM, mult=FLAT_MULT):
    """the section-aligned (flat) deploy grid: R, the seam record's R, the pads, and the brain and photographed mask rotated
    (flatten_seam.py run('flat'), flat_pad, write_padded, flatten_dir_norm; reblend_prep.py rotations)"""
    aff = U["fine_affine"]
    ax, _ = ap_axis(aff)

    def to_axis(nn):
        tgt = aff[:3, ax] / np.linalg.norm(aff[:3, ax])
        tgt = -tgt if np.dot(nn, tgt) < 0 else tgt
        return rotation_to(nn, tgt), float(np.degrees(np.arccos(np.clip(abs(np.dot(nn, tgt)), 0, 1))))
    R, ang = to_axis(n)
    n4 = np.array([round(float(x), NORMAL_DECIMALS) for x in n], float); n4 /= np.linalg.norm(n4)
    R_rec, ang_rec = to_axis(n4)
    assert abs(ang_rec - round(ang, 3)) < 0.02, (ang, ang_rec)
    # pads (flat_pad): the brain rotated about the deploy grid's center keeps >= pad_mm (+1 voxel), sizes a multiple of `mult`
    A = U["affine"][:3, :3]; sh = np.array(U["shape"]); c = 0.5 * (sh - 1)
    Minv = np.linalg.inv(A) @ R @ A
    out = (np.argwhere(U["brain"]).astype(np.float64) - c) @ Minv.T + c
    lo = np.floor(out.min(0)).astype(int); hi = (sh - 1 - np.ceil(out.max(0))).astype(int)
    need = np.ceil(pad_mm / np.sqrt((A ** 2).sum(0)) - 1e-9).astype(int) + 1
    q = np.maximum(0, need - np.minimum(lo, hi)); tot = sh + 2 * q
    extra = (np.ceil(tot / mult) * mult).astype(int) - tot if mult > 1 else np.zeros(3, int)
    q_lo, q_hi = q + extra // 2, q + extra - extra // 2
    Afl = U["affine"].copy(); Afl[:3, 3] -= U["affine"][:3, :3] @ np.asarray(q_lo, float); Afl = _stored(Afl)
    pw = [(int(a), int(b)) for a, b in zip(q_lo, q_hi)]
    brain = apply_R(np.pad(U["brain"].astype(np.uint8), pw).astype(np.float32), R, Afl, 0) > 0.5
    sup = apply_R(np.pad(U["real_mask"], pw).astype(np.float32), R, Afl, 3)
    real = (sup >= SUPPORT_THR) & brain
    return dict(R=R, angle_deg=ang, R_rec=R_rec, angle_rec=ang_rec, normal=n, normal_rec=n4, affine=Afl,
                shape=tuple(int(s) for s in brain.shape), pad_lo=q_lo.tolist(), pad_hi=q_hi.tolist(), brain=brain, real_mask=real)


def register_trim(Fl, register):
    """the register's label 1 on the flat grid, nearest through world coordinates (mri_trim.py, register mode)"""
    b, A = Fl["brain"], Fl["affine"]
    if register is None:
        return np.zeros_like(b)
    rimg = nib.load(register); lab1 = np.asanyarray(rimg.dataobj) == 1
    gi = np.indices(b.shape).reshape(3, -1).T
    v = np.rint(_world_vox(_to_unrot(_vox_world(gi.astype(np.float64), A), A, b.shape, Fl["R_rec"]), rimg.affine)).astype(int)
    del gi
    ok = np.all((v >= 0) & (v < np.array(rimg.shape[:3])), axis=1)
    tv = np.zeros(len(v), bool); tv[ok] = lab1[tuple(v[ok].T)]
    return tv.reshape(b.shape)


def detect_mri_only(Fl):
    """tissue the MRI shows and the photographs lack, on the flat grid (axis 1 caudal -> rostral; mri_trim.py with its height rule):
    `stump` = unphotographed brain voxels within the photographed extent, opened by a ball, the components at least min_ml large that
    begin within touch_mm of the photographed end and lie in the lower part of the brain, grown back by 2 voxels; `tail` = the unphotographed
    brain caudal of the stump, under its caudal-most foot_mm (dilated 1 mm in-plane). Photographed tissue is never removed."""
    T = P["trim"]
    A, b = Fl["affine"], Fl["brain"]; rm = Fl["real_mask"] & b
    ph = int(np.flatnonzero(rm.any(axis=(0, 2))).min())                   # the photographed (caudal) end plane
    Mx = b & ~rm; Mx[:, :ph, :] = False
    r = int(round(T["open_mm"] / ISO)); g = np.arange(-r, r + 1); X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    lab, n = ndi.label(ndi.binary_opening(Mx, structure=(X ** 2 + Y ** 2 + Z ** 2) <= r * r))
    zax = int(np.argmax(np.abs(A[2, :3]))); bz = np.flatnonzero(b.any(axis=tuple(x for x in range(3) if x != zax)))
    zlo, zhi = (bz.min(), bz.max()) if A[2, zax] > 0 else (bz.max(), bz.min())
    keep = np.zeros_like(Mx)
    for k, sz in enumerate(ndi.sum(np.ones_like(lab), lab, range(1, n + 1)) if n else [], 1):
        if sz * ISO ** 3 / 1000 < T["min_ml"]:
            continue
        c = lab == k
        if np.flatnonzero(c.any(axis=(0, 2))).min() <= ph + int(round(T["touch_mm"] / ISO)) \
                and (ndi.center_of_mass(c)[zax] - zlo) / (zhi - zlo) <= T["max_height"]:
            keep |= c
    stump = ndi.binary_dilation(keep, iterations=2) & Mx
    tail = np.zeros_like(b)
    if stump.any():
        y0 = int(np.flatnonzero(stump.any(axis=(0, 2))).min())
        foot = ndi.binary_dilation(stump[:, y0:y0 + int(round(T["foot_mm"] / ISO)), :].any(axis=1), iterations=int(round(1.0 / ISO)))
        tail[:, :y0, :] = (b & ~rm)[:, :y0, :] & foot[:, None, :]
    return stump | tail


def trim_on_frame(trim, Fl, mask_img):
    """the flat grid's trim as labels on the b0 frame, inside the brain mask (export_cutoff_L4.py): a frame voxel is 1 when at least 4 of its 8
    sub-points fall in the trim; detached fragments under frame_min_vox voxels are dropped"""
    A, sh, R = Fl["affine"], trim.shape, Fl["R_rec"]
    f = _world_vox(_to_unrot(_vox_world(np.argwhere(trim).astype(np.float64), A), A, sh, R), mask_img.affine)
    lo = np.maximum(np.floor(f.min(0)).astype(int) - 2, 0); hi = np.minimum(np.ceil(f.max(0)).astype(int) + 3, mask_img.shape[:3])
    box = np.indices(tuple(hi - lo)).reshape(3, -1).T + lo
    hit = np.zeros(len(box), np.float32)
    offs = [(i, j, k) for i in (-0.25, 0.25) for j in (-0.25, 0.25) for k in (-0.25, 0.25)]
    for o in offs:
        v = np.rint(_world_vox(_from_unrot(_vox_world(box + np.array(o), mask_img.affine), A, sh, R), A)).astype(int)
        ok = np.all((v >= 0) & (v < np.array(sh)), axis=1)
        t = np.zeros(len(v), bool); t[ok] = trim[tuple(v[ok].T)]; hit += t
    frac = np.zeros(mask_img.shape[:3], np.float32); frac[tuple(box.T)] = hit / len(offs)
    lab = (frac >= 0.5) & (np.asanyarray(mask_img.dataobj) > 0)
    pl, pn = ndi.label(lab, structure=np.ones((3, 3, 3)))
    if pn > 1:
        for k, sz in enumerate(ndi.sum(lab, pl, range(1, pn + 1)).astype(int), 1):
            if sz < P["trim"]["frame_min_vox"]:
                lab[pl == k] = False
    return lab


def clean_face(Fl, trim, tau=CLEAN["tau"], erode_mm=CLEAN["erode_mm"], window_mm=CLEAN["window_mm"], margin_mm=CLEAN["margin_mm"]):
    """z_f and d* on the flat planes (clean_face.py)"""
    ax, ant_hi = ap_axis(Fl["affine"])
    assert ax == 1 and ant_hi, "the flat grid's axis 1 must run caudal -> rostral"
    b = Fl["brain"] & ~trim
    rm = Fl["real_mask"] & b
    be = ndi.binary_erosion(b, iterations=int(round(erode_mm / ISO)))
    ne = be.sum(axis=(0, 2)); nre = (rm & be).sum(axis=(0, 2))
    ok = np.where(ne > 0, nre / np.maximum(ne, 1), 0.0) >= tau
    yb = np.flatnonzero(b.any(axis=(0, 2))); yr = np.flatnonzero(rm.any(axis=(0, 2)))
    L = int(round(window_mm / ISO))
    zc = next((y for y in range(int(yr.min()), int(yr.max()) - L) if ok[y:y + L].all()), None)
    if zc is None:
        raise SystemExit(f"no plane from which the photographs are complete over {window_mm} mm (tau {tau})")
    zf = zc + int(round(margin_mm / ISO))
    miss = int(b[:, :zf, :].sum())
    n = int(np.searchsorted(np.cumsum(rm.sum(axis=(0, 2))[zf:]), miss)) + 1
    assert ok[zf:zf + n + int(CFG["recal_mm"] / ISO)].all(), "the slab d* and the 8 mm band rostral of it are not all complete"
    return dict(z_c=int(zc), z_f=int(zf), dstar_planes=n, dstar_mm=round(n * ISO, 2), photographed_end=int(yr.min()),
                brain_caudal_end=int(yb.min()), missing_pct=round(100 * miss / int(b.sum()), 2), trim_voxels=int(trim.sum()))


# --------------------------------------------------------------------------- the folder

def _placeholder(p, root):
    """a path under the data root as $LINC_DATA/... (linc.paths.rerooted resolves it), else the path itself"""
    p = Path(p).absolute()
    try:
        return "$LINC_DATA/" + str(p.relative_to(Path(root).absolute()))
    except ValueError:
        return str(p)


def inputs_of(subject):
    """the brain's input templates: its row of configs/completion.yaml prepare.inputs.of_record; a new brain's own registration
    (`linc-register mri`, new_brain) when it exists, else what `linc-register direct-placement` wrote (new_brain_stand_in)"""
    from .. import paths
    for row in P["inputs"]["of_record"]:
        if subject in row["brains"]:
            return {k: v for k, v in row.items() if k != "brains"}
    new, stand_in = P["inputs"]["new_brain"], P["inputs"]["new_brain_stand_in"]
    at = lambda t: Path(t.format(ID=subject, work=paths.work_root(), mri=paths.data_root() / f"sub-{subject}" / "mri"))  # noqa: E731
    if not (at(new["nonlin"]) / f"{subject}_bf_nonlin.nii.gz").exists() and (at(stand_in["nonlin"]) / f"{subject}_bf_nonlin.nii.gz").exists():
        return dict(stand_in)
    return dict(new)


def prepare(subject, out_dir, *, data_root=None, nonlin, matrix, brain_mask, register=None, frame=FRAME, detect_trim=False,
            b0=B0, stand_in_b0=None, lo0=False, pad_mm=PAD_MM, flat_pad_mm=FLAT_PAD_MM, flat_mult=FLAT_MULT, keep=None, log=print):
    """write the prepared folder <out_dir>/<subject>/ (seam.json, coarse/, fine/) from the brain's products; returns seam.json's record.
    Path arguments are templates ({ID}, {mri}, {bf}, {nonlin}; module docstring). register: None = no trim. stand_in_b0: a brain
    without MRI, the b0 and the frame are the stand-in, and the photographed fractions are placed by the linear start alone (no
    warp). lo0: the stand-in network's deployed input, lower limit STAND_IN_LO (configs/completion.yaml stand_in_lo). keep: a folder
    to also write the intermediate grids to (deploy/ on the deploy grid, flat/ on the flat grid)."""
    from .. import paths
    t0 = time.time()
    S = subject
    root = Path(data_root) if data_root else paths.data_root()
    if lo0 and stand_in_b0 is None:
        raise ValueError("lo0 is the stand-in network's input scaling: give stand_in_b0")
    dst = Path(out_dir) / S
    if dst.exists():
        raise FileExistsError(f"{dst} exists: a prepared folder is never written over")
    mri, bfd = root / f"sub-{S}" / "mri", root / f"sub-{S}" / "blockface"
    fmt = lambda t, **k: Path(str(t).format(ID=S, mri=mri, bf=bfd, work=paths.work_root(), **k))  # noqa: E731
    nl = fmt(nonlin)
    if stand_in_b0 is not None:
        b0_p = frame_p = fmt(stand_in_b0, nonlin=nl); warp = None
    else:
        frame_p = fmt(frame, nonlin=nl); b0_p = fmt(b0, nonlin=nl)
        b0_p = b0_p if b0_p.exists() else frame_p
        warp = nl / f"{S}_warp.nii.gz"
    ins = dict(fine_grid=nl / f"{S}_mri.nii.gz", bf=nl / f"{S}_bf_nonlin.nii.gz", bf_rgb=nl / f"{S}_bf_nonlin_rgb.nii.gz",
               warp=warp, matrix=fmt(matrix, nonlin=nl), frame=frame_p, b0=b0_p, brain_mask=fmt(brain_mask, nonlin=nl),
               register=None if register is None else fmt(register, nonlin=nl), blockface_store=bfd / f"{S}_blockface.ome.zarr",
               blockface_info=bfd / f"{S}.info", holes=bfd / f"{S}_holes.nii.gz", iso_grey=bfd / f"{S}_blockface_iso_grey.nii.gz",
               mask_iso=bfd / f"{S}_mask_iso.nii.gz")
    if not ins["holes"].exists():                                         # provenance.py: a brain without filled tubes
        ins["holes"] = None
    missing = [f"{k}: {p}" for k, p in ins.items() if p is not None and not p.exists()]
    if missing:
        raise FileNotFoundError("missing inputs:\n  " + "\n  ".join(missing))
    bi, fi = nib.load(b0_p), nib.load(frame_p)
    if bi.shape != fi.shape or not np.allclose(bi.affine, fi.affine):                # preprocess.py: <ID>_mri_full on the frame
        raise ValueError(f"{b0_p} is not on the frame {frame_p}")

    log(f"{S}: 1/6 the working grid"); U = unrotated(ins["fine_grid"], ins["bf"], b0_p, ins["brain_mask"], pad_mm)
    log(f"{S}: 2/6 section normal and flat grid")
    normal = section_normal(ins["iso_grey"], ins["mask_iso"], frame_p, ins["matrix"])
    Fl = flat(U, normal, flat_pad_mm, flat_mult)
    if detect_trim:                                                       # a new brain: as a register would give it, label 1 and the brain without it
        found = detect_mri_only(Fl)
        if found.any():
            mimg = nib.load(ins["brain_mask"]); lab = trim_on_frame(found, Fl, mimg)
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            hdr = mimg.header.copy(); hdr.set_data_dtype(np.uint8)
            ins["register"] = Path(out_dir) / f"{S}_mri_only_tissue.nii.gz"; ins["brain_mask"] = Path(out_dir) / f"{S}_brain_without_mri_only_tissue.nii.gz"
            nib.save(nib.Nifti1Image(lab.astype(np.uint8), mimg.affine, hdr), ins["register"])
            nib.save(nib.Nifti1Image(((np.asanyarray(mimg.dataobj) > 0) & ~lab).astype(np.uint8), mimg.affine, hdr), ins["brain_mask"])
            log(f"{S}: tissue only the MRI shows: {int(lab.sum())} voxels on the b0 frame -> {ins['register']}")
            U = unrotated(ins["fine_grid"], ins["bf"], b0_p, ins["brain_mask"], pad_mm); Fl = flat(U, normal, flat_pad_mm, flat_mult)
    log(f"{S}: 3/6 photographed fraction"); pf = photo_fraction(
        U, store=ins["blockface_store"], info=ins["blockface_info"], holes=ins["holes"], iso_grey=ins["iso_grey"], frame=frame_p,
        matrix=ins["matrix"], warp=warp)
    log(f"{S}: 4/6 color"); rgb = color(U, ins["bf_rgb"])
    log(f"{S}: 5/6 MRI-side trim and junction plane"); trim = register_trim(Fl, ins["register"]); cf = clean_face(Fl, trim)
    if keep:
        kd = Path(keep) / S
        for sub, arrs, A in (("deploy", dict(brain=U["brain"], real_bf=U["real_bf"], real_mask=U["real_mask"], synth=U["synth"],
                                                 photo_fraction=pf, real_bf_rgb=rgb), U["affine"]),
                             ("flat", dict(brain=Fl["brain"], real_mask=Fl["real_mask"], trim_mri=trim), Fl["affine"])):
            (kd / sub).mkdir(parents=True, exist_ok=True)
            for k, a in arrs.items():
                nib.save(nib.Nifti1Image(a.astype(np.uint8) if a.dtype == bool else a, A), str(kd / sub / f"{k}.nii.gz"))
        json.dump(dict(mlo=U["mlo"], mhi=U["mhi"], blo=U["blo"], bhi=U["bhi"], deploy_core_shape=list(U["core_shape"]),
                       deploy_pad=U["pad"], flat_pad=dict(lo=Fl["pad_lo"], hi=Fl["pad_hi"]), R=Fl["R"].tolist(),
                       R_seam_record=Fl["R_rec"].tolist(), normal=Fl["normal"].tolist(), angle_deg=Fl["angle_deg"], clean_face=cf),
                  open(kd / "steps.json", "w"), indent=1)

    log(f"{S}: 6/6 the folder")
    fr = nib.load(b0_p); Af = fr.affine
    bm = _data(ins["brain_mask"]) > 0
    Ad, shd = U["affine"], np.array(U["shape"])
    M = np.linalg.solve(Af, Ad)                                           # deploy index -> frame index
    assert np.allclose(M[:3, :3], (ISO / FRAME_MM) * np.eye(3), atol=1e-6), "the deploy grid is not the frame's axis-aligned 2x lattice"
    f0 = M[:3, 3]
    assert np.allclose(2 * f0, np.round(2 * f0), atol=1e-4), f"the deploy grid is off the frame's 2x lattice ({f0})"
    ic0 = np.ceil(f0 - 1e-6).astype(int); ic1 = np.floor(f0 + 0.5 * (shd - 1) + 1e-6).astype(int)
    Nc = ic1 - ic0 + 1; Nf = 2 * Nc
    jd0 = np.round(2 * (ic0 - f0)).astype(int)                            # deploy index of fine voxel 0
    Ac = Af.copy(); Ac[:3, 3] = Af[:3, :3] @ ic0 + Af[:3, 3]
    Afi = Ac.copy(); Afi[:3, :3] = Af[:3, :3] / 2.0
    assert np.allclose(Afi @ np.r_[0, 0, 0, 1.0], Ad @ np.r_[jd0, 1.0], atol=1e-5), "fine box misplaced"
    x = np.asanyarray(fr.dataobj).astype(np.float32)
    lo, hi = (STAND_IN_LO if lo0 else U["mlo"]), U["mhi"]
    cin = box_copy((np.clip((x - lo) / max(1e-6, hi - lo), 0, 1) * bm).astype(np.float32), ic0, Nc)
    relo = None
    if lo0:                                                               # canon_relo.py: the same input, re-scaled from lo = 0
        old = box_copy((np.clip((x - U["mlo"]) / max(1e-6, hi - U["mlo"]), 0, 1) * bm).astype(np.float32), ic0, Nc)
        m = (old > 0) & (old < 1) & (cin > 0) & (cin < 1)
        r = float(np.corrcoef(old[m], cin[m])[0, 1])
        assert r > 0.99999, f"the lo = {lo} input is not an affine re-scaling of the percentile one (r {r})"
        relo = dict(lo=lo, hi=hi, replaced_mlo=U["mlo"], why="lo fixed to match the stand-in network's training inputs (lo = 0)",
                    check_r_unclipped=r)
    cbr = box_copy(bm.astype(np.uint8), ic0, Nc)
    fine = dict(bf=box_copy(U["real_bf"], jd0, Nf), real_mask=box_copy(U["real_mask"].astype(np.uint8), jd0, Nf) > 0,
                photo_fraction=box_copy(pf, jd0, Nf), brain=box_copy(U["brain"].astype(np.uint8), jd0, Nf) > 0,
                bf_rgb=box_copy(rgb, jd0, Nf))
    rm = fine["real_mask"]
    dmm = colwise_depth_mm(rm, Afi)
    valid = rm & np.isfinite(dmm) & (dmm >= VALID_TRIM_MM)
    Afl, shfl, R = Fl["affine"], Fl["shape"], Fl["R_rec"]
    ax_fl, _ = ap_axis(Afl)

    def ydep(k):                                                          # the flat plane coordinate of fine voxel k
        return _world_vox(_from_unrot(_vox_world(np.atleast_2d(np.asarray(k, float)), Afi), Afl, shfl, R), Afl)[:, ax_fl]
    y0 = float(ydep([0, 0, 0])[0]); g = [float(ydep(e)[0] - y0) for e in np.eye(3)]
    kt = np.array([[17, 41, 29], [Nf[0] - 3, 5, Nf[2] - 7]], float)
    assert np.allclose(ydep(kt), y0 + kt @ np.array(g), atol=1e-6), "y_dep is not affine"

    rel = lambda p: _placeholder(p, root)  # noqa: E731
    inputs = {}
    for k, p in ins.items():
        if p is None:
            continue
        if k == "blockface_store":
            _, _, meta = zarr_level_geometry(p, BF_LEVEL)
            inputs[k] = dict(path=rel(p), read="the level-1 shape and the multiscales metadata only",
                             metadata_md5={str(q.relative_to(p)): _md5(q) for q in meta})
        else:
            inputs[k] = dict(path=rel(p), md5=_md5(p))
    rec = dict(subject=S, frame=FRAME_NAME, nonlin=nl.name, matrix=rel(ins["matrix"]),
               placement=dict(kind="registration to the brain's own b0" if stand_in_b0 is None else
                              "direct placement into the MRI template (stand-in b0, no warp)", product_dir=rel(nl), matrix=rel(ins["matrix"])),
               coarse_box=dict(lo=ic0.tolist(), shape=Nc.tolist()), fine_shape=Nf.tolist(), deploy_index_of_fine0=jd0.tolist(),
               mlo=lo, mhi=hi, blo=U["blo"], bhi=U["bhi"],
               seam=dict(z_f=cf["z_f"], dstar_planes=cf["dstar_planes"], dstar_mm=cf["dstar_mm"],
                         y_dep=dict(y0=y0, g=g, what="flat deploy A-P plane coordinate of fine voxel k = y0 + g . k"),
                         flat=dict(affine=Afl.tolist(), shape=list(map(int, shfl)), R=R.tolist(), ap_axis=int(ax_fl))),
               counts=dict(coarse_brain=int(cbr.sum()), fine_brain=int(fine["brain"].sum()), real_mask=int(rm.sum()),
                           valid=int(valid.sum()), valid_trim_removed=int(rm.sum() - valid.sum())),
               sources=dict(b0=rel(b0_p), b0_md5=inputs["b0"]["md5"], brain=rel(ins["brain_mask"]), brain_md5=inputs["brain_mask"]["md5"]),
               **({} if relo is None else dict(input_relo=relo)),
               provenance=dict(made_by="linc.completion.prepare", inputs=inputs, stand_in=stand_in_b0 is not None, lo0=bool(lo0),
                               trim=dict(register=None if ins["register"] is None else rel(ins["register"]), voxels=cf["trim_voxels"]),
                               clean_face=dict(z_c=cf["z_c"], photographed_end=cf["photographed_end"],
                                               brain_caudal_end=cf["brain_caudal_end"], missing_pct=cf["missing_pct"]),
                               parameters=dict(pad_mm=pad_mm, flat_pad_mm=flat_pad_mm, flat_mult=flat_mult, crop_mm=CROP_MM,
                                               b0_percentiles=list(B0_PCT), bf_percentiles=list(BF_PCT), rgb_percentile=RGB_PCT,
                                               support_threshold=SUPPORT_THR, valid_trim_mm=VALID_TRIM_MM, clean_face=dict(CLEAN)),
                               seam_angle_deg=Fl["angle_deg"], wall_s=None),
               created=dt.datetime.now().astimezone().isoformat(timespec="seconds"))
    tmp = Path(out_dir) / f".{S}.tmp{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "coarse").mkdir(parents=True); (tmp / "fine").mkdir()
    try:                                                                  # written in a temporary folder, then renamed
        sv = lambda a, A, p, t: nib.save(nib.Nifti1Image(a.astype(t), A), str(tmp / p))  # noqa: E731
        sv(cin, Ac, "coarse/in.nii.gz", np.float32); sv(cbr, Ac, "coarse/brain.nii.gz", np.uint8)
        sv(fine["bf"], Afi, "fine/bf.nii.gz", np.float32); sv(rm, Afi, "fine/real_mask.nii.gz", np.uint8)
        sv(fine["photo_fraction"], Afi, "fine/photo_fraction.nii.gz", np.float32)
        sv(fine["brain"], Afi, "fine/brain.nii.gz", np.uint8); sv(valid, Afi, "fine/valid.nii.gz", np.uint8)
        sv(fine["bf_rgb"], Afi, "fine/bf_rgb.nii.gz", np.float32)
        rec["provenance"]["wall_s"] = round(time.time() - t0, 1)
        json.dump(rec, open(tmp / "seam.json", "w"), indent=1)
        os.rename(tmp, dst)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    log(f"{S}: coarse {Nc.tolist()} fine {Nf.tolist()} z_f {cf['z_f']} d* {cf['dstar_planes']} valid {rec['counts']['valid']} "
        f"-> {dst} ({rec['provenance']['wall_s']} s)")
    return rec


def stand_in_training_folder(subject, out_dir, targets_from, stand_in_b0, data_root=None):
    """the stand-in network's training folder of a brain with MRI (build_canon_pseudo.py): its own prepared folder with only the network's
    input replaced by the stand-in b0, normalized by that image's own percentiles over the brain, so that training sees what a brain
    without MRI gets; coarse/brain and fine/ are links to the own folder (the targets do not change)"""
    from .. import paths
    S = subject
    src, dst = (Path(targets_from) / S).resolve(), Path(out_dir) / S
    if dst.exists():
        raise FileExistsError(f"{dst} exists: a prepared folder is never written over")
    root = Path(data_root) if data_root else paths.data_root()
    mri = root / f"sub-{S}" / "mri"
    rec = json.loads((src / "seam.json").read_text())
    p = Path(str(stand_in_b0).format(ID=S, mri=mri, bf=root / f"sub-{S}" / "blockface", work=paths.work_root()))
    if not p.exists():
        raise FileNotFoundError(f"missing stand-in b0: {p}")
    im = nib.load(p); own = nib.load(paths.rerooted(rec["sources"]["b0"]))
    if im.shape != own.shape or not np.allclose(im.affine, own.affine, atol=1e-5):
        raise ValueError(f"{p} is not on the brain's b0 frame")
    bm = _data(paths.rerooted(rec["sources"]["brain"])) > 0
    x = np.asanyarray(im.dataobj).astype(np.float32)
    lo, hi = (float(np.percentile(x[bm], q)) for q in B0_PCT)
    cin = (np.clip((x - lo) / max(1e-6, hi - lo), 0, 1) * bm).astype(np.float32)
    cin = box_copy(cin, np.array(rec["coarse_box"]["lo"]), np.array(rec["coarse_box"]["shape"]))
    (dst / "coarse").mkdir(parents=True)
    nib.save(nib.Nifti1Image(cin, nib.load(src / "coarse" / "in.nii.gz").affine), dst / "coarse" / "in.nii.gz")
    os.symlink(src / "coarse" / "brain.nii.gz", dst / "coarse" / "brain.nii.gz"); os.symlink(src / "fine", dst / "fine")
    rec["input"] = dict(kind="the stand-in b0", path=str(p), md5=_md5(p), norm=[lo, hi], replaces=str(src / "coarse" / "in.nii.gz"),
                        rule="own 0.5/99.5 percentiles over the brain; masked by it")
    (dst / "seam.json").write_text(json.dumps(rec, indent=1))
    print(f"{S}: stand-in input written (norm {lo:.4g}/{hi:.4g}) -> {dst}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="linc-complete prepare", formatter_class=argparse.RawDescriptionHelpFormatter, description=HELP)
    ap.add_argument("--subject", required=True, help="the Specimen ID (e.g. MF278)")
    ap.add_argument("--out", required=True, type=Path, help="the folder of prepared folders: writes <out>/<ID>/")
    ap.add_argument("--targets-from", type=Path, help="to train the stand-in network: a folder of own-b0 prepared folders, whose targets this "
                    "brain's folder links to; only the network's input changes, to --stand-in-b0")
    ap.add_argument("--stand-in-b0", help="with --targets-from: the path template of the brain's stand-in b0 (default: what `linc-register direct-placement` wrote, {mri}/stand_in/{ID}_mri_full.nii.gz)")
    # one path, overriding the brain's row of configs/completion.yaml (path templates as there)
    ap.add_argument("--nonlin"); ap.add_argument("--matrix"); ap.add_argument("--brain-mask"); ap.add_argument("--register", help="'none': no trim")
    ap.add_argument("--lo0", action="store_true", help="with --stand-in-b0: the deployed input, x = clip(b0 / p99.5, 0, 1); a brain's row implies it")
    a = ap.parse_args(argv)
    ins = inputs_of(a.subject)
    for k in ("nonlin", "matrix", "brain_mask", "register", "stand_in_b0"):
        if getattr(a, k):
            ins[k] = None if getattr(a, k) == "none" else getattr(a, k)
    if a.register:                                                       # a register given (or "none"): no detection
        ins.pop("detect_trim", None)
    try:
        if a.targets_from:
            stand_in_training_folder(a.subject, a.out, a.targets_from, a.stand_in_b0 or P["inputs"]["new_brain_stand_in"]["stand_in_b0"])
            return 0
        prepare(a.subject, a.out, lo0=a.lo0 or (bool(ins.get("stand_in_b0")) and not a.stand_in_b0), **ins)
    except (FileExistsError, FileNotFoundError, ValueError) as e:
        sys.exit(f"linc-complete prepare: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
