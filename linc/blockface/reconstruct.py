"""The reconstruction itself (Methods, "Blockface reconstruction", steps 3, 5 and 6, and the final volume, Fig. 3).

You need
  stage 3 (the jump-corrected _work stacks, nii/02_blockface_bg_L2: prepare.py) and the curated brain mask and marker tubes
  (curated/: a hand-edited <S>_mask_edited or <S>_holes_edited wins over the models' <S>_mask, <S>_holes); parameters:
  configs/blockface.yaml `reconstruction`.
What it does (L1 .. L4: the photographs reduced 2, 4, 8 and 16 times in-plane)
  First    the L4 grid: aligned to the curated mask by world coordinates and cropped tight around it; the L4 mask and tubes
           (label 2 marks interpolated sections).
  Then     the L1 pixels: the stack with its background rebuilt at L1 and matched to that grid; the missing sections
           interpolated again by multi-resolution demons; the brain masked; a per-section degree-2 illumination field divided
           out (MR263, as its record: then a z-referenced field, configs/subjects/MR263.yaml shading); the brightness steps between
           sessions removed (destep.py) and each section's color matched to one reference
           (a gain per channel); the marker tubes filled (their dark halo grown at L1, a biharmonic fill at L4, feathered in).
You get
  <S>_blockface.ome.zarr (L1 .. L4), <S>_mask and <S>_holes (L4 NIfTI), <S>.info, steps/<S>_03_color.ome.zarr (before the tubes
  are filled), <S>_brightness.png and <S>_anterior10.png.

Ported unchanged from the analysis project's code/recon.py (reconstruct and its helpers).
"""
import copy
import os

import cv2
import numpy as np
import zarr

from linc.blockface import destep, spacing, zarrio
from linc.blockface.prepare import _poly_terms, blockface_bg
from linc.blockface.subject import params

R = params()["reconstruction"]
ZREF = R["zref"]                    # the z-referenced illumination field (configs/blockface.yaml reconstruction.zref)


# =====================================================================================================
# Forward reconstruction - the FINAL processed blockface --------------------------------------------
# Off the hand-curated brain mask (+ holes), at L4 in-plane (/16) / NATIVE through-plane resolution. This
# is the canonical deliverable path and SUPERSEDES the old stage-3 tail (remove_background -> correct_shading
# -> harmonize_intensity): stages 1->register_jumps are shared, then this rebuilds the background-intact twin
# and runs, in the principled order: crop -> demons re-interpolation of missing slices -> mask -> shading ->
# de-step intensity -> per-channel white-balance -> 3D biharmonic punch-hole inpaint. All tuning lives in
# config (Z_TARGET_UM, HARMONIZE_MM, CROP_*, GAIN_CLIP, DEMONS, HOLE_INPAINT, W_LUM, ORIENT_CORR_MIN).
#
#   Inputs : jump-corrected _work/blockface_raw.ome.zarr (stage 3), nii/02_blockface_bg_L2.nii.gz (affine +
#            voxel-order reference), cfg.curated_mask, cfg.curated_holes (optional), xlsx missing/vx/vz.
#   Outputs: cfg.final('blockface'|'mask'|'holes'|'bg'), cfg.steps_dir/, cfg.info_file, <S>_brightness.png,
#            <S>_anterior10.png  - all in the subject's blockface modality folder.
# Everything is at L4 in-plane / native-z, so the curated mask & holes need no in-plane resample (no
# non-zero-mean trick). Usage: python recon.py reconstruct --subject <S> [--no-holes].
# =====================================================================================================
def _save_nii(path, vol, A):
    import nibabel as nib
    nib.save(nib.Nifti1Image(np.ascontiguousarray(vol), A), path)


# ---------- geometry helpers ----------
def _blockmean2(a, axes):
    """Block-mean by 2 along the given (in-plane) axes of a float array."""
    for ax in axes:
        n = a.shape[ax] - a.shape[ax] % 2
        a = np.take(a, np.arange(n), axis=ax)
        sh = list(a.shape); sh[ax:ax + 1] = [n // 2, 2]
        a = a.reshape(sh).mean(ax + 1)
    return a


def _pad_axis(a, b, n, m):
    """Grow [a,b) symmetrically to a length that is a multiple of m, clamped inside [0,n) by shifting."""
    L = b - a
    P = ((L + m - 1) // m) * m
    if P > n:
        P = (n // m) * m
    a2 = max(0, min(a - (P - L) // 2, n - P))
    return a2, a2 + P


def _orient_to_nii(bg_zarr_CHzW, cfg):
    """Map the rebuilt uncorrected bg zarr (C,H2,nz,W2) into the SAME voxel order as 02_blockface_bg_L2.nii.gz
    so it aligns in-plane with the curated mask (nz is the slice axis in both). The in-plane permute is fixed
    (2,1,0); the flip is resolved by max spatial correlation against the (framework-exported) bg L2 nii - a
    deterministic auto-alignment (same inputs -> same flip), guarded by ORIENT_CORR_MIN, not a random guess."""
    import nibabel as nib
    Nbg = np.asanyarray(nib.load(cfg.snap("02_blockface_bg_L2")).dataobj).astype(np.float32)
    Nbg = Nbg.mean(-1) if Nbg.ndim == 4 else Nbg
    Z = bg_zarr_CHzW.astype(np.float32).mean(0)
    perm = (2, 1, 0)
    Zt = Z.transpose(perm)                                       # full-res (level RECON_LEVEL) in nii order
    # The perm/flip decision is resolution-independent; the rebuilt bg may be FINER than the L2 reference
    # (L1 -> 2x in-plane), so block-mean a copy down to the L2 nii's in-plane shape for the CORRELATION only
    # (apply() flips the full-res volume). nz may differ (a caudal-cutoff change) -> compare the overlapping
    # leading z-range (trailing drops don't renumber, so slice z is the same physical slice).
    Zc = Zt
    while Zc.shape[0] > Nbg.shape[0] and Zc.shape[2] > Nbg.shape[2]:
        Zc = _blockmean2(Zc, (0, 2))
    assert Zc.shape[0] == Nbg.shape[0] and Zc.shape[2] == Nbg.shape[2], (Zc.shape, Nbg.shape, Zt.shape)
    nzc = min(Zc.shape[1], Nbg.shape[1])
    zs = [nzc // 4, nzc // 2, 3 * nzc // 4]
    best, bflip = -2, None
    for flips in [(), (0,), (2,), (0, 2)]:
        cand = np.flip(Zc, flips) if flips else Zc
        c = np.mean([np.corrcoef(cand[:, z, :].ravel(), Nbg[:, z, :].ravel())[0, 1] for z in zs])
        if c > best: best, bflip = c, flips
    assert best > R['orient_corr_min'], f"orientation corr only {best:.3f}"
    def apply(bg_C):
        return np.stack([np.flip(bg_C[c].transpose(perm), bflip) if bflip else bg_C[c].transpose(perm)
                         for c in range(bg_C.shape[0])], axis=-1)
    return apply, best


def _affine_level(cfg, f, level, native_z=True):
    """Output affine at in-plane pyramid `level`, from 02_blockface_bg_L2's affine (which is at L2). From L2
    to `level`: n = level - 2. n>0 -> |n| COARSER block-centre halvings (L2->L3->L4); n<0 -> |n| FINER steps
    (L2->L1). native_z=True keeps the through-plane resolution (no z scaling)."""
    import nibabel as nib
    A = nib.load(cfg.snap("02_blockface_bg_L2")).affine.copy()
    sc = np.linalg.norm(A[:3, :3], axis=0)
    a, b, zax = min([(0, 1, 2), (0, 2, 1), (1, 2, 0)], key=lambda t: abs(sc[t[0]] - sc[t[1]]))
    n = level - 2
    for _ in range(abs(n)):
        for ax in (a, b):
            if n > 0:
                A[:3, 3] += A[:3, ax] * 0.5; A[:3, ax] *= 2       # coarser (L2->L4)
            else:
                A[:3, ax] /= 2; A[:3, 3] -= A[:3, ax] * 0.5       # finer   (L2->L1)
    if f > 1 and not native_z:
        A[:3, 3] += A[:3, zax] * ((f - 1) / 2.0); A[:3, zax] *= f
    return A, zax


def _finer_affine(A, in_axes, steps):
    """Refine an affine in-plane by `steps` block-centre halvings (coarse->fine) along `in_axes`."""
    A = A.astype(float).copy()
    for _ in range(steps):
        for ax in in_axes:
            A[:3, ax] /= 2; A[:3, 3] -= A[:3, ax] * 0.5
    return A


def _upsample_inplane(a, r):
    """Block-replicate (nearest, label-preserving) along the two in-plane axes 0 & 2 by integer `r`."""
    return np.repeat(np.repeat(a, r, axis=0), r, axis=2) if r > 1 else a


def _downsample_inplane(a, r):
    """In-plane block-MEAN reduce along axes 0 & 2 by integer `r` (analytic inverse of _upsample_inplane; slice
    axis 1 and any trailing channel axis kept). Trims a ragged remainder. For a boolean mask, `>0` on the result
    is 'any voxel in the block was set'. Returns float for an image, so the caller controls dtype."""
    if r <= 1:
        return a
    D0, nz, D1 = a.shape[:3]; rest = a.shape[3:]
    D0c, D1c = (D0 // r) * r, (D1 // r) * r
    out = np.empty((D0c // r, nz, D1c // r, *rest), np.float32)
    for z0 in range(0, nz, 64):                                  # reduce in z-chunks: a full-volume float cast is
        z1 = min(nz, z0 + 64)                                    # ~4x the L1 array (tens of GB on a tall subject ->
        blk = a[:D0c, z0:z1, :D1c].astype(np.float32)           # swap-thrash); a chunk is bounded (~1-2 GB).
        out[:, z0:z1] = blk.reshape(D0c // r, r, z1 - z0, D1c // r, r, *rest).mean(axis=(1, 4))
    return out


def _write_pyramid(zarr_path, rgb, A1, reduce, depth):
    """Write an RGB uint8 (W,nz,H,3) volume as an OME-Zarr pyramid of `depth` levels (level 0 = the input's
    resolution = RECON_LEVEL; deeper levels halve in-plane, slice axis kept). A1 = its nibabel affine
    (positive-diagonal, isotropic in-plane). Filled per slice to cap peak RAM. Returns the group."""
    W, nz, H, _ = rgb.shape
    bz = zarrio.new_group(zarr_path)
    a0 = zarrio.create_level(bz, 0, [3, H, nz, W])                    # (c, rows=H, slices=nz, cols=W)
    for s in range(nz):
        a0[:, :, s, :] = np.ascontiguousarray(np.transpose(rgb[:, s, :, :], (2, 1, 0)))   # (W,H,3)->(3,H,W)
    zarrio.add_pyramid(bz, reduce, nlevels=depth)
    sx, sy, sz = (float(np.linalg.norm(A1[:3, i])) for i in range(3))  # mm: cols(x0), slices(y1), rows(z2)
    # A-P: A1's slice axis is negative for blockface, so hand the mapping over explicitly rather than as a
    # positive y_trans (which would silently flip the volume back to the pre-2026-09-19 convention).
    zarrio.write_metadata(bz, sx * 1000.0, sy * 1000.0, reduce, nlevels=depth,
                          y_trans=abs(A1[1, 3]) / sy, z_off=A1[2, 3] / sz - 0.5, x_off=A1[0, 3] / sx - 0.5,
                          y_anterior0=(float(A1[1, 3]) if A1[1, 1] < 0 else None))
    return bz


def _export_view_nii(cfg):
    """Write a VIEWABLE RGB nifti of the FINAL blockface at R['view_level'] (for fsleyes/freeview QC and
    mask/holes curation) -> <S>_blockface_L{VIEW_LEVEL}.nii.gz. The canonical multi-level data stays the zarr;
    this is a convenience derivative. Called after every blockface-pyramid write (reconstruct + reinpaint)."""
    idx = int(R['view_level']) - int(R['level'])                    # pyramid array index for VIEW_LEVEL
    arr, A = zarrio.read_level(cfg.final_zarr("blockface"), idx)
    _save_nii(cfg.final(f"blockface_L{int(R['view_level'])}"), np.ascontiguousarray(arr), A)
    return arr.shape


def _inplane_offset(A_vol, A_ref):
    """Affine-only counterpart of _match_inplane: the integer in-plane (axes 0 & 2) shift that places A_vol's
    origin onto A_ref's grid by world coords. Returns (offsets dict, shifted affine, max sub-voxel residual).
    Used in Phase A to align the L4 grid to the curated mask WITHOUT materialising any bg pixels."""
    A_out = A_vol.astype(float).copy(); off = {}; resid = 0.0
    for a in (0, 2):
        col = A_vol[:3, a]
        off_f = float(np.dot(A_ref[:3, 3] - A_vol[:3, 3], col) / np.dot(col, col))
        o = int(round(off_f)); resid = max(resid, abs(off_f - o)); off[a] = o
        A_out[:3, 3] += col * o
    return off, A_out, resid


def _match_inplane(vol, A_vol, A_ref, wm, hm):
    """Place `vol` (in-plane axes 0 & 2, slice axis 1, optional trailing channel axis) onto the reference
    in-plane grid (wm x hm) defined by `A_ref`, matched by WORLD coordinates - so the curated/model mask,
    not a (possibly drifted) auto-segmentation footprint, DEFINES the reconstruction grid. Both grids are
    the same L4 in-plane lattice, so this is an integer-voxel crop/pad (a zero shift in the reproducible
    case); any sub-voxel residual from a footprint drift is returned for logging. The slice axis is left
    untouched (native-z). Returns (vol_on_ref_grid, A_out, max_residual_vox) with A_ref's in-plane origin
    and vol's native-z column."""
    out = np.zeros((wm, vol.shape[1], hm) + vol.shape[3:], vol.dtype)
    A_out = A_vol.astype(float).copy()
    src = [slice(None)] * vol.ndim
    dst = [slice(None)] * out.ndim
    resid = 0.0
    for a, n_out in ((0, wm), (2, hm)):
        col = A_vol[:3, a]
        off_f = float(np.dot(A_ref[:3, 3] - A_vol[:3, 3], col) / np.dot(col, col))    # vol_idx = ref_idx + off
        o = int(round(off_f)); resid = max(resid, abs(off_f - o))
        n_in = vol.shape[a]
        lo, hi = max(0, -o), min(n_out, n_in - o)                                     # valid ref indices
        dst[a], src[a] = (slice(lo, hi), slice(lo + o, hi + o)) if hi > lo else (slice(0, 0), slice(0, 0))
        A_out[:3, 3] += col * o
    out[tuple(dst)] = vol[tuple(src)]
    return out, A_out, resid


# ---------- multi-resolution demons re-interpolation (native-z) ----------
def _lum(rgb):
    return rgb.astype(np.float32) @ np.asarray(R['w_lum'], np.float32)


def _smooth_resample(image, shrink, sigma):
    import SimpleITK as sitk
    sm = sitk.SmoothingRecursiveGaussian(image, sigma) if sigma > 0 else image
    osz, osp = image.GetSize(), image.GetSpacing()
    nsz = [int(s / float(shrink) + 0.5) for s in osz]
    nsp = [((osz[i] - 1) * osp[i]) / (nsz[i] - 1) if nsz[i] > 1 else osp[i] for i in range(image.GetDimension())]
    return sitk.Resample(sm, nsz, sitk.Transform(), sitk.sitkLinear, image.GetOrigin(), nsp,
                         image.GetDirection(), 0.0, image.GetPixelID())


def _ms_field(fixed, moving, shrinks, sigmas, iters, dev):
    """Multi-resolution symmetric-forces demons; returns the displacement field (moving->fixed)."""
    import SimpleITK as sitk
    reg = sitk.FastSymmetricForcesDemonsRegistrationFilter()
    reg.SetNumberOfIterations(iters); reg.SetStandardDeviations(dev)
    fpyr, mpyr = [fixed], [moving]
    for s, sg in reversed(list(zip(shrinks, sigmas))):
        fpyr.append(_smooth_resample(fpyr[0], s, sg)); mpyr.append(_smooth_resample(mpyr[0], s, sg))
    field = sitk.Image(fpyr[-1].GetSize(), sitk.sitkVectorFloat64, fpyr[-1].GetDimension())
    field.CopyInformation(fpyr[-1])
    for fx, mv in reversed(list(zip(fpyr, mpyr))):
        field = sitk.Resample(field, fx)
        field = reg.Execute(fx, mv, field)
    return field


def _warp_rgb(rgb, df, fr):
    import SimpleITK as sitk
    a = sitk.GetArrayFromImage(df) * float(fr); s = sitk.GetImageFromArray(a, isVector=True); s.CopyInformation(df)
    tx = sitk.DisplacementFieldTransform(sitk.Cast(s, sitk.sitkVectorFloat64))
    o = np.empty_like(rgb, dtype=np.float32)
    for c in range(3):
        im = sitk.GetImageFromArray(rgb[..., c].astype(np.float32))
        o[..., c] = sitk.GetArrayFromImage(sitk.Resample(im, im, tx, sitk.sitkLinear, 0.0))
    return o


def _demons_fill(bg, mask_native, interp):
    """Overwrite the given native missing slices of bg (D0,nz,D1,3) with multi-res, tissue-focused demons
    interpolation between the nearest present (non-interp) slices. R['demons'] is tuned at level L4, so the
    voxel-based params (shrink pyramid, smoothing sigma, mask dilation) are scaled by US=2**(4-RECON_LEVEL)
    to keep the SAME physical schedule at the reconstruction level: the deformation field is estimated at
    the (validated) L4 physical scale but WARPS the full-res L1 neighbour slices, so the filled slices carry
    L1 detail without a coarse-then-upsample step. Returns the set actually filled."""
    import SimpleITK as sitk
    from scipy import ndimage as ndi
    D = R['demons']
    US = 2 ** (4 - int(R['level']))
    shrink = tuple(x * US for x in D["shrink"]); sigma = tuple(x * US for x in D["sigma"]); dilate = D["dilate"] * US
    interp = sorted(interp)
    if not interp:
        return set()
    nz = bg.shape[1]
    pres = [z for z in range(nz) if z not in set(interp)]
    def brk(z):
        lo = [r for r in pres if r < z]; hi = [r for r in pres if r > z]
        return (max(lo) if lo else None, min(hi) if hi else None)
    gaps = {}
    for z in interp:
        za, zb = brk(z)
        if za is not None and zb is not None:
            gaps.setdefault((za, zb), []).append(z)
    for (za, zb), zlist in gaps.items():
        A, B = bg[:, za, :, :], bg[:, zb, :, :]
        reg = ndi.binary_dilation(mask_native[:, za, :] | mask_native[:, zb, :], iterations=dilate)[..., None]
        Ag, Bg = sitk.GetImageFromArray(_lum(A * reg)), sitk.GetImageFromArray(_lum(B * reg))
        fBA = _ms_field(Ag, Bg, shrink, sigma, D["iters"], D["dev"])
        fAB = _ms_field(Bg, Ag, shrink, sigma, D["iters"], D["dev"])
        for z in zlist:
            t = (z - za) / (zb - za)
            bg[:, z, :, :] = np.clip(np.rint((1 - t) * _warp_rgb(A, fAB, t) + t * _warp_rgb(B, fBA, 1 - t)),
                                     0, 255).astype(np.uint8)
    return set(interp)


# ---------- corrections (off the curated mask) ----------
def _correct_shading_np(bg, masked, mask, deg=2):
    """Per-slice deg-`deg` 2D log-luminance illumination field inside the curated mask, mean-1 within the
    brain (level-preserving), divided out of `masked` (and `bg` too if given). Slice axis=1. In-place.
    Pass bg=None to correct only `masked` (the L1 pipeline drops the background-intact twin). Reuses _poly_terms."""
    D0, nz, D1, _ = masked.shape
    arrs = (masked,) if bg is None else (bg, masked)
    lum = masked.astype(np.float32).mean(-1)
    Y, X = np.mgrid[0:D0, 0:D1]
    terms = _poly_terms(Y / D0 - 0.5, X / D1 - 0.5, deg)
    for i in range(nz):
        mk = mask[:, i, :]
        if mk.sum() < 200:
            continue
        ys, xs = np.where(mk)
        Amat = np.stack(_poly_terms(ys / D0 - 0.5, xs / D1 - 0.5, deg), 1)
        c = np.linalg.lstsq(Amat, np.log(np.clip(lum[:, i, :][mk], 1, None)), rcond=None)[0]
        fld = np.exp(sum(c[k] * terms[k] for k in range(len(c)))).astype(np.float32)
        fld /= fld[mk].mean()
        for arr in arrs:
            img = arr[:, i, :, :].astype(np.float32)
            arr[:, i, :, :] = np.clip(np.where(img > 0, img / fld[..., None], 0), 0, 255).astype(np.uint8)


def _zref_lowpass(A, sig):
    """Gaussian smooth of a field with holes: NaN means 'not measured here' and must not be smoothed in
    as zero, so normalise by the smoothed validity mask (normalised convolution)."""
    from scipy.ndimage import gaussian_filter
    m = np.isfinite(A).astype(np.float32)
    num = gaussian_filter(np.where(m > 0, A, 0).astype(np.float32), sig, mode="nearest")
    den = gaussian_filter(m, sig, mode="nearest")
    return np.where(den > 0.15, num / np.maximum(den, 1e-6), np.nan)


def _zref_shading_np(masked, mask, mm_inplane, mm_z, per_channel=True):
    """Z-REFERENCED illumination correction, after _correct_shading_np (configs/subjects/<ID>.yaml shading: poly2+zref; MR263).
    Slice axis=1, in place.

    `_correct_shading_np` fits each slice on its own, so whatever it removes it removes from anatomy too,
    and it cannot see a lighting change as anything other than tissue. This instead asks what is DIFFERENT
    about this slice's illumination: estimate the smooth log field per slice, take the median of that field
    over the slices within +/- window_mm, and divide out only the difference. Anatomy is shared with the
    neighbours, so it cancels and survives; a lighting wobble is not shared, so it goes. Per channel (the
    wobble carries a colour cast a single gain cannot reach) and exactly level-preserving per slice, leaving
    the cross-slice gain to harmonize/destep as before."""
    D0, nz, D1, C = masked.shape
    P = ZREF
    F = max(1, int(round(P["coarse_mm"] / max(mm_inplane, 1e-6))))              # coarse block, ~0.64 mm
    W = max(1, int(round(P["window_mm"] / max(mm_z, 1e-6))))                    # z half-window in slices
    sig = P["sigma_mm"] / (F * mm_inplane)                                      # smoothing, in coarse px
    h, w = D0 // F, D1 // F
    if h < 4 or w < 4 or nz < 3:
        print(f"zref shading: grid too small ({h}x{w}, {nz} slices) - falling back to poly2")
        return _correct_shading_np(None, masked, mask)
    S = np.zeros((nz, h, w, C), np.float32); N = np.zeros((nz, h, w), np.float32)
    for i in range(nz):
        mk = mask[:h * F, i, :w * F].astype(np.float32)
        S[i] = (masked[:h * F, i, :w * F, :].astype(np.float32) * mk[..., None]).reshape(h, F, w, F, C).sum((1, 3))
        N[i] = mk.reshape(h, F, w, F).sum((1, 3))
    valid = N >= 0.85 * F * F                                                   # interior blocks only
    chans = range(C) if per_channel else [None]
    fields = {}
    for c in chans:
        v = (S[..., c] if c is not None else S.mean(-1)) / np.maximum(N, 1)
        L = np.where(valid & (v > 5), np.log(np.maximum(v, 1)), np.nan)
        with np.errstate(invalid="ignore"):
            L -= np.nan_to_num(np.nanmean(L.reshape(nz, -1), 1), nan=0.0)[:, None, None]
        Fl = np.stack([_zref_lowpass(L[i], sig) for i in range(nz)])
        R = np.empty_like(Fl)
        for i in range(nz):
            with np.errstate(invalid="ignore"):
                R[i] = np.nanmedian(Fl[max(0, i - W):min(nz, i + W + 1)], 0)    # the local z-trend
        Dv = np.where(np.isfinite(Fl - R), Fl - R, 0.0)
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(np.where(np.isfinite(Fl), Dv, np.nan).reshape(nz, -1), 1)
        fields[c] = np.nan_to_num(Dv - np.nan_to_num(mu, nan=0.0)[:, None, None], nan=0.0)
    for i in range(nz):
        mk = mask[:, i, :]
        if mk.sum() < 200:
            continue
        src = masked[:, i, :, :].astype(np.float32)
        out = np.empty_like(src)
        for c in range(C):
            f = fields[c if per_channel else None][i]
            g = cv2.resize(f.astype(np.float32), (D1, D0), interpolation=cv2.INTER_LINEAR)
            out[..., c] = src[..., c] / np.exp(g)
            a, b = src[..., c][mk].mean(), out[..., c][mk].mean()               # the field is mean-zero over
            if b > 1:                                                           # INTERIOR blocks, so renormalise
                out[..., c] *= a / b                                            # on the mask levels are read on
        masked[:, i, :, :] = np.clip(np.where(src > 0, out, 0), 0, 255).astype(np.uint8)
    print(f"zref shading: {nz} slices, {h}x{w} field (F={F}, sigma={sig:.1f} px), z-window +/-{W}")


def _channel_levels(masked, mask, trim=5):
    """Per-slice per-channel robust (trimmed-mean) brain level; tiny/empty slices interpolated. (nz,C)."""
    D0, nz, D1, C = masked.shape
    lev = np.full((nz, C), np.nan)
    for i in range(nz):
        mk = mask[:, i, :]
        if mk.sum() >= 100:
            for c in range(C):
                d = masked[:, i, :, c][mk]
                lo, hi = np.percentile(d, [trim, 100 - trim])
                lev[i, c] = d[(d >= lo) & (d <= hi)].mean()
    ok = ~np.isnan(lev[:, 0]) & (lev[:, 0] > 0)   # all-black brain slices (lev==0) -> interpolate, not keep 0
    for c in range(C):
        lev[:, c] = np.interp(np.arange(nz), np.where(ok)[0], lev[ok, c])
    return lev


def _apply_gain(arr, g):
    """arr (D0,nz,D1,3) *= per-slice(-per-channel) gain g: (nz,) scalar or (nz,C)."""
    g = np.asarray(g, np.float32)
    gb = g[None, :, None, None] if g.ndim == 1 else g[None, :, None, :]
    arr[:] = np.clip(arr.astype(np.float32) * gb, 0, 255).astype(np.uint8)


# ---------- punch-hole inpaint (3D biharmonic RGB) ----------
def _grow_halo(vol, holes, brain):
    """Grow the hole mask by each punch-hole's antialiased dark HALO. The curated/model holes live on the L4
    (0.32mm) grid; upsampled x US=2**(4-RECON_LEVEL) to the reconstruction level they become BLOCKY, so at L1
    the ring of dark voxels just outside the block boundary (the hole's soft edge) is left unmasked and the
    biharmonic leaves a faint residual (a 'partially inpainted' hole - the L4 pipeline had no sub-voxel edge to
    miss, so it looked perfect). Per connected hole, on a padded bbox (cheap): add the dark voxels within US of
    the hole (< HOLE_INPAINT['halo_frac'] x the LOCAL brain level) that are CONNECTED to the hole - so the fill
    tracks the true soft edge at the native level, WITHOUT a large geometric collar (which would merge holes and
    explode the biharmonic) and WITHOUT swallowing a nearby dark sulcus/vessel that merely sits close by. No-op
    at L4 (US=1). Returns the grown boolean mask."""
    from scipy import ndimage as ndi
    US = 2 ** (4 - int(R['level']))
    if US <= 1:
        return holes
    frac = R['tube_inpainting'].get("halo_frac", 0.75)
    W = np.asarray(R['w_lum'], np.float32)
    out = holes.copy()
    lab, n = ndi.label(holes)
    st = np.ones((3, 3, 3))
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        pad = US + 2
        b = tuple(slice(max(0, s.start - pad), min(D, s.stop + pad)) for s, D in zip(sl, holes.shape))
        core = lab[b] == k                                   # this hole only
        br = brain[b]; lu = vol[b].astype(np.float32) @ W    # luminance on the LOCAL bbox only (no full-vol float cast)
        env = ndi.binary_dilation(core, st, iterations=US)   # search band around the hole
        outside = br & ~ndi.binary_dilation(core, st, iterations=US + 2)   # ref ring set back 2 vox off the rim
        ref = np.median(lu[outside]) if outside.any() else (np.median(lu[br]) if br.any() else 255.0)
        dark = (env & br & (lu < frac * ref)) | core         # candidate dark halo + the core
        cl, _ = ndi.label(dark)                              # keep only the dark blob touching the core
        keep = [x for x in np.unique(cl[core]) if x]
        out[b] |= np.isin(cl, keep)
    return out


def _biharmonic_fill(vol, holes, brain, dilate, margin):
    """3D biharmonic RGB inpaint of `holes` in `vol` (D0,nz,H,3): dilate the mask, crop ONE bbox (+margin),
    background-extend the exterior to nearest tissue (so a hole touching the brain rim does not bleed black in),
    then a SINGLE `inpaint_biharmonic(split_into_regions=True)` solve (skimage splits + solves each connected hole
    region internally - far less Python/setup overhead than looping per hole). Cheap here because it runs on the
    downsampled L4 volume (the pre-migration L4 fill did all subjects in ~seconds). Returns a float32 copy."""
    from scipy import ndimage as ndi
    from skimage.restoration import inpaint_biharmonic
    out = vol.astype(np.float32).copy()
    Hd = ndi.binary_dilation(holes, structure=np.ones((3, 3, 3)), iterations=dilate) & (brain | holes)
    if not Hd.any():
        return out
    xs, ys, zs = np.where(Hd)
    sl = tuple(slice(max(0, c.min() - margin), min(d, c.max() + 1 + margin))
               for c, d in zip((xs, ys, zs), vol.shape[:3]))
    Bc = out[sl]; Hc = Hd[sl]; brc = brain[sl]; tissue = brc & ~Hc
    ind = ndi.distance_transform_edt(~tissue, return_distances=False, return_indices=True)
    nearest = Bc[ind[0], ind[1], ind[2]]
    canvas = Bc.copy(); canvas[~brc] = nearest[~brc]; canvas /= 255.0
    try:
        filled = inpaint_biharmonic(canvas, Hc, channel_axis=-1, split_into_regions=True)
    except TypeError:
        filled = inpaint_biharmonic(canvas, Hc, channel_axis=-1)
    reg = out[sl]; reg[Hc] = np.clip(filled * 255.0, 0, 255)[Hc]; out[sl] = reg
    return out


def _feather_composite(vol, filled_low, M, brain, US, band):
    """Composite a COARSE-level fill into the native volume over the grown hole mask `M`, feathered so there is no
    seam. Per connected hole: bilinearly upsample the coarse (L4) fill patch to native, then alpha-blend it in as
    alpha*fill + (1-alpha)*tissue, with alpha = 1 inside M and a smoothstep ramp to 0 across a `band`-voxel ring of
    real tissue just OUTSIDE M (alpha forced to 0 off-brain). So the hole+rim is fully replaced by the smooth fill
    and the transition into tissue is gradual; tissue beyond the band is untouched. Returns uint8."""
    from scipy import ndimage as ndi
    out = vol.copy()                                                          # stays uint8; only hole crops touched
    lab, _ = ndi.label(M)
    D0, nz, D1 = M.shape
    for s in ndi.find_objects(lab):
        if s is None:
            continue
        y0, y1 = max(0, s[0].start - band), min(D0, s[0].stop + band)          # native bbox + feather ring
        x0, x1 = max(0, s[2].start - band), min(D1, s[2].stop + band)
        z0, z1 = s[1].start, s[1].stop
        yl0, yl1 = y0 // US, -(-y1 // US); xl0, xl1 = x0 // US, -(-x1 // US)    # covering coarse (L4) region
        up = ndi.zoom(filled_low[yl0:yl1, z0:z1, xl0:xl1], (US, 1, US, 1), order=1)   # -> native patch (bilinear)
        up = up[y0 - yl0 * US:y0 - yl0 * US + (y1 - y0), :, x0 - xl0 * US:x0 - xl0 * US + (x1 - x0)]
        Mb = M[y0:y1, z0:z1, x0:x1]; brb = brain[y0:y1, z0:z1, x0:x1]
        al = np.clip(1.0 - ndi.distance_transform_edt(~Mb).astype(np.float32) / float(band), 0, 1)
        al = al * al * (3 - 2 * al); al[Mb] = 1.0; al[~brb & ~Mb] = 0.0
        a3 = al[..., None]
        reg = out[y0:y1, z0:z1, x0:x1].astype(np.float32)                      # small crop cast, NOT the whole volume
        hy, hz, hx = min(up.shape[0], reg.shape[0]), min(up.shape[1], reg.shape[1]), min(up.shape[2], reg.shape[2])
        reg[:hy, :hz, :hx] = a3[:hy, :hz, :hx] * up[:hy, :hz, :hx] + (1 - a3[:hy, :hz, :hx]) * reg[:hy, :hz, :hx]
        out[y0:y1, z0:z1, x0:x1] = np.clip(np.rint(reg), 0, 255).astype(np.uint8)
    return out


def _inpaint_holes(vol, holes, dilate=None, margin=None):
    """Fill the drilled punch-hole tubes of a masked brain (D0,nz,H,3) with a MULTISCALE biharmonic fill.

    The holes live on the L4 (0.32 mm) grid; at the reconstruction level (L1, US=2**(4-RECON_LEVEL)) their soft
    dark rim is RESOLVED and pokes beyond the block-upsampled mask, and a direct L1 biharmonic over the big blocky
    hole overshoots (a faint bright 'box'). So instead: (1) GROW the mask at native res by its intensity-gated dark
    halo (_grow_halo: dilated voxels darker than halo_frac x local tissue, connected to the hole -> covers the
    resolved rim without eating sulci); (2) SOLVE the biharmonic at the COARSE L4 level, where each hole is tiny
    and the problem is well-conditioned (smooth, no overshoot) - the pre-migration fill that looked clean; (3)
    bilinearly UPSAMPLE that coarse fill and FEATHER-COMPOSITE it over the grown mask so the transition into tissue
    is seamless. No texture/grain is synthesised: for a registration template a smooth fill injects no spurious
    features. At L4 (US=1) it degenerates to a plain biharmonic. Params from R['tube_inpainting']. Returns
    (filled_vol_uint8, n_filled_voxels)."""
    if dilate is None: dilate = R['tube_inpainting']["dilate"]
    if margin is None: margin = R['tube_inpainting']["margin"]
    holes = np.asarray(holes) > 0                            # MUST be boolean (an int mask fancy-indexes -> 6-D blowup)
    if holes.sum() == 0:
        return vol, 0
    brain0 = vol.any(-1) > 0
    M = _grow_halo(vol, holes, brain0)                      # native-res, intensity-gated grown hole mask
    US = 2 ** (4 - int(R['level']))
    if US <= 1:                                             # already coarsest: plain biharmonic on the grown mask
        out = _biharmonic_fill(vol, M, brain0, dilate, margin)
        return np.clip(np.rint(out), 0, 255).astype(np.uint8), int(M.sum())
    vol_low = _downsample_inplane(vol, US)                  # native -> L4 (in-plane mean)
    holes_low = _downsample_inplane(holes, US) > 0         # original holes at L4 (chunked cast -> no full uint8 copy)
    brain_low = vol_low.any(-1) > 0
    filled_low = _biharmonic_fill(vol_low, holes_low, brain_low, dilate, margin)   # coarse smooth solve
    band = int(R['tube_inpainting'].get("feather_band", 3))
    out = _feather_composite(vol, filled_low, M, brain0, US, band)
    return out, int(M.sum())


# ---------- anterior QC montage ----------
def montage(cfg, n=10, target=300):
    """One image per subject: the `n` anterior-most coronal slices (axis 1) of the canonical blockface
    pyramid (deepest = L4), in-plane strided for speed, rotated 180 to natural orientation -> <S>_anterior10.png."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    arr, A = zarrio.read_level(cfg.final_zarr("blockface"), 4 - int(R['level']))   # deepest array = L4
    nz = arr.shape[1]
    vz = float(np.round(np.linalg.norm(A[:3, 1]), 3))
    st = max(1, round(max(arr.shape[0], arr.shape[2]) / target))
    k0 = max(0, nz - n); idx = list(range(k0, nz))
    sl = np.asarray(arr[::st, k0:nz, ::st, :])                    # (d0, n, d1, 3)
    fig, ax = plt.subplots(2, 5, figsize=(15, 6.6))
    for a, j, k in zip(ax.ravel(), range(len(idx)), idx):
        panel = np.transpose(sl[:, j, :, :], (1, 0, 2))[::-1, ::-1]   # display transpose + 180deg -> natural
        a.imshow(panel); a.set_title(f"slice {k}", fontsize=9); a.set_xticks([]); a.set_yticks([])
    for a in ax.ravel()[len(idx):]:
        a.axis("off")
    plt.suptitle(f"{cfg.subject}  -  anterior-most {n} coronal slices (native z = {vz:.3f} mm, in-plane x{st} down)",
                 fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(cfg.root, f"{cfg.subject}_anterior10.png"); plt.savefig(out, dpi=100); plt.close()
    print(f"  montage: nz={nz} vz={vz} anterior slices {idx[0]}-{idx[-1]} -> {os.path.basename(out)}")


# ---------- driver ----------
def reconstruct(cfg, do_holes=True):
    """FINAL blockface reconstruction off the curated mask (+holes). The IMAGE is a canonical OME-Zarr PYRAMID
    at in-plane level R['level'] (L1) down to L4 -> cfg.final_zarr('blockface'); the MASK/HOLES finals
    stay tiny L4 niftis (the hand-edit surface). Phase A computes the L4 grid + curated finals (affine/mask
    only, no bg pixels); Phase B builds the L1 pixels (demons/shading/de-step/inpaint NATIVELY at L1) and
    writes the pyramids. Per-step wall times append to <S>_timing.tsv. See the module banner."""
    import time, shutil
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import nibabel as nib
    W_LUM = np.asarray(R['w_lum'], np.float32)
    t0 = time.time()
    LV = int(R['level']); US = 2 ** (4 - LV); DEPTH = 4 - LV + 1     # L1: US=8, DEPTH=4 (arrays L1..L4)
    odir = cfg.root; sdir = cfg.steps_dir; s = cfg.subject
    # purge stale pre-refactor deliverables: the OLD L4 <S>_blockface.nii.gz / <S>_bg.nii.gz (a different grid
    # than the canonical zarr, and the name a human would grab) and any old steps/*.nii.gz must not linger.
    for p in (cfg.final("blockface"), cfg.final("bg")):
        if os.path.exists(p):
            os.remove(p)
    shutil.rmtree(sdir, ignore_errors=True)
    os.makedirs(sdir, exist_ok=True)
    _tt = os.path.join(odir, f"{s}_timing.tsv"); _prev = [time.time()]
    def _lap(step):
        now = time.time()
        try:
            with open(_tt, "a") as fh:
                fh.write(f"{s}\treconstruct\t{step}\t{now - _prev[0]:.2f}\n")
        except Exception:
            pass
        _prev[0] = now
    print(f"=== reconstruct {s} -> {odir}  (level L{LV}, US={US}, pyramid depth {DEPTH}) ===", flush=True)
    # max(1, ...): a subject whose native z is already coarser than Z_TARGET_UM rounds to 0 and np.repeat(cur, 0)
    # then yields an EMPTY mask (MF204 at vz=810 um, 2026-09-19). Same floor as segment._f.
    f = cfg.curation_f()                                          # curation z-factor: pinned per brain (configs/subjects)

    # Work on the REFERENCE extent (full native slice count) so a caudal-cutoff change never shifts the
    # in-plane crop the curated mask sits on; the cutoff becomes a pure z-truncation. Copy cfg (never mutate).
    cfg = copy.copy(cfg)
    nz_ref = zarr.open_group(cfg.raw_stack, mode="r")["0"].shape[2]
    new_cut = min([b for b, a in cfg.missing if a < 0], default=None)     # caudal cutoff (native index) or None
    orig_missing = cfg.missing                                            # keep the REAL cutoff for the bg crop
    cfg.missing = tuple(p for p in cfg.missing if p[1] >= 0) + ((nz_ref - 1, -1),)

    # ===== PHASE A: L4 grid + curated MASK/HOLES finals (affine/mask only - no bg pixels) =====
    # The curated (edited-else-auto) mask DEFINES the grid; we align the L4 output affine to it by world
    # coords (a footprint drift never has to reproduce the grid), tight-crop off its footprint, and write the
    # tiny L4 mask/holes niftis (the hand-edit round-trip surface). Phase B derives the L1 pixel grid from this.
    if not os.path.exists(cfg.curated_mask):
        raise SystemExit(f"no mask for {s}: expected the auto {cfg.mask_model} (run `run.py {s} segment`) "
                         f"or a hand {cfg.mask_edited}")
    A4, zax = _affine_level(cfg, f, 4, native_z=True)
    mim = nib.load(cfg.curated_mask)
    cur = (np.asanyarray(mim.dataobj) > 0)                                   # (wm, z_zds, hm) curated L4 grid
    wm, hm = cur.shape[0], cur.shape[2]
    # align A4 onto the mask grid, by world coordinates at the reconstruction spacing: a hand edit of record carries the calibrated
    # spacing (its header was rescaled with the products), which in-plane (MF5-MF7, MM15: 26 -> 27 um) would shift the grid
    _, A4, resid = _inplane_offset(A4, spacing.to_reconstruction_spacing(mim.affine, cfg))
    # resid is the sub-voxel part of the L4 offset (bounded [0,0.5]); the integer rounding absorbs it exactly
    # as the original pipeline did (a benign global <=0.5-L4-voxel world shift, consistent across the volume).
    # Phase B is self-consistent regardless (its A1 grid derives from this A4). Warn only if notably large.
    if resid > 0.2:
        print(f"  [note: curated mask {resid:.3f} L4-voxel off the rebuilt lattice (rounded; benign)]", flush=True)
    nz = nz_ref
    mask_native = np.repeat(cur, f, axis=1)[:, :nz, :]                       # 0.40mm mask -> native-z
    keep = min(new_cut + 1, nz) if new_cut is not None else nz             # caudal cutoff = z-truncation
    if keep < nz:
        mask_native = mask_native[:, :keep]; nz = keep
    fp = (mask_native > 0).any(axis=1)                                      # tight CROP_MULTIPLE crop at L4
    ii = np.where(fp.any(1))[0]; jj = np.where(fp.any(0))[0]
    mgn = int(round(R['crop_margin_mm'] * 1000.0 / (cfg.vx * 16.0)))      # margin in L4 voxels
    i0, i1 = _pad_axis(max(0, ii[0] - mgn), min(wm, ii[-1] + 1 + mgn), wm, R['crop_multiple'])
    j0, j1 = _pad_axis(max(0, jj[0] - mgn), min(hm, jj[-1] + 1 + mgn), hm, R['crop_multiple'])
    mask_native = mask_native[i0:i1, :, j0:j1]
    A4[:3, 3] += A4[:3, 0] * i0 + A4[:3, 2] * j0                            # A4 = FINAL L4 grid (aligned+cropped)
    W4, H4 = i1 - i0, j1 - j0
    interp_all = sorted({z for b0, b1 in cfg.missing if b0 >= 0 for z in range(b0 + 1, b1) if 0 <= z < nz})
    interp_native = [z for z in interp_all if mask_native[:, z, :].any()]   # slices demons will fill (label 2)
    mask = mask_native.astype(np.uint8)
    for z in interp_native:
        mask[:, z, :][mask[:, z, :] > 0] = 2
    mbin = mask > 0
    hole_native = np.zeros(mask.shape, np.uint8)
    if do_holes and os.path.exists(cfg.curated_holes):
        hcur = (np.asanyarray(nib.load(cfg.curated_holes).dataobj) > 0)
        assert hcur.shape[0] == wm and hcur.shape[2] == hm, (hcur.shape, (wm, hm))
        hole_native = (np.repeat(hcur[i0:i1, :, j0:j1], f, axis=1)[:, :nz, :] & mbin).astype(np.uint8)
    _save_nii(cfg.final("mask"), mask, A4)                                  # L4 mask final (labels 0/1/2)
    _save_nii(cfg.final("holes"), hole_native, A4)                         # L4 holes final
    _lap("grid_L4")
    print(f"  [PhaseA L4 {W4}x{nz}x{H4}  regrid-resid {resid:.3f}  crop i{i0}:{i1}_j{j0}:{j1} "
          f"{time.time() - t0:.0f}s]", flush=True)

    # ===== PHASE B: L1 pixels (native-L1 corrections/demons/inpaint) + OME-Zarr pyramids =====
    # Build the bg on the SAME crop as stage-2 (02_blockface_bg_L2, the orient reference + the A1_bg affine
    # source): use the ORIGINAL caudal cutoff, NOT the full extent. With the cutoff removed the tissue footprint
    # over the dropped tail slices can push the in-plane bbox over a CROP_MULTIPLE boundary, making the rebuilt
    # bg one crop-unit wider than the reference -> _orient_to_nii shape-assert failure (and, latently, an A1_bg
    # origin that no longer matches the wider bg -> _match_inplane misplacement). The caudal cutoff is applied
    # to the bg as a pure z-truncation below, exactly as before.
    cfg_bg = copy.copy(cfg); cfg_bg.missing = orig_missing
    blockface_bg(cfg_bg, level=LV); print(f"  [bg L{LV} rebuilt {time.time() - t0:.0f}s]", flush=True)
    bz = zarr.open_group(cfg.blockface_bg, mode="r")["0"]
    orient, corr = _orient_to_nii(np.asarray(bz), cfg)
    bg1 = orient(np.asarray(bz))                                            # (W_l1, nz_ref, H_l1, 3) nii order
    shutil.rmtree(cfg.blockface_bg, ignore_errors=True)
    _lap("bg_L1")
    A1_bg, _ = _affine_level(cfg, f, LV, native_z=True)
    A1_target = _finer_affine(A4, (0, 2), 4 - LV)                           # final L4 grid -> L1 (x US)
    bg1, A1, residL1 = _match_inplane(bg1, A1_bg, A1_target, W4 * US, H4 * US)
    bg1 = bg1[:, :nz]                                                       # caudal z-truncation
    assert bg1.shape[0] == W4 * US and bg1.shape[2] == H4 * US and residL1 < 0.05, (bg1.shape, residL1)
    print(f"  [orient corr={corr:.3f}, match-L1 {bg1.shape} resid {residL1:.3f} {time.time() - t0:.0f}s]", flush=True)
    _lap("match_L1")
    mask_L1 = _upsample_inplane(mask_native, US); mbinL1 = mask_L1 > 0
    interp_native = _demons_fill(bg1, mask_L1, [z for z in interp_native if mbinL1[:, z, :].any()])
    _lap("demons"); print(f"  [demons {len(interp_native)} L1 slices {time.time() - t0:.0f}s]", flush=True)
    masked = (bg1 * mbinL1[..., None]).astype(np.uint8); del bg1
    shading = cfg.yaml.get("shading", "poly2")                              # per brain (configs/subjects/<ID>.yaml)
    assert shading in ("poly2", "poly2+zref"), shading
    _correct_shading_np(None, masked, mbinL1)
    if shading == "poly2+zref":                         # MR263's record (the author's request, 25 Sep 2026): deg-2 THEN zref
        _zref_shading_np(masked, mbinL1, float(np.linalg.norm(A1[:3, 0])), cfg.vz / 1000.0)
    _lap("shading")
    print(f"  [shading {time.time() - t0:.0f}s]", flush=True)
    lev = _channel_levels(masked, mbinL1)                                   # (nz,C) on the de-shadowed brain
    lum = lev @ W_LUM; vpos = lum > 0
    rho = np.array([np.median((lev[:, c] / lum)[vpos]) for c in range(masked.shape[-1])])  # global ref colour
    sig = R['harmonize_mm'] / (cfg.vz / 1000.0)                          # de-step sigma as a physical width
    L_ref, steps, _ = destep.destep_reference(lum, smooth=sig)
    total = np.clip(L_ref[:, None] * rho[None, :] / np.maximum(lev, 1e-6), *R['gain_clip'])
    _apply_gain(masked, total); _lap("harmonize")
    print(f"  [harmonize destep steps@{steps} {time.time() - t0:.0f}s]", flush=True)
    _write_pyramid(cfg.color_store, masked, A1, cfg.reduce, DEPTH); _lap("write_03color")   # pre-inpaint anchor
    print(f"  [wrote steps/03_color pyramid L{LV}..L4 {time.time() - t0:.0f}s]", flush=True)
    nholes = 0
    if hole_native.any():
        h1 = _upsample_inplane(hole_native, US) & mbinL1
        masked, nholes = _inpaint_holes(masked, h1)
        _lap("inpaint"); print(f"  [inpaint {nholes} hole voxels {time.time() - t0:.0f}s]", flush=True)
    _write_pyramid(cfg.final_zarr("blockface"), masked, A1, cfg.reduce, DEPTH); _lap("write_pyramid")
    _export_view_nii(cfg)                                                   # viewable RGB nifti for QC/curation
    print(f"  [wrote blockface pyramid L{LV}..L4 + L{R['view_level']} view nifti {time.time() - t0:.0f}s]", flush=True)

    # provenance + brightness QC + montage. native_nz: the stack's frame count, or, when the brain pins stack_origin_frames, the frame
    # count of the stack of record, which the stacks' world origin counts (configs/subjects/<ID>.yaml; the volume itself is unchanged)
    open(cfg.info_file, "w").write(
        f"subject={s}\nlevel=L{LV}\nrecon_level={LV}\nslice_axis=1\nfactor={f}\nnative_nz={cfg.stack_origin_frames or nz_ref}\n"
        f"z_out={nz}\nnative_z=True\nL4_shape={W4}x{nz}x{H4}\nL{LV}_shape={W4 * US}x{nz}x{H4 * US}\n"
        f"L0_inplane={W4 * 16}x{H4 * 16}\ncrop=i{i0}:{i1}_j{j0}:{j1}\n"
        f"intensity_steps={','.join(map(str, steps))}\nharmonize_sigma_slices={sig:.2f}\n"
        f"interp_native={','.join(map(str, sorted(interp_native)))}\n"
        f"hole_voxels={nholes}\norient_corr={corr:.3f}\nregrid_resid_L4={resid:.3f}\nregrid_resid_L1={residL1:.3f}\n")
    target = total * lev                                         # corrected per-channel level
    fig, ax = plt.subplots(figsize=(11, 3.4))
    for c, col, nm in [(0, "red", "R"), (1, "green", "G"), (2, "blue", "B")]:
        ax.plot(lev[:, c], lw=0.5, color=col, alpha=0.45)
        ax.plot(target[:, c], lw=1.4, color=col, label=nm)
    for j in sorted(interp_native):
        ax.axvline(j, color="0.85", lw=0.4, zorder=0)
    for p in steps:
        ax.axvline(p, color="orange", lw=1.0, ls=":", zorder=1, label="exposure step" if p == steps[0] else None)
    ax.set_title(f"{s}  per-channel level: raw (thin) vs corrected (thick)  |  L{LV}, destep steps@{steps}")
    ax.set_xlabel("slice"); ax.set_ylabel("channel level"); ax.legend(fontsize=8, ncol=4); plt.tight_layout()
    plt.savefig(os.path.join(odir, f"{s}_brightness.png"), dpi=95); plt.close()
    montage(cfg); _lap("finalize")
    print(f"=== DONE {s} in {time.time() - t0:.0f}s "
          f"(flicker std {np.nanstd(lum):.2f} -> {np.nanstd(target @ W_LUM):.2f}) ===", flush=True)
