"""The registered blockface products, made by ONE resample from the native blockface pyramid through the composed map (grayscale,
color and mask), and the anti-alias filter used wherever a finer image is pulled onto a coarser grid.

Two options of the product step, both used by the registration to the brain's own MRI (the products of record were made with them;
the direct routes use neither):
  * support normalization (`norm=True`): the photographed support M (any channel > 0) is pulled through the SAME anti-alias filter
    and map, and divides the pulled channels where it is >= 0.5 (elsewhere 0); the mask is restricted to that support. Without it,
    every resample averages the last photographed tissue with the empty space beyond it (a dark rim at the photographed end and at
    the brain surface). For a constant image the division is exact whatever the linear operator.
  * the native end crop (`crop_mm`): the last round(crop_mm / vz) photographed sections of a brain whose own last sections are
    darker are dropped before the resample (configs/subjects/<ID>.yaml `native_end_crop_mm`).

Ported from the analysis project: `antialias` = featurereg/r1_rewarp.py; `pull_rgb`, `divide`, `last_slice`, `crop_native`,
`crop_slices` = code/masknorm.py; `bf_products` = the product step of featurereg/r1_rewarp.py `one_resample_products` (with norm and
crop as its records were run: code/masknorm.py switched on by synthesis/v2/v2b_env.sh) and of bftemplate/direct/register_direct.py and
featurereg/d2_s2_register.py (norm and crop off); unchanged.
"""
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

from linc.blockface import zarrio
from linc.registration import xfmtools as X

W_LUM = np.array([0.299, 0.587, 0.114], np.float32)                # RGB -> luminance (Rec. 601)
SUPPORT_THR = 0.5                                                  # pulled support needed for a normalized voxel


def antialias(arr, src_vox, dst_vox):
    """Gaussian pre-filter along each axis where the target voxel is coarser than the source one."""
    sig = [0.5 * np.sqrt(max((d / s) ** 2 - 1.0, 0.0)) for s, d in zip(src_vox, dst_vox)]
    return ndi.gaussian_filter(arr, sig) if any(x > 0 for x in sig) else arr


def divide(num, den, thr=SUPPORT_THR):
    """num = resampled I*M (shape S or S + (C,)), den = resampled M (shape S), through the same operator.
    -> (I_out float32, support bool): num / den where den >= thr, else 0."""
    sup = den >= thr
    d = np.where(sup, den, 1.0).astype(np.float32)
    if num.ndim == den.ndim + 1:
        out = np.where(sup[..., None], num / d[..., None], 0.0)
    else:
        out = np.where(sup, num / d, 0.0)
    return out.astype(np.float32), sup


def last_slice(mask, min_px=50):
    """Highest slice index (axis 1) with >= min_px mask pixels (the last photographed section)."""
    per = (mask > 0).sum(axis=tuple(a for a in range(mask.ndim) if a != 1))
    js = np.where(per >= min_px)[0]
    assert js.size, "no photographed slice"
    return int(js.max())


def crop_native(arr, last, n):
    """Zero slices last-n+1 .. end along axis 1 (the last n photographed slices and anything past them)."""
    if n <= 0:
        return arr
    out = arr.copy()
    out[:, last - n + 1:] = 0
    return out


def crop_slices(mm, vz_mm):
    return int(round(mm / vz_mm)) if mm > 0 else 0


def pull_rgb(rgb, A, out_shape, out_affine, fmap, dst_vox, norm=False):
    """ONE anti-aliased pull of a masked native RGB level (W, nz, H, 3) onto a target grid through `fmap` (each channel as is).
    norm=True: the photographed support M = (rgb > 0 in any channel) is pulled through the SAME anti-alias + map and divides the
    channels. -> (rgb_out float32 (..., 3), support bool or None)."""
    src_vox = np.abs(np.diag(A)[:3])
    out = np.zeros(tuple(out_shape[:3]) + (3,), np.float32)
    for c in range(3):
        out[..., c] = X.apply_once(antialias(rgb[..., c].astype(np.float32), src_vox, dst_vox), A, out_shape, out_affine, fmap, order=1)
    if not norm:
        return out, None
    M = (rgb.astype(np.uint16).sum(-1) > 0).astype(np.float32)
    Mp = X.apply_once(antialias(M, src_vox, dst_vox), A, out_shape, out_affine, fmap, order=1)
    return divide(out, Mp)


def bf_products(final_zarr, mask_path, shape, affine, fmap, level=1, header_map=None, norm=False, crop_mm=0.0, vz_mm=None):
    """Grayscale (float32), RGB (uint8) and mask (bool) of the blockface on the grid (shape, affine), pulled once through `fmap`
    (target world -> blockface world) from pyramid level `level` of the final reconstruction and from its L4 mask.
    header_map: an optional 4x4 applied to the pyramid level's affine first (a header-only spacing correction).
    norm: support normalization; crop_mm (with vz_mm, the section step in mm): the native end crop (module docstring)."""
    rgb, A = zarrio.read_level(final_zarr, level)
    if header_map is not None:
        A = header_map @ A
    mimg = nib.load(str(mask_path))
    marr = np.asanyarray(mimg.dataobj) > 0
    if crop_mm > 0:                                   # both stores share the slice axis (axis 1): same nz, step and origin
        assert rgb.shape[1] == marr.shape[1] and np.isclose(A[1, 1], mimg.affine[1, 1]) and np.isclose(A[1, 3], mimg.affine[1, 3])
        last, n = last_slice(marr), crop_slices(crop_mm, float(vz_mm))
        rgb, marr = crop_native(rgb, last, n), crop_native(marr, last, n)
    out, sup = pull_rgb(rgb, A, shape, affine, fmap, np.abs(np.diag(affine)[:3]), norm=norm)
    mk = X.apply_once(marr.astype(np.float32), mimg.affine, shape, affine, fmap, order=1) > 0.5
    if sup is not None:
        mk &= sup
    grey = (out @ W_LUM) * mk
    return grey.astype(np.float32), np.clip(out * mk[..., None], 0, 255).astype(np.uint8), mk
