"""Image agreement over a region (paper: Methods, caudal completion, "compared with the photographs by the structural similarity
index"): SSIM (scikit-image, 2-D, library defaults: 7 x 7 uniform window, K1 0.01, K2 0.03, data range 1) per coronal slice over the
bounding box of the scored voxels, the local SSIM map averaged over the scored voxels only, the slice values averaged; l1 and PSNR
over the scored voxels; the high-frequency ratio and correlation (image minus its Gaussian blur, sigma 0.5 mm).
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.metrics import structural_similarity as ssim2d

from . import CFG

_E = CFG["evaluation"]


def region_metrics(pred, bf, mask):
    """{l1, psnr, ssim} of pred against bf (both on [0, 1]) over mask; {} for an empty mask"""
    m = mask > 0
    if m.sum() == 0:
        return {}
    l1 = float(np.abs(pred[m] - bf[m]).mean())
    mse = float(((pred[m] - bf[m]) ** 2).mean())
    psnr = 99.0 if mse < 1e-12 else float(10 * np.log10(1.0 / mse))
    ys = np.where(m.any(axis=(0, 2)))[0]           # coronal slices (axis 1)
    ss = []
    for y in ys:
        sm = m[:, y, :]
        if sm.sum() < _E["ssim_min_slice_vox"]:
            continue
        xs, zs = np.where(sm)
        x0, x1, z0, z1 = xs.min(), xs.max() + 1, zs.min(), zs.max() + 1
        g = bf[:, y, :][x0:x1, z0:z1]; p = pred[:, y, :][x0:x1, z0:z1]; smb = sm[x0:x1, z0:z1]
        if min(g.shape) < _E["ssim_min_box"]:
            continue
        _, S = ssim2d(g, p, data_range=1.0, full=True)
        ss.append(float(S[smb].mean()))
    return {"l1": l1, "psnr": psnr, "ssim": float(np.mean(ss)) if ss else float("nan")}


def sharp(pred, real, m, iso=CFG["output_mm"]):
    """high-frequency std ratio (pred / real) and correlation over m; {} under min_sharp_vox voxels"""
    if m.sum() < _E["min_sharp_vox"]:
        return {}
    hf = lambda v: v - ndi.gaussian_filter(v, _E["sharp_sigma_mm"] / iso)  # noqa: E731
    hp, hr = hf(pred), hf(real)
    return dict(hf_ratio_s0_5=float(hp[m].std() / max(hr[m].std(), 1e-9)), hf_corr_s0_5=float(np.corrcoef(hp[m], hr[m])[0, 1]))
