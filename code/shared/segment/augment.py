"""Data augmentation on the GPU, per batch (nnU-Net's default probabilities and ranges, applied with torch).

Spatial, one grid_sample per batch (image linear, label nearest), from the context box down to the patch, in voxel units (rotations stay
true for non-cubic patches; with no rotation or scale the patch is an exact crop of the context center):
  rotation  p 0.2, each axis +-30 deg (2D: one angle)          scale  p 0.2, 0.8-1.2 (isotropic; +-20 %, the spacing spread)
  mirroring p 0.5 per axis (axes in `mirror_axes`)
Intensity, per sample:
  noise p 0.1 (variance 0-0.1)        blur p 0.2 (sigma 0.5-1 voxel)       brightness p 0.15 (x 0.75-1.25)
  contrast p 0.15 (0.75-1.25, clipped to the range before)      low resolution p 0.25 (zoom 0.5-1; block means, back linear)
  inverted gamma p 0.1, then gamma p 0.3 (0.7-1.5, statistics kept), as two transforms (nnU-Net)
DEFAULTS can be overridden per task (spec "augment").
"""
import math

import torch
import torch.nn.functional as F

DEFAULTS = dict(p_rot=0.2, rot_deg=30.0, p_scale=0.2, scale=(0.8, 1.2), p_mirror=0.5, mirror_axes=None,
                p_noise=0.1, noise_var=0.1, p_blur=0.2, blur_sigma=(0.5, 1.0), p_bright=0.15, bright=(0.75, 1.25),
                p_contrast=0.15, contrast=(0.75, 1.25), p_lowres=0.25, lowres=(0.5, 1.0), p_gamma=0.3, gamma=(0.7, 1.5),
                p_gamma_inv=0.1)


def _u(g, n, lo, hi, dev):
    return lo + (hi - lo) * torch.rand(n, generator=g, device=dev)


def _rot(angles, dim):
    """rotation matrices (B, dim, dim) in tensor-axis order"""
    B = angles.shape[0]; dev = angles.device
    if dim == 2:
        c, s = torch.cos(angles[:, 0]), torch.sin(angles[:, 0])
        return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
    R = torch.eye(3, device=dev).expand(B, 3, 3).clone()
    for ax in range(3):
        c, s = torch.cos(angles[:, ax]), torch.sin(angles[:, ax])
        i, j = [a for a in range(3) if a != ax]
        M = torch.eye(3, device=dev).expand(B, 3, 3).clone()
        M[:, i, i] = c; M[:, i, j] = -s; M[:, j, i] = s; M[:, j, j] = c
        R = M @ R
    return R


def spatial(x, y, patch, g, P):
    """x (B, C, *ctx) float, y (B, *ctx) -> (B, C, *patch), (B, *patch) long"""
    B, dim, dev = x.shape[0], x.ndim - 2, x.device
    ctx = list(x.shape[2:])
    ang = torch.zeros(B, 1 if dim == 2 else 3, device=dev)
    do = torch.rand(B, generator=g, device=dev) < P["p_rot"]
    ang[do] = (torch.rand(int(do.sum()), ang.shape[1], generator=g, device=dev) * 2 - 1) * math.radians(P["rot_deg"])
    s = torch.ones(B, device=dev)
    do = torch.rand(B, generator=g, device=dev) < P["p_scale"]
    s[do] = _u(g, int(do.sum()), *P["scale"], dev)
    A = _rot(ang, dim) / s[:, None, None]                                  # voxel offsets: in = A @ out
    pt = torch.tensor(patch, dtype=torch.float32, device=dev); ct = torch.tensor(ctx, dtype=torch.float32, device=dev)
    A = A * pt[None, None, :] / ct[None, :, None]                         # normalized: n_in = diag(1/ctx) A diag(patch) n_out
    A = A.flip(1).flip(2)                                                  # affine_grid wants (x, y, z) = reversed tensor axes
    theta = torch.cat([A, torch.zeros(B, dim, 1, device=dev)], 2)
    grid = F.affine_grid(theta, [B, 1] + list(patch), align_corners=False)
    xo = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    yo = F.grid_sample(y[:, None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)[:, 0].long()
    axes = P["mirror_axes"] if P["mirror_axes"] is not None else list(range(dim))
    for ax in axes:
        do = (torch.rand(B, generator=g, device=dev) < P["p_mirror"]).nonzero()[:, 0]
        if len(do):
            xo[do] = xo[do].flip(ax + 2); yo[do] = yo[do].flip(ax + 1)       # spatial axis ax of (n, C, *S) and (n, *S)
    return xo, yo


def _gauss_blur(x, sigma):
    """x (C, *S), separable Gaussian with sigma voxels"""
    dim = x.ndim - 1
    r = max(1, int(math.ceil(3 * sigma)))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) / sigma) ** 2); k = k / k.sum()
    conv = F.conv2d if dim == 2 else F.conv3d
    y = x[:, None]
    for ax in range(dim):
        shape = [1, 1] + [1] * dim; shape[2 + ax] = 2 * r + 1
        pad = [0] * (2 * dim); pad[2 * (dim - 1 - ax)] = r; pad[2 * (dim - 1 - ax) + 1] = r
        y = conv(F.pad(y, pad, mode="replicate"), k.view(shape))
    return y[:, 0]


def lowres(x, z):
    """simulated low resolution, (n, C, *S): down by zoom z with block means (area), back up linearly; both steps center-aligned,
    so the content stays in place (a mode="nearest" reduction would take each block's first voxel: +0.66 voxel when applied)"""
    S = list(x.shape[2:])
    small = F.interpolate(x, size=[max(1, int(round(s * z))) for s in S], mode="area")
    return F.interpolate(small, size=S, mode="bilinear" if x.ndim == 4 else "trilinear", align_corners=False)


def intensity(x, g, P):
    B, dim, dev = x.shape[0], x.ndim - 2, x.device
    r = lambda: torch.rand(B, generator=g, device=dev)  # noqa: E731
    for b in (r() < P["p_noise"]).nonzero()[:, 0].tolist():
        sd = math.sqrt(float(_u(g, 1, 0.0, P["noise_var"], dev)))
        x[b] = x[b] + sd * torch.randn(x[b].shape, generator=g, device=dev)
    for b in (r() < P["p_blur"]).nonzero()[:, 0].tolist():
        x[b] = _gauss_blur(x[b], float(_u(g, 1, *P["blur_sigma"], dev)))
    m = r() < P["p_bright"]
    if m.any():
        x[m] = x[m] * _u(g, int(m.sum()), *P["bright"], dev).view((-1,) + (1,) * (dim + 1))
    for b in (r() < P["p_contrast"]).nonzero()[:, 0].tolist():
        f = float(_u(g, 1, *P["contrast"], dev)); lo, hi = x[b].min(), x[b].max(); mu = x[b].mean()
        x[b] = ((x[b] - mu) * f + mu).clamp(lo, hi)
    for b in (r() < P["p_lowres"]).nonzero()[:, 0].tolist():
        x[b] = lowres(x[b:b + 1], float(_u(g, 1, *P["lowres"], dev)))[0]
    for invert, p in ((True, P["p_gamma_inv"]), (False, P["p_gamma"])):
        for b in (r() < p).nonzero()[:, 0].tolist():
            v = -x[b] if invert else x[b]
            mu, sd = v.mean(), v.std()
            lo, hi = v.min(), v.max()
            v = ((v - lo) / (hi - lo + 1e-7)).clamp(0, 1) ** float(_u(g, 1, *P["gamma"], dev)) * (hi - lo) + lo
            v = (v - v.mean()) / (v.std() + 1e-7) * sd + mu
            x[b] = -v if invert else v
    return x


def augment(x, y, patch, g, params=None):
    P = dict(DEFAULTS, **(params or {}))
    xo, yo = spatial(x.float(), y, patch, g, P)
    return intensity(xo, g, P), yo


def center_crop(x, y, patch):
    """validation: the patch at the context center, no augmentation"""
    sl = tuple(slice((c - p) // 2, (c - p) // 2 + p) for c, p in zip(x.shape[2:], patch))
    return x[(slice(None), slice(None)) + sl].float(), y[(slice(None),) + sl].long()
