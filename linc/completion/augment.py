"""Training augmentation: ONE physical-space transform per sample (rotation about each axis +-rot, per-axis scale +-scale,
translation +-trans_mm, a smooth elastic field of amplitude elastic_mm on an 8^3 lattice over the patch), evaluated on each grid's
own voxel centers and sampled with grid_sample(align_corners=True), so the 0.5 mm input and the 0.25 mm target stay exactly
paired; then an intensity augmentation of the input only (contrast, brightness, smooth multiplicative bias field, noise).
"""
import torch
import torch.nn.functional as F

from . import CFG

VC, VF = CFG["input_mm"], CFG["output_mm"]
_A, _I = CFG["training"]["augment"], CFG["training"]["intensity"]


def _rot3(N, dev, rot):
    def R(axis, a):
        c, s = torch.cos(a), torch.sin(a); z = torch.zeros_like(c); o = torch.ones_like(c)
        r = {0: [o, z, z, z, c, -s, z, s, c], 1: [c, z, s, z, o, z, -s, z, c], 2: [c, -s, z, s, c, z, z, z, o]}[axis]
        return torch.stack(r, 1).view(N, 3, 3)
    a = [(torch.rand(N, device=dev) * 2 - 1) * rot for _ in range(3)]
    return R(0, a[0]) @ R(1, a[1]) @ R(2, a[2])


def _grid(nv, vox, off, N, A, t, noise, half, elastic_mm, dev):
    """normalized sampling grid (N, nv, nv, nv, 3; grid_sample's x/y/z order) of one grid's voxel centers after the transform"""
    p1 = (torch.arange(nv, device=dev, dtype=torch.float32) - off) * vox
    P = torch.stack(torch.meshgrid(p1, p1, p1, indexing="ij"), -1)[None].expand(N, -1, -1, -1, -1)
    Q = torch.einsum("nij,nxyzj->nxyzi", A, P) + t[:, None, None, None, :]
    if elastic_mm > 0:
        e = F.grid_sample(noise, (P / half).flip(-1), mode="bilinear", padding_mode="border", align_corners=True)
        Q = Q + e.permute(0, 2, 3, 4, 1) * elastic_mm
    idx = Q / vox + off
    return (idx / (nv - 1) * 2 - 1).flip(-1)


def intensity(x, dev, contrast=_I["contrast"], brightness=_I["brightness"], noise=_I["noise"], bias=_I["bias"]):
    N, C = x.shape[:2]
    ext = (1,) * (x.dim() - 2)
    x = x * (1 + (torch.rand(N, C, *ext, device=dev) * 2 - 1) * contrast)
    x = x + (torch.rand(N, C, *ext, device=dev) * 2 - 1) * brightness
    low = torch.randn(N, C, *([4] * (x.dim() - 2)), device=dev)                # smooth multiplicative bias field
    mode = "bilinear" if x.dim() == 4 else "trilinear"
    bf = F.interpolate(low, size=x.shape[2:], mode=mode, align_corners=False)
    x = x * (1 + bf * bias)
    x = x + torch.randn_like(x) * noise
    return x


def two_grid_augment(x, y, m, rot=_A["rot"], scale=_A["scale"], trans_mm=_A["trans_mm"], elastic_mm=_A["elastic_mm"],
                     intens=_A["intens"], low=8):
    """x (N, 1, n, n, n) on the input grid; y, m (N, 1, 2n, 2n, 2n) on the output grid"""
    N, dev = x.shape[0], x.device
    n = y.shape[-1] // 2
    A = _rot3(N, dev, rot) * (1 + (torch.rand(N, 3, device=dev) * 2 - 1) * scale)[:, None, :]
    t = (torch.rand(N, 3, device=dev) * 2 - 1) * trans_mm
    noise = torch.randn(N, 3, low, low, low, device=dev)
    half = n * VC / 2.0
    gf = _grid(2 * n, VF, float(n - 1), N, A, t, noise, half, elastic_mm, dev)
    gc = _grid(n, VC, (n - 1) / 2.0, N, A, t, noise, half, elastic_mm, dev)
    x = F.grid_sample(x, gc, mode="bilinear", padding_mode="zeros", align_corners=True)
    y = F.grid_sample(y, gf, mode="bilinear", padding_mode="zeros", align_corners=True)
    m = F.grid_sample(m, gf, mode="nearest", padding_mode="zeros", align_corners=True)
    if intens:
        x = intensity(x, dev)
    return x, y, m
