#!/usr/bin/env python3
"""Symmetric feature-based registration of two images A and B: one solve, both directions, neither image privileged
(paper: Methods, "Image registration"; used by the link and by the mappings to published templates).

Port of the analysis project's mri/code/symreg_upstream.py (with the helpers it took from mri/code/symreg.py and the input check of
code/input_mask_check.py). Behavior is unchanged; only paths, the environment overrides and the docstrings differ (see the end).

Protocol:
  * halfway space: the affine S (B world -> A world, RAS) is split as S = H H (H = expm(logm(S)/2)); A is resampled through H and B
    through H^-1 onto one mid grid (axis-aligned, both brain boxes + 4 mm), each image interpolated once;
  * features = upstream anatomix (anatomix-dev-vit, all 32 channels, L2, no PCA) gated by the hole-filled mask + 12 MIND-SSC (radius 1,
    dilation 2), both from the UNMASKED normalized image (clip at 0 and at the in-mask 99th percentile, min-max; same rule for A and B);
  * ONE stationary velocity field v: A_mid o exp(-v/2) vs B_mid o exp(+v/2) (symmetric log-domain SyN), LNCC summed over channel chunks
    (exact gradient) over the WHOLE mid grid, normalized by the union of the masks and scaled by 20 / n_channels;
  * optimizer = the FireANTs fork's WarpAdam step rule on v (lr 1.0, half a voxel), gradient smoothed (sigma 1.5) and v smoothed after each
    step (--smooth-warp-sigma); FFT pyramid 8,4,2,1 x 200 iterations, LNCC kernels 17,9,5,3; convergence monitor (10 iterations, 1e-6);
  * stage 2 (reextract): both images resampled to the halfway point by the stage-1 half-warps, features re-extracted, a residual symmetric
    field w solved (2,1 x 200, cc 5,3). F = exp(v/2) o exp(w) o exp(v/2) (A_mid -> B_mid), G = F^-1 (composed numerically).
Outputs in OUT/: H.txt, Hinv.txt (4x4 RAS), F_ras_disp.nii.gz, G_ras_disp.nii.gz (RAS pull displacement on the mid grid), mid_A, mid_B,
mid_Amask, mid_Bmask (.nii.gz), metrics.json. Pull maps in world coordinates (see maps()):
    B -> A (resample B onto A's grid):  a |-> H^-1 F H^-1 (a)        A -> B:  b |-> H G H (b)

    python -m template.link.symreg A Amask B Bmask S_ras_B_to_A.txt OUT [--res 0.25] [--device cuda:0]

Needs a GPU (or a very long CPU run) and the registration environment (settings/env/README.txt: torch, the FireANTs fork, anatomix). halfway()
and maps() are numpy-only and run anywhere.

Differences from the analysis script: the run-time switches LINC_CLIP_NEG (pre-2-Oct clipping) and LINC_ALLOW_UNMASKED are gone (the
clip at 0 of record is fixed; --allow-unmasked REASON replaces the override); defaults come from settings/link.yaml (`symreg`), with the
same values; the halfway step is a function (halfway()) so it can be checked without a GPU.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from scipy.linalg import expm, logm

UPSTREAM_VERSION = "neel-dey/anatomix@4c86aee + neel-dey/FireANTs@21ae072"
REF_CHANNELS = 20            # the landmark weight was calibrated against 20 channels (8 PCA + 12 MIND)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------------------------------ input check (fail-closed)
class InputNotMasked(RuntimeError):
    pass


def check_masked(image, mask, tol=0.001, label="", allow_reason=None, dist_mm=2.0, rel=0.01):
    """Raise InputNotMasked when real signal lies away from the mask: voxels farther than dist_mm from the mask whose |value| exceeds
    rel x the in-brain median |value|, over more than tol of the image's nonzero voxels (analysis project code/input_mask_check.py)."""
    im = nib.load(str(image)) if not hasattr(image, "dataobj") else image
    mi = nib.load(str(mask)) if not hasattr(mask, "dataobj") else mask
    d = np.asarray(im.dataobj); m = np.asarray(mi.dataobj) > 0.5
    a = np.sqrt((np.asarray(d, np.float64) ** 2).sum(axis=tuple(range(3, d.ndim)))) if d.ndim > 3 else np.abs(np.asarray(d, np.float64))
    nz = a > 0
    if nz.shape != m.shape:
        raise InputNotMasked(f"{label}: image grid {nz.shape} != mask grid {m.shape} (resample the mask first)")
    n = int(nz.sum()); out = float((nz & ~ndi.binary_dilation(m)).sum() / max(n, 1))
    vox = np.sqrt((np.asarray(im.affine)[:3, :3] ** 2).sum(0))
    far = ndi.distance_transform_edt(~m, sampling=vox) > dist_mm
    med = float(np.median(a[m & nz])) if (m & nz).any() else 0.0
    bad = float((far & (a > rel * med)).sum() / max(n, 1))
    rec = dict(label=label, image=str(image), mask=str(mask), nonzero=n, out_dilated=round(out, 5), far_signal=round(bad, 5),
               rule=f"> {dist_mm} mm from the mask and > {rel:g} x in-brain median, over > {tol:g} of nonzero", tol=tol)
    if bad > tol:
        msg = (f"{label}: {100 * bad:.2f} % of nonzero voxels carry signal (> {100 * rel:g} % of the in-brain median) more than {dist_mm:g} mm "
               f"from the brain mask (> {100 * tol:.2f} %): input not masked")
        if allow_reason:
            print(f"[input_mask_check] OVERRIDDEN ({allow_reason}): {msg}", flush=True)
            rec["override"] = allow_reason
        else:
            raise InputNotMasked(msg + " (multiply the image by its brain mask)")
    return rec


# ------------------------------------------------------------------------------------------ halfway space (numpy)
def half_affine(S):
    L = logm(S)
    if np.abs(L.imag).max() > 1e-8:
        raise SystemExit("affine has no real logarithm (reflection or rotation >= 180 deg)")
    return np.real(expm(0.5 * np.real(L)))


def bbox_corners(mask_path):
    i = nib.load(str(mask_path))
    m = np.asanyarray(i.dataobj) > 0.5
    idx = np.argwhere(m)
    lo, hi = idx.min(0), idx.max(0)
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)
    return c @ i.affine[:3, :3].T + i.affine[:3, 3]


def mid_grid(H, Hinv, Amask, Bmask, res, pad=4.0):
    ca = bbox_corners(Amask) @ Hinv[:3, :3].T + Hinv[:3, 3]          # A corners -> mid (m = H^-1 a)
    cb = bbox_corners(Bmask) @ H[:3, :3].T + H[:3, 3]                # B corners -> mid (m = H b)
    c = np.vstack([ca, cb])
    lo, hi = c.min(0) - pad, c.max(0) + pad
    shape = tuple(int(np.ceil((h - l) / res)) + 1 for l, h in zip(lo, hi))
    M = np.diag([res, res, res, 1.0]); M[:3, 3] = lo
    return shape, M


def resample(src_path, pull, shape, M, order):
    """Sample src at pull(m) for every mid voxel m (pull: 4x4 RAS mid world -> src world)."""
    i = nib.load(str(src_path))
    v = np.asanyarray(i.dataobj).astype(np.float32)
    v = v.reshape(v.shape[:3])
    T = np.linalg.inv(i.affine) @ pull @ M                          # mid voxel -> src voxel
    return ndi.affine_transform(v, T[:3, :3], T[:3, 3], output_shape=shape, order=order, mode="constant", cval=0.0)


def halfway(A, Amask, B, Bmask, S, out, res, pad=4.0, pairs=()):
    """H, Hinv and the mid-grid images (mid_A, mid_Amask, mid_B, mid_Bmask, extra pairs). Returns (H, Hinv, shape, M, paths)."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    S = np.loadtxt(S) if not isinstance(S, np.ndarray) else S
    H = half_affine(S); Hinv = np.linalg.inv(H)
    np.savetxt(out / "H.txt", H); np.savetxt(out / "Hinv.txt", Hinv)
    shape, M = mid_grid(H, Hinv, Amask, Bmask, res, pad)
    log(f"mid grid {shape} @ {res} mm; |H H - S| = {np.abs(H @ H - S).max():.2e}")
    paths = {}
    for key, src, pull in (("mid_A", A, H), ("mid_Amask", Amask, H), ("mid_B", B, Hinv), ("mid_Bmask", Bmask, Hinv)):
        v = resample(src, pull, shape, M, 1)
        if key.endswith("mask"):
            v = (v > 0.5).astype(np.float32)
        p = out / f"{key}.nii.gz"
        nib.save(nib.Nifti1Image(v.astype(np.float32), M), str(p))
        paths[key] = p
    for k, (a2, b2) in enumerate(pairs):
        for key, src, pull in ((f"mid_A{k + 2}", a2, H), (f"mid_B{k + 2}", b2, Hinv)):
            p = out / f"{key}.nii.gz"
            nib.save(nib.Nifti1Image(resample(src, pull, shape, M, 1).astype(np.float32), M), str(p))
            paths[key] = p
    log(f"halfway resampling done ({1 + len(pairs)} channel pair(s))")
    return H, Hinv, shape, M, paths


def maps(run):
    """(a2b, b2a) world point maps of a finished run: a2b pulls a point of A to its corresponding point of B (resamples B onto A),
    b2a the reverse. (Analysis project: mri/code/pair_benchmark.py maps_symreg.)"""
    from shared import xfm
    run = Path(run)
    H, Hi = xfm.Affine(np.loadtxt(run / "H.txt")), xfm.Affine(np.loadtxt(run / "Hinv.txt"))
    Fw, Gw = xfm.Warp(run / "F_ras_disp.nii.gz", "ras_disp"), xfm.Warp(run / "G_ras_disp.nii.gz", "ras_disp")
    a2b = lambda X: Hi.pull(Fw.pull(Hi.pull(X)))     # noqa: E731  point in A -> corresponding point in B
    b2a = lambda X: H.pull(Gw.pull(H.pull(X)))       # noqa: E731  point in B -> corresponding point in A
    return a2b, b2a


# ------------------------------------------------------------------------------------------ field helpers (torch)
def compose(u_outer, v_inner):
    """Displacement of x |-> phi_outer(psi(x)), psi(x) = x + v_inner(x), phi_outer(y) = y + u_outer(y)."""
    import torch
    from torch.nn import functional as F
    S = v_inner.shape[1:-1]
    idg = F.affine_grid(torch.eye(3, 4, device=v_inner.device, dtype=v_inner.dtype)[None], (1, 3) + tuple(S),
                        align_corners=True)
    return v_inner + F.grid_sample(u_outer.permute(0, 4, 1, 2, 3), idg + v_inner, mode="bilinear",
                                   padding_mode="border", align_corners=True).permute(0, 2, 3, 4, 1)


def folds(u, t2p):
    """Fraction of voxels where x |-> x + u(x) folds: det(J) has the opposite sign to the identity map's det(J)."""
    import torch
    L = t2p[:3, :3]
    S = u.shape[1:-1]
    g = torch.stack(torch.meshgrid(*[torch.linspace(-1, 1, s, device=u.device, dtype=u.dtype) for s in S],
                                   indexing="ij"), -1).flip(-1)       # normalized (x, y, z) at array (i, j, k)
    X = (g + u[0]) @ L.T                                              # physical position of each mapped voxel
    J = torch.stack([torch.stack(torch.gradient(X[..., c]), -1) for c in range(3)], -2)[1:-1, 1:-1, 1:-1]
    J0 = torch.stack([torch.stack(torch.gradient((g @ L.T)[..., c]), -1) for c in range(3)], -2)[1, 1, 1]
    return float(((torch.linalg.det(J) * torch.sign(torch.linalg.det(J0))) <= 0).float().mean())


def to_ras_disp_nifti(u, t2p, M, out):
    """Normalized displacement [1, Z, Y, X, 3] -> RAS physical displacement NIfTI on the mid grid (X, Y, Z, 3)."""
    d = (u[0] @ t2p[:3, :3].T).detach().cpu().numpy()                 # LPS physical, array order (Z, Y, X)
    d = np.transpose(d, (2, 1, 0, 3)) * np.array([-1.0, -1.0, 1.0], np.float32)
    img = nib.Nifti1Image(d.astype(np.float32), M)
    img.set_sform(M, 1); img.set_qform(M, 1)
    nib.save(img, str(out))


def fireants_user_lock():
    """FireANTs' global /tmp/fireants_image_lock -> a per-user file."""
    import fcntl
    from contextlib import contextmanager
    import fireants.io.image as fimg
    path = os.environ.get("FIREANTS_LOCK", f"/tmp/fireants_image_lock_{os.getuid()}")

    @contextmanager
    def get_lock():
        with open(path, "w") as fh:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
    if hasattr(fimg, "get_lock"):
        fimg.get_lock = get_lock


# ------------------------------------------------------------------------------------------ symmetric SyN (fork API)
def symsyn(fa, fb, union, masked, scales, iters, cc, a, v0=None, lm=None):
    """Symmetric log-domain SyN on the fork's primitives. fa, fb: [1, C, Z, Y, X] features of A and B on the mid grid.
    union: [1, 1, Z, Y, X] union of the two masks: the loss region if `masked`, else the whole grid; the image loss is
    normalized by its voxel count in both cases. Returns v [1, Z, Y, X, 3] (normalized coordinates)."""
    import torch
    from torch.nn import functional as F
    from fireants.interpolator import fireants_interpolator
    from fireants.losses.cc import LocalNormalizedCrossCorrelationLoss, gaussian_1d, separable_filtering
    from fireants.registration.optimizers.adam import adam_update_fused
    from fireants.utils.globals import MIN_IMG_SIZE
    from fireants.utils.imageutils import downsample, scaling_and_squaring
    from fireants.utils.util import ConvergenceMonitor, grad_smoothing_hook

    dev, dt = fa.device, fa.dtype
    full = list(fa.shape[2:])
    C = fa.shape[1]
    chunk = a.chunk
    chunks = [(c, min(c + chunk, C)) for c in range(0, C, chunk)]
    loss_fn = LocalNormalizedCrossCorrelationLoss(spatial_dims=3, kernel_size=list(cc), reduction="none").to(dev)
    loss_fn.set_scales(list(scales)); loss_fn.set_iterations(list(iters))
    clamp = {k: (float(x.min()), float(x.max())) for k, x in (("A", fa), ("B", fb))}
    eye = torch.eye(3, 4, device=dev, dtype=dt)[None]
    gs = lambda s: [gaussian_1d(t, truncated=2) for t in torch.zeros(3, device=dev, dtype=dt) + s]  # noqa: E731
    g_grad, g_warp = gs(a.smooth_grad_sigma), gs(a.smooth_warp_sigma)
    monitor = ConvergenceMonitor(10, a.tolerance)
    imgperm, vperm = (0, 4, 1, 2, 3), (0, 2, 3, 4, 1)

    def level(x, key, size):
        """The fork's pyramid image (FFT downsample, clamped to the full image's range), a few channels at a time."""
        if size == full:
            return x
        out = torch.empty([1, x.shape[1], *size], device=dev, dtype=dt)
        step = max(1, int(2 ** 25 // int(np.prod(full))))
        for c0 in range(0, x.shape[1], step):
            out[:, c0:c0 + step] = downsample(x[:, c0:c0 + step], size=size, mode="trilinear", clamp_range=clamp[key])
        return out

    v = torch.zeros([1, *full, 3], device=dev, dtype=dt) if v0 is None else v0.detach().clone()
    size_prev = list(v.shape[1:-1])
    m1, m2 = torch.zeros_like(v), torch.zeros_like(v)
    for scale, n_it in zip(scales, iters):
        monitor.reset()
        loss_fn.set_current_scale_and_iterations(scale, n_it)
        size = [max(int(s / scale), MIN_IMG_SIZE) for s in full] if scale > 1 else full
        la, lb = level(fa, "A", size), level(fb, "B", size)
        Ud = union if size == full else (F.interpolate(union, size=size, mode="trilinear", align_corners=True) > 0.5).to(dt)
        w_img = (REF_CHANNELS / C) / float(Ud.sum())
        region = Ud if masked else None
        if size != size_prev:                        # StationaryVelocity.set_size / WarpAdam.set_data_and_size
            rs = lambda t: F.interpolate(t.permute(*imgperm), size=size, mode="trilinear",  # noqa: E731
                                         align_corners=True).permute(*vperm).contiguous()
            v, m1, m2 = rs(v), rs(m1), rs(m2)
            size_prev = size
        t = 0
        grid = F.affine_grid(eye, [1, 3, *size], align_corners=True)
        half_res = 1.0 / (max(size) - 1)
        for it in range(n_it):
            with torch.no_grad():                    # half-warps as leaves; their graphs are rebuilt one at a time below
                lB = scaling_and_squaring(0.5 * v, grid, n=a.integrator_n).requires_grad_(True)
                lA = scaling_and_squaring(-0.5 * v, grid, n=a.integrator_n).requires_grad_(True)
            loss_val = 0.0
            for c0, c1 in chunks:                    # exact gradient, accumulated over channel chunks
                Bw = fireants_interpolator(lb[:, c0:c1].contiguous(), affine=eye, grid=lB, mode="bilinear",
                                           align_corners=True, is_displacement=True)
                Aw = fireants_interpolator(la[:, c0:c1].contiguous(), affine=eye, grid=lA, mode="bilinear",
                                           align_corners=True, is_displacement=True)
                val = loss_fn.forward_util(Bw, Aw)
                part = (val.sum() if region is None else (val * Ud).sum()) * w_img
                part.backward()
                loss_val += part.item()
                del Bw, Aw, val, part
            if lm is not None:                       # symmetric landmark term, in the mid space
                pa, pb, Lm, lam, sig = lm
                xa = pa[0, 0, 0] + F.grid_sample(lB.permute(*imgperm), pa, mode="bilinear", padding_mode="border",
                                                 align_corners=True)[0, :, 0, 0].T
                xb = pb[0, 0, 0] + F.grid_sample(lA.permute(*imgperm), pb, mode="bilinear", padding_mode="border",
                                                 align_corners=True)[0, :, 0, 0].T
                lml = lam * ((((xa - xb) @ Lm.T) ** 2).sum(-1) / sig ** 2).mean()
                lml.backward()
                loss_val += lml.item()
            v.requires_grad_(True)                   # chain rule into v, one integration graph in memory at a time
            for cst, lg in ((0.5, lB.grad), (-0.5, lA.grad)):
                scaling_and_squaring(cst * v, grid, n=a.integrator_n).backward(lg)
            with torch.no_grad():
                g = grad_smoothing_hook(v.grad, g_grad) if a.smooth_grad_sigma > 0 else v.grad
                t += 1                               # WarpAdam: Adam direction, max-norm clamp, half-voxel step
                m1.mul_(0.9).add_(g, alpha=0.1); m2.mul_(0.99).addcmul_(g, g, value=0.01)
                d = g.clone()
                adam_update_fused(d, m1, m2, 1 - 0.9 ** t, 1 - 0.99 ** t, 1e-8)
                gm = torch.clamp(1e-8 + d.norm(dim=-1).max(), min=1.0)
                vn = v.detach() - d * (a.lr * half_res / gm)
                if a.smooth_warp_sigma > 0:
                    vn = separable_filtering(vn.permute(*imgperm).contiguous(), g_warp).permute(*vperm).contiguous()
            v = vn
            del lB, lA, g, d
            if monitor.converged(loss_val):
                break
        log(f"  level x{scale} {size}: {it + 1}/{n_it} iterations, loss {loss_val:.4f}")
    return v.detach()


# ------------------------------------------------------------------------------------------ CLI
def parser():
    from template.link.common import config
    c = config()["symreg"]
    j = lambda x: ",".join(str(v) for v in x)  # noqa: E731
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("A"); ap.add_argument("Amask"); ap.add_argument("B"); ap.add_argument("Bmask")
    ap.add_argument("S", help="4x4 RAS text matrix mapping B world -> A world (push), e.g. a symmetric affine")
    ap.add_argument("out")
    ap.add_argument("--res", type=float, default=0.25)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--scales", default=j(c["scales"])); ap.add_argument("--iterations", default=j(c["iterations"]))
    ap.add_argument("--cc", default=j(c["cc"]))
    ap.add_argument("--stage2-scales", default=j(c["stage2_scales"])); ap.add_argument("--stage2-iterations", default=j(c["stage2_iterations"]))
    ap.add_argument("--stage2-cc", default=j(c["stage2_cc"]))
    ap.add_argument("--stage2", choices=["reextract", "continue"], default=c["stage2"])
    ap.add_argument("--lr", type=float, default=c["lr"])
    ap.add_argument("--smooth-grad-sigma", type=float, default=c["smooth_grad_sigma"])
    ap.add_argument("--smooth-warp-sigma", type=float, default=c["smooth_warp_sigma"])
    ap.add_argument("--masked-loss", action="store_true", help="loss on the union of the masks instead of the whole grid")
    ap.add_argument("--pca-channels", type=int, default=None, help="accepted for compatibility; NOT applied")
    ap.add_argument("--clip-percentile", type=float, default=c["clip_percentile"])
    ap.add_argument("--backbone", default=c["backbone"])
    ap.add_argument("--mindssc-params", default=j(c["mindssc"]), help="radius,dilation")
    ap.add_argument("--sw-params", default=j(c["sliding_window"]), help="window,batch,overlap,mode,sigma")
    ap.add_argument("--tolerance", type=float, default=c["tolerance"])
    ap.add_argument("--seed", type=int, default=c["seed"])
    ap.add_argument("--dump", action="store_true", help="save the two half-warps (debug)")
    ap.add_argument("--max-gpu-frac", type=float, default=c["max_gpu_frac"], help="hard cap on this process's share of the GPU")
    ap.add_argument("--channel-chunk", type=int, default=c["channel_chunk"], help="loss channels per chunk")
    ap.add_argument("--channel-groups", type=int, default=None, help="if set, chunk = ceil(C / groups)")
    ap.add_argument("--landmarks", default=None, help='JSON {"names": [...], "A": [[x,y,z],...], "B": [[x,y,z],...]} world RAS')
    ap.add_argument("--lm-exclude", nargs="*", default=[])
    ap.add_argument("--lm-sigma", type=float, default=c["lm_sigma"])
    ap.add_argument("--lm-weight", type=float, default=c["lm_weight"])
    ap.add_argument("--integrator-n", type=int, default=c["integrator_n"])
    ap.add_argument("--pair", nargs=2, action="append", default=[], metavar=("A2", "B2"),
                    help="extra channel pair: an image in A's world and one in B's world (features stacked; LNCC per channel)")
    ap.add_argument("--no-mind", action="store_true", help="features = anatomix only")
    ap.add_argument("--allow-unmasked", default=None, metavar="REASON", help="explicit, logged override of the input mask check")
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    from template.link.common import config
    mc = config()["symreg"]
    for _img, _msk in ((a.A, a.Amask), (a.B, a.Bmask)):          # every registration input must be brain-masked
        check_masked(_img, _msk, label=f"symreg {Path(_img).name}", allow_reason=a.allow_unmasked, **mc["input_mask_check"])
    ints = lambda s: [int(x) for x in s.split(",")]  # noqa: E731
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    # ---- halfway space
    H, Hinv, shape, M, paths = halfway(a.A, a.Amask, a.B, a.Bmask, a.S, out, a.res, mc["mid_grid_pad_mm"], a.pair)

    # FireANTs' compiled kernels off BEFORE anything imports FireANTs (reproducibility)
    from anatomix.registration.registration_infrastructure.fused_ops import configure as _ffo_configure
    _ffo_configure(["--fused-ops", "off"])
    import torch
    from torch.nn import functional as F
    from fireants.io import Image
    from fireants.interpolator import fireants_interpolator
    from anatomix.registration.registration_infrastructure.features import (combine_feature_channels, load_backbone,
                                                                            minmax_normalize, prepare_feature_channels)
    fireants_user_lock()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    dev = torch.device(a.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
        torch.cuda.set_per_process_memory_fraction(a.max_gpu_frac, dev)   # shared cards: fail THIS process rather than starve others

    IA, IB = Image.load_file(str(paths["mid_A"]), device=dev), Image.load_file(str(paths["mid_B"]), device=dev)
    MA, MB = Image.load_file(str(paths["mid_Amask"]), device=dev), Image.load_file(str(paths["mid_Bmask"]), device=dev)
    MAa, MBa = (MA.array > 0).float(), (MB.array > 0).float()
    t2p = IA.torch2phy[0].double()
    c = (t2p @ torch.tensor([-1.0, -1.0, -1.0, 1.0], dtype=t2p.dtype, device=t2p.device))[:3].cpu().numpy()
    if np.abs(c - M[:3, 3] * np.array([-1, -1, 1])).max() > 1e-3:
        raise SystemExit(f"torch2phy convention check failed: {c} vs {M[:3, 3]}")
    fill = lambda m: torch.from_numpy(ndi.binary_fill_holes(m[0, 0].cpu().numpy() > 0)).to(m)[None, None]  # noqa: E731
    GA, GB = fill(MAa), fill(MBa)                     # feature gates (the masks with their cavities filled)
    norm = {}
    for I, Mk, k in ((IA, MAa, "A"), (IB, MBa, "B")):  # same rule for both images: clip at 0 and the in-mask percentile, min-max
        x = I.array.float()
        vals = x[Mk > 0]
        hi = float(torch.quantile(vals[::max(1, vals.numel() // 2_000_000)], a.clip_percentile / 100.0))
        norm[k] = minmax_normalize(x, 0.0, hi, name=k)
    for k in range(len(a.pair)):                      # extra pairs: same rule, each with its side's mask (no clip at 0)
        for side, Mk in (("A", MAa), ("B", MBa)):
            key = f"{side}{k + 2}"
            x = Image.load_file(str(paths[f"mid_{key}"]), device=dev).array.float()
            vals = x[Mk > 0]
            hi = float(torch.quantile(vals[::max(1, vals.numel() // 2_000_000)], a.clip_percentile / 100.0))
            norm[key] = minmax_normalize(x, None, hi, name=key)
    fam = "anatomix" if a.no_mind else "anatomix+mindssc"
    sw = a.sw_params.split(",")
    rad, dil = ints(a.mindssc_params)
    cfg = dict(features=fam, isotropic=True, window=int(sw[0]), sw_batch=int(sw[1]), overlap=float(sw[2]),
               sw_mode=sw[3], sigma=float(sw[4]), feature_normalization="l2", mindssc_radius=rad,
               mindssc_dilation=dil, verbose=False)
    spacing = (a.res,) * 3
    model = load_backbone(a.backbone, dev)

    def features(xn, gate):
        """Network + MIND-SSC on the unmasked image, network gated by the mask."""
        p, m = prepare_feature_channels(xn, spacing, model, **cfg)
        f = combine_feature_channels(p, m, gate, fam, append_mask=False)
        del p, m; torch.cuda.empty_cache()
        return f

    gib = lambda: round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)  # noqa: E731
    t0 = time.time()
    keysA = ["A"] + [f"A{k + 2}" for k in range(len(a.pair))]
    keysB = ["B"] + [f"B{k + 2}" for k in range(len(a.pair))]
    feats = lambda keys, gate, u=None: torch.cat([features(norm[k] if u is None else warp_(norm[k], u), gate) for k in keys], 1)  # noqa: E731
    warp_ = None
    fA, fB = feats(keysA, GA), feats(keysB, GB)
    feat_s = [round(time.time() - t0, 1)]
    C = fA.shape[1]
    a.chunk = math.ceil(C / a.channel_groups) if a.channel_groups else a.channel_chunk
    log(f"features: {C} channels ({time.time() - t0:.1f} s); peak GiB {gib()}; loss chunk {a.chunk}")

    # ---- landmarks (optional): points in normalized mid coordinates
    LM, lm_used = None, None
    p2t = torch.linalg.inv(t2p)

    def norm_pts(P):
        P = torch.tensor(np.c_[P * np.array([-1.0, -1.0, 1.0]), np.ones(len(P))], dtype=t2p.dtype, device=t2p.device)
        return (P @ p2t.T)[:, :3].float().reshape(1, 1, 1, -1, 3)
    if a.landmarks:
        J = json.loads(Path(a.landmarks).read_text())
        keep = [i for i, n in enumerate(J["names"]) if n not in a.lm_exclude]
        lm_used = [J["names"][i] for i in keep]
        PA = np.asarray(J["A"], float)[keep]; PB = np.asarray(J["B"], float)[keep]
        MA_ = (np.c_[PA, np.ones(len(PA))] @ Hinv.T)[:, :3]           # A point -> A_mid coords (m = H^-1 a)
        MB_ = (np.c_[PB, np.ones(len(PB))] @ H.T)[:, :3]              # B point -> B_mid coords (m = H b)
        LM = (norm_pts(MA_), norm_pts(MB_), t2p[:3, :3].float(), float(a.lm_weight), float(a.lm_sigma))
        log(f"landmark term: {len(keep)} landmarks, sigma {a.lm_sigma} mm, weight {a.lm_weight}")

    U1 = torch.maximum(MAa, MBa)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    v1 = symsyn(fA, fB, U1, a.masked_loss, ints(a.scales), ints(a.iterations), ints(a.cc), a, lm=LM)
    rt = [round(time.time() - t0, 1)]
    log(f"stage 1 {rt[0]} s; peak GiB {gib()}")

    full = list(IA.array.shape[2:])                   # array order (Z, Y, X)
    grid_full = F.affine_grid(torch.eye(3, 4, device=dev)[None], [1, 3, *full], align_corners=True)
    from fireants.utils.imageutils import scaling_and_squaring
    expv = lambda vel, cst: scaling_and_squaring(cst * vel, grid_full, n=a.integrator_n)  # noqa: E731
    eye = torch.eye(3, 4, device=dev)[None]
    warp = lambda x, u, mode="bilinear": fireants_interpolator(x, affine=eye, grid=u.contiguous(), mode=mode,  # noqa: E731
                                                                align_corners=True, is_displacement=True)
    samp = lambda u, p: p[0, 0, 0] + F.grid_sample(u.permute(0, 4, 1, 2, 3), p, mode="bilinear", padding_mode="border",  # noqa: E731
                                                   align_corners=True)[0, :, 0, 0].T
    if a.stage2 == "reextract":
        with torch.no_grad():
            uA1, uB1 = expv(v1, -0.5), expv(v1, 0.5)
            del fA, fB; torch.cuda.empty_cache()
            t0 = time.time()
            GA2, GB2 = (warp(GA, uA1, "nearest") > 0.5).float(), (warp(GB, uB1, "nearest") > 0.5).float()
            warp_ = warp
            fA2, fB2 = feats(keysA, GA2, uA1), feats(keysB, GB2, uB1)
            feat_s.append(round(time.time() - t0, 1))
            U2 = torch.maximum((warp(MAa, uA1) > 0.5).float(), (warp(MBa, uB1) > 0.5).float())
            LM2 = None
            if LM is not None:                        # landmarks carried to the halfway point of stage 1
                LM2 = (samp(uB1, LM[0]).reshape(1, 1, 1, -1, 3), samp(uA1, LM[1]).reshape(1, 1, 1, -1, 3)) + LM[2:]
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        w = symsyn(fA2, fB2, U2, a.masked_loss, ints(a.stage2_scales),
                   ints(a.stage2_iterations), ints(a.stage2_cc), a, lm=LM2)
        rt.append(round(time.time() - t0, 1))
        del fA2, fB2; torch.cuda.empty_cache()
    else:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        v1 = symsyn(fA, fB, U1, a.masked_loss, ints(a.stage2_scales), ints(a.stage2_iterations), ints(a.stage2_cc), a,
                    v0=v1, lm=LM)
        rt.append(round(time.time() - t0, 1))
        del fA, fB; torch.cuda.empty_cache()
    log(f"stage 2 {rt[1]} s; peak GiB {gib()}")
    del model; torch.cuda.empty_cache()

    with torch.no_grad():
        if a.stage2 == "reextract":
            ew, emw = expv(w, 0.5), expv(w, -0.5)
            uB, uA = compose(uB1, ew), compose(uA1, emw)                # phi_B = e(v/2) o e(w/2), phi_A likewise
            Fd = compose(uB1, compose(expv(w, 1.0), uB1))              # e(v/2) o e(w) o e(v/2)
            Gd = compose(uA1, compose(expv(w, -1.0), uA1))             # e(-v/2) o e(-w) o e(-v/2)
            half_cons = None
        else:
            uB, uA = expv(v1, 0.5), expv(v1, -0.5)
            Fd, Gd = expv(v1, 1.0), expv(v1, -1.0)
            half_cons = compose(uB, uB) - Fd                           # integration numerics
        if a.dump:
            torch.save({"uA": uA.cpu(), "uB": uB.cpu(), "t2p": t2p.cpu(), "MA": MAa.cpu()}, out / "halfwarps.pt")
        t2pf = t2p.float()
        mm = lambda u: (u[0] @ t2pf[:3, :3].T).norm(dim=-1)  # noqa: E731
        mA, mB = MAa[0, 0] > 0, MBa[0, 0] > 0
        gf, fg = mm(compose(Gd, Fd)), mm(compose(Fd, Gd))               # G(F(x)) - x, F(G(x)) - x
        q99 = lambda e, m: float(torch.quantile(e[m][::7], 0.99))  # noqa: E731
        met = {"mid_shape": list(shape), "res_mm": a.res, "feature_seconds": round(sum(feat_s), 1),
               "feature_seconds_per_stage": feat_s, "stage_runtimes_s": rt,
               "exp_half_squared_vs_exp_max_mm": None if half_cons is None else float(mm(half_cons).max()),
               "roundtrip_GF_mm_in_Amask": {"median": float(gf[mA].median()), "p99": q99(gf, mA), "max": float(gf[mA].max())},
               "roundtrip_FG_mm_in_Bmask": {"median": float(fg[mB].median()), "p99": q99(fg, mB), "max": float(fg[mB].max())},
               "folds_fraction": {"phi_A": folds(uA, t2pf), "phi_B": folds(uB, t2pf),
                                  "F": folds(Fd, t2pf), "G": folds(Gd, t2pf)},
               "disp_mm_in_Amask_F": {"median": float(mm(Fd)[mA].median()), "max": float(mm(Fd)[mA].max())}}
        d = lambda x, y: float(2 * (x & y).sum() / (x.sum() + y.sum()))  # noqa: E731
        met["mask_dice_mid_before"] = d(MAa > 0.5, MBa > 0.5)
        met["mask_dice_mid_after"] = d(warp(MAa, uA) > 0.5, warp(MBa, uB) > 0.5)
        if LM is not None:                            # in-sample fit: the two landmarks' positions at the halfway point
            xa, xb = ((samp(ew, LM2[0]), samp(emw, LM2[1])) if a.stage2 == "reextract" else
                      (samp(uB, LM[0]), samp(uA, LM[1])))
            r = ((xa - xb) @ LM[2].T).norm(dim=-1).cpu().numpy()
            met["landmark_residual_mid_mm"] = {"median": float(np.median(r)), "max": float(r.max()),
                                               "per_landmark": dict(zip(lm_used, [round(float(x), 4) for x in r]))}
        to_ras_disp_nifti(Fd, t2pf, M, out / "F_ras_disp.nii.gz")
        to_ras_disp_nifti(Gd, t2pf, M, out / "G_ras_disp.nii.gz")
    if a.landmarks:
        met["landmarks_used"] = lm_used
        met["landmark_sigma_mm"], met["landmark_weight"] = a.lm_sigma, a.lm_weight
    met["wall_s"] = round(time.time() - t_all, 1)
    met["peak_gpu_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    met["channel_groups"] = math.ceil(C / a.chunk)
    met["engine"] = "upstream-sym"
    met["engine_version"] = UPSTREAM_VERSION
    met["engine_doc"] = ("halfway affine (S = H H) + upstream anatomix(32, no PCA)+MIND-SSC(12) features + symmetric "
                         "log-domain registration on the FireANTs fork (A o exp(-v/2) vs B o exp(v/2)); "
                         + ("stage 2 re-extracted at the halfway point: F = e(v/2) e(w) e(v/2), G = F^-1"
                            if a.stage2 == "reextract" else "one field: F = exp(v), G = exp(-v)"))
    met["settings"] = {"loss": "masked_cc (union of masks)" if a.masked_loss else "cc (whole mid grid, background included)",
                       "image_loss_scale": f"({REF_CHANNELS}/{C}) * sum_channels sum_region(-LNCC) / |union|",
                       "channels": (f"{1 + len(a.pair)} pair(s) x " if a.pair else "")
                                   + ("anatomix only" if a.no_mind else "anatomix + mindssc(12)") + f", {C} total",
                       "channel_pairs": [[a.A, a.B]] + [list(p) for p in a.pair], "backbone": a.backbone,
                       "mindssc_radius_dilation": [rad, dil], "sliding_window": a.sw_params, "feature_normalization": "l2",
                       "clip_percentile_both": a.clip_percentile, "pca_channels_requested": a.pca_channels,
                       "pca_note": "not applied: upstream registers every anatomix channel (loss chunked)",
                       "optimizer": "WarpAdam step rule on v (additive, log-domain)", "lr": a.lr,
                       "smooth_grad_sigma": a.smooth_grad_sigma, "smooth_warp_sigma": a.smooth_warp_sigma,
                       "scales": a.scales, "iterations": a.iterations, "cc": a.cc, "stage2": a.stage2,
                       "stage2_scales": a.stage2_scales, "stage2_iterations": a.stage2_iterations, "stage2_cc": a.stage2_cc,
                       "integrator_n": a.integrator_n, "tolerance": a.tolerance, "channel_chunk": a.chunk,
                       "pyramid": "FFT downsample per level (fork)", "fused_ops": "off", "seed": a.seed}
    met["argv"] = sys.argv[1:] if argv is None else list(argv)
    (out / "metrics.json").write_text(json.dumps(met, indent=1))
    log(json.dumps({k: v for k, v in met.items() if k not in ("settings", "argv")}))


if __name__ == "__main__":
    main()
