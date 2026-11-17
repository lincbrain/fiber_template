"""Inference: a model folder (model.pt = state_dict, spec.json) applied to an image, the mask written in the image's grid.

With a target spacing in the spec, the image is first resampled to it (linear), and the class probabilities are resampled back to the
image's own grid (linear) before the argmax, so the mask is in the input's grid. Sliding window over the z-scored image: patch = the training patch, step 0.5 of it, each window's softmax weighted by a Gaussian (sigma
1/8 of the patch, nnU-Net's importance map); optional mirroring (the mean over every combination of flips along `mirror_axes`);
float32, cuDNN deterministic. Optional postprocessing: keep the largest connected component of each class.
The trainer's held-out validation calls predict_case() itself, so a separate run of this module reproduces its scores.

    python -m shared.segment.predict --model <run folder> --out <mask> IMAGE [IMAGE ...]           # one case (one file per channel)
    python -m shared.segment.predict --model <run folder> --task <spec.json> --fold <k> --out-dir <dir> [--score]   # a fold's held-out cases
"""
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code")); __package__ = "shared.segment"
from . import io  # noqa: E402
from shared.segment.net import UNet  # noqa: E402


def load(model_dir, device="cuda"):
    model_dir = Path(model_dir)
    spec = json.load(open(model_dir / "spec.json"))
    net = UNet.from_spec(spec["net"])
    net.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True))
    return net.to(device).eval(), spec


def gaussian(patch, device, sigma_scale=1 / 8):
    g = None
    for p in patch:
        c = (p - 1) / 2; s = p * sigma_scale
        v = torch.exp(-0.5 * ((torch.arange(p, device=device, dtype=torch.float32) - c) / s) ** 2)
        g = v if g is None else g[..., None] * v
    g = g / g.max()
    return g.clamp_min(g[g > 0].min())


def _starts(S, P, step):
    if S <= P:
        return [0]
    n = int(math.ceil((S - P) / (P * step))) + 1
    return sorted({int(round(v)) for v in np.linspace(0, S - P, n)})


@torch.no_grad()
def sliding_window(net, img, patch, device, step=0.5, mirror_axes=(), batch=4):
    """img (C, *S) float32 numpy, z-scored -> class probabilities (K, *S) float32 numpy"""
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    C, S = img.shape[0], list(img.shape[1:])
    pad = [max(0, p - s) for p, s in zip(patch, S)]
    x = torch.from_numpy(np.ascontiguousarray(img)).to(device)
    if any(pad):
        x = torch.nn.functional.pad(x, [v for p in reversed(pad) for v in (p // 2, p - p // 2)])
    Sp = list(x.shape[1:])
    K = net.n_classes
    acc = torch.zeros([K] + Sp, device=device); wsum = torch.zeros(Sp, device=device)
    gw = gaussian(patch, device)
    boxes = [tuple(slice(a, a + p) for a, p in zip(st, patch)) for st in itertools.product(*[_starts(s, p, step) for s, p in zip(Sp, patch)])]
    flips = [()] + [c for r in range(1, len(mirror_axes) + 1) for c in itertools.combinations(mirror_axes, r)]
    for i in range(0, len(boxes), batch):
        bb = boxes[i:i + batch]
        xb = torch.stack([x[(slice(None),) + b] for b in bb])
        pr = 0
        for fl in flips:
            d = [a + 2 for a in fl]
            o = net(xb.flip(d) if fl else xb).float().softmax(1)
            pr = pr + (o.flip(d) if fl else o)
        pr = pr / len(flips)
        for b, p in zip(bb, pr):
            acc[(slice(None),) + b] += p * gw; wsum[b] += gw
    out = acc / wsum
    if any(pad):
        out = out[(slice(None),) + tuple(slice(p // 2, p // 2 + s) for p, s in zip(pad, S))]
    return out.cpu().numpy()


def postprocess_mask(mask, pp, n_classes, case_mask=None):
    """pp: None / "none", "largest_component", or a dict {"kind": ...}:
      largest_cc_fill        the largest 26-connected component, interior holes filled (code/blockface/mask_rules.py largest_cc_fill)
      fill_holes             enclosed holes filled (scipy binary_fill_holes on the binary mask; code/blockface/mask_rules.py
                             fill_photo_mask, the pipeline's rule for the photograph masks); on the output grid, so at native size
                             for a native photograph
      tubes_left_hemisphere  drop voxels at/right of the brain mask's x-centroid + left_tol_vox (axis 0 increasing = anatomical
                             right), clamp to the brain mask, drop 26-connected components < min_vox (code/blockface/mask_rules.py
                             tube_rule); needs the case's brain mask on the same grid"""
    from scipy import ndimage as ndi
    kind = pp.get("kind") if isinstance(pp, dict) else pp
    if kind in (None, "none"):
        return mask
    if kind == "largest_component":
        return largest_component(mask, n_classes)
    if kind == "fill_holes":
        return ndi.binary_fill_holes(mask > 0).astype(np.uint8)
    if kind == "largest_cc_fill":
        M = mask > 0
        lab, n = ndi.label(M, structure=np.ones((3,) * M.ndim))
        if n == 0:
            return M.astype(np.uint8)
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        return ndi.binary_fill_holes(lab == int(np.argmax(sizes))).astype(np.uint8)
    if kind == "tubes_left_hemisphere":
        assert case_mask is not None, "tubes_left_hemisphere needs the brain mask (--mask, or the case's mask)"
        P = mask > 0; brain = case_mask > 0
        assert brain.shape == P.shape, (brain.shape, P.shape)
        if brain.any():
            midx = ndi.center_of_mass(brain)[0] + float(pp.get("left_tol_vox", 3.0))
            P = P & ~(np.arange(P.shape[0])[:, None, None] >= midx)
        P = P & brain
        lab, n = ndi.label(P, structure=np.ones((3, 3, 3)))
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        return np.isin(lab, np.where(sizes >= int(pp.get("min_vox", 3)))[0]).astype(np.uint8)
    raise ValueError(f"unknown postprocess {pp!r}")


def largest_component(mask, n_classes):
    from scipy import ndimage as ndi
    out = np.zeros_like(mask)
    for c in range(1, n_classes):
        lab, n = ndi.label(mask == c)
        if n:
            out[lab == (np.bincount(lab.ravel())[1:].argmax() + 1)] = c
    return out


def predict_case(net, spec, image, device="cuda", mirror=None, postprocess=None, native=False, case_mask=None, raw=False):
    """image: path(s) of one case -> (mask uint8 in the image's grid, meta, seconds). native: the image is at native resolution and is
    first downsampled by the spec's native_downsample (block means), the network's working level; the mask comes back at native size."""
    t0 = time.time()
    img, meta = io.read_image(image, spec["net"]["dim"], spec.get("spacing"))
    shape = list(img.shape[1:]); ts = spec.get("target_spacing")
    if native:
        f = spec["native_downsample"]
        img = io.resample(img, [s // f for s in shape], "area")
    elif ts:                                                            # the network's grid; the probabilities come back linearly
        img = io.resample(img, io.target_shape(shape, meta["spacing"], ts), "linear")
    mirror = spec["inference"]["mirror"] if mirror is None else mirror
    axes = tuple(range(spec["net"]["dim"])) if mirror else ()
    pr = sliding_window(net, io.zscore(img, spec.get("normalization", "zscore")), spec["patch"], device, spec["inference"]["step"], axes)
    if ts or native:
        pr = io.resample(pr, shape, "linear")
    mask = pr.argmax(0).astype(np.uint8)
    pp = spec["inference"]["postprocess"] if postprocess is None else postprocess
    cm = io.read_label(case_mask, spec["net"]["dim"])[0] if isinstance(case_mask, (str, Path)) else case_mask
    out = postprocess_mask(mask, pp, spec["net"]["n_classes"], cm)
    if raw:
        return out, meta, time.time() - t0, mask
    return out, meta, time.time() - t0
