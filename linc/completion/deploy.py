"""Prediction and the join with the photographs (paper: Methods, caudal completion).

  predict   sliding window over the whole 0.5 mm input (64^3 -> 128^3 windows, overlap 0.5, Gaussian blending, batch 1); the output
            on CPU, clipped to [0, 1]
  regions   the voxel sets around the cut of one prepared brain (the cut: the clean cut face z_f for the completed volumes,
            "full"; the rostral edge of the withheld slab z_f + dstar_planes for the evaluation, "lso")
  join      one isotonic (PAVA) map prediction -> photographs, fitted on the first 8 mm of retained tissue rostral of the cut that is
            fully photographed (photo fraction >= 0.999); then a cosine cross-fade over 2 mm from the photographs (rostral) into
            the mapped prediction, on retained tissue: w = 0.5 - 0.5 cos(pi * clip(d / 2 mm, 0, 1)), d = mm rostral of the cut;
            completed = w * photographs + (1 - w) * mapped prediction, times the brain
  slab_scores  the withheld-slab measures of one prediction (SSIM / l1 / PSNR, sharpness, per 1 mm of depth past the cut)
"""
import json
import time

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference

from . import CFG
from .data import y_dep_volume
from .metrics import region_metrics, sharp

ISO = CFG["output_mm"]
RECAL_MM, FADE_MM = CFG["recal_mm"], CFG["fade_mm"]


def predict(net, x, device="cuda", n=CFG["window"]):
    """net's output for the input x (1, X, Y, Z) on the output grid (2X, 2Y, 2Z), float32 on [0, 1]"""
    t = torch.from_numpy(x[None].astype(np.float32))
    for k in range(30):                                                   # a shared GPU: wait for memory rather than fail
        try:
            with torch.no_grad():
                out = sliding_window_inference(t, (n, n, n), CFG["window_batch"], net, overlap=CFG["overlap"], mode=CFG["blend"],
                                               sw_device=torch.device(device), device=torch.device("cpu"))
            return np.clip(out[0, 0].numpy(), 0, 1).astype(np.float32)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); print(f"  OOM, retry {k + 1}/30 in 60 s", flush=True); time.sleep(60)
    raise RuntimeError("prediction kept running out of GPU memory")


def pava_fit(x, y, nbins=CFG["pava_bins"]):
    """isotonic regression of y on x over quantile bins of x: (bin centers, non-decreasing bin values) for np.interp"""
    edges = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
    which = np.clip(np.digitize(x, edges[1:-1]), 0, edges.size - 2)
    cnt = np.bincount(which, minlength=edges.size - 1).astype(float)
    sm = np.bincount(which, weights=y, minlength=edges.size - 1)
    ok = cnt > 0
    v, w, c = [], [], []
    for yi, wi in zip(sm[ok] / cnt[ok], cnt[ok]):
        v.append(yi); w.append(wi); c.append(1)
        while len(v) > 1 and v[-2] > v[-1]:
            t = w[-2] + w[-1]; v[-2] = (v[-2] * w[-2] + v[-1] * w[-1]) / t; w[-2] = t; c[-2] += c[-1]
            v.pop(); w.pop(); c.pop()
    return 0.5 * (edges[:-1] + edges[1:])[ok], np.repeat(v, c)


def regions(D, protocol):
    """the voxel sets of prepared brain D around the cut (protocol "full": the cut is z_f; "lso": z_f + dstar_planes)"""
    rec = json.load(open(f"{D}/seam.json"))
    ld = lambda p: np.asanyarray(nib.load(f"{D}/{p}").dataobj)  # noqa: E731
    brain = ld("fine/brain.nii.gz") > 0; rm = ld("fine/real_mask.nii.gz") > 0; pf = ld("fine/photo_fraction.nii.gz")
    ydep = y_dep_volume(rec, brain.shape); yy = np.rint(ydep)
    zf, n = rec["seam"]["z_f"], rec["seam"]["dstar_planes"]
    photo = rm & brain & (pf >= CFG["fully_photographed"])
    cut = zf + n if protocol != "full" else zf
    slab_planes = (yy >= zf) & (yy < zf + n - 1)
    retained = (yy >= cut) & rm & brain
    recal = retained & (yy < cut + int(round(RECAL_MM / ISO))) & (pf >= CFG["fully_photographed"])
    score = photo & slab_planes if protocol != "full" else np.zeros_like(photo)
    depth = ((cut - ydep) * ISO).astype(np.float32)
    return dict(brain=brain, retained=retained, recal=recal, score=score, depth=depth, ydep=ydep, real_mask=rm,
                z_f=zf, n=n, cut=cut, affine=nib.load(f"{D}/fine/bf.nii.gz").affine)


def join(raw, target, rg):
    """(mapped prediction, cross-fade weight w, completed): raw = the prediction times the brain; target = the photographs in the
    units the result should have"""
    cx, cy = pava_fit(raw[rg["recal"]], target[rg["recal"]])
    synth = (np.interp(raw, cx, cy) * rg["brain"]).astype(np.float32)
    w = np.clip((-rg["depth"]) / FADE_MM, 0, 1); w = ((0.5 - 0.5 * np.cos(np.pi * w)) * rg["retained"]).astype(np.float32)
    completed = ((w * target + (1 - w) * synth) * rg["brain"]).astype(np.float32)
    return synth, w, completed


def slab_scores(raw, T, rg):
    """the withheld-slab measures of one prediction: raw = the prediction times the brain, T = the photographs normalized as in
    training (data.load_subject's target); joined as the completed volumes, scored on the slab's photographed voxels"""
    _, _, dep = join(raw, T, rg)
    m = rg["score"]
    out = dict(**region_metrics(dep, T, m), **sharp(dep, T, m))
    bins = []
    for k in range(int(np.ceil(rg["n"] * ISO)) + 1):
        mk = m & (rg["depth"] > k) & (rg["depth"] <= k + 1)
        if mk.sum() >= CFG["evaluation"]["min_bin_vox"]:
            mt = region_metrics(dep, T, mk); bins.append(dict(mm_past_cut=k + 0.5, n=int(mk.sum()), ssim=mt.get("ssim"), l1=mt.get("l1")))
    out["by_depth"] = bins
    out["raw"] = region_metrics(raw, T, m)
    return out
