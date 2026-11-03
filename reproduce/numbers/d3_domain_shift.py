#!/usr/bin/env python3
"""Completion accuracy on withheld tissue, from the brain's own b0 and from a stand-in b0 (paper: Methods and Results, caudal
completion; the paper reads summary.ssim.mean_own, the withheld-slab SSIM, and summary.ssim.n).

The leave-slab-out network (d3_lso_A_s0) never saw any brain's withheld slab. For each of the 12 evaluation brains it predicts the
slab twice, from
  (a) the brain's own b0 (the prepared input: normalized by its own 0.5/99.5 percentiles over the completion brain, masked by it)
  (b) a stand-in b0 (the template's b0 carried into the brain by the direct placement, on the b0's frame grid), cropped to the same
      box, normalized by ITS OWN 0.5/99.5 percentiles over the same brain and masked by it (what a brain without MRI gets)
and both are joined and scored as the completed volumes (linc.completion.deploy.slab_scores: one isotonic map on the 8 mm rostral
of the simulated cut, 2 mm cross-fade, SSIM / l1 / PSNR on the slab's fully photographed voxels, per 1 mm of depth).
The paired difference (b) - (a) per brain, mean with a 95 % bootstrap CI over the brains, is the expected cost of having no MRI.
Needs a GPU (or --device cpu, slower); the record was made on an NVIDIA Ada GPU.
-> $LINC_NUMBERS/d3_domain_shift.json
    python reproduce/numbers/d3_domain_shift.py [--device cuda]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from linc import paths, provenance  # noqa: E402
from linc.completion import CFG, data, deploy, network  # noqa: E402

RUN = "d3_lso_A_s0"                     # the leave-slab-out network with the own b0 as input
PROTOCOL_DIR = "canon_d3"               # the prepared brains with MRI (own b0)
STAND_IN = "nonlinear_s2val"            # the stand-in b0 per brain: data/sub-<ID>/mri/<STAND_IN>/<ID>_mri_full.nii.gz
SEED = 20261004

# the record's own texts, verbatim (they are part of the record's data)
WHAT = ("d3 domain shift: the leave-slab-out model's withheld-slab completion from the brain's own b0 vs from a pseudo-b0 "
        "(the template b0 pulled into the brain by S2, as the brains without MRI get), same brain mask, same scoring")
READING = "summary.ssim.mean_diff = the expected change in slab SSIM for a brain completed from a pseudo-b0"
CAVEATS = ["the pseudo-b0s come from S2 dev_s125 (template b0 v1, md5 8624a5af...); for these MRI brains S2's placement "
           "is validated against R1 o R5 at a 0.71 mm median (validated 4 Oct); for the 8 brains without MRI the shape match "
           "may be slightly worse or better, depending on their anatomy",
           "real vs pseudo-b0 intensity correlation inside both masks 0.43-0.79 (another animal, scanner and protocol): "
           "that contrast difference is what this test measures"]

md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()  # noqa: E731


def box_copy(src, lo, n, fill=0):
    """src[lo : lo + n] per axis (any lo, n), zero outside src"""
    out = np.full(tuple(n) + src.shape[3:], fill, dtype=src.dtype)
    s_lo = np.maximum(lo, 0); s_hi = np.minimum(lo + n, src.shape[:3])
    if np.any(s_hi <= s_lo):
        return out
    d_lo = s_lo - lo
    out[d_lo[0]:d_lo[0] + s_hi[0] - s_lo[0], d_lo[1]:d_lo[1] + s_hi[1] - s_lo[1], d_lo[2]:d_lo[2] + s_hi[2] - s_lo[2]] = \
        src[s_lo[0]:s_hi[0], s_lo[1]:s_hi[1], s_lo[2]:s_hi[2]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    P = paths.products("products_completion")
    root = P["completion_inputs"] / PROTOCOL_DIR
    ckp = P["completion_runs"] / RUN / "last.pt"
    net, cfg = network.load(ckp, a.device)
    assert cfg["protocol"] == "lso", "the test needs the leave-slab-out network (it never saw the slabs)"
    p_lo, p_hi = CFG["input_percentiles"]
    rows, inputs = {}, [ckp]
    for S in CFG["training"]["members"]:
        D = root / S; rec = json.load(open(D / "seam.json"))
        rg = deploy.regions(D, "lso")
        sd = data.load_subject(D, "lso", cfg.get("photographed_only"))
        T = sd["y"]                                                       # the photographs, normalized as in training
        xa = sd["x"]
        pp = P["brain_data"] / f"sub-{S}/mri/{STAND_IN}/{S}_mri_full.nii.gz"; pim = nib.load(pp)
        own = nib.load(data.recorded_path(rec["sources"]["b0"]))
        assert pim.shape == own.shape and np.allclose(pim.affine, own.affine, atol=1e-5), f"{S}: the stand-in b0 is not on the b0's frame grid"
        bm = np.asanyarray(nib.load(data.recorded_path(rec["sources"]["brain"])).dataobj) > 0
        pb = np.asanyarray(pim.dataobj).astype(np.float32)
        lo, hi = float(np.percentile(pb[bm], p_lo)), float(np.percentile(pb[bm], p_hi))
        xb_full = np.clip((pb - lo) / max(1e-6, hi - lo), 0, 1) * bm
        xb = box_copy(xb_full.astype(np.float32), np.array(rec["coarse_box"]["lo"]), np.array(rec["coarse_box"]["shape"]))[None]
        ra = deploy.predict(net, xa, a.device); rb = deploy.predict(net, xb, a.device)
        sa, sb = deploy.slab_scores(ra * rg["brain"], T, rg), deploy.slab_scores(rb * rg["brain"], T, rg)
        rows[S] = dict(own=sa, pseudo=sb, diff=dict(ssim=sb["ssim"] - sa["ssim"], l1=sb["l1"] - sa["l1"], psnr=sb["psnr"] - sa["psnr"]),
                       pseudo_norm=[lo, hi], pseudo_b0=str(pp), pseudo_b0_md5=md5(pp), score_vox=int(rg["score"].sum()))
        inputs.append(pp)
        print(f"{S}: slab SSIM own {sa['ssim']:.4f} stand-in {sb['ssim']:.4f} (diff {sb['ssim'] - sa['ssim']:+.4f}); "
              f"l1 own {sa['l1']:.4f} stand-in {sb['l1']:.4f}", flush=True)
        if a.device != "cpu":
            torch.cuda.empty_cache()
    rng = np.random.default_rng(SEED); subs = sorted(rows); summ = {}
    for m in ("ssim", "l1", "psnr"):
        d = np.array([rows[S]["diff"][m] for S in subs])
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(CFG["evaluation"]["n_boot"])]
        summ[m] = dict(mean_own=float(np.mean([rows[S]["own"][m] for S in subs])), mean_pseudo=float(np.mean([rows[S]["pseudo"][m] for S in subs])),
                       mean_diff=float(d.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n=len(d))
    out = dict(what=WHAT, run=RUN, root=str(root), pseudo_dir=STAND_IN, per_brain=rows, summary=summ, reading=READING, caveats=CAVEATS)
    provenance.write_numbers("d3_domain_shift", out, inputs=inputs, md5=False, cmd="python reproduce/numbers/d3_domain_shift.py")
    for m, v in summ.items():
        print(f"{m}: own {v['mean_own']:.4f} stand-in {v['mean_pseudo']:.4f} diff {v['mean_diff']:+.4f} "
              f"CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")


if __name__ == "__main__":
    main()
