#!/usr/bin/env python3
"""Train a synthesis network: the 3D U-Net (an encoder-decoder convolutional network with skip connections) that predicts blockface
tissue from a b0 image, used where a brain has no photographs (configs/completion.yaml, training).

You need
  a folder of prepared brains (`linc-complete prepare`), one folder per brain named by its Specimen ID, holding the brains that
  configs/completion.yaml lists under training (members and extra_train: the 13 brains with MRI). Own b0: `linc-complete prepare`. Stand-in
  b0: the same with --targets-from and --stand-in-b0. The targets are the brain's own photographs; no hand labels. A CUDA GPU with about 8 GB.

What it does
  1. reads each brain's input (the b0 at 0.5 mm) and targets (the photographs' grey at 0.25 mm, normalized by their 1st / 99th percentiles,
     on photographed tissue: fine/valid, minus voxels with a photographed fraction under 0.5; protocol lso also withholds a slab, to
     evaluate the prediction on tissue the network never saw)
  2. each epoch draws 500 random pairs of patches, pooled over the brains and centered on target voxels: 64^3 input voxels and the 128^3
     output voxels they cover (32 mm); batch 2
  3. applies one random spatial transform to both grids of a pair, so that they stay aligned, and intensity changes to the input only,
     so that the network does not depend on one scanner's contrast
  4. minimizes l1 + 0.2 gradient-l1 + 0.6 (1 - multi-scale SSIM, the structural similarity index), on target voxels only, with Adam
     (learning rate 2e-4, a 5-epoch linear warm-up, cosine decay, gradient norm at most 1.0) for 80 epochs
  5. keeps an exponential moving average of the weights (decay 0.999): these are the weights a network ships with

You get  (--out, default $LINC_WORK/completion/runs/<prepared folder>_<protocol>_s<seed>)
  last.pt (`linc-complete --model <out>` reads it), config.json, train_log.jsonl and DONE. A stopped run resumes from ckpt.pt, saved every 5 epochs.

    cli/linc-complete train --root $LINC_WORK/prepared                   # protocol full, seed 0: the network that synthesizes tissue
    cli/linc-complete train --root $LINC_WORK/prepared --plan            # the brains, missing files and expected GPU hours; no training

The networks of record (models/completion.md; configs/completion.yaml runs_of_record): d3_full_A_s0 (own b0) and d3p_full_A_s0 (stand-in b0), protocol
full; d3_lso_A_s0 and d3p_lso_A_s0, protocol lso. A retrained network has the same recipe, not the same bits: the first epoch's losses differ in the
third or fourth digit from the record's (a different GPU; cuDNN's benchmark mode, as in the runs of record, is not deterministic).
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import CFG, augment, data, losses, network


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)


def recipe():
    """the training configuration of record (the keys a checkpoint's cfg carries)"""
    T = CFG["training"]
    return dict(in_res_mm=CFG["input_mm"], out_res_mm=CFG["output_mm"], features=list(CFG["features"]), head=CFG["head"],
                patch_coarse=CFG["window"], batch=T["batch"], length=T["patches_per_epoch"], epochs=T["epochs"], lr=T["lr"],
                num_workers=T["num_workers"], w_l1=T["w_l1"], w_ssim=T["w_ssim"], w_grad=T["w_grad"], cosine=True, warmup=T["warmup"],
                ema=True, ema_decay=T["ema_decay"], clip=T["clip"], in_channels=1, aug_params=dict(T["augment"]))


def plan(a, members):
    """what a run reads, and how long the runs of record took; trains nothing"""
    T = CFG["training"]
    NEED = ("seam.json", "coarse/in.nii.gz", "fine/bf.nii.gz", "fine/valid.nii.gz", "fine/photo_fraction.nii.gz", "fine/brain.nii.gz")
    missing = [f"{S}/{f}" for S in members for f in NEED if not (a.root / S / f).exists()]
    print(f"{a.root}: {len(members)} brains ({', '.join(members)}); protocol {a.protocol}, seed {a.seed}; "
          f"{T['epochs']} epochs x {T['patches_per_epoch']} patches, batch {T['batch']}")
    print(f"input files: {len(members) * len(NEED) - len(missing)} of {len(members) * len(NEED)} found" + (f"; missing, e.g. {missing[:3]}" if missing else ""))
    for n, r in T["runs_of_record"].items():
        print(f"  {n}: {r['input']}, protocol {r['protocol']}: run of record {r['hours']} h on one {r['gpu']}")
    return 1 if missing else 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="linc-complete train", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path, help="the folder of prepared brains")
    ap.add_argument("--plan", action="store_true", help="print the brains, missing files and expected GPU hours; train nothing")
    ap.add_argument("--protocol", choices=["full", "lso"], default="full"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path); ap.add_argument("--device", default="cuda")
    ap.add_argument("--quick", action="store_true", help="smoke test: one epoch of two patches, no data-loader processes")
    a = ap.parse_args(argv)
    T = CFG["training"]
    members = list(T["members"]) + [S for S in T["extra_train"] if S not in T["members"]]
    if a.plan:
        return plan(a, members)
    a.out = a.out or Path(os.environ.get("LINC_WORK", "work")) / "completion" / "runs" / f"{a.root.resolve().name}_{a.protocol}_s{a.seed}"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    DEV = a.device
    cfg = recipe()
    if a.quick:
        cfg.update(epochs=1, length=2, num_workers=0)
    rd = a.out; rd.mkdir(parents=True, exist_ok=True)
    if (rd / "DONE").exists():
        print(f"[{rd.name}] already DONE"); return
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    subjects, info = [], {}
    for S in members:
        sd = data.load_subject(a.root / S, a.protocol, T["photographed_only"])
        info[S] = sd["info"]; subjects.append(sd)
        print(f"  {S}: valid {sd['info']['valid']} -> kept {sd['info']['after_keep']} -> {sd['info']['after_all']}; "
              f"norm {np.round(sd['info']['target_norm'], 3).tolist()}", flush=True)
    cfg.update(protocol=a.protocol, held_out=[], seed=a.seed, train_subs=members, extra_train=list(T["extra_train"]),
               photographed_only=T["photographed_only"], root=str(a.root), subjects=info, name=rd.name)
    json.dump(cfg, open(rd / "config.json", "w"), indent=1, default=list)
    ds = data.Patches(subjects, n=cfg["patch_coarse"], length=cfg["length"], seed=a.seed)
    seed = a.seed

    def wif(wid):
        torch.utils.data.get_worker_info().dataset.rng = np.random.default_rng([seed, 1000 + wid])

    g = torch.Generator(); g.manual_seed(a.seed)
    loader = DataLoader(ds, batch_size=cfg["batch"], shuffle=True, num_workers=cfg["num_workers"], pin_memory=True, drop_last=True,
                        persistent_workers=cfg["num_workers"] > 0, worker_init_fn=wif if cfg["num_workers"] > 0 else None, generator=g)
    G = network.build(cfg).to(DEV)
    opt = torch.optim.Adam(G.parameters(), lr=cfg["lr"])
    ssim_term = losses.MSSSIMTerm(3)
    ema = EMA(G, cfg["ema_decay"])
    warm = cfg["warmup"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda ep: (ep + 1) / warm if ep < warm
                                              else 0.5 * (1 + math.cos(math.pi * (ep - warm) / max(1, cfg["epochs"] - warm))))
    start = 0
    ck = rd / "ckpt.pt"
    if ck.exists():
        s = torch.load(ck, map_location=DEV, weights_only=False)
        G.load_state_dict(s["G"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        ema.shadow = s["ema"]; start = s["epoch"] + 1
        torch.set_rng_state(s["rng_cpu"].cpu())
        if DEV == "cuda":
            torch.cuda.set_rng_state(s["rng_cuda"].cpu())
        print(f"[{rd.name}] resumed at epoch {start}", flush=True)
    for epoch in range(start, cfg["epochs"]):
        G.train(); t0 = time.time(); agg = {}
        for batch in loader:
            x = batch["input"].to(DEV, non_blocking=True)
            y = batch["target"].to(DEV, non_blocking=True)
            m = batch["mask"].to(DEV, non_blocking=True)
            x, y, m = augment.two_grid_augment(x, y, m, **cfg["aug_params"])
            opt.zero_grad(set_to_none=True)
            pred = G(x)
            l1 = losses.masked_l1(pred, y, m)
            grad = losses.masked_grad_l1(pred, y, m)
            ss = ssim_term(pred, y, m)
            loss = cfg["w_l1"] * l1 + cfg["w_grad"] * grad + cfg["w_ssim"] * ss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), cfg["clip"])
            opt.step(); ema.update(G)
            for k, v in dict(l1=l1.item(), grad=grad.item(), ssim=ss.item(), G=loss.item()).items():
                agg[k] = agg.get(k, 0.0) + v
        sched.step()
        row = dict(epoch=epoch, s=round(time.time() - t0, 1), lr=opt.param_groups[0]["lr"],
                   gpu_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if DEV == "cuda" else None, host=os.uname().nodename,
                   **{k: round(v / len(loader), 5) for k, v in agg.items()})
        print(json.dumps(row), flush=True)
        with open(rd / "train_log.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")
        if (epoch + 1) % 5 == 0 or epoch == cfg["epochs"] - 1:
            torch.save(dict(G=G.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), ema=ema.shadow, epoch=epoch,
                            rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state() if DEV == "cuda" else None), str(ck) + ".tmp")
            os.replace(str(ck) + ".tmp", ck)
    torch.save({"G": ema.shadow, "cfg": cfg, "epoch": cfg["epochs"] - 1}, rd / "last.pt")
    (rd / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + f" {os.uname().nodename}\n")
    if not a.quick:
        ck.unlink()
    print(f"DONE [{rd.name}]", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
