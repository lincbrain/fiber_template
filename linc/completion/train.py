#!/usr/bin/env python3
"""Train a completion network: the recipe of the networks of record (configs/completion.yaml, training).

    python -m linc.completion.train --root <prepared brains> --protocol full|lso --seed 0 --out <dir>/<name> [--device cuda]
--root holds one prepared folder per brain (layout: linc/completion/__init__.py), named by LINC ID; the brains are the config's
members plus extra_train. Per epoch, 500 random paired patches (64^3 input at 0.5 mm -> 128^3 output at 0.25 mm, 32 mm), batch 2,
pooled over the brains, each centered on a training-target voxel; the paired two-grid augmentation (augment.py); loss l1 + 0.2
gradient-l1 + 0.6 (1 - multi-scale SSIM) on the output, masked to the targets (data.load_subject); Adam 2e-4 with a 5-epoch linear
warm-up and cosine decay, gradient-norm clipping 1.0, EMA 0.999 of the weights; 80 epochs.
The networks of record: protocol full, seed 0 (d3_full_A_s0 on the own b0; d3p_full_A_s0 on the stand-in b0 of the same 13 brains);
protocol lso, seed 0 (d3_lso_A_s0). A retrained network matches them in recipe, not bit for bit (GPU nondeterminism, data-loader
order). Writes <out>/{config.json, train_log.jsonl, last.pt, DONE}; resumes from <out>/ckpt.pt (saved every 5 epochs).
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path); ap.add_argument("--protocol", choices=["full", "lso"], required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quick", action="store_true", help="smoke test: 2 epochs of 20 patches")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    DEV = a.device
    cfg = recipe()
    if a.quick:
        cfg.update(epochs=2, length=20, num_workers=2)
    rd = a.out; rd.mkdir(parents=True, exist_ok=True)
    if (rd / "DONE").exists():
        print(f"[{rd.name}] already DONE"); return
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    T = CFG["training"]
    members = list(T["members"]) + [S for S in T["extra_train"] if S not in T["members"]]
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
    main()
