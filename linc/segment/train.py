"""Train one segmentation model of a task: one fold (its held-out brains predicted and scored at the end) or "all" (every brain;
the shipped model). The recipe is nnU-Net's (Isensee et al. 2021), reimplemented with the augmentation on the GPU.

You need
  a task file: the cases (each an image and its hand-made label), the five folds (the brains each fold holds out) and the task's
  settings. The four of record are configs/segment/<model>.json (mri_brainmask, bf_mask2d, bf_mask3d, bf_holes); their images and
  labels are read under $LINC_PRODUCTS. A CUDA GPU: about 2 h per run on an RTX 6000 Ada.

What it does
  1. resamples each image to the task's target spacing (if it has one) and normalizes each channel to zero mean, unit variance
  2. cuts training patches, one in three centered on a labeled voxel, and augments them (rotation, scale, mirroring, noise, blur,
     brightness, contrast, low resolution, gamma)
  3. trains a U-Net (an encoder-decoder convolutional network with skip connections; nnU-Net's layout) for 200 epochs x 250
     iterations, batch 2 unless the task file says otherwise: loss = Dice loss (one minus the overlap between prediction and label)
     + cross-entropy, at every decoder level; SGD with Nesterov momentum 0.99, learning rate 1e-2 decaying polynomially to 0
  4. keeps the final epoch's weights (no selection on held-out data)
  5. predicts every held-out case with the release's inference (linc/segment/predict.py) and scores it against its label (Dice
     coefficient, average symmetric surface distance, 95th-percentile Hausdorff distance, in mm)

You get  ($LINC_WORK/segment/runs/<task>/fold_<k>/, or --out)
  model.pt and spec.json (a model folder: `linc-segment predict --model <folder>` reads it), record.json (the recipe, every
  epoch's loss, the held-out scores), val_pred/ (the held-out masks) and DONE. A stopped run resumes from its last.pt.

    cli/linc-segment train --task configs/segment/<model>.json --fold <0-4|all>
    cli/linc-segment train --task configs/segment/<model>.json --plan        # the cases, folds, missing files, GPU hours; no training

The runs of record made the four shipped models (fold all; $LINC_MODELS/segment/<model>) and the held-out scores of their records
(folds 0-4; reproduce/numbers/segmodels_<model>_2026-11.py). On one GPU the recipe is deterministic: two runs gave bitwise identical
weights (mri_brainmask, 2 epochs, RTX 4000 Ada, 3 Nov 2026). Whether another GPU model gives the same bits has not been
measured.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2])); __package__ = "linc.segment"
from . import augment, data, io, losses, metrics, predict  # noqa: E402
from .net import UNet  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]                            # the repository
FEATURES = {2: (32, 64, 128, 256, 512, 512, 512), 3: (32, 64, 128, 256, 320, 320)}
PATCH = {2: (512, 512), 3: (128, 128, 128)}
CTX = {2: 1.4, 3: 1.3}


def now():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def save_atomic(obj, path, as_json=False):
    tmp = Path(str(path) + f".tmp{os.getpid()}")
    if as_json:
        tmp.write_text(json.dumps(obj, indent=1, default=float))
    else:
        torch.save(obj, tmp)
    os.replace(tmp, path)


def git_state():
    """the commit of the repository and whether linc/segment differs from it"""
    import subprocess
    try:
        run = lambda *c: subprocess.run(["git", "-C", str(ROOT)] + list(c), capture_output=True, text=True, timeout=30).stdout.strip()  # noqa: E731
        dirty = run("status", "--porcelain", "--", "linc/segment")
        return dict(commit=run("rev-parse", "HEAD"), dirty=bool(dirty), dirty_files=dirty.splitlines())
    except (OSError, subprocess.SubprocessError) as ex:
        return dict(error=repr(ex))


def code_md5():
    return {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).resolve().parent.glob("*.py"))}


def dice_counts(logits, y, K):
    p = logits.argmax(1)
    out = []
    for c in range(1, K):
        a, b = p == c, y == c
        out.append([int((a & b).sum()), int(a.sum()), int(b.sum())])
    return np.array(out)


def work_root():
    from ..paths import work_root as w
    return w()


def plan(task, a, own):
    """what a run would read and how long it took on the record's GPU, scaled by epochs x iterations x batch; trains nothing"""
    files = [q for c in task["cases"] for q in (c["image"] if isinstance(c["image"], list) else [c["image"]]) + [c["label"]]
             + ([c["mask"]] if "mask" in c else [])]
    missing = [q for q in files if not os.path.exists(q)]
    subs = sorted({c["subject"] for c in task["cases"]})
    print(f"{task['task']}: {len(task['cases'])} cases from {len(subs)} brains ({task['dim']}D, {task.get('channels', 1)} channel(s)); "
          f"target spacing {task.get('target_spacing') or 'the images own'}; {a.epochs} epochs x {a.iters} iterations x batch {a.batch}")
    print(f"input files: {len(files) - len(missing)} of {len(files)} found" + (f"; missing, e.g. {missing[:3]}" if missing else ""))
    ref = 200 * 250 * own.get("batch", 2)                               # the run of record's epochs x iterations x batch
    scale = a.epochs * a.iters * a.batch / ref
    for k in ([a.fold] if a.fold else list(task["folds"]) + ["all"]):
        tr, va = data.split(task, k)
        r = task.get("record_runs", {}).get(str(k))
        est = f"run of record {r['hours']} h on one {r['gpu']}" + (f" (x {scale:.3g} here: {r['hours'] * scale:.1f} h)" if scale != 1 else "") \
            if r else "no run of record: time one epoch"
        held = f"held out {', '.join(task['folds'][str(k)])}" if k != "all" else "every brain"
        print(f"  fold {k}: {len(tr)} training / {len(va)} held-out cases ({held}); {est}")
    return 1 if missing else 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="linc-segment train", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, type=Path); ap.add_argument("--fold")
    ap.add_argument("--plan", action="store_true", help="print the cases, folds, missing files and expected GPU hours; train nothing")
    ap.add_argument("--out", type=Path); ap.add_argument("--cache", type=Path)
    ap.add_argument("--epochs", type=int, default=200); ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--batch", type=int); ap.add_argument("--patch", type=int, nargs="+")
    ap.add_argument("--features", type=int, nargs="+"); ap.add_argument("--no-deep-supervision", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-2); ap.add_argument("--weight-decay", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int)
    ap.add_argument("--val-patches", type=int, default=64); ap.add_argument("--device", default="cuda")
    ap.add_argument("--mirror", choices=("on", "off"), default="on", help="mirroring at inference (the model's spec default)")
    ap.add_argument("--mem-fraction", type=float, default=None, help="cap this process's share of the GPU memory (two jobs per card)")
    ap.add_argument("--postprocess", choices=("task", "none", "largest_component", "largest_cc_fill"), default="task",
                    help="task = the task's own postprocess block (none if it has none)")
    ap.add_argument("--max-hours", type=float, default=None, help="stop (resumable) after this wall time")
    ap.add_argument("--ckpt-every", type=int, default=5, help="write last.pt every this many epochs (and at the end)")
    ap.add_argument("--remap", nargs="*", default=[], help="SRC=DST: read the task's files from DST instead (a local copy)")
    ap.add_argument("--normalization", choices=("zscore", "zscore_nonzero"), help="overrides the task's normalization")
    ap.add_argument("--target-spacing", nargs="+", help="resample to this voxel size (mm) for training and prediction, or 'median' "
                    "(the per-axis median over the task's cases, nnU-Net's rule); overrides the task's target_spacing")
    a = ap.parse_args(argv)
    if a.fold is None and not a.plan:
        ap.error("give --fold <0-4|all> (or --plan)")
    pairs = [tuple(r.split("=", 1)) for r in a.remap]
    try:
        task = data.load_task(a.task, pairs)
    except (OSError, RuntimeError) as ex:
        sys.exit(f"{a.task}: {ex}")
    own = task.get("train", {})                                       # the task's own arguments where they differ from the defaults
    a.batch = a.batch or own.get("batch", 2); a.workers = own.get("workers", 6) if a.workers is None else a.workers
    if a.plan:
        return plan(task, a, own)
    if a.target_spacing:
        task["target_spacing"] = "median" if a.target_spacing == ["median"] else [float(v) for v in a.target_spacing]
    if task.get("target_spacing") == "median":
        task["target_spacing_rule"] = "median per axis over the task's cases (nnU-Net)"
        task["target_spacing"] = data.median_spacing(task)
    if a.normalization:
        task["normalization"] = a.normalization
    name, dim, K = task["task"], int(task["dim"]), len(task["classes"])
    out = a.out or work_root() / "segment" / "runs" / name / f"fold_{a.fold}"; out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE").exists():
        print(out, "is DONE"); return 0
    ts_tag = "native" if not task.get("target_spacing") else "ts_" + "_".join(f"{v:.4f}" for v in task["target_spacing"])
    ts_tag += "_lab2_" + hashlib.md5(json.dumps([[c["image"], c["label"]] for c in task["cases"]]).encode()).hexdigest()[:8]
    # _lab2: io.resample_label's rule (nearest-exact; block mean >= 0.5 at even integer reductions); the hash of the case files makes
    # a task repointed to restaged data build its own cache (cache_case keeps any case whose meta.json exists)
    cache = a.cache or work_root() / "segment" / "cache" / name / ts_tag
    patch = list(a.patch or task.get("patch") or PATCH[dim])
    feats = tuple(a.features or task.get("features") or FEATURES[dim])
    aug = dict(augment.DEFAULTS, **task.get("augment", {}))
    tr, va = data.split(task, a.fold)
    t0 = time.time()
    whole = bool(task.get("whole_image"))
    if whole:                                                          # photographs: read directly by the loader workers, no cache
        assert not task.get("target_spacing"), "whole_image tasks are trained at their images' own grid"
        shapes = {}
        C = int(task["channels"])
    else:
        shapes = {c["id"]: data.cache_case(c, dim, cache, task.get("spacing"), task.get("target_spacing"), task["normalization"])
                  for c in tr + va}
        nch = {m["channels"] for m in shapes.values()}; assert len(nch) == 1, f"channel counts differ: {nch}"
        C = nch.pop()
    print(f"{name} fold {a.fold}: {len(tr)} training / {len(va)} validation cases; cache {time.time() - t0:.0f} s", flush=True)

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(a.device)
    if a.mem_fraction and dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(a.mem_fraction, dev.index if dev.index is not None else torch.cuda.current_device())
    net = UNet(dim, C, K, feats, deep_supervision=not a.no_deep_supervision).to(dev)
    div = net.divisor(); assert all(p % d == 0 for p, d in zip(patch, div)), f"patch {patch} not divisible by {div}"
    opt = torch.optim.SGD(net.parameters(), lr=a.lr, momentum=0.99, nesterov=True, weight_decay=a.weight_decay)
    ctx = [p + 2 * int(round((CTX[dim] - 1) * p / 2)) for p in patch]
    recipe = dict(network=net.to_spec(), patch=patch, context=ctx, batch=a.batch, epochs=a.epochs, iterations_per_epoch=a.iters,
                  optimizer=dict(name="SGD", nesterov=True, momentum=0.99, lr=a.lr, weight_decay=a.weight_decay,
                                 schedule="poly (1 - epoch/epochs)^0.9", grad_norm_clip=12.0, autocast="bfloat16"),
                  loss="Dice (foreground classes, batch) + cross-entropy; deep supervision weights 1, 1/2, 1/4, ... (lowest 0)",
                  normalization=task["normalization"], foreground_oversampling="1/3 of the patches centered on a foreground voxel",
                  augmentation=aug, seed=a.seed,
                  inference=dict(step=0.5, mirror=a.mirror == "on",
                                 postprocess=task.get("postprocess") if a.postprocess == "task" else a.postprocess))
    rec = dict(task=name, fold=str(a.fold), task_spec=str(a.task), started=now(), folds=task.get("folds"), fold_rule=task.get("fold_rule"),
               target_spacing=task.get("target_spacing"), target_spacing_rule=task.get("target_spacing_rule"),
               data=dict(train_cases=len(tr), train_subjects=sorted({c["subject"] for c in tr}),
                         val_cases=len(va), val_subjects=sorted({c["subject"] for c in va}),
                         train_voxels=int(sum(np.prod(shapes[c["id"]]["shape"]) for c in tr)) if shapes else None, whole_image=whole),
               recipe=recipe, epochs=[], code_md5=code_md5(), git=git_state(),
               hardware=dict(host=socket.gethostname().split(".")[0], gpu=torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
                             torch=torch.__version__, cuda=torch.version.cuda, python=platform.python_version()))
    start = 0
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev, weights_only=False)
        net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"]); start = ck["epoch"] + 1
        rec["epochs"] = ck["record"]["epochs"]; rec["started"] = ck["record"]["started"]
        rec.setdefault("resumed", []).extend(ck["record"].get("resumed", []) + [dict(epoch=start, at=now(), host=rec["hardware"]["host"])])
        print(f"resumed at epoch {start}", flush=True)
    ds = (data.WholeImages(tr, dim, ctx, task["normalization"], task.get("spacing"), seed=a.seed * 10007 + start) if whole
          else data.Patches(tr, cache, ctx, seed=a.seed * 10007 + start))
    dl = iter(torch.utils.data.DataLoader(ds, batch_size=a.batch, num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0,
                                          prefetch_factor=4 if a.workers else None))
    vpatch = []
    if va:
        vds = iter(data.WholeImages(va, dim, ctx, task["normalization"], task.get("spacing"), seed=12345) if whole
                   else data.Patches(va, cache, ctx, seed=12345))
        vpatch = [next(vds) for _ in range(a.val_patches)]
    g = torch.Generator(device=dev); g.manual_seed(a.seed * 7919 + start)
    t_run = time.time()
    for ep in range(start, a.epochs):
        te = time.time()
        for pg in opt.param_groups:
            pg["lr"] = a.lr * (1 - ep / a.epochs) ** 0.9
        net.train(); tot = 0.0
        for it in range(a.iters):
            x, y = next(dl)
            x, y = augment.augment(x.to(dev, non_blocking=True), y.to(dev, non_blocking=True), patch, g, aug)
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                outs = net(x)
            lo = losses.loss([o.float() for o in outs] if isinstance(outs, list) else outs.float(), y)
            opt.zero_grad(set_to_none=True); lo.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 12.0); opt.step()
            tot += float(lo)
        row = dict(epoch=ep, loss=tot / a.iters, lr=opt.param_groups[0]["lr"], s=round(time.time() - te, 1))
        if vpatch:
            net.eval(); cnt = 0
            with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16):
                for i in range(0, len(vpatch), a.batch):
                    xb = torch.stack([v[0] for v in vpatch[i:i + a.batch]]).to(dev); yb = torch.stack([v[1] for v in vpatch[i:i + a.batch]]).to(dev)
                    xb, yb = augment.center_crop(xb, yb, patch)
                    cnt = cnt + dice_counts(net(xb).float(), yb, K)
            row["val_pseudo_dice"] = [float(2 * t / max(p + r, 1)) for t, p, r in cnt]
        rec["epochs"].append(row)
        stop = a.max_hours and (time.time() - t_run) / 3600 > a.max_hours and ep + 1 < a.epochs
        if (ep + 1) % a.ckpt_every == 0 or ep + 1 == a.epochs or stop:
            save_atomic(dict(net=net.state_dict(), opt=opt.state_dict(), epoch=ep, record=rec), out / "last.pt")
        save_atomic(rec, out / "record.json", as_json=True)
        print(f"ep {ep + 1}/{a.epochs} loss {row['loss']:.4f} " + (f"val {row.get('val_pseudo_dice')} " if vpatch else "") +
              f"lr {row['lr']:.2e} {row['s']} s", flush=True)
        if stop:
            print("max-hours reached: stopping (resumable)", flush=True); return 3
    # the model as shipped: state_dict + spec
    spec = dict(task=name, fold=str(a.fold), net=net.to_spec(), patch=patch, classes=task["classes"], channels=C,
                normalization=task["normalization"], spacing=task.get("spacing"), target_spacing=task.get("target_spacing"),
                whole_image=whole, native_downsample=task.get("native_downsample"),
                inference=recipe["inference"],
                trained=dict(epochs=a.epochs, iterations_per_epoch=a.iters, batch=a.batch, train_cases=len(tr),
                             train_subjects=len(rec["data"]["train_subjects"]), finished=now()))
    save_atomic(net.state_dict(), out / "model.pt"); save_atomic(spec, out / "spec.json", as_json=True)
    rec["training_wall_s"] = round(sum(r["s"] for r in rec["epochs"]), 1)
    # held-out cases through the inference path itself
    if va:
        pnet, pspec = predict.load(out, a.device)
        rows = {}
        pp = pspec["inference"].get("postprocess")
        has_pp = pp not in (None, "none")
        rows_raw = {}
        for c in va:
            mk = c.get(pp.get("mask_key", "mask")) if isinstance(pp, dict) and pp.get("kind") == "tubes_left_hemisphere" else None
            mask, meta, s, mraw = predict.predict_case(pnet, pspec, c["image"], a.device, case_mask=mk, raw=True)
            ext = ".nii.gz" if meta["kind"] == "nifti" else ".png"
            io.write_mask(mask, meta, out / "val_pred" / f"{c['id']}{ext}")
            ref, _ = io.read_label(c["label"], dim)
            sp = meta["spacing"] if meta["kind"] == "nifti" else (c.get("spacing") or meta["spacing"])
            rows[c["id"]] = dict(subject=c["subject"], counted=c.get("counted", True), seconds=round(s, 1),
                                 **metrics.score(mask, ref, sp, range(1, K)))
            if has_pp:
                io.write_mask(mraw, meta, out / "val_pred_raw" / f"{c['id']}{ext}")
                rows_raw[c["id"]] = dict(subject=c["subject"], counted=c.get("counted", True),
                                         **metrics.score(mraw, ref, sp, range(1, K)))
            print(c["id"], json.dumps(rows[c["id"]]), flush=True)
        rec["validation"] = rows                                          # through the spec's inference (with its post-processing)
        if has_pp:
            rec["validation_raw"] = rows_raw
            rec["validation_raw_summary"] = metrics.summarize(rows_raw, K)
        rec["validation_summary"] = metrics.summarize(rows, K)                 # the counted cases (counted: false = flagged)
        rec["validation_summary_all"] = metrics.summarize(rows, K, counted_only=False)
    rec["finished"] = now()
    save_atomic(rec, out / "record.json", as_json=True)
    (out / "DONE").write_text(now() + "\n")
    print("DONE", out, json.dumps(rec.get("validation_summary", {})), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
