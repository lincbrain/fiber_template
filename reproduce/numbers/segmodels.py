#!/usr/bin/env python
"""The record of one segmentation model (Supplementary S5; Table S8): segmodels_<model>_2026-11, from the products of its training.

You need
  env/deep.yml (torch counts the parameters; CPU, up to 16 processes; bf_mask2d's 16,078 photographs take a few minutes). Under configs/products_segmentation.yaml's
  `segmodels` folder: tasks/<model>.json (cases, folds, the earlier network's held-out predictions, card notes) and the six runs runs/<model>/fold_{0..4,all}/
  (record.json with the trainer's held-out scores, spec.json, model.pt, val_pred/; predict_separate/ of the fold of the predictor check; for bf_mask2d, every fold's
  validation_fill/, the held-out photographs predicted again with the shipped post-processing). The cases' images and labels and the earlier network's masks are the
  paths the task JSON records, re-rooted through configs/path_placeholders*.yaml.

What it does
  1. gathers each held-out case's scores from the runs (the Dice coefficient, the surface distances); cases with "counted": false are listed, not pooled
  2. scores the earlier network's predictions with linc.segment.metrics on the label grid, because a comparison needs the same measures on the same cases
  3. for the marker tubes, computes the centerline Dice of both networks, because one wall voxel lowers the Dice coefficient of a thin tube
  4. summarizes them pooled and per brain, on the brains held out by both networks, by position in the stack and for empty labels. Nothing is trained or predicted.

You get
  $LINC_NUMBERS/segmodels_<model>_2026-11.json (and its provenance); reproduce/check_numbers.py compares it with the paper's record.

    python reproduce/numbers/segmodels_<model>_2026-11.py     (one script per model)
"""
import argparse
import hashlib
import os
import inspect
import json
import tempfile
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from linc import paths, provenance  # noqa: E402
from linc.segment import data as D, io, metrics, predict  # noqa: E402

INCUMBENT = {
    "mri_brainmask": ("incumbent_dataset521", "Dataset521_MRImaskV2 (3D nnU-Net, the recipe of Dataset520 under these folds)",
                      "Dataset520, the model in use, was trained on every brain here (in-sample everywhere); Dataset521 is its recipe "
                      "trained under these five folds, so its held-out predictions are the fair comparison"),
    "bf_mask3d": ("incumbent_dataset506", "Dataset506_BFmask5 (3D nnU-Net ResEnc-M, fold 0)", None),
    "bf_holes": ("incumbent_dataset508", "Dataset508_holes3 (3D nnU-Net ResEnc-M, fold 0)", None),
    "bf_mask2d": ("incumbent_dataset500", "Dataset500_SLICE (2D nnU-Net, fold 0)", None),
}
FOLDS_ALL = "all"
MARGIN = 0.01        # the shipping rule: a paired median Dice difference to the earlier network >= -MARGIN
# the decided reading of a comparison in which the new model is below the earlier network on most brains (3 Nov 2026)
COMPARISON_NOTE = {
    "bf_mask3d": ("consistent with its one-voxel extension at anterior-facing surfaces, where the masks of record (made from the "
                  "earlier network's own predictions and then edited) follow the cut face"),
}
# post-processing decided after training and applied post hoc: every fold's held-out cases re-predicted with it (scored_by); the runs'
# own validation (without it) is kept as each case's "raw"; the shipped spec.json carries the rule
POSTHOC = {
    "bf_mask2d": dict(postprocess={"kind": "fill_holes"}, scored_by="models_wrapup/post/t2_fill_score.py", dir="validation_fill",
                      main="label", outline="label_filled", without="native_raw",
                      empty_label_rule=True, position_first=80,           # both fixed on 3 Nov 2026
                      position_band="first 80 photographs by stack index, ten label sections",
                      decision="decided on 3 Nov 2026, after training: the pipeline's rule of record (code/tasks.py normalize_masks: "
                               "binary_fill_holes on every mask of the earlier network at native size); reconstruction has only read filled masks"),
}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def q(v, p):
    return float(np.percentile(v, p)) if len(v) else None


def med(v):
    return float(np.median(v)) if len(v) else None


def pooled(rows):
    d = [r["dice"] for r in rows]; h = [r["hd95_mm"] for r in rows if r["hd95_mm"] is not None]
    return dict(dice_median=med(d), dice_mean=float(np.mean(d)) if d else None, dice_min=float(min(d)) if d else None, dice_q10=q(d, 10),
                hd95_median=med(h), hd95_q90=q(h, 90),
                assd_median=med([r["assd_mm"] for r in rows if r["assd_mm"] is not None]),
                n_cases=len(rows), n_brains=len({r["subject"] for r in rows}))


CLDICE_TASKS = ("bf_holes",)                # thin tubular labels: Dice falls with one wall voxel; centerline Dice follows the tube's course
CLDICE_RULE = ("centerline Dice (clDice; Shit et al. 2021): 2 Tprec Tsens / (Tprec + Tsens), Tprec = |skel(P) & L| / |skel(P)|, "
               "Tsens = |skel(L) & P| / |skel(L)|; skeletons by skimage's 3D skeletonize on the label grid")


def cldice(pred, label):
    """centerline Dice of two binary volumes on one grid (CLDICE_RULE)"""
    from skimage.morphology import skeletonize
    P, L = pred > 0, label > 0
    assert P.shape == L.shape, (P.shape, L.shape)
    sp, sl = skeletonize(P) > 0, skeletonize(L) > 0
    tp = float((sp & L).sum() / sp.sum()) if sp.any() else 0.0
    ts = float((sl & P).sum() / sl.sum()) if sl.any() else 0.0
    return dict(cldice=2 * tp * ts / (tp + ts) if tp + ts else 0.0, tprec=tp, tsens=ts)


def _cldice_job(job):
    cid, pred, label, dim = job
    return cid, cldice(io.read_label(str(pred), dim)[0], io.read_label(str(label), dim)[0])


def parameters(model_pt):
    import torch
    sd = torch.load(model_pt, map_location="cpu", weights_only=True)
    return int(sum(v.numel() for k, v in sd.items() if not k.endswith("num_batches_tracked")))


def rerooted_task(tp):
    """the task JSON with every recorded path re-rooted on this machine (paths.rerooted): a copy in a temporary folder, for
    linc.segment.data.load_task (which reads the dataset folder it names); the record keeps the md5 of the file as released"""
    raw = json.load(open(tp))
    if raw.get("nnunet_raw"):
        raw["nnunet_raw"] = paths.rerooted(raw["nnunet_raw"])
    for c in raw.get("cases") or []:
        c["image"] = [paths.rerooted(p) for p in c["image"]] if isinstance(c["image"], list) else paths.rerooted(c["image"])
        c["label"] = paths.rerooted(c["label"])
    for key in [k for k in raw if k.startswith("incumbent_") and isinstance(raw[k], dict)] + ["corrected_labels"]:
        for e in (raw.get(key) or {}).values():
            if isinstance(e, dict) and e.get("path"):
                e["path"] = paths.rerooted(e["path"])
    d = Path(tempfile.mkdtemp(prefix="segmodels_")); local = d / tp.name
    local.write_text(json.dumps(raw)); task = D.load_task(str(local))
    return raw, task


def main(model):
    class a:                                                              # the record's own model; no options
        task, partial = model, False
    MW = paths.products("products_segmentation")["segmodels"]; ROOT = paths.products_root()
    tp = MW / "tasks" / f"{a.task}.json"
    raw, task = rerooted_task(tp)
    dim, K = int(task["dim"]), len(task["classes"])
    cases = {c["id"]: c for c in task["cases"]}
    folds = {str(k): v for k, v in task["folds"].items()}
    fold_of = {s: k for k, ss in folds.items() for s in ss}
    runs = {k: MW / "runs" / a.task / f"fold_{k}" for k in list(folds) + [FOLDS_ALL]}
    done = {k: (r / "DONE").exists() for k, r in runs.items()}
    if not a.partial:
        assert all(done.values()), f"unfinished runs: {[k for k, d in done.items() if not d]}"
    rec = {k: json.load(open(r / "record.json")) for k, r in runs.items() if (r / "record.json").exists()}
    inputs = [tp] + [r / "record.json" for k, r in runs.items() if k in rec]
    ref_run = FOLDS_ALL if done.get(FOLDS_ALL) else next((k for k in runs if done.get(k)), None)
    spec = json.load(open(runs[ref_run] / "spec.json")) if ref_run else {}

    # ---- input / output
    out = dict(task=a.task, task_json_md5=md5(tp))                       # the task JSON as read now (card_notes are added after training)
    notes = raw.get("card_notes") or []                                    # the per-model findings, one sentence each
    assert isinstance(notes, list) and all(isinstance(s, str) and s.strip() for s in notes), "card_notes: a list of sentences"
    out["card_notes"] = [s.strip() for s in notes]
    out["input"] = dict(dim=dim, channels=int(task["channels"]), normalization=task["normalization"],
                        target_spacing=spec.get("target_spacing"),
                        spacing_rule=("median per axis over the task's cases (nnU-Net)" if a.task in ("bf_mask3d", "bf_holes") else
                                      "fixed 0.5 mm isotropic" if a.task == "mri_brainmask" else "the photographs at /4 (no resampling)"),
                        whole_image=bool(raw.get("whole_image")), native_downsample=raw.get("native_downsample"),
                        patch=spec.get("patch"))
    out["output"] = dict(classes=task["classes"], grid="the input image's own grid (class probabilities resampled back linearly)")
    out["postprocess"] = spec.get("inference", {}).get("postprocess")
    inf = spec.get("inference", {})                                       # linc.segment.predict.predict_case, as the shipped spec sets it
    out["inference"] = dict(step=inf.get("step"), mirror=inf.get("mirror"), mirror_copies=2 ** dim if inf.get("mirror") else 1,
                            gaussian_sigma_of_patch=inspect.signature(predict.gaussian).parameters["sigma_scale"].default,
                            probabilities_back=("linear, to the input's grid" if spec.get("target_spacing") else
                                                "linear, to native size (native photographs only)" if raw.get("whole_image") else None),
                            postprocess=inf.get("postprocess"),
                            identical_across_runs=len({json.dumps(r.get("recipe", {}).get("inference"), sort_keys=True)
                                                       for r in rec.values()}) == 1)
    if not a.partial:
        assert out["inference"]["identical_across_runs"], {k: r.get("recipe", {}).get("inference") for k, r in rec.items()}
    ph = POSTHOC.get(a.task)
    if ph:
        out["postprocess"] = out["inference"]["postprocess"] = ph["postprocess"]
        out["inference"]["postprocess_posthoc"] = dict(ph, runs_spec_postprocess=inf.get("postprocess"))
        if not a.partial:                                                 # the shipped spec must run: linc.segment.predict knows the rule
            predict.postprocess_mask(np.zeros([4] * dim, np.uint8), ph["postprocess"], K)

    # ---- data
    per_brain = {}
    for c in task["cases"]:
        b = per_brain.setdefault(c["subject"], dict(n_images=0, label_kind=set()))
        b["n_images"] += 1; b["label_kind"].add(c.get("label_kind"))
    for b in per_brain.values():
        b["label_kind"] = sorted(x for x in b["label_kind"] if x)
    prov = {}
    for c in task["cases"]:
        s = "; ".join(x.strip() for x in (c.get("provenance") or "").split(";")[:2] if "/" not in x)[:160]   # the description, not the per-case path
        prov[s] = prov.get(s, 0) + 1
    dat = dict(brains=len(per_brain), images_total=len(cases), images_counted=sum(1 for c in task["cases"] if c.get("counted", True)),
               per_brain=per_brain, label_kind={k: sum(1 for c in task["cases"] if c.get("label_kind") == k)
                                                for k in sorted({c.get("label_kind") for c in task["cases"] if c.get("label_kind")})},
               label_provenance=[dict(text=s, n=n) for s, n in sorted(prov.items(), key=lambda x: -x[1]) if s])
    if any("modality" in c for c in task["cases"]):
        dat["per_modality"] = {m: sum(1 for c in task["cases"] if c.get("modality") == m) for m in sorted({c.get("modality") for c in task["cases"]})}
    if dim == 2:
        dat["photographs"] = len(cases)
        dat["label_quantization_mm"] = raw.get("label_quantization_mm")
    out["data"] = dat

    # ---- recipe, folds
    r0 = rec.get(ref_run, {})
    rcp = r0.get("recipe", {})
    out["recipe"] = {k: rcp.get(k) for k in ("network", "patch", "batch", "epochs", "iterations_per_epoch", "optimizer", "loss",
                                             "augmentation", "normalization", "seed", "foreground_oversampling", "inference")}
    if ref_run and (runs[ref_run] / "model.pt").exists():
        out["recipe"]["parameters"] = parameters(runs[ref_run] / "model.pt")
    out["folds"] = folds
    out["fold_rule"] = raw.get("fold_rule") or raw.get("folds_note")

    # ---- cross-validation (each fold's held-out cases, scored through the predictor in the run)
    per_case = []
    for k in folds:
        v = rec.get(k, {}).get("validation") or {}
        vr = rec.get(k, {}).get("validation_raw") or {}
        vo = {}
        if ph:                                                            # the shipped rule's scores; the trainer's become "raw"
            f = runs[k] / ph["dir"] / "scores.json"
            if f.exists():
                assert json.load(open(f.parent / "meta.json"))["model_md5"] == md5(runs[k] / "model.pt"), f"{f}: other weights"
                vo = json.load(open(f)); inputs.append(f)
                assert set(vo) == set(v), f"{f}: cases differ from the trainer's"
            else:
                assert a.partial, f"{f} missing ({ph['scored_by']})"
            vr, v = v, {cid: r[ph["main"]] for cid, r in vo.items()}
        for cid, row in v.items():
            c = cases[cid]; m = row["1"]
            pc = dict(case=cid, subject=c["subject"], fold=k, dice=m["dice"], hd95_mm=m["hd95_mm"], assd_mm=m["assd_mm"],
                      counted=c.get("counted", True), label_kind=c.get("label_kind"))
            if "modality" in c:
                pc["modality"] = c["modality"]
            if dim == 2:
                pc.update(components=m.get("components_pred"), components_ref=m.get("components_ref"), fp_area_mm2=m.get("fp_area_mm2"),
                          vox_pred=m["vox_pred"], label_empty=m["vox_ref"] == 0, stack_index=int(cid.rsplit("_", 1)[1]))
            if cid in vr:
                pc["raw"] = dict(dice=vr[cid]["1"]["dice"], hd95_mm=vr[cid]["1"]["hd95_mm"])
            if cid in vo:
                for blk in ("outline", "without"):
                    pc[blk] = dict(dice=vo[cid][ph[blk]]["1"]["dice"], hd95_mm=vo[cid][ph[blk]]["1"]["hd95_mm"])
            per_case.append(pc)
    counted = [r for r in per_case if r["counted"]]
    empty_rule = bool(ph and ph.get("empty_label_rule"))                  # an empty label: listed, not pooled (its Dice measures the label)
    empty = [r for r in counted if empty_rule and r["label_empty"]]
    counted = [r for r in counted if not (empty_rule and r["label_empty"])]
    cv = dict(per_case=per_case, pooled=pooled(counted),
              per_brain={b: dict(dice_median=med([r["dice"] for r in counted if r["subject"] == b]),
                                 hd95_median=med([r["hd95_mm"] for r in counted if r["subject"] == b and r["hd95_mm"] is not None]),
                                 n=sum(1 for r in counted if r["subject"] == b)) for b in sorted({r["subject"] for r in counted})},
              not_pooled=[r["case"] for r in per_case if not r["counted"]],
              folds_scored=sorted({r["fold"] for r in per_case}))
    for blk in ("raw", "outline", "without"):                             # without: the shipped path minus the post-processing
        if any(blk in r for r in counted):
            cv[f"pooled_{blk}"] = pooled([dict(r, **r[blk], assd_mm=None) for r in counted if blk in r])
    corr = raw.get("corrected_labels") or {}                              # masks of record corrected after training (a separate block)
    if corr:
        cv["corrected_labels"] = {}
        for cid, e in corr.items():
            r0 = next((r for r in per_case if r["case"] == cid), None)
            if r0 is None:
                assert a.partial, f"corrected label for {cid}: no held-out prediction"
                continue
            pf = next((runs[r0["fold"]] / (ph["dir"] + "/masks" if ph else "val_pred")).glob(f"{cid}.*"))   # the shipped mask
            pred, _ = io.read_label(str(pf), dim); new, meta = io.read_label(e["path"], dim); inputs.append(Path(e["path"]))
            pred = (pred > 0).astype(np.uint8)                            # the post-hoc masks are saved 0/255 for viewing
            assert new.shape == pred.shape, (cid, new.shape, pred.shape)
            m = metrics.score(pred, new, meta["spacing"] if meta["kind"] == "nifti" else cases[cid].get("spacing"), [1])["1"]
            cv["corrected_labels"][cid] = dict(fold=r0["fold"], label=os.path.relpath(e["path"], ROOT), reason=e.get("reason"),
                                               label_md5=md5(e["path"]),
                                               as_trained=dict(dice=r0["dice"], hd95_mm=r0["hd95_mm"]),
                                               against_corrected=dict(dice=m["dice"], hd95_mm=m["hd95_mm"], assd_mm=m["assd_mm"]))
    if cv.get("corrected_labels"):                                       # per brain: as trained vs against the corrected labels
        cl = cv["corrected_labels"]
        cv["corrected_labels_summary"] = {
            b: dict(n=len(rr), as_trained=dict(dice_median=med([x["as_trained"]["dice"] for x in rr]),
                                               hd95_median=med([x["as_trained"]["hd95_mm"] for x in rr if x["as_trained"]["hd95_mm"] is not None])),
                    against_corrected=dict(dice_median=med([x["against_corrected"]["dice"] for x in rr]),
                                           hd95_median=med([x["against_corrected"]["hd95_mm"] for x in rr if x["against_corrected"]["hd95_mm"] is not None])),
                    reason=rr[0]["reason"], folds=sorted({x["fold"] for x in rr}))
            for b in sorted({cases[c]["subject"] for c in cl}) for rr in [[cl[c] for c in cl if cases[c]["subject"] == b]]}
    cv["worst_per_fold"] = {k: dict(case=w["case"], subject=w["subject"], dice=w["dice"], hd95_mm=w["hd95_mm"])
                            for k in folds for w in [min((r for r in counted if r["fold"] == k), key=lambda r: r["dice"], default=None)] if w}
    if any("modality" in r for r in per_case):
        cv["per_modality"] = {m: pooled([r for r in counted if r.get("modality") == m]) for m in sorted({r.get("modality") for r in counted})}
    cv["per_brain_summary"] = brain_summary(cv["per_brain"])
    if a.task in CLDICE_TASKS:                                            # the masks as shipped: val_pred, after the task's rule
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(min(16, max(1, len(counted)))) as ex:
            cl = dict(ex.map(_cldice_job, [(r["case"], next((runs[r["fold"]] / "val_pred").glob(f"{r['case']}.*")), cases[r["case"]]["label"], dim)
                                           for r in counted]))
        for r in counted:
            r.update(cl[r["case"]])
        for b, v in cv["per_brain"].items():
            v["cldice"] = med([r["cldice"] for r in counted if r["subject"] == b])
        c = [v["cldice"] for v in cv["per_brain"].values()]
        cv["per_brain_summary"].update(cldice_median=med(c), cldice_min=min(c), cldice_max=max(c))
        cv["cldice_rule"] = CLDICE_RULE
    for blk in ("outline", "without"):                                    # per brain, each brain counting once
        if any(blk in r for r in counted):
            cv[f"per_brain_{blk}"] = {b: dict(dice_median=med([r[blk]["dice"] for r in rr]),
                                              hd95_median=med([r[blk]["hd95_mm"] for r in rr if r[blk]["hd95_mm"] is not None]), n=len(rr))
                                      for b in sorted({r["subject"] for r in counted})
                                      for rr in [[r for r in counted if r["subject"] == b and blk in r]] if rr}
    for blk in ("outline", "without"):
        if cv.get(f"per_brain_{blk}"):
            cv[f"per_brain_{blk}_summary"] = brain_summary(cv[f"per_brain_{blk}"])
    if empty_rule:
        cv["empty_label"] = empty_block(empty)
        ks = {}                                                           # where they sit: photographs from the nearer end of the stack
        for r in per_case:
            ks.setdefault(r["subject"], []).append(r["stack_index"])
        ks = {b: sorted(v) for b, v in ks.items()}
        cv["empty_label"]["max_from_nearer_end"] = max((min(ks[r["subject"]].index(r["stack_index"]),
                                                            len(ks[r["subject"]]) - 1 - ks[r["subject"]].index(r["stack_index"]))
                                                        for r in empty), default=None)
        cv["position"] = dict(first=ph["position_first"], band=ph["position_band"],
                              rule="the first photographs of each stack by index (the case number) vs the rest",
                              label=position_block(counted, lambda r: r["dice"], ph["position_first"]),
                              **({"outline": position_block(counted, lambda r: r["outline"]["dice"], ph["position_first"])}
                                 if all("outline" in r for r in counted) else {}))
    out["cv"] = cv

    # ---- incumbent, scored here on the same cases with the same metrics
    key, name, caveat = INCUMBENT[a.task]
    inc = raw.get(key) or {}
    in_sample, jobs, no_pred = set(), [], {}
    for cid, e in inc.items():
        p = e.get("path") if isinstance(e, dict) else e
        sub = cases[cid]["subject"] if cid in cases else cid
        if not p:                                                         # in the incumbent's training set, or no prediction of its own
            why = (e.get("reason") if isinstance(e, dict) else None) or ""
            if "in-sample" in why or not why:
                in_sample.add(sub)
            else:
                no_pred[cid] = why
            continue
        if cid not in cases:
            continue
        jobs.append((cid, cases[cid], p, dim, K, bool(ph))); inputs.append(Path(p))
    from concurrent.futures import ProcessPoolExecutor                    # T2: 16,078 photographs (~40 min in one process)
    with ProcessPoolExecutor(min(16, max(1, len(jobs)))) as ex:
        rows = list(ex.map(_score_incumbent, jobs, chunksize=max(1, len(jobs) // 64)))   # small tasks: one case per worker
    new_by = {r["case"]: r for r in counted}
    fair = [r for r in rows if r["counted"] and r["case"] in new_by]
    out["incumbent"] = dict(
        name=name, in_sample_brains=sorted(in_sample), no_prediction=no_pred,
        fair_subset=dict(brains=sorted({r["subject"] for r in fair}), n_cases=len(fair),
                         incumbent=dict(dice_median=med([r["dice"] for r in fair]), hd95_median=med([r["hd95_mm"] for r in fair if r["hd95_mm"] is not None])),
                         new=dict(dice_median=med([new_by[r["case"]]["dice"] for r in fair]),
                                  hd95_median=med([new_by[r["case"]]["hd95_mm"] for r in fair if new_by[r["case"]]["hd95_mm"] is not None])),
                         paired_dice_diff_median=med([new_by[r["case"]]["dice"] - r["dice"] for r in fair]),
                         new_higher=sum(new_by[r["case"]]["dice"] > r["dice"] for r in fair)),
        per_case=rows, caveat=caveat or raw.get("incumbent_note"))
    out["incumbent"]["fair_subset"]["per_brain"] = fair_per_brain(fair, new_by, lambda x: x["dice"])
    out["incumbent"]["fair_subset"]["per_brain_summary"] = fair_summary(out["incumbent"]["fair_subset"]["per_brain"])
    fs = out["incumbent"]["fair_subset"]
    if a.task in CLDICE_TASKS:                                            # the same measure on the brains held out by both networks
        p_of = {j[0]: j[2] for j in jobs}
        with ProcessPoolExecutor(min(16, max(1, len(fair)))) as ex:
            ci = dict(ex.map(_cldice_job, [(r["case"], p_of[r["case"]], cases[r["case"]]["label"], dim) for r in fair]))
        for r in fair:
            r.update(ci[r["case"]])
        pb = fs["per_brain"]
        for b, v in pb.items():
            rr = [r for r in fair if r["subject"] == b]
            v["incumbent_cldice"] = med([r["cldice"] for r in rr]); v["new_cldice"] = med([new_by[r["case"]]["cldice"] for r in rr])
        fs["per_brain_summary"].update(incumbent_cldice_median=med([v["incumbent_cldice"] for v in pb.values()]),
                                       new_cldice_median=med([v["new_cldice"] for v in pb.values()]),
                                       paired_cldice_diff_median=med([v["new_cldice"] - v["incumbent_cldice"] for v in pb.values()]),
                                       new_cldice_higher_brains=sum(v["new_cldice"] > v["incumbent_cldice"] for v in pb.values()))
    if fs["paired_dice_diff_median"] is not None:
        fs["margin"] = MARGIN; fs["within_margin"] = bool(fs["paired_dice_diff_median"] >= -MARGIN)
        if a.task in COMPARISON_NOTE:
            fs["reading"] = COMPARISON_NOTE[a.task]
    if empty_rule:                                                        # the same two rules for the incumbent
        ec = {r["case"] for r in empty}
        out["incumbent"]["empty_label"] = empty_block([r for r in rows if r["case"] in ec])
        fs = [dict(r, stack_index=int(r["case"].rsplit("_", 1)[1])) for r in fair]
        out["incumbent"]["position"] = dict(first=ph["position_first"], band=ph["position_band"], label=position_block(fs, lambda r: r["dice"], ph["position_first"]),
                                            **({"outline": position_block(fs, lambda r: r["outline"]["dice"], ph["position_first"])}
                                               if all("outline" in r for r in fs) else {}))
    if ph and fair and all("outline" in new_by[r["case"]] for r in fair):
        ob = lambda x, key: x["outline"][key]  # noqa: E731
        out["incumbent"]["fair_subset"]["outline"] = dict(
            incumbent=dict(dice_median=med([ob(r, "dice") for r in fair]), hd95_median=med([ob(r, "hd95_mm") for r in fair if ob(r, "hd95_mm") is not None])),
            new=dict(dice_median=med([ob(new_by[r["case"]], "dice") for r in fair]),
                     hd95_median=med([ob(new_by[r["case"]], "hd95_mm") for r in fair if ob(new_by[r["case"]], "hd95_mm") is not None])),
            paired_dice_diff_median=med([ob(new_by[r["case"]], "dice") - ob(r, "dice") for r in fair]),
            per_brain=fair_per_brain(fair, new_by, lambda x: x["outline"]["dice"]))
        o = out["incumbent"]["fair_subset"]["outline"]; o["per_brain_summary"] = fair_summary(o["per_brain"])

    # ---- ceiling (T1)
    if raw.get("annotator_ceiling"):                                      # the task key's name is historical: masks, not annotators
        out["ceiling"] = dict(raw["annotator_ceiling"], what="three hand-edited masks per brain, on the dwi, T2 and micro images of MF283-286, from the same procedure, compared on one grid (not three annotators)",
                              source="MF283-286, three hand-edited masks each (dwi, T2, micro), compared on one grid")

    # ---- the shipped ("all") model
    if done.get(FOLDS_ALL):
        ra = rec[FOLDS_ALL]
        out["all_model"] = dict(run=str(runs[FOLDS_ALL].relative_to(ROOT)), epochs=len(ra["epochs"]), final_loss=ra["epochs"][-1]["loss"],
                                wall_h=round(ra.get("training_wall_s", 0) / 3600, 2), gpu=ra["hardware"]["gpu"],
                                weights_md5=md5(runs[FOLDS_ALL] / "model.pt"), spec_md5=md5(runs[FOLDS_ALL] / "spec.json"),
                                parameters=out["recipe"].get("parameters"), git=ra.get("git"))

    # ---- predictor check: a separate predict process vs the trainer's own validation (masks and Dice), on one of THIS task's finished
    # folds (predict_separate/), predicted after its final weights were written; no fallback to another run
    chk = [k for k in folds if done.get(k) and (runs[k] / "predict_separate" / "scores.json").exists()]
    if not chk:
        assert a.partial, f"no predict_separate on a finished fold of {a.task}: not falling back"
        out["predictor_check"] = dict(missing=True)
    else:
        src = runs[chk[0]]
        rv = json.load(open(src / "record.json"))["validation"]; sv = json.load(open(src / "predict_separate" / "scores.json"))
        assert set(sv) == set(rv), f"{src}: predict_separate cases differ from the trainer's"
        same = 0
        for cid in rv:
            f = next((src / "val_pred").glob(f"{cid}.*")); g = next((src / "predict_separate").glob(f"{cid}.*"))
            same += int(np.array_equal(_arr(f), _arr(g)))
        out["predictor_check"] = dict(source_run=str(src.relative_to(ROOT)), fold=chk[0], n_cases=len(rv), n_identical=same,
                                      max_abs_dice_diff=float(max(abs(rv[c]["1"]["dice"] - sv[c]["1"]["dice"]) for c in rv)))
        inputs += [src / "predict_separate" / "scores.json"]

    # ---- hardware
    out["hardware"] = dict(runs={k: dict(gpu=r["hardware"]["gpu"], wall_h=round(r.get("training_wall_s", 0) / 3600, 2),
                                         resumed=[{a: b for a, b in x.items() if a != "host"} for x in r.get("resumed") or []] or None)
                                for k, r in rec.items()},
                           torch=r0.get("hardware", {}).get("torch"), cuda=r0.get("hardware", {}).get("cuda"),
                           wall_hours_total=round(sum(r.get("training_wall_s", 0) for r in rec.values()) / 3600, 2))
    # ---- code: every run trained by the same training code (linc_seg in the training tree = linc/segment); the commit may differ
    md5s = {k: r.get("code_md5") for k, r in rec.items()}
    out["code"] = dict(linc_seg_md5_identical=len({json.dumps(v, sort_keys=True) for v in md5s.values()}) == 1,
                       commits={k: (r.get("git") or {}).get("commit") for k, r in rec.items()},
                       dirty={k: (r.get("git") or {}).get("dirty") for k, r in rec.items()})
    if not a.partial:
        assert out["code"]["linc_seg_md5_identical"], f"the runs differ in their training code: {md5s}"
    out["status"] = dict(runs_done=sorted(k for k, d in done.items() if d), runs_missing=sorted(k for k, d in done.items() if not d))

    assert '"host"' not in json.dumps(out, default=_default), "a record names a machine: keep the GPU model only"
    name_rec = f"segmodels_{a.task}_2026-11"
    provenance.write_numbers(name_rec, out, [str(p) for p in inputs], md5=False, cmd=f"python reproduce/numbers/{name_rec}.py")
    p = out["cv"]["pooled"]
    print(a.task, "cv", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()}, "| incumbent fair",
          out["incumbent"]["fair_subset"]["n_cases"], out["incumbent"]["fair_subset"]["incumbent"], out["incumbent"]["fair_subset"]["new"])


def brain_summary(pb):
    """each brain counting once (the paper's Table S8): the median and range over brains of each brain's median"""
    d = [v["dice_median"] for v in pb.values() if v.get("dice_median") is not None]
    h = [v["hd95_median"] for v in pb.values() if v.get("hd95_median") is not None]
    return dict(n_brains=len(d), dice_median=med(d), dice_min=min(d) if d else None, dice_max=max(d) if d else None,
                hd95_median=med(h), hd95_min=min(h) if h else None, hd95_max=max(h) if h else None)


def fair_summary(pb):
    i = [v["incumbent_dice_median"] for v in pb.values()]; n = [v["new_dice_median"] for v in pb.values()]
    return dict(n_brains=len(pb), incumbent_dice_median=med(i), new_dice_median=med(n),
                new_higher_brains=sum(v["new_dice_median"] > v["incumbent_dice_median"] for v in pb.values()))


def fair_per_brain(fair, new_by, val):
    """each brain's median over its cases in the fair subset, incumbent and new (each brain counts once)"""
    return {b: dict(incumbent_dice_median=med([val(r) for r in rr]), new_dice_median=med([val(new_by[r["case"]]) for r in rr]), n=len(rr))
            for b in sorted({r["subject"] for r in fair}) for rr in [[r for r in fair if r["subject"] == b]]}


def empty_block(rows):
    """photographs whose label is empty : listed, not pooled; the response: how many were marked, the median marked area"""
    def s(rr):
        mk = [r for r in rr if r["vox_pred"] > 0]
        return dict(n=len(rr), marked=len(mk), marked_area_mm2_median=med([r["fp_area_mm2"] for r in mk]))
    return dict(s(rows), per_brain={b: s([r for r in rows if r["subject"] == b]) for b in sorted({r["subject"] for r in rows})},
                cases=sorted(r["case"] for r in rows))


def position_block(rows, val, first):
    """Dice by position in the stack (the band fixed on 3 Nov 2026): the first `first` photographs (stack index < first) vs the rest"""
    def s(rr):
        d = [val(r) for r in rr]
        return dict(n=len(d), dice_median=med(d), n_below_0p9=int(sum(x < 0.9 for x in d)))
    out = {}
    for name, sel in ((f"first_{first}", lambda r: r["stack_index"] < first), ("rest", lambda r: r["stack_index"] >= first)):
        rr = [r for r in rows if sel(r)]
        out[name] = dict(pooled=s(rr), per_brain={b: s([r for r in rr if r["subject"] == b]) for b in sorted({r["subject"] for r in rr})})
    return out


def _score_incumbent(job):
    """one incumbent prediction against its label (and, for a post-hoc task, block 2: both hole-filled at /4)"""
    cid, c, p, dim, K, outline = job
    ref, meta = io.read_label(c["label"], dim); pred, _ = io.read_label(p, dim)
    assert pred.shape == ref.shape, f"{cid}: incumbent {pred.shape} vs label {ref.shape}"
    sp = meta["spacing"] if meta["kind"] == "nifti" else (c.get("spacing") or meta["spacing"])
    m = metrics.score(pred, ref, sp, range(1, K))["1"]
    row = dict(case=cid, subject=c["subject"], counted=c.get("counted", True), dice=m["dice"], hd95_mm=m["hd95_mm"], assd_mm=m["assd_mm"],
               vox_pred=m["vox_pred"], vox_ref=m["vox_ref"], fp_area_mm2=m.get("fp_area_mm2"))
    if outline:
        mo = metrics.score(fill(pred), fill(ref), sp, range(1, K))["1"]
        row["outline"] = dict(dice=mo["dice"], hd95_mm=mo["hd95_mm"])
    return row


def fill(m):
    """enclosed holes filled (the photograph rule, linc.segment.predict fill_holes)"""
    return predict.postprocess_mask(m, {"kind": "fill_holes"}, 2)


def _arr(p):
    p = str(p)
    if p.endswith((".nii", ".nii.gz")):
        import nibabel as nib
        return np.asanyarray(nib.load(p).dataobj)
    from PIL import Image
    return np.asarray(Image.open(p))


def _default(o):
    if isinstance(o, set):
        return sorted(o)
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    return str(o)


if __name__ == "__main__":
    sys.exit("run one of reproduce/numbers/segmodels_<model>_2026-11.py")
