"""Score any predictions against a task's labels, on the label's grid: Dice, ASSD and HD95 in mm (and for 2D images the connected
components and the false-positive / false-negative area per image), per case and summarized over the counted cases (a case with
"counted": false is scored and flagged). For the new models' held-out predictions and for the incumbents' alike.

    python -m linc.segment.score --task <spec.json> --pred <folder | map.json> [--fold k] --out <scores.json> [--remap SRC=DST ...]
--pred: a folder of <case id><ext> files (ext as the label's), or a JSON {case id: prediction path}. --fold: only that fold's held-out
cases; otherwise every case that has a prediction. A prediction must be on its label's grid (same shape).
"""
import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2])); __package__ = "linc.segment"
from . import data, io, metrics  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, type=Path); ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--fold"); ap.add_argument("--out", required=True, type=Path); ap.add_argument("--remap", nargs="*", default=[])
    a = ap.parse_args(argv)
    task = data.load_task(a.task, [tuple(r.split("=", 1)) for r in a.remap])
    dim, K = int(task["dim"]), len(task["classes"])
    cases = data.split(task, a.fold)[1] if a.fold is not None else task["cases"]
    pmap = json.load(open(a.pred)) if a.pred.is_file() else None
    rows, missing = {}, []
    for c in cases:
        if pmap is not None:
            pp = pmap.get(c["id"])
        else:
            ext = ".nii.gz" if str(c["label"]).endswith(".nii.gz") else Path(c["label"]).suffix
            pp = a.pred / f"{c['id']}{ext}"
            pp = pp if pp.exists() else None
        if not pp:
            missing.append(c["id"]); continue
        ref, meta = io.read_label(c["label"], dim)
        pred, _ = io.read_label(pp, dim)
        assert pred.shape == ref.shape, f"{c['id']}: prediction {pred.shape} is not on the label's grid {ref.shape}"
        sp = meta["spacing"] if meta["kind"] == "nifti" else (c.get("spacing") or meta["spacing"])
        rows[c["id"]] = dict(subject=c["subject"], counted=c.get("counted", True), prediction=str(pp),
                             **metrics.score(pred, ref, sp, range(1, K)))
    out = dict(task=task["task"], fold=a.fold, predictions=str(a.pred), n_scored=len(rows), missing=missing, cases=rows,
               summary=metrics.summarize(rows, K), summary_all=metrics.summarize(rows, K, counted_only=False))
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(out, indent=1))
    print(f"{len(rows)} scored, {len(missing)} without a prediction -> {a.out}")
    print(json.dumps({k: v for k, v in out["summary"].items() if v}, indent=1)[:1500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
