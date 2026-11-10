"""Write a task's folds into its spec, once, before any training (data.make_folds: seed 2026, grouped by subject; 5 folds when the
task has >= 10 subjects, else leave one subject out). Existing folds are never changed.

    python -m shared.segment.folds --task <spec.json>
"""
import argparse
import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code")); __package__ = "shared.segment"
from . import data  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, type=Path)
    a = ap.parse_args(argv)
    raw = json.load(open(a.task))
    if "folds" in raw:
        print(f"{a.task}: folds already written ({len(raw['folds'])}); not changed"); return 0
    folds, rule = data.make_folds(data.load_task(a.task))
    raw["folds"], raw["fold_rule"] = folds, rule
    tmp = a.task.with_name(a.task.name + f".tmp{os.getpid()}"); tmp.write_text(json.dumps(raw, indent=1)); os.replace(tmp, a.task)
    print(rule); [print(" ", k, v) for k, v in folds.items()]
    return 0


if __name__ == "__main__":
    sys.exit(main())
