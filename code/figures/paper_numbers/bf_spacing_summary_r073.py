#!/usr/bin/env python3
"""The blockface pixel size and section step over the 21 sectioned brains: median, quartiles and range (paper: Methods, "Sectioning
and blockface photography").

    python code/figures/paper_numbers/bf_spacing_summary_r073.py

You need
  $LINC_NUMBERS/bf_spacing_used.json (run code/figures/paper_numbers/bf_spacing_used.py first).
What it does
  The median, the 25th and 75th percentiles (numpy's default, linear) and the range of bf_spacing_used.per_brain's vx_um (pixel
  size) and vz_um (section step).
You get
  $LINC_NUMBERS/bf_spacing_summary_r073.json.

Ported from the analysis project's paper/r073_l1/bf_spacing_summary_r073.py.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
from shared import paths, provenance  # noqa: E402


def summary(v):
    v = np.asarray(v, float)
    return dict(n=int(v.size), median=float(np.median(v)), q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75)),
                min=float(v.min()), max=float(v.max()))


def main():
    src = paths.numbers_root() / "bf_spacing_used.json"
    pb = json.loads(src.read_text())["data"]["per_brain"]
    assert len(pb) == 21
    out = dict(vx_um=summary([b["vx_um"] for b in pb.values()]), vz_um=summary([b["vz_um"] for b in pb.values()]),
               note="over the 21 sectioned brains, from bf_spacing_used.per_brain (vx_um: pixel size; vz_um: section step)")
    print(json.dumps(out, indent=1))
    provenance.write_numbers("bf_spacing_summary_r073", out, inputs=[src], md5=False, cmd="python code/figures/paper_numbers/bf_spacing_summary_r073.py")


if __name__ == "__main__":
    main()
