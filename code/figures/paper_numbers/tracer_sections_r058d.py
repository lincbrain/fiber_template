#!/usr/bin/env python3
"""How many histology sections carry the tracer labels shown in the template, on the final labels (paper: Results, tracer labels).
You need
  push_tracer_r058d (settings/internal/products_mri.yaml): each case's record of the labels pushed to the blockface, per piece and class.
What it does
  Counts the sections as code/figures/paper_numbers/tracer_sections_r054.py does, on the labels after the two corrections of Supplementary S2.
You get
  $LINC_NUMBERS/tracer_sections_r058d_r17.json
    python code/figures/paper_numbers/tracer_sections_r058d.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
from shared import paths, provenance  # noqa: E402
from tracer_sections_r054 import CASES, LEFT_OUT, count_sections  # noqa: E402

# the record's own method text, verbatim (it is part of the record's data)
METHOD = ["sections carrying a shown tracer class (any class but Fibers) with >= 1 L2 pixel after the registration, per case; "
          "labels/tracer_sections_r054.py on the final labels labels/push_tracer_r058d"]


def main():
    P = paths.products("products_mri")
    out = dict(method=METHOD, cases={}); ins = []
    for S in CASES:
        p = P["push_tracer_r058d"] / S / f"push_tracer_{S}.json"; ins.append(p)
        J = json.loads(p.read_text())
        out["cases"][S] = dict(shown_classes=count_sections(J, "px_L2", lambda c: c != LEFT_OUT),
                               with_fibers=count_sections(J, "px_L2", lambda c: True),
                               drawn_shown_classes=count_sections(J, "px_section", lambda c: c != LEFT_OUT))
    out["total"] = {k: sum(v[k] for v in out["cases"].values()) for k in out["cases"][CASES[0]]}
    provenance.write_numbers("tracer_sections_r058d_r17", out, inputs=ins, md5=False, cmd="python code/figures/paper_numbers/tracer_sections_r058d.py")


if __name__ == "__main__":
    main()
