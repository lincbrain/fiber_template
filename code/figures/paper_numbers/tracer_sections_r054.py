#!/usr/bin/env python3
"""How many histology sections carry the tracer labels shown in the template (paper: Results, tracer labels in the template; Fig. 10).
Per case, from the per-piece tracer records of the labels pushed to the blockface (products: push_tracer/<ID>/push_tracer_<ID>.json; per
piece and class, the pixels on the blockface at level L2). A section counts when any of its registered pieces carries at least one L2
pixel of a shown class (injection sites, fiber bundles, terminals; the single-fiber "Fibers" class left out). Also the same count with
"Fibers" included, and the count over every class drawn on the section (px_section > 0), whether or not it survived the registration.
-> $LINC_NUMBERS/tracer_sections_r054.json
    python code/figures/paper_numbers/tracer_sections_r054.py"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
from shared import paths, provenance  # noqa: E402

CASES = ["MR243", "MR252", "MR256", "MF191"]                       # LINC IDs of MM1, MM2, MM3, MF8 (the tracer cases)
LEFT_OUT = "Fibers"                                                # single-fiber class, not shown in Fig. 10

# the record's own method text, verbatim (it is part of the record's data)
METHOD = [
 "terminals; the single-fiber \"Fibers\" class left out), per case, from labels/push_tracer/<S>/push_tracer_<S>.json (per piece and class:",
 "pixels on the blockface at L2). A section counts when any of its registered pieces carries >= 1 L2 pixel of a shown class. Also the",
 "same count with Fibers, and with every class drawn on the section (px_section > 0) whether or not it survived the registration.",
 "-> paper/numbers/tracer_sections_r054.json"
]


def count_sections(J, key, keep):
    """Number of distinct sections with at least one piece whose class c (keep(c) true) has J[...][key] > 0."""
    return len({r["section"] for r in J["pieces"] for c, v in r["classes"].items() if keep(c) and v.get(key, 0) > 0})


def main():
    P = paths.products("products_mri")
    out = dict(method=METHOD, cases={}); ins = []
    for S in CASES:
        p = P["push_tracer"] / S / f"push_tracer_{S}.json"; ins.append(p)
        J = json.loads(p.read_text())
        out["cases"][S] = dict(shown_classes=count_sections(J, "px_L2", lambda c: c != LEFT_OUT),
                               with_fibers=count_sections(J, "px_L2", lambda c: True),
                               drawn_shown_classes=count_sections(J, "px_section", lambda c: c != LEFT_OUT))
    out["total"] = {k: sum(v[k] for v in out["cases"].values()) for k in out["cases"][CASES[0]]}
    provenance.write_numbers("tracer_sections_r054", out, inputs=ins, md5=False, cmd="python code/figures/paper_numbers/tracer_sections_r054.py")


if __name__ == "__main__":
    main()
