#!/usr/bin/env python3
"""Which section pairs the label interpolation filled, per tracer case (Supplementary S2).

You need (configs/products_mri.yaml)
  tracer_v2_bfnative: each case's interp_v2.json and its group sheets on the blockface grid; numbers_of_record: the earlier
  accounting record (tracer_interp_accounting_r058d), for the comparison.

What it does
  sorts every pair of consecutive charted sections into interpolated, adjacent (nothing to fill) or skipped (footprints differ), counts
  the labeled sections, and lists the pairs gained over the earlier interpolation.

You get
  $LINC_NUMBERS/tracer_interp_accounting_v2.json

    python reproduce/numbers/tracer_interp_accounting_v2.py
"""
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from linc import paths, provenance  # noqa: E402

NAME = "tracer_interp_accounting_v2"
CASES = ["MR243", "MR252", "MR256", "MF191"]
SKIP = "skipped: footprints differ (partial section)"
# the record's method text, fixed (it was this script's docstring when the record was made; the docstring may change)
METHOD = [
    'The between-section interpolation of the tracer labels, accounted per case (paper: Results, tracer labels; Supplementary S2): the',
    'successor of tracer_interp_accounting_r058d for the chain of record since 3 Nov 2026 (the per-hemisphere demons interpolation,',
    'cli/linc-labels interpolate). Keys the paper reads: cases.<S>.{slice_mm, sections, labeled_sections, pairs_by_status.<status>.pairs}.',
    'The interpolation pairs section slices PER HEMISPHERE. For the whole-section view of this record, each pair of consecutive section',
    'slices (a, b) is "interpolated" when a pair covered it in at least one hemisphere, "adjacent" when b - a = 1 (nothing to fill) and',
    'otherwise "skipped: footprints differ (partial section)" (the per-hemisphere footprint Dice < 0.8 in both hemispheres; no pair is',
    'skipped for its gap). The per-hemisphere statuses and the pairs gained over the earlier record (interpolated now, not before) are kept',
    'beside. labeled_sections = sections whose sheets hold a bundle, terminal-field or injection-site label (the single-fiber class left out).',
    "Inputs (products_mri): `tracer_v2_bfnative`/<S>/interp_v2.json (the interpolation's own record) and <S>_{Bundles,Terminals,Injection_site}",
    '_bfnative_L2.nii.gz (the group sheets on the blockface grid); `numbers_of_record`/tracer_interp_accounting_r058d.json (the earlier',
    'record, for slice_mm and the comparison).',
]


def main():
    P = paths.products("products_mri"); V2 = P["tracer_v2_bfnative"]; REC = P["numbers_of_record"] / "tracer_interp_accounting_r058d.json"
    old = json.loads(REC.read_text())["data"]["cases"]
    rec = dict(method=METHOD, cases={}); inputs = [REC]
    for S in CASES:
        J = json.loads((V2 / S / "interp_v2.json").read_text()); inputs.append(V2 / S / "interp_v2.json")
        hp = J["hemisphere_pairs"]; done = {(p["a"], p["b"]) for p in J["pairs"]}
        slices = sorted({r["a"] for rows in hp.values() for r in rows} | {r["b"] for rows in hp.values() for r in rows})
        st = {"interpolated": [], "adjacent": [], SKIP: []}
        for a, b in zip(slices[:-1], slices[1:]):
            if b - a < 2:
                st["adjacent"].append([a, b])
            elif any((a_, b_) in done for (a_, b_) in [(a, b)]) or any(a <= p[0] < b and a < p[1] <= b for p in done):
                st["interpolated"].append([a, b])                           # a pair covers this step (in one hemisphere at least)
            else:
                st[SKIP].append([a, b])
        lab = None
        for g in ("Bundles", "Terminals", "Injection_site"):
            p = V2 / S / f"{S}_{g}_bfnative_L2.nii.gz"
            if p.exists():
                m = np.asanyarray(nib.load(str(p)).dataobj) > 0; inputs.append(p)
                lab = m.any(axis=(0, 2)) if lab is None else (lab | m.any(axis=(0, 2)))
        labeled = int(lab.sum()) if lab is not None else 0
        rec_old = old[S]
        rec["cases"][S] = dict(slice_mm=rec_old["slice_mm"], sections=len(slices), labeled_sections=labeled,
                               pairs_by_status={k: dict(pairs=len(v)) for k, v in st.items()},
                               per_hemisphere={h: {k: sum(r["status"] == k for r in rows) for k in ("interpolated", "adjacent", "skipped: gap > 2.5 mm", SKIP)} for h, rows in hp.items()},
                               v2_pairs=len(done), slices_interpolated=J["summary"]["slices_interpolated"],
                               gained_vs_record=sorted([list(p) for p in done - {(p["a"], p["b"]) for p in rec_old.get("pairs", []) if p["status"] == "interpolated"}]),
                               record=dict(sections=rec_old["sections"], labeled_sections=rec_old["labeled_sections"], pairs_by_status={k: v["pairs"] for k, v in rec_old["pairs_by_status"].items()}))
        print(S, rec["cases"][S]["sections"], labeled, {k: len(v) for k, v in st.items()}, "gained", rec["cases"][S]["gained_vs_record"], flush=True)
    print(provenance.write_numbers(NAME, rec, inputs=inputs, md5=False, cmd="python reproduce/numbers/tracer_interp_accounting_v2.py"))


if __name__ == "__main__":
    main()
