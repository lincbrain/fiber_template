#!/usr/bin/env python3
"""Tube offsets of the histology -> blockface registrations (paper: Results "Histology registered to the blockface", Table 3, Fig. 6).

From the per-piece records of linc/evaluation tube detection (products: tube_landmarks/r054hole_<ID>_native.json; tubes hidden from
the registration, blockface tube center = centroid of the hole's dark core), the six histology brains.
Per brain and pooled, two variants {n, median_mm, p95_mm, mean_mm, detected, searched, rate}:
  hole_hidden   the paper's: hole-only blockface center; every piece in which the hole was found with the tubes hidden
  mask_hidden   the earlier center (centroid of the whole tube mask), same pieces (for comparison)
searched = pieces whose blockface slice carries a tube; detected = hole found within 1 mm; rate = detected / searched.
Also per brain: how often the hole-only rule applied (refined), the shift of the center (mm; piece grid 0.034 mm/px), and the share of
pieces with a tube-sized hole (>= 0.10 mm^2) within 1 / 2 mm of the carried center, and the median area of the detected holes.
-> $LINC_NUMBERS/tube_offsets_r054.json
    python reproduce/numbers/tube_offsets_r054.py"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from linc import paths, provenance  # noqa: E402

BRAINS = ["MR243", "MR252", "MR256", "MF278", "MF275", "MF191"]     # LINC IDs of MM1, MM2, MM3, MF5, MF6, MF8
FIX_MM = 0.034                                                     # TIRL piece grid (Methods, histology registration)

# the record's own method text, verbatim (it is part of the record's data)
METHOD = [
 "only), the six histology brains.",
 "Variants (pooled.<v> and per_subject.<S>.<v>, each {n, median_mm, p95_mm, mean_mm, detected, searched, rate}):",
 "  hole_hidden   the 14 Oct 2026 revision's rule: blockface center = centroid of the hole-only voxels; every piece in which the hole was found with the",
 "                tubes hidden (NOT paired with the tubes-visible arm)",
 "  mask_hidden   the earlier center alone (centroid of the whole tube mask), the same hidden-only set (reproduces the",
 "                inpainted arm of histo_tube_landmarks.json on its unpaired set)",
 "searched = pieces whose blockface slice carries a tube; detected = hole found within 1 mm; rate = detected / searched.",
 "Also per brain: how often the hole-only rule applied (refined), the shift of the center (mm, fixed grid 0.034 mm/px), and",
 "detection diagnostics in a 4 mm window around the mapped hole-only center (hole-like components of 0.03-1.5 mm^2; tube-sized",
 "= >= 0.10 mm^2): the share of pieces with any tube-sized hole within 1 mm / within 2 mm, and the median area of the detected holes.",
 "-> paper/numbers/tube_offsets_r054.json",
 "    python3 labels/tube_offsets_r054.py"
]


def stats(rows, key):
    s = [r for r in rows if r.get("bf_tube")]; e = [r[key]["error_mm"] for r in s if r[key]["found"]]
    q = lambda f: round(float(f(e)), 3) if e else None  # noqa: E731
    return dict(n=len(e), median_mm=q(np.median), p95_mm=q(lambda x: np.percentile(x, 95)), mean_mm=q(np.mean),
                detected=len(e), searched=len(s), rate=round(len(e) / len(s), 4) if s else None)


def main():
    P = paths.products("products_mri")
    out = dict(method=METHOD, per_subject={}, pooled={}); allrows = []; ins = []
    for S in BRAINS:
        p = P["tube_landmarks"] / f"r054hole_{S}_native.json"; ins.append(p)
        rows = json.loads(p.read_text())["pieces"]; allrows += rows
        s = [r for r in rows if r.get("bf_tube")]; h = [r["hole"] for r in s]
        shift = [r["center_shift_fixed_px"] * FIX_MM for r in s]
        out["per_subject"][S] = dict(
            hole_hidden=stats(rows, "hole"), mask_hidden=stats(rows, "mask"),
            refined_share=round(float(np.mean([r["refined"] for r in s])), 4),
            center_shift_mm_median=round(float(np.median(shift)), 3), center_shift_mm_p90=round(float(np.percentile(shift, 90)), 3),
            hole_to_mask_area_median=round(float(np.median([r["hole_px"] / r["mask_px"] for r in s if r["refined"]])), 3),
            diag_share_tube_sized_within_1mm=round(float(np.mean([(x["diag_nearest_tube_sized_mm"] or 9) <= 1.0 for x in h])), 4),
            diag_share_tube_sized_within_2mm=round(float(np.mean([(x["diag_nearest_tube_sized_mm"] or 9) <= 2.0 for x in h])), 4),
            detected_hole_area_mm2_median=(round(float(np.median([x["hole_area_mm2"] for x in h if x["found"]])), 3)
                                           if any(x["found"] for x in h) else None))
    out["pooled"] = dict(hole_hidden=stats(allrows, "hole"), mask_hidden=stats(allrows, "mask"))
    s = [r for r in allrows if r.get("bf_tube")]
    out["pooled"]["center_shift_mm_median"] = round(float(np.median([r["center_shift_fixed_px"] * FIX_MM for r in s])), 3)
    out["pooled"]["refined_share"] = round(float(np.mean([r["refined"] for r in s])), 4)
    provenance.write_numbers("tube_offsets_r054", out, inputs=ins, md5=False, cmd="python reproduce/numbers/tube_offsets_r054.py")


if __name__ == "__main__":
    main()
