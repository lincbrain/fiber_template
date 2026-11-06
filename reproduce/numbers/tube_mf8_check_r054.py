#!/usr/bin/env python3
"""MF8's low tube detection: a detection failure or a ground-truth problem? (paper: Results, "Histology registered to the blockface").
From $LINC_NUMBERS/tube_offsets_r054.json (run reproduce/numbers/tube_offsets_r054.py first): the share of pieces with a tube-sized
hole (>= 0.10 mm^2) within 2 mm of the carried hole-only center, MF8 against the other five brains (min-max), and the median area of
the holes that were detected. A tube hole absent from the section cannot be found by any registration.
-> $LINC_NUMBERS/tube_mf8_check_r054.json
    python reproduce/numbers/tube_mf8_check_r054.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from linc import paths, provenance  # noqa: E402

MF8 = "MF191"                                                      # LINC ID of MF8

# the record's own method text, verbatim (it is part of the record's data)
METHOD = [
 "ground-truth problem? From paper/numbers/tube_offsets_r054.json: the share of pieces with a tube-sized hole (>= 0.10 mm^2, the hole",
 "rules of the detector) within 2 mm of the carried hole-only center, MF8 against the other five brains (min-max), and the median area of",
 "the holes that were detected. A tube hole absent from the section cannot be found by any registration.",
 "-> paper/numbers/tube_mf8_check_r054.json"
]


def check(src, method):
    """The record's data from a tube offsets record (src)."""
    P = json.loads(src.read_text())["data"]["per_subject"]
    oth = [S for S in P if S != MF8]
    k, a = "diag_share_tube_sized_within_2mm", "detected_hole_area_mm2_median"
    return dict(method=method,
                MF8=dict(share_tube_sized_within_2mm=P[MF8][k], detected_hole_area_mm2_median=P[MF8][a], rate=P[MF8]["hole_hidden"]["rate"]),
                others=dict(brains=oth, share_tube_sized_within_2mm_min=min(P[S][k] for S in oth),
                            share_tube_sized_within_2mm_max=max(P[S][k] for S in oth),
                            detected_hole_area_mm2_median_min=min(P[S][a] for S in oth),
                            detected_hole_area_mm2_median_max=max(P[S][a] for S in oth)))


def main():
    src = paths.numbers_root() / "tube_offsets_r054.json"
    out = check(src, METHOD)
    provenance.write_numbers("tube_mf8_check_r054", out, inputs=[src], md5=False, cmd="python reproduce/numbers/tube_mf8_check_r054.py")


if __name__ == "__main__":
    main()
