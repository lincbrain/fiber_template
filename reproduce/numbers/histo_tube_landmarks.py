#!/usr/bin/env python3
"""Tube offsets of the histology -> blockface registrations, both blockface targets on the same pieces (paper: Results "Histology
registered to the blockface", Table 3, Fig. 6b-c; Abstract; Methods, Fig. 1 caption for the number of brains).

The offset of a piece is the distance (mm, histology frame) between the blockface marker tube mapped through the piece's registration
and the tube hole detected in the histology within 1 mm. From the per-piece records of tube detection (products:
tube_landmarks/inp_<ID>_native.json, tubes hidden from the registration by inpainting; tube_landmarks/preinp_<ID>_native.json, the
pre-inpaint blockface with the tubes visible), the six histology brains. Both targets are scored on the SAME pieces (hole found in
both). Per brain: pieces searched (blockface slice carries a tube), found per target, paired, {n, median_mm, p95_mm, mean_mm} per
target, the median paired difference (pre-inpaint minus inpainted) and how many pieces got better / worse. Pooled: the same over all
pieces, with the detection denominators.
-> $LINC_NUMBERS/histo_tube_landmarks.json
    python reproduce/numbers/histo_tube_landmarks.py"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from linc import paths, provenance  # noqa: E402

BRAINS = ["MR243", "MR252", "MR256", "MF278", "MF275", "MF191"]     # LINC IDs of MM1, MM2, MM3, MF5, MF6, MF8

# the record's own text fields, verbatim (they are part of the record's data)
MEASURE = ("distance (mm, histology frame) between the blockface tube mapped through the piece's TIRL chain and the tube hole "
           "detected in the histology at JP2 level 3 within 1 mm (labels/tube_landmarks_native.py); paired on pieces found in both arms")
DENOMINATOR_RULE = "searched = pieces whose blockface slice carries a tube; found = hole detected within 1 mm"
TARGET_OF_RECORD = "pre-inpaint (user decision 2026-09-26, physical grounds)"


def load(T, tag, S):
    p = T / f"{tag}_{S}_native.json"
    return {r["piece"]: r for r in json.loads(p.read_text())["pieces"]} if p.exists() else None


def stats(e):
    e = np.array(e)
    return dict(n=int(len(e)), median_mm=round(float(np.median(e)), 3) if len(e) else None,
                p95_mm=round(float(np.percentile(e, 95)), 3) if len(e) else None, mean_mm=round(float(e.mean()), 3) if len(e) else None)


def denoms(per):
    """Detection denominators: a piece is SEARCHED when its blockface slice carries a tube (pieces_with_bf_tube)."""
    ok = [v for v in per.values() if "missing" not in v]
    n = sum(v["pieces_with_bf_tube"] for v in ok); fp = sum(v["found_preinpaint"] for v in ok)
    return dict(searched_pieces_with_bf_tube=n, found_preinpaint=fp, found_inpainted=sum(v["found_inpainted"] for v in ok),
                paired=sum(v["paired"] for v in ok), detection_rate_preinpaint=round(fp / n, 3) if n else None,
                denominator_rule=DENOMINATOR_RULE)


def main():
    T = paths.products("products_mri")["tube_landmarks"]
    per, pooled_i, pooled_p, ins = {}, [], [], []
    for S in BRAINS:
        a, b = load(T, "inp", S), load(T, "preinp", S)
        if a is None or b is None:
            per[S] = dict(missing="inp" if a is None else "preinp"); continue
        ins += [T / f"inp_{S}_native.json", T / f"preinp_{S}_native.json"]
        common = [p for p in a if p in b and a[p].get("found") and b[p].get("found")]
        ei, ep = [a[p]["error_mm"] for p in common], [b[p]["error_mm"] for p in common]
        d = np.array(ep) - np.array(ei)
        per[S] = dict(pieces_with_bf_tube=sum(1 for r in b.values() if r.get("bf_tube")),
                      found_inpainted=sum(1 for r in a.values() if r.get("found")), found_preinpaint=sum(1 for r in b.values() if r.get("found")),
                      paired=len(common), inpainted=stats(ei), preinpaint=stats(ep),
                      paired_diff_median_mm=round(float(np.median(d)), 3) if len(d) else None,
                      preinpaint_better=int((d < 0).sum()), preinpaint_worse=int((d > 0).sum()))
        pooled_i += ei; pooled_p += ep
    d = np.array(pooled_p) - np.array(pooled_i)
    data = dict(measure=MEASURE,
                per_subject=per, pooled=dict(**denoms(per), inpainted=stats(pooled_i), preinpaint=stats(pooled_p),
                                             paired_diff_median_mm=round(float(np.median(d)), 3) if len(d) else None,
                                             preinpaint_better=int((d < 0).sum()), preinpaint_worse=int((d > 0).sum())),
                target_of_record=TARGET_OF_RECORD)
    provenance.write_numbers("histo_tube_landmarks", data, inputs=ins, md5=False, cmd="python reproduce/numbers/histo_tube_landmarks.py")


if __name__ == "__main__":
    main()
