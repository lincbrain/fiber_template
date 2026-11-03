#!/usr/bin/env python3
"""How accurate the cross-modal alignment is, for the 13 brains with MRI: compared with their MRI-based alignment (Methods and
Results; linc/evaluation/direct_placement.py).

    python reproduce/numbers/d2_s2_validation_prod_s125.py

You need
  the cross-modal alignments of record (configs/products_blockface.yaml `direct_placement`), the blockface-to-b0 registrations of
  record (the registry entries of the manifest of record) and each brain's b0 -> MRI template registration of record.
What it does
  linc/evaluation/direct_placement.py per brain, then the summary with the tuning brains (used to choose the smoothing: MF278,
  MR250, MR252) apart.
You get
  $LINC_NUMBERS/d2_s2_validation_prod_s125.json: per brain the distances, and the held-out median of the per-brain median distances
  and the median 95th percentile.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from linc import paths, provenance  # noqa: E402
from linc.evaluation import direct_placement as DP  # noqa: E402

MRI_BRAINS = ["MF275", "MF278", "MF282", "MR243", "MR249", "MR250", "MR251", "MR252", "MR253", "MR254", "MR256", "MR257", "MR261"]
TUNING = ["MR250", "MR252", "MF278"]
R5_RECIPE = "D22"


def main():
    prod = paths.products("products_blockface")
    man = json.loads((prod["numbers_of_record"] / "r1_current_2026-10-03.json").read_text())["data"]["brains"]
    root = paths.products_root()
    rows = []
    for S in MRI_BRAINS:
        r1 = man[S]["registry"]
        r1 = str(root / r1.split("LINC_template/", 1)[1]) if "LINC_template/" in r1 else r1            # the manifest's entry, under $LINC_PRODUCTS
        r5 = prod["data"] / f"sub-{S}" / "xfm" / "b0-native__to__mri-template"
        rows.append(DP.one(S, str(prod["direct_placement"] / S), r1, str(r5), man[S].get("variant"), man[S].get("qc"), R5_RECIPE))
        print(json.dumps(rows[-1])[:200], flush=True)
    tune = [r for r in rows if r["subject"] in TUNING]; rest = [r for r in rows if r["subject"] not in TUNING]
    provenance.write_numbers("d2_s2_validation_prod_s125", dict(rows=rows, tuning=TUNING, summary_tuning=DP.summary(tune),
                                                                 summary_held_out=DP.summary(rest), expected_error_for_no_mri=DP.summary(rest)),
                             cmd="python reproduce/numbers/d2_s2_validation_prod_s125.py", md5=False)


if __name__ == "__main__":
    main()
