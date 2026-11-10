#!/usr/bin/env python3
"""Per-piece quality of the histology -> blockface registrations, frozen on 1 Oct 2026 (paper: Results "Histology registered to the
blockface", Table 3 and Fig. 6a: number of pieces per brain and the median tissue Dice before registration, after the affine stage
and after the nonlinear stage).

The registration-QC file of the analysis (products: numbers_of_record/registration_qc.json) holds one row per registration of every
kind; it kept growing as other registrations were rerun, so the paper cites this dated snapshot of its histology -> blockface rows
(family "histo2bf": brain, piece, blockface slice, gate, and the tissue Dice of each registration stage). Selection rule: every row
of that family, in file order. Later appends to the file added rows of other families only, so this rule reproduces the 1 Oct
snapshot exactly. The source file is recorded under the provenance "extra" block (not as an input), so later appends to it do not
fail the snapshot's provenance check.
-> $LINC_NUMBERS/registration_qc_histo2bf_20261001.json
    python code/figures/paper_numbers/registration_qc_histo2bf_20261001.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
from shared import paths, provenance  # noqa: E402

FAMILY = "histo2bf"                                                # histology -> blockface rows
SNAPSHOT_DATE = "20261001"                                         # the date of the snapshot the paper cites
NAME = f"registration_qc_{FAMILY}_{SNAPSHOT_DATE}"


def main():
    src = paths.products("products_mri")["numbers_of_record"] / "registration_qc.json"
    rows = json.loads(src.read_text())
    rec = provenance.file_record(src, md5=True)
    rows = rows.get("data", rows) if isinstance(rows, dict) else rows
    keep = [r for r in rows if r.get("family") == FAMILY]
    provenance.write_numbers(NAME, keep, inputs=(), md5=False, extra={"frozen_from": rec, "family": FAMILY, "n_rows": len(keep)},
                             cmd=f"python code/figures/paper_numbers/{NAME}.py")


if __name__ == "__main__":
    main()
