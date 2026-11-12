#!/usr/bin/env python3
"""The histology numbers of the paper's registrations, each component's nonlinear stiffness chosen from its affine-stage tissue Dice
(settings/histology.yaml tirl.stiff_nonlinear, adopted 12 Nov 2026; paper: Results "Histology registered to the blockface", Table 4,
Discussion).

You need
  the products (products_mri): numbers_of_record/registration_qc_histo2bf_h9.json, the registration QC's rows of these registrations
  (per component: blockface slice, gate and the tissue Dice of each TIRL stage; read as registration_qc_histo2bf_20261001.py reads the
  QC of record), and tube_landmarks_h9 (h9hole_<ID>_native.json: the tube landmarks of these registrations, the tubes hidden).
What it does
  registration_qc_histo2bf_h9  the QC rows as recorded (the six histology brains' 395 components)
  tube_offsets_h9              tube_offsets_r054.py's computation on these landmarks, without MF8's mf191FR_c01_s46nDF_0002 (dropped as
                               unmatched, 5 Nov 2026 revision, as in mf8_s46_drop_r068.py); per brain, the components whose tube search
                               stopped (search window outside the section image) are listed (scorer_errors)
  tube_mf8_check_h9            tube_mf8_check_r054.py's computation on tube_offsets_h9
You get
  $LINC_NUMBERS/registration_qc_histo2bf_h9.json, tube_offsets_h9.json, tube_mf8_check_h9.json.
    python code/figures/paper_numbers/histology_h9.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "code")); sys.path.insert(0, str(HERE))
from shared import paths, provenance  # noqa: E402
import tube_mf8_check_r054 as MF8CHECK  # noqa: E402
import tube_offsets_r054 as OFFSETS  # noqa: E402

DROP = "mf191FR_c01_s46nDF_0002"
WHY = ("dropped as unmatched (5 Nov 2026 revision): the matching rule classed this cerebellum-and-brainstem component as the right "
       "hemisphere, so it was registered to the right-hemisphere region of photograph 66; never validly registered")
STIFF = "the registrations with the per-component nonlinear stiffness (settings/histology.yaml tirl.stiff_nonlinear, 12 Nov 2026)"
CMD = "python code/figures/paper_numbers/histology_h9.py"


def main():
    P = paths.products("products_mri")
    src = P["numbers_of_record"] / "registration_qc_histo2bf_h9.json"
    rows = json.loads(src.read_text())["data"]
    provenance.write_numbers("registration_qc_histo2bf_h9", rows, inputs=(), md5=False, cmd=CMD,
                             extra=dict(frozen_from=provenance.file_record(src, md5=True), family="histo2bf", n_rows=len(rows)))
    out, ins = OFFSETS.offsets([f"tube_offsets_r054's computation on {STIFF}, without {DROP}: {WHY}."] + OFFSETS.METHOD,
                               drop={DROP}, landmarks="tube_landmarks_h9", tag="h9hole")
    for S in OFFSETS.BRAINS:
        out["per_subject"][S]["scorer_errors"] = json.loads((P["tube_landmarks_h9"] / f"h9hole_{S}_native.json").read_text())["summary"]["scorer_errors"]
    provenance.write_numbers("tube_offsets_h9", out, inputs=ins, md5=False, cmd=CMD,
                             extra=dict(derived_from="tube_offsets_r054's computation", removed=[DROP], why=WHY))
    src = paths.numbers_root() / "tube_offsets_h9.json"
    out = MF8CHECK.check(src, ["tube_mf8_check_r054's computation on tube_offsets_h9."] + MF8CHECK.METHOD)
    provenance.write_numbers("tube_mf8_check_h9", out, inputs=[src], md5=False, cmd=CMD, extra=dict(derived_from="tube_mf8_check_r054"))


if __name__ == "__main__":
    main()
