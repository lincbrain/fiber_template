#!/usr/bin/env python3
"""The histology numbers without MF8's component mf191FR_c01_s46nDF_0002 (paper: Results "Histology registered to the blockface",
Table 4, Discussion).

You need
  $LINC_NUMBERS/registration_qc_histo2bf_20261001.json (code/figures/paper_numbers/registration_qc_histo2bf_20261001.py) and the tube
  landmarks of the products (products_mri `tube_landmarks`, as code/figures/paper_numbers/tube_offsets_r054.py reads them).
What it does
  Drops the one component as unmatched (5 Nov 2026 revision): the matching rule classed this cerebellum-and-brainstem component as
  the right hemisphere, as it did s47 and s49, which its score floor had already left unmatched, so it was registered to the
  right-hemisphere region of photograph 66 and never validly registered. Nothing else changes: the tissue-Dice rows without its row,
  and tube_offsets_r054.py's and tube_mf8_check_r054.py's computations without it. The paper's match table without it:
  labels/match_tables/num_MF191_r068.txt.
You get
  $LINC_NUMBERS/registration_qc_histo2bf_r068.json, tube_offsets_r068.json, tube_mf8_check_r068.json.
    python code/figures/paper_numbers/mf8_s46_drop_r068.py
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
CMD = "python code/figures/paper_numbers/mf8_s46_drop_r068.py"


def main():
    num = paths.numbers_root()
    src = num / "registration_qc_histo2bf_20261001.json"
    rows = json.loads(src.read_text())["data"]
    keep = [r for r in rows if r.get("tag") != DROP]
    assert len(rows) - len(keep) == 1, f"expected one row of {DROP}, found {len(rows) - len(keep)}"
    provenance.write_numbers("registration_qc_histo2bf_r068", keep, inputs=[src], md5=False, cmd=CMD,
                             extra=dict(derived_from="registration_qc_histo2bf_20261001", removed=[DROP], why=WHY, n_rows=len(keep)))
    out, ins = OFFSETS.offsets([f"tube_offsets_r054's computation without {DROP}: {WHY}."] + OFFSETS.METHOD, drop={DROP})
    provenance.write_numbers("tube_offsets_r068", out, inputs=ins, md5=False, cmd=CMD,
                             extra=dict(derived_from="tube_offsets_r054", removed=[DROP], why=WHY))
    src = num / "tube_offsets_r068.json"
    out = MF8CHECK.check(src, [f"tube_mf8_check_r054's computation on tube_offsets_r068 (without {DROP})."] + MF8CHECK.METHOD)
    provenance.write_numbers("tube_mf8_check_r068", out, inputs=[src], md5=False, cmd=CMD,
                             extra=dict(derived_from="tube_mf8_check_r054", removed=[DROP], why=WHY))


if __name__ == "__main__":
    main()
