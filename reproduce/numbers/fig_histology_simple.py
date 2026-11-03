#!/usr/bin/env python3
"""The histology section shown in the paper's histology figure, and its tissue Dice (paper: Results "Histology registered to the
blockface", Fig. 6a caption: "tissue Dice ..., the median of MR256").

The section is chosen by rule, with no tube criterion: of the MR256 histology -> blockface pieces, the one whose nonlinear tissue Dice
is closest to the median of the brain (ties: the first in file order). Read from the dated registration-QC snapshot
$LINC_NUMBERS/registration_qc_histo2bf_20261001.json (run reproduce/numbers/registration_qc_histo2bf_20261001.py first); the figure
originally read the live registration-QC file, whose MR256 rows are identical. Only the number is ported here, not the figure.
data = {subject, piece, piece_dice, subject_median_dice, n_pieces, rule}.
-> $LINC_NUMBERS/fig_histology_simple.json
    python reproduce/numbers/fig_histology_simple.py"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from linc import paths, provenance  # noqa: E402

SUBJ = "MR256"                                                     # LINC ID of MM3, the brain shown in the figure
FAMILY = "histo2bf"                                                # histology -> blockface rows

# the record's own rule text, verbatim (it is part of the record's data)
RULE = "MR256 histo2bf piece whose nonlinear tissue Dice is closest to the brain's median (registration_qc.json)"


def main():
    src = paths.numbers_root() / "registration_qc_histo2bf_20261001.json"
    rq = [r for r in json.loads(src.read_text())["data"] if r.get("family") == FAMILY and r["subject"] == SUBJ]
    dices = [r["metrics"]["dice_nonlinear"] for r in rq]
    med = float(np.median(dices))
    ra = min(rq, key=lambda r: abs(r["metrics"]["dice_nonlinear"] - med))
    data = dict(subject=SUBJ, piece=ra["tag"], piece_dice=ra["metrics"]["dice_nonlinear"], subject_median_dice=round(med, 4),
                n_pieces=len(rq), rule=RULE)
    provenance.write_numbers("fig_histology_simple", data, inputs=[src], md5=False, cmd="python reproduce/numbers/fig_histology_simple.py")


if __name__ == "__main__":
    main()
