#!/usr/bin/env python3
"""The sanity check of the 21 reconstructed blockface volumes (Results, "All 21 photographed stacks were reconstructed").

    python reproduce/numbers/bf_sanity.py

You need
  the 21 reconstructions ($LINC_DATA/sub-<ID>/blockface).
What it does
  linc/blockface/sanity.py on all 21: orientation, grayscale against mask, the side of the marker tubes, size against the cohort,
  alignment through the stack.
You get
  $LINC_NUMBERS/bf_sanity.json.
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from linc import paths, provenance  # noqa: E402
from linc.blockface import sanity  # noqa: E402

SUBJ = ["MF191", "MF196", "MF204", "MF275", "MF278", "MF282", "MR243", "MR244", "MR247", "MR248", "MR249", "MR250",
        "MR251", "MR252", "MR253", "MR254", "MR256", "MR257", "MR261", "MR263", "MR273"]
CHECKS = "One sanity pass over every reconstructed blockface subject: is it oriented, consistent and sane?\n\n`audit_structure.py` says whether the FILES are there. This says whether their CONTENTS are what we think,\nand it checks the things that have actually gone wrong in this project rather than a generic wish list:\n\n  ORIENT   the store's slice axis must run anterior -> posterior (negative step, slice 0 frontal). That is\n           the convention since 2026-09-19; see code/ORIENTATION.md.\n  MIRROR   the iso grey must agree with its own brain mask in WORLD space. A stray\n           `fix_ap_orientation.py apply` after a rebuild mirrors an already-correct grey against its mask,\n           which is invisible in any single volume and destroys every registration that uses the grey\n           (MF191 and MF204, 2026-09-20: dice 0.61 instead of 0.97, R3 NCC -0.013).\n  CHIRAL   the drill holes are in the LEFT hemisphere, so their centroid must sit at NEGATIVE world x\n           relative to the brain's. Left-right cannot be settled by fit quality on a near-symmetric brain\n           (MR261's two x-flips scored Dice 0.904 vs 0.902), so this marker is the only cheap check.\n  SIZE     L-R and D-V extent against the cohort. A subject 20 % off is a voxel-size error, not a brain.\n           (A-P is NOT checked: the cohort is truncated by varying amounts, which is real.)\n  ALIGN    frame-to-frame brain-centroid step through the stack - the alignment quality that the camera-jump\n           correction exists to deliver.\n"                     # the record's `checks` text, verbatim


def main():
    data = str(paths.products("products_blockface")["data"])
    rows = []
    for S in SUBJ:
        try:
            rows.append(sanity.check(S, data))
        except Exception as e:  # noqa: BLE001
            rows.append(dict(subject=S, findings=[f"ERROR {type(e).__name__}: {e}"]))
        print(S, rows[-1].get("findings"), flush=True)
    mine = copy.deepcopy(rows); ml, md = sanity.judge_size(mine)
    provenance.write_numbers("bf_sanity", dict(
        checks=CHECKS, rows=sorted(mine, key=lambda x: x["subject"]),
        n=len(mine), n_clean=sum(not r["findings"] for r in mine), cohort_median_LR_mm=ml, cohort_median_DV_mm=md,
        findings={r["subject"]: r["findings"] for r in mine if r["findings"]}), cmd="python reproduce/numbers/bf_sanity.py")


if __name__ == "__main__":
    main()
