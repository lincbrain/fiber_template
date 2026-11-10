#!/usr/bin/env python3
"""The comparison of the blockface-based and the MRI-based alignment (Results, "Blockface-based versus MRI-based alignment").

    python code/figures/paper_numbers/route_test_1c.py

You need
  the per-target evaluation rows (settings/internal/products_blockface.yaml `route_test`: <target>/rows.json and <target>/link/).
What it does
  code/figures/evaluation/route_test.py `summary` on them.
You get
  $LINC_NUMBERS/route_test_1c.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
from shared import paths, provenance  # noqa: E402
from figures.evaluation import route_test  # noqa: E402


def main():
    res, inputs = route_test.summary(str(paths.products("products_blockface")["route_test"]))
    provenance.write_numbers("route_test_1c", res, inputs=inputs, cmd="python code/figures/paper_numbers/route_test_1c.py", md5=False)


if __name__ == "__main__":
    main()
