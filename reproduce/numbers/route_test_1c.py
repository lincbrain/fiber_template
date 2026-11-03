#!/usr/bin/env python3
"""The comparison of the blockface-based and the MRI-based alignment (Results, "Blockface-based versus MRI-based alignment").

    python reproduce/numbers/route_test_1c.py

You need
  the per-target evaluation rows (configs/products_blockface.yaml `route_test`: <target>/rows.json and <target>/link/).
What it does
  linc/evaluation/route_test.py `summary` on them.
You get
  $LINC_NUMBERS/route_test_1c.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from linc import paths, provenance  # noqa: E402
from linc.evaluation import route_test  # noqa: E402


def main():
    res, inputs = route_test.summary(str(paths.products("products_blockface")["route_test"]))
    provenance.write_numbers("route_test_1c", res, inputs=inputs, cmd="python reproduce/numbers/route_test_1c.py", md5=False)


if __name__ == "__main__":
    main()
