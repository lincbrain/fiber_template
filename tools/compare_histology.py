#!/usr/bin/env python3
"""Compare a brain's histology registration with the paper's, component by component.

    $PY tools/compare_histology.py <your --out folder>/tirl <the paper's tirl folder> [--file moving4_nonlinear.png]

You need your `linc-histology all` output and the paper's registration (for the paper's brains,
$LINC_PRODUCTS/data_arm_final/sub-<ID>/blockface/06_histo2bf/tirl).
What it does: for every component the paper registered (a folder holding the file, by default the registered section
moving4_nonlinear.png), compares your file with the paper's, byte for byte.
You get one line per component that differs or is missing, then a last line "<n> of <N> components identical"; the exit status is
0 only when all are identical.
"""
import argparse
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mine"); ap.add_argument("paper"); ap.add_argument("--file", default="moving4_nonlinear.png")
    a = ap.parse_args()
    refs = sorted(p for p in Path(a.paper).glob(f"*/{a.file}"))
    bad = 0
    for ref in refs:
        f = Path(a.mine) / ref.parent.name / a.file
        if not f.exists():
            bad += 1
            print(f"{ref.parent.name}: missing")
        elif f.read_bytes() != ref.read_bytes():
            bad += 1
            print(f"{ref.parent.name}: differs")
    print(f"{len(refs) - bad} of {len(refs)} components identical ({a.file})")
    sys.exit(0 if refs and not bad else 1)


if __name__ == "__main__":
    main()
