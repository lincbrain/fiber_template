#!/usr/bin/env python3
"""Compare a brain's histology registration with the paper's, component by component.

    $PY tests/compare/compare_histology.py <your --out folder>/tirl <the paper's tirl folder> [--file moving4_nonlinear.png]
    $PY_TIRL tests/compare/compare_histology.py <your --out folder>/tirl <the paper's tirl folder> --max-mm 0.1

You need your `linc-histology all` output and the paper's registration (for the paper's brains,
$LINC_DATA_TREE/brains/sub-<ID>/intermediate/histology/tirl); with --max-mm, the TIRL environment ($PY_TIRL).
What it does: for every component the paper registered (a folder holding the file, by default the registered section
moving4_nonlinear.png), compares your file with the paper's, byte for byte. With --max-mm, a component whose file differs is
accepted when the two final transforms (fixed4_nonlinear.timg) map every pixel of the photograph to points at most that far apart, in
TIRL's mm: TIRL's last stage can end a few round-off steps apart from the paper's run (steps.yaml, the histology notes). TIRL's mm
are nominal: both images are given a 0.034 mm pixel (settings/histology.yaml tirl.resolution_mm), so 0.1 is 2.9 pixels; a pixel is
half the brain's L2 voxel of tissue (0.040 mm for the brains photographed at 20 um, 0.054 mm at 27 um), 0.1 TIRL mm 0.12 or 0.16 mm.
You get one line per component that differs or is missing, then a last line with the counts; the exit status is 0 only when every
component is identical, or within --max-mm when it is given.
"""
import argparse
import sys
from pathlib import Path


def map_distance_mm(a, b):
    """Largest distance (mm) between the points two TIRL transforms of one photograph map its pixels to."""
    import numpy as np
    import tirl
    x, y = tirl.load(str(a)), tirl.load(str(b))
    pa, pb = x.domain.get_physical_coordinates(), y.domain.get_physical_coordinates()
    if pa.shape != pb.shape:
        return float("inf")
    return float(np.linalg.norm(pa - pb, axis=-1).max())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mine"); ap.add_argument("paper"); ap.add_argument("--file", default="moving4_nonlinear.png")
    ap.add_argument("--max-mm", type=float, help="accept a differing component whose final transform is within this distance (TIRL's mm, 0.034 per pixel)")
    a = ap.parse_args()
    refs = sorted(p for p in Path(a.paper).glob(f"*/{a.file}"))
    if not refs:
        print(f"nothing to compare with: the data folder holds no registration of the paper's ({a.paper}/*/{a.file})")
        sys.exit(3)
    bad = close = 0
    worst = 0.0
    for ref in refs:
        f = Path(a.mine) / ref.parent.name / a.file
        if not f.exists():
            bad += 1
            print(f"{ref.parent.name}: missing")
        elif f.read_bytes() != ref.read_bytes():
            if a.max_mm is None:
                bad += 1
                print(f"{ref.parent.name}: differs")
                continue
            d = map_distance_mm(f.parent / "fixed4_nonlinear.timg", ref.parent / "fixed4_nonlinear.timg")
            worst = max(worst, d)
            ok = d <= a.max_mm
            close += ok
            bad += not ok
            print(f"{ref.parent.name}: differs, transforms at most {d:.3f} mm apart{'' if ok else ' (too far)'}")
    same = len(refs) - bad - close
    if a.max_mm is None:
        print(f"{same} of {len(refs)} components identical ({a.file})")
    else:
        print(f"{same} of {len(refs)} components identical, {close} within {a.max_mm:g} mm (largest {worst:.3f} mm), {bad} not ({a.file})")
    sys.exit(0 if refs and not bad else 1)


if __name__ == "__main__":
    main()
