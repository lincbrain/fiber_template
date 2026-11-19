#!/usr/bin/env python3
"""Compare NIfTI images file by file, e.g. yours with the paper's.

    $PY tests/compare/compare_nifti.py <your folder> <the other folder> <file> [<file> ...]

You need two folders that hold files of the same names.
What it does: loads each pair and compares their voxel values and their affines.
You get one line per file, "identical" or how the two differ, then a last line "<n> of <N> identical"; the exit status is 0 when all
are identical, 3 ("nothing to compare with") when the other folder holds none of the files.
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    mine, other, names = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    if not any((other / name).exists() for name in names):          # none of the paper's files: nothing to compare with
        print(f"nothing to compare with: {other} holds none of the {len(names)} files")
        sys.exit(3)
    bad = 0
    for name in names:
        a, b = mine / name, other / name
        missing = [str(p) for p in (a, b) if not p.exists()]
        if missing:
            print(f"{name}: missing {', '.join(missing)}")
            bad += 1
            continue
        x, y = nib.load(a), nib.load(b)
        X, Y = np.asanyarray(x.dataobj), np.asanyarray(y.dataobj)
        same_affine = np.array_equal(x.affine, y.affine)
        if X.shape != Y.shape:
            print(f"{name}: differs (shape {X.shape} against {Y.shape})")
            bad += 1
        elif np.array_equal(X, Y) and same_affine:
            print(f"{name}: identical")
        else:
            d = np.abs(X.astype(np.float64) - Y.astype(np.float64))
            print(f"{name}: differs ({int((d > 0).sum())} voxels, largest difference {d.max():g}; "
                  f"affine {'equal' if same_affine else 'differs'})")
            bad += 1
    print(f"{len(names) - bad} of {len(names)} identical")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
