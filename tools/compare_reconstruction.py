#!/usr/bin/env python3
"""Compare a reconstruction with another (e.g. yours with the paper's), to check that a rerun reproduces it.

    $PY tools/compare_reconstruction.py <ID> <your sub-<ID>/blockface> <the other sub-<ID>/blockface>
    e.g. $PY tools/compare_reconstruction.py MR252 $LINC_REBUILD_DATA/sub-MR252/blockface $LINC_RECORDS/data/sub-MR252/blockface

You need
  the two blockface folders.
What it does
  It compares every level of the OME-Zarr stores (the arrays and their scale and translation), the NIfTI products (arrays and
  affines) and <ID>.info. The stores are read one block of sections at a time, so a few GB of memory suffice.
You get
  one line per item (whether the arrays are identical and the transforms or affines equal, or which side is missing); exit status 0
  when all are identical.
"""
import os
import sys
import numpy as np, nibabel as nib, zarr
S, new, rec = sys.argv[1], sys.argv[2], sys.argv[3]
bad = 0
SECTIONS = 32                                            # sections read at a time (the stores' axis 2)


def ome(p):
    g = zarr.open_group(p, mode="r"); m = g.attrs.asdict().get("ome", g.attrs.asdict()).get("multiscales") or g.attrs.asdict().get("multiscales")
    return g, m[0]["datasets"]


def same_array(A, B):
    """Two zarr arrays equal, read in blocks of SECTIONS sections along axis 2 (c, rows, sections, columns)."""
    if A.shape != B.shape or A.dtype != B.dtype:
        return False
    n = A.shape[2]
    return all(np.array_equal(A[:, :, s:s + SECTIONS], B[:, :, s:s + SECTIONS]) for s in range(0, n, SECTIONS))


for st in (f"{S}_blockface.ome.zarr", "02_blockface.ome.zarr", "02_blockface_mask.ome.zarr", f"steps/{S}_03_color.ome.zarr"):
    a, b = os.path.join(new, st), os.path.join(rec, st)
    if not (os.path.exists(a) and os.path.exists(b)):
        print(f"{st:34s} missing: new {os.path.exists(a)} record {os.path.exists(b)}"); bad += 1; continue
    (ga, da), (gb, db) = ome(a), ome(b)
    for x, y in zip(da, db):
        same = same_array(ga[x["path"]], gb[y["path"]])
        tr = x.get("coordinateTransformations") == y.get("coordinateTransformations")
        bad += not (same and tr)
        print(f"{st:34s} L{x['path']}: array identical {same} | transforms equal {tr}")
for f in (f"{S}_mask.nii.gz", f"{S}_holes.nii.gz", f"{S}_blockface_L2.nii.gz"):
    a, b = os.path.join(new, f), os.path.join(rec, f)
    if not (os.path.exists(a) and os.path.exists(b)):
        print(f"{f:34s} missing: new {os.path.exists(a)} record {os.path.exists(b)}"); bad += 1; continue
    ia, ib = nib.load(a), nib.load(b)
    A, B = np.asanyarray(ia.dataobj), np.asanyarray(ib.dataobj)
    same, aff = A.shape == B.shape and np.array_equal(A, B), np.allclose(ia.affine, ib.affine, atol=1e-6)
    bad += not (same and aff)
    print(f"{f:34s} array identical {same} | affine equal {aff}")
a, b = os.path.join(new, f"{S}.info"), os.path.join(rec, f"{S}.info")
same = os.path.exists(a) and os.path.exists(b) and open(a).read() == open(b).read()
bad += not same
print(f"{S}.info identical {same}")
sys.exit(1 if bad else 0)
