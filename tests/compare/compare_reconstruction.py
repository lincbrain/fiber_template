#!/usr/bin/env python3
"""Compare a reconstruction with another (e.g. yours with the paper's), to check that a rerun reproduces it.

    $PY tests/compare/compare_reconstruction.py <ID> <your sub-<ID>/blockface> <the other sub-<ID>/blockface>
                                        [--min-r R --min-dice D --max-shift-mm M [--stack-min-r R2 --stack-min-dice D2]]
    e.g. $PY tests/compare/compare_reconstruction.py MR252 $LINC_REBUILD_DATA/sub-MR252/blockface $LINC_RECORDS/data/sub-MR252/blockface

You need
  the two blockface folders.
What it does
  It compares every level of the OME-Zarr stores (the arrays and their scale and translation), the NIfTI products (arrays and
  affines) and <ID>.info. The stores are read one block of sections at a time, so a few GB of memory suffice.
  With the three tolerance options, an item that is not identical is measured instead: its arrays by Pearson r (the volumes, over
  the voxels nonzero in either) or Dice (the masks and tubes, voxels > 0), its grid by the largest difference in origin (mm; the
  scales must be equal). It is within the tolerance when r >= R or Dice >= D, and the shift <= M. <ID>.info is then listed, not
  tested. For a brain whose reconstruction of record cannot be reproduced exactly from the photographs (see its plan note).
  Two stores cropped differently (their shapes differ) are placed on the union of their grids by world position, zero outside
  each crop, and measured there; their shift is then the offset between the grids beyond whole pixels. --stack-min-r and
  --stack-min-dice set other limits for the stack that `prepare` writes (02_blockface, 02_blockface_mask), which is cropped and
  masked by the photograph masks, when those are made again by the network (default: R and D).
You get
  one line per item (whether the arrays are identical and the transforms or affines equal, or which side is missing; the
  measures when they differ under a tolerance), then a last line "<ID>: <n> of <N> items identical" (and how many within the
  tolerance); exit status 0 when all are identical, or identical or within the tolerance.
"""
import argparse
import os
import sys
import numpy as np, nibabel as nib, zarr

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("S"); ap.add_argument("new"); ap.add_argument("rec")
ap.add_argument("--min-r", type=float); ap.add_argument("--min-dice", type=float); ap.add_argument("--max-shift-mm", type=float)
ap.add_argument("--stack-min-r", type=float); ap.add_argument("--stack-min-dice", type=float)
a = ap.parse_args()
S, new, rec = a.S, a.new, a.rec
tol = (a.min_r, a.min_dice, a.max_shift_mm)
if any(t is not None for t in tol) and not all(t is not None for t in tol):
    sys.exit("give all three of --min-r, --min-dice and --max-shift-mm, or none")
tol = all(t is not None for t in tol)
if not tol and (a.stack_min_r is not None or a.stack_min_dice is not None):
    sys.exit("--stack-min-r and --stack-min-dice go with --min-r, --min-dice and --max-shift-mm")
STACK = ("02_blockface.ome.zarr", "02_blockface_mask.ome.zarr")   # the stack `prepare` writes, cropped and masked by the photograph masks
bad = n = within = 0
SECTIONS = 32                                            # sections read at a time (the stores' axis 2)
AGREE = 4                                                # ... and measured at a time under a tolerance (float64)


def ome(p):
    g = zarr.open_group(p, mode="r"); m = g.attrs.asdict().get("ome", g.attrs.asdict()).get("multiscales") or g.attrs.asdict().get("multiscales")
    return g, m[0]["datasets"]


def same_array(A, B):
    """Two zarr arrays equal, read in blocks of SECTIONS sections along axis 2 (c, rows, sections, columns)."""
    if A.shape != B.shape or A.dtype != B.dtype:
        return False
    n = A.shape[2]
    return all(np.array_equal(A[:, :, s:s + SECTIONS], B[:, :, s:s + SECTIONS]) for s in range(0, n, SECTIONS))


class Agreement:
    """Pearson r (over voxels nonzero in either) or Dice (voxels > 0), accumulated block by block."""
    def __init__(self, dice, stack=False):
        self.dice, self.s = dice, np.zeros(6)
        r, d = (a.stack_min_r, a.stack_min_dice) if stack else (None, None)
        self.limit = (a.min_dice if d is None else d) if dice else (a.min_r if r is None else r)

    def add(self, x, y):
        if self.dice:
            x, y = x > 0, y > 0
            self.s[:3] += (np.count_nonzero(x & y), np.count_nonzero(x), np.count_nonzero(y))
        else:
            m = (x != 0) | (y != 0); x, y = x[m].astype(np.float64), y[m].astype(np.float64)
            self.s += (x.size, x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum())

    def value(self):
        if self.dice:
            i, p, q = self.s[:3]; return 2 * i / (p + q) if p + q else 1.0
        k, sx, sy, sxx, syy, sxy = self.s
        return (k * sxy - sx * sy) / np.sqrt(max((k * sxx - sx * sx) * (k * syy - sy * sy), 1e-300))

    def words(self):
        return f"{'Dice' if self.dice else 'r'} {self.value():.6f}"

    def ok(self):
        return self.value() >= self.limit


def union_agree(A, B, ta, tb, dice, stack):
    """Two stores cropped differently (shapes differ, scales equal): both placed on the union of their grids by world position
    (zero outside each crop) and measured there, block by block. The offset between the grids is rounded to whole pixels; the
    leftover fraction of a pixel is returned with the Agreement."""
    o = np.array(ta, float) - np.array(tb, float)             # B index = A index + o (world = (index + translation) * scale)
    k = np.rint(o).astype(int)
    lo = np.minimum(0, -k); hi = np.maximum(A.shape, np.array(B.shape) - k)   # the union, in A's indices
    agree = Agreement(dice, stack)
    for z0 in range(lo[2], hi[2], AGREE):
        z1 = min(z0 + AGREE, hi[2])
        shape = (*(hi[:2] - lo[:2]), z1 - z0, hi[3] - lo[3])
        U = []
        for X, off in ((A, np.zeros(4, int)), (B, -k)):         # X's index 0 sits at A index `off`
            u = np.zeros(shape, X.dtype)
            a0, a1 = max(z0, off[2]), min(z1, off[2] + X.shape[2])
            if a1 > a0:
                p = off - lo
                u[p[0]:p[0] + X.shape[0], p[1]:p[1] + X.shape[1], a0 - z0:a1 - z0, p[3]:p[3] + X.shape[3]] = \
                    np.asarray(X[:, :, a0 - off[2]:a1 - off[2]])
            U.append(u)
        agree.add(*U)
    return agree, np.abs(o - k)


def judge(label, same, grid_equal, agree, shift, scales_equal, how="origin shift"):
    """Count one item: identical, within the tolerance, or differs; print its line."""
    global bad, n, within
    n += 1
    words = f"{label} array identical {same} | {'transforms' if '.zarr' in label else 'affine'} equal {grid_equal}"
    if same and grid_equal:
        print(words); return
    if tol and agree is not None and scales_equal:
        ok = (same or agree.ok()) and shift <= a.max_shift_mm
        within += ok; bad += not ok
        print(f"{words} | {'identical' if same else agree.words()}, {how} {shift:.4f} mm: {'within' if ok else 'outside'} "
              "the tolerance")
        return
    bad += 1; print(words)


for st in (f"{S}_blockface.ome.zarr", "02_blockface.ome.zarr", "02_blockface_mask.ome.zarr", f"steps/{S}_03_color.ome.zarr"):
    pa, pb = os.path.join(new, st), os.path.join(rec, st)
    if not (os.path.exists(pa) and os.path.exists(pb)):
        print(f"{st:34s} missing: new {os.path.exists(pa)} record {os.path.exists(pb)}"); bad += 1; n += 1; continue
    (ga, da), (gb, db) = ome(pa), ome(pb)
    for x, y in zip(da, db):
        A, B = ga[x["path"]], gb[y["path"]]
        same = same_array(A, B)
        tx, ty = x.get("coordinateTransformations"), y.get("coordinateTransformations")
        agree = shift = scales_equal = None
        label, how = f"{st:34s} L{x['path']}:", "origin shift"
        if tol and not (same and tx == ty) and A.shape == B.shape:
            agree = Agreement(dice="mask" in st, stack=st in STACK)
            if not same:
                for s in range(0, A.shape[2], AGREE):
                    agree.add(np.asarray(A[:, :, s:s + AGREE]), np.asarray(B[:, :, s:s + AGREE]))
            sa, sb = np.array(tx[1]["scale"], float), np.array(ty[1]["scale"], float)
            scales_equal = np.allclose(sa, sb, rtol=1e-9)
            shift = float(np.abs(sa * np.array(tx[0]["translation"], float) - sb * np.array(ty[0]["translation"], float)).max()) / 1000.0
        elif tol and A.ndim == B.ndim == 4 and A.shape[0] == B.shape[0]:   # cropped differently: compared by world position
            sa, sb = np.array(tx[1]["scale"], float), np.array(ty[1]["scale"], float)
            scales_equal = np.allclose(sa, sb, rtol=1e-9)
            if scales_equal:
                agree, frac = union_agree(A, B, tx[0]["translation"], ty[0]["translation"], "mask" in st, st in STACK)
                shift = float((frac * np.abs(sa)).max()) / 1000.0
                how = f"cropped differently ({'x'.join(map(str, A.shape[1:]))} and {'x'.join(map(str, B.shape[1:]))}), " \
                      "measured by world position; grid offset beyond whole pixels"
        judge(label, same, tx == ty, agree, shift, scales_equal, how)
for f in (f"{S}_mask.nii.gz", f"{S}_holes.nii.gz", f"{S}_blockface_L2.nii.gz"):
    pa, pb = os.path.join(new, f), os.path.join(rec, f)
    if not (os.path.exists(pa) and os.path.exists(pb)):
        print(f"{f:34s} missing: new {os.path.exists(pa)} record {os.path.exists(pb)}"); bad += 1; n += 1; continue
    ia, ib = nib.load(pa), nib.load(pb)
    A, B = np.asanyarray(ia.dataobj), np.asanyarray(ib.dataobj)
    same, aff = A.shape == B.shape and np.array_equal(A, B), np.allclose(ia.affine, ib.affine, atol=1e-6)
    agree = shift = scales_equal = None
    if tol and not (same and aff) and A.shape == B.shape:
        agree = Agreement(dice="_blockface" not in f)
        if not same:
            for s in range(0, A.shape[1], 4 * AGREE):                  # NIfTI order (W, sections, H[, c])
                agree.add(A[:, s:s + 4 * AGREE], B[:, s:s + 4 * AGREE])
        scales_equal = np.allclose(ia.affine[:3, :3], ib.affine[:3, :3], atol=1e-6)
        shift = float(np.abs(ia.affine[:3, 3] - ib.affine[:3, 3]).max())
    judge(f"{f:34s}", same, aff, agree, shift, scales_equal)
pa, pb = os.path.join(new, f"{S}.info"), os.path.join(rec, f"{S}.info")
same = os.path.exists(pa) and os.path.exists(pb) and open(pa).read() == open(pb).read()
if tol and not same and os.path.exists(pa) and os.path.exists(pb):
    la, lb = dict(l.split("=", 1) for l in open(pa).read().split()), dict(l.split("=", 1) for l in open(pb).read().split())
    diff = [f"{k} {la.get(k)} (record {lb.get(k)})" for k in sorted(set(la) | set(lb)) if la.get(k) != lb.get(k)]
    print(f"{S}.info identical False | listed, not tested: {'; '.join(diff)}")
else:
    bad += not same; n += 1
    print(f"{S}.info identical {same}")
if tol:
    stk = "" if a.stack_min_r is None and a.stack_min_dice is None else \
        f"; the stack r >= {a.min_r if a.stack_min_r is None else a.stack_min_r:g}, Dice >= {a.min_dice if a.stack_min_dice is None else a.stack_min_dice:g}"
    print(f"{S}: {n - bad - within} of {n} items identical, {within} within the tolerance (r >= {a.min_r:g}, Dice >= {a.min_dice:g}, "
          f"origin shift <= {a.max_shift_mm:g} mm{stk}), {bad} outside")
else:
    print(f"{S}: {n - bad} of {n} items identical")
sys.exit(1 if bad else 0)
