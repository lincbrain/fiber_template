#!/usr/bin/env python3
"""Compare a registration with the paper's: identical, or within a stated tolerance where a GPU rerun differs by floating point.

    $PY tests/compare/compare_registration.py entry <your entry> <the paper's entry> [--median-mm 0.05] [--p99-mm 0.5] [--qc]
    $PY tests/compare/compare_registration.py warp  <your warp> <the paper's warp> [--median-mm 0.05] [--p99-mm 0.5]
    $PY tests/compare/compare_registration.py image <your image> <the paper's image> [--mask <the paper's mask>] [--min-r 0.9999]

You need
  entry  two entries of the transform registry (code/shared/xfm.py), e.g. $LINC_DATA/sub-<ID>/xfm/b0-native__to__mri-template and the
         paper's, $LINC_DATA/sub-<ID>/xfm_record/b0-native__to__mri-template;
  warp   two displacement fields on the same grid (NIfTI, three components in mm);
  image  two images on the same grid (e.g. a template), and optionally the voxels to compare (a mask of the paper's).
What it does
  entry  every file the paper's meta.json lists: a text file (an affine) must be identical; a warp must be identical (bytes or
         voxels) or within the tolerance below. With --qc, your entry must also hold QC "pass" (the registration's own check).
  warp   the distance between the two displacements at each voxel of the grid, in mm: its median and 99th percentile must be at
         most --median-mm and --p99-mm (the largest is reported, not tested: single voxels can differ by more).
  image  identical voxels, or Pearson r inside the mask (else where either image is nonzero) of at least --min-r.
  The default tolerance is above what reruns of the paper's registrations gave on RTX 6000 Ada and RTX 4000 Ada cards (medians
  under 0.03 mm, 99th percentiles under 0.5 mm: README.txt, Exact reproduction), and far below a failed registration.
You get
  one line per file, then a last line with the evidence; exit status 0 only when every file is identical or within the tolerance.
"""
import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def load(p):
    h = nib.load(str(p))
    return h, np.asanyarray(h.dataobj)


def warp_diff(mine, paper):
    """The distance between two displacement fields per voxel (mm): None if their grids differ."""
    (hm, A), (hp, B) = load(mine), load(paper)
    if A.shape != B.shape or not np.allclose(hm.affine, hp.affine, atol=1e-6):
        return None
    A = A.reshape(*A.shape[:3], -1).astype(np.float64); B = B.reshape(*B.shape[:3], -1).astype(np.float64)
    return np.sqrt(((A - B) ** 2).sum(-1)).ravel()


def same_file(a, b):
    if Path(a).read_bytes() == Path(b).read_bytes():
        return True
    if str(a).endswith((".nii", ".nii.gz")):
        (ha, A), (hb, B) = load(a), load(b)
        return A.shape == B.shape and np.array_equal(A, B) and np.array_equal(ha.affine, hb.affine)
    return False


def judge_warp(mine, paper, a):
    """(status, words) for one warp: identical, within, or differs."""
    if same_file(mine, paper):
        return "identical", "identical"
    d = warp_diff(mine, paper)
    if d is None:
        return "differs", "differs (another grid)"
    med, p99, mx = np.median(d), np.percentile(d, 99), d.max()
    ok = med <= a.median_mm and p99 <= a.p99_mm
    return ("within" if ok else "differs"), f"median {med:.4f} mm, 99th percentile {p99:.4f} mm, largest {mx:.3f} mm"


def label(p):
    """The file with its folder (two maps can share a name)."""
    return "/".join(Path(p).parts[-2:])


def finish(ok, line):
    print(line)
    sys.exit(0 if ok else 1)


def entry(a):
    mine, paper = Path(a.mine), Path(a.paper)
    files = json.loads((paper / "meta.json").read_text())["files"]
    count = {"identical": 0, "within": 0, "differs": 0, "missing": 0}
    for name in sorted(files):
        f, ref = mine / name, paper / name
        if not f.exists():
            st, words = "missing", "missing"
        elif name.endswith((".nii", ".nii.gz")) and load(ref)[1].shape[-1] == 3:
            st, words = judge_warp(f, ref, a)
        else:
            st = "identical" if same_file(f, ref) else "differs"; words = st
        count[st] += 1
        print(f"{name}: {words}")
    qc_ok, qc = True, ""
    if a.qc:
        q = json.loads((mine / "meta.json").read_text()).get("qc", {}) if (mine / "meta.json").exists() else {}
        status = q.get("status") if isinstance(q, dict) else q
        qc_ok, qc = status == "pass", f"; QC {status}"
    finish(count["differs"] == 0 and count["missing"] == 0 and qc_ok,
           f"{mine.name}: {count['identical']} of {len(files)} files identical, {count['within']} within tolerance (median <= "
           f"{a.median_mm:g} mm, 99th percentile <= {a.p99_mm:g} mm), {count['differs']} outside, {count['missing']} missing{qc}")


def warp(a):
    st, words = judge_warp(a.mine, a.paper, a)
    finish(st != "differs", f"{label(a.mine)}: {st}: {words} (limits: median {a.median_mm:g} mm, 99th percentile {a.p99_mm:g} mm)")


def image(a):
    if same_file(a.mine, a.paper):
        finish(True, f"{label(a.mine)}: identical")
    (hm, A), (hp, B) = load(a.mine), load(a.paper)
    if A.shape != B.shape or not np.allclose(hm.affine, hp.affine, atol=1e-6):
        finish(False, f"{label(a.mine)}: differs (another grid)")
    m = (load(a.mask)[1] > 0) if a.mask else ((A != 0) | (B != 0))
    x, y = A[m].astype(np.float64), B[m].astype(np.float64)
    r = float(np.corrcoef(x, y)[0, 1]); top = max(float(np.percentile(np.abs(y), 99)), 1e-12)
    finish(r >= a.min_r, f"{label(a.mine)}: Pearson r {r:.7f} over {int(m.sum())} voxels (at least {a.min_r:g}); absolute difference "
                         f"median {100 * np.median(np.abs(x - y)) / top:.4f} %, largest {100 * np.abs(x - y).max() / top:.2f} % of the paper's "
                         f"99th percentile")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    for name in ("entry", "warp", "image"):
        p = sub.add_parser(name); p.add_argument("mine"); p.add_argument("paper")
        if name != "image":
            p.add_argument("--median-mm", type=float, default=0.05); p.add_argument("--p99-mm", type=float, default=0.5)
        if name == "entry":
            p.add_argument("--qc", action="store_true", help="your entry must hold QC pass")
        if name == "image":
            p.add_argument("--mask"); p.add_argument("--min-r", type=float, default=0.9999)
    a = ap.parse_args()
    {"entry": entry, "warp": warp, "image": image}[a.what](a)


if __name__ == "__main__":
    main()
