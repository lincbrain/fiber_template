#!/usr/bin/env python3
"""Compare a step of the MRI template's build with the paper's files, as docs/tutorials/01_mri.md and code/template/README.md state them.

    $PY tests/compare/compare_mri_build.py staged  <your sub-<ID>/mri> <the paper's sub-<ID>> [--min 17]
    $PY tests/compare/compare_mri_build.py nmt     <your sub-<ID>/mri> <the paper's sub-<ID>> [--min-agree 93]
    $PY tests/compare/compare_mri_build.py field   t2|micro <your sub-<ID>> <the paper's sub-<ID>> [--median 0.05] [--p99 0.3] [--largest L]
    $PY tests/compare/compare_mri_build.py frame   <your sub-<ID>/mri> <the paper's sub-<ID>> [--deg 1.0] [--mm 0.5]
    $PY tests/compare/compare_mri_build.py prepare <your prep/<ID>> <the paper's prep folder> [--t2 0.88 --b0 0.87 --frac 0.84 --tensor 0.98 --dice 0.97]
    $PY tests/compare/compare_mri_build.py run     <your run>/out <the paper's run>/out [--parts affine_step1 affine_step2 nln_step_01] [--min 202]
    $PY tests/compare/compare_mri_build.py run     <your run> <the paper's run> --parts config subject_ids.csv standardspace.nii.gz --min 16
    $PY tests/compare/compare_mri_build.py entries <your xfm folder> <the paper's xfm folder> <entry> [<entry> ...]

You need the step's outputs and the paper's (the paper's MRI folder of a brain is $LINC_PRODUCTS/mri/data/sub-<ID>, its run
$LINC_PRODUCTS/mri/tc/rb_final, its link entries $LINC_PRODUCTS/mri/xfm_rb_final/sub-<ID>/xfm).
What it does:
  staged   every staged image (dwi/, t2/, micro/) that the paper also has: voxel values and affine;
  nmt      the share of the brain's voxels (your b0 mask) where your SARM labels equal the paper's;
  field    the within-brain field (t2 or micro to the b0): distance between your field and the paper's inside your b0 mask, its
           median and 99th percentile (the largest, a single voxel, is reported and checked only with --largest);
  frame    the canonical frame: the angle between the two rotations and the mean distance between the two maps over the brain;
  prepare  the construction inputs: Pearson r inside both final masks for T2, b0, the two fractions and the tensor, and the masks' Dice;
  run      the MMORF run: every file of the named parts (folders or files) of the paper's run, byte for byte or voxel for voxel
           (--skip-ini leaves out the parameter files);
  entries  transform entries: every file but the JSON metadata, byte for byte or voxel for voxel; and QC "pass" in your entry.
You get one line per item and a last line with the evidence; the exit status is 0 only when the stated identity or tolerance holds.
"""
import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def load(p):
    return np.asanyarray(nib.load(str(p)).dataobj)


def same_image(a, b):
    x, y = nib.load(str(a)), nib.load(str(b))
    X, Y = np.asanyarray(x.dataobj), np.asanyarray(y.dataobj)
    return X.shape == Y.shape and np.array_equal(X, Y) and np.allclose(x.affine, y.affine, rtol=0, atol=0)


def same_file(a, b):
    if Path(a).read_bytes() == Path(b).read_bytes():
        return True
    return str(a).endswith((".nii.gz", ".nii")) and same_image(a, b)


def finish(ok, line):
    print(line)
    sys.exit(0 if ok else 1)


def staged(a):
    mine, paper = Path(a.mine), Path(a.paper)
    n = bad = 0
    for sub in ("dwi", "t2", "micro"):
        for f in sorted((mine / sub).glob("*.nii.gz")):
            ref = paper / sub / f.name
            if not ref.exists():
                continue
            n += 1
            ok = same_image(f, ref)
            bad += not ok
            print(f"{sub}/{f.name}: {'identical' if ok else 'differs'}")
    finish(bad == 0 and n >= a.min, f"{n - bad} of {n} staged images identical to the paper's (at least {a.min} expected)")


def nmt(a):
    mine, paper = Path(a.mine), Path(a.paper)
    m = load(mine / "masks" / "dwi_brain_mask.nii.gz") > 0
    x, y = load(mine / "landmarks" / "SARM6_in_subject.nii.gz"), load(paper / "landmarks" / "SARM6_in_subject.nii.gz")
    agree = 100 * float((x[m] == y[m]).mean())
    finish(agree >= a.min_agree, f"SARM labels agree with the paper's on {agree:.1f} % of the brain's voxels (at least {a.min_agree:g} % required)")


def field(a):
    mine, paper = Path(a.mine), Path(a.paper)
    entry = f"xfm/{a.kind}-native__to__b0-native"
    f = lambda d: sorted((d / entry).glob("*.nii.gz"))[0]                       # noqa: E731
    x, y = np.squeeze(load(f(mine))).astype(np.float64), np.squeeze(load(f(paper))).astype(np.float64)
    m = load(mine / "mri" / "masks" / "dwi_brain_mask.nii.gz") > 0
    d = np.linalg.norm(x[m] - y[m], axis=-1)
    med, p99, top = float(np.median(d)), float(np.percentile(d, 99)), float(d.max())
    print(f"{a.kind} field against the paper's, inside the b0 mask: median {med:.3f} mm, 99th percentile {p99:.3f} mm, largest {top:.3f} mm")
    ok = med <= a.median and p99 <= a.p99 and (a.largest is None or top <= a.largest)
    finish(ok, f"{a.kind} field: median {med:.3f} mm (at most {a.median:g}), 99th percentile {p99:.3f} mm (at most {a.p99:g}), largest {top:.2f} mm"
               + (f" (at most {a.largest:g})" if a.largest is not None else ""))


def frame(a):
    mine, paper = Path(a.mine), Path(a.paper)
    T = lambda d: np.asarray(json.loads((d / "landmarks" / "canonical.json").read_text())["T_subject_to_canonical"], float)  # noqa: E731
    A, B = T(mine), T(paper)
    R = A[:3, :3] @ np.linalg.inv(B[:3, :3])
    deg = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    img = nib.load(str(mine / "masks" / "dwi_brain_mask.nii.gz"))
    ijk = np.argwhere(np.asanyarray(img.dataobj) > 0)[::50]
    xyz = np.c_[ijk, np.ones(len(ijk))] @ img.affine.T                        # world (RAS) coordinates of the brain's voxels
    mm = float(np.linalg.norm((xyz @ A.T - xyz @ B.T)[:, :3], axis=1).mean())
    finish(deg <= a.deg and mm <= a.mm, f"canonical frame against the paper's: {deg:.2f} deg (at most {a.deg:g}), mean move {mm:.2f} mm over the brain (at most {a.mm:g})")


def prepare(a):
    mine, paper = Path(a.mine), Path(a.paper)
    x, y = load(mine / "model_mask.nii.gz") > 0, load(paper / "model_mask.nii.gz") > 0
    dice = 2 * float((x & y).sum()) / float(x.sum() + y.sum())
    m = x & y
    need = {"t2_modelmask": a.t2, "b0": a.b0, "frac1": a.frac, "frac2": a.frac, "tensor_vr": a.tensor}
    got, ok = {}, dice >= a.dice
    for f, lo in need.items():
        X, Y = load(mine / f"{f}.nii.gz"), load(paper / f"{f}.nii.gz")
        r = float(np.corrcoef(X[m].ravel(), Y[m].ravel())[0, 1])
        got[f] = r
        ok &= r >= lo
        print(f"{f}: r {r:.3f} (at least {lo:g})")
    finish(ok, "inside both masks r: T2 {t2_modelmask:.3f}, b0 {b0:.3f}, fractions {frac1:.3f} and {frac2:.3f}, tensor {tensor_vr:.3f}; "
           "mask Dice {d:.3f} (at least {dm:g})".format(d=dice, dm=a.dice, **got))


def run(a):
    mine, paper = Path(a.mine), Path(a.paper)
    if not paper.exists():
        finish(False, f"nothing to compare with: the data folder does not hold the paper's run ({paper})")
    n = bad = missing = 0
    for part in a.parts:
        top = paper / part
        for ref in ([top] if top.is_file() else sorted(top.rglob("*"))):
            if not ref.is_file() or (a.skip_ini and ref.suffix == ".ini"):
                continue
            n += 1
            f = mine / ref.relative_to(paper)
            if not f.exists():
                missing += 1
                print(f"{ref.relative_to(paper)}: missing")
            elif not same_file(f, ref):
                bad += 1
                print(f"{ref.relative_to(paper)}: differs")
    same = n - bad - missing
    finish(bad == 0 and missing == 0 and n >= a.min,
           f"{same} of {n} files of {', '.join(a.parts)} identical to the paper's run ({missing} missing; at least {a.min} expected)")


def entries(a):
    mine, paper = Path(a.mine), Path(a.paper)
    n = bad = 0
    for e in a.entries:
        meta = mine / e / "meta.json"
        qc = json.loads(meta.read_text()).get("qc", {}) if meta.exists() else {}
        status = qc.get("status") if isinstance(qc, dict) else qc
        if status != "pass":
            bad += 1
            print(f"{e}: QC {status!r} in {meta}")
        for ref in sorted((paper / e).iterdir()):
            if ref.suffix == ".json":
                continue
            n += 1
            f = mine / e / ref.name
            ok = f.exists() and same_file(f, ref)
            bad += not ok
            print(f"{e}/{ref.name}: {'identical' if ok else 'differs or missing'}")
    finish(bad == 0 and n > 0, f"{n - bad} of {n} files of {len(a.entries)} entries identical to the paper's, QC pass" if bad == 0
           else f"{bad} problems in {len(a.entries)} entries ({n} files compared)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    p = sub.add_parser("staged"); p.add_argument("mine"); p.add_argument("paper"); p.add_argument("--min", type=int, default=17)
    p = sub.add_parser("nmt"); p.add_argument("mine"); p.add_argument("paper"); p.add_argument("--min-agree", type=float, default=93.0)
    p = sub.add_parser("field"); p.add_argument("kind", choices=["t2", "micro"]); p.add_argument("mine"); p.add_argument("paper")
    p.add_argument("--median", type=float, default=0.05); p.add_argument("--p99", type=float, default=0.3)
    p.add_argument("--largest", type=float, default=None)
    p = sub.add_parser("frame"); p.add_argument("mine"); p.add_argument("paper")
    p.add_argument("--deg", type=float, default=1.0); p.add_argument("--mm", type=float, default=0.5)
    p = sub.add_parser("prepare"); p.add_argument("mine"); p.add_argument("paper")
    for k, v in (("t2", 0.88), ("b0", 0.87), ("frac", 0.84), ("tensor", 0.98), ("dice", 0.97)):
        p.add_argument(f"--{k}", type=float, default=v)
    p = sub.add_parser("run"); p.add_argument("mine"); p.add_argument("paper")
    p.add_argument("--parts", nargs="+", default=["affine_step1", "affine_step2", "nln_step_01"]); p.add_argument("--min", type=int, default=202)
    p.add_argument("--skip-ini", action="store_true")
    p = sub.add_parser("entries"); p.add_argument("mine"); p.add_argument("paper"); p.add_argument("entries", nargs="+")
    a = ap.parse_args()
    {"staged": staged, "nmt": nmt, "field": field, "frame": frame, "prepare": prepare, "run": run, "entries": entries}[a.what](a)


if __name__ == "__main__":
    main()
