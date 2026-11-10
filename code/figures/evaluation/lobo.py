"""Leave one brain out (LOBO) for the MRI template: the held-out brain against the brains a fold was built from (paper: Results
"MRI template", out-of-sample check; Methods, "Leave one brain out").

Each fold leaves one of the four MRI brains (and its mirror) out of construction. Every input of the fold, held-out and included, is
then registered the same way (T2 only, the fold's step-8 schedule) to the same target, the fold's step-8 iteration-01 T2 template.
Each input's map is composed as premat (its affine into the fold's standard space) + its raw MMORF warp, without the fold's unbiasing
average inverse warp (identically for held-out and included inputs, so the comparison stays matched).

Metrics (`score_one`, inside the target template's brain mask, the same mask for every input of the fold):
  r            Pearson correlation of the warped input with the template
  nmad         mean |input - template| / mean |template| (scale-sensitive where r is not)
  dice         warped brain mask (> 0.5) vs the template brain mask
  jac_p1/p99   1st / 99th percentile of the Jacobian determinant (finite, > 0)
  jac_outside  % of those voxels with determinant outside [0.5, 2]
Ported unchanged from the analysis project's held-out evaluator. FSL (6.0.7.4) is called through $FSLDIR/bin.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

MASK_THRESHOLD = 0.5                   # brain masks binarized at > 0.5
JAC_RANGE = (0.5, 2.0)                 # Jacobian determinants outside this range count as large deformation


def fsl(tool):
    d = os.environ.get("FSLDIR")
    if not d:
        raise RuntimeError("set FSLDIR (FSL 6.0.7.4, see README.txt)")
    return str(Path(d) / "bin" / tool)


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode:
        print(f"  ! {cmd[:110]}\n    {r.stderr[:300]}", file=sys.stderr)
    return r.returncode == 0


def load(p):
    return np.asanyarray(nib.load(str(p)).dataobj).astype(np.float32)


def ref_grid(fold_dir, construction_root):
    """The --ref grid the construction driver used for the fold's warp compositions (last *_nlnT_concat_unbiased_warps_mods.sh;
    the driver writes it relative to the construction root, the parent of the runs folder)."""
    concat = sorted((fold_dir / "out" / "scripts").glob("*_nlnT_concat_unbiased_warps_mods.sh"))
    for line in concat[-1].read_text().split(";"):
        if "--ref=" in line:
            return construction_root / line.split("--ref=")[1].split()[0]
    raise RuntimeError(fold_dir.name)


def jobs(runs, folds, final_run="rb_final"):
    """Every input of every fold: the held-out brain + mirror (images and affines from the held-out registration, inputs from the
    final run's data) and the fold's included inputs (from the fold's data and its affine step 2)."""
    out = []
    for fold in folds:
        F = runs / fold
        B = fold.split("lobo_")[-1]; ids = json.loads((F / "run_spec.json").read_text())["subject_ids"]
        for sid in [B, B + "f"] + ids:
            held = sid.rstrip("f") == B
            d = (runs / final_run / "data") if held else (F / "data")
            aff = (F / "out" / "heldout" / "affine" / f"{sid}_to_standardspace.mat") if held else \
                  (F / "out" / "affine_step2" / sid / "scalar_mod_01_to_standardspace.mat")
            out.append(dict(fold=fold, sid=sid, group="held-out" if held else "included", aff=aff,
                            img=d / "T2M" / sid / "T2M" / "T2M_unbiased_brain.nii.gz", msk=d / "T2M" / sid / "T2M" / "T2M_brain_mask.nii.gz"))
    return out


def compose_one(j, runs, out_root):
    """premat + raw MMORF warp (convertwarp --relout), then the T2 (spline) and the brain mask (trilinear) through it, into
    <out_root>/<fold>/. The raw warp is read from <fold>/out/heldout_t2/<sid>_to_template_warp.nii.gz."""
    F = runs / j["fold"]; ref = ref_grid(F, runs.parent); sid = j["sid"]
    od = Path(out_root) / j["fold"]; od.mkdir(parents=True, exist_ok=True)
    raw = F / "out" / "heldout_t2" / f"{sid}_to_template_warp.nii.gz"; full = od / f"{sid}_full_premat_warp.nii.gz"
    ok = sh(f"{fsl('convertwarp')} --ref={ref} --out={full} --premat={j['aff']} --warp1={raw} --relout")
    ok &= sh(f"{fsl('applywarp')} -i {j['img']} -r {ref} -o {od}/{sid}_T2_to_template.nii.gz -w {full} --interp=spline")
    ok &= sh(f"{fsl('applywarp')} -i {j['msk']} -r {ref} -o {od}/{sid}_mask_to_template.nii.gz -w {full} --interp=trilinear")
    return sid, j["fold"], ok


def score_one(img_p, mask_p, jac_p, T, M):
    """r / nmad / dice / Jacobian statistics of one warped input against the fold's template T inside its mask M."""
    V = load(img_p)
    out = {}
    a, b = V[M], T[M]
    out["r"] = float(np.corrcoef(a, b)[0, 1])
    out["nmad"] = float(np.abs(a - b).mean() / np.abs(b).mean())
    if Path(mask_p).exists():
        S = load(mask_p) > MASK_THRESHOLD
        out["dice"] = float(2 * (S & (M)).sum() / (S.sum() + M.sum()))
    if jac_p and Path(jac_p).exists():
        J = load(jac_p)[M]
        J = J[np.isfinite(J) & (J > 0)]
        out["jac_p1"] = float(np.percentile(J, 1))
        out["jac_p99"] = float(np.percentile(J, 99))
        out["jac_outside"] = float(((J < JAC_RANGE[0]) | (J > JAC_RANGE[1])).mean() * 100)
    return out
