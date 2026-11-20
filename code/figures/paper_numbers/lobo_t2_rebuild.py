#!/usr/bin/env python3
"""Leave one brain out: does a brain the template never saw align as well as the brains it was built from? (the out-of-sample check)

You need (settings/internal/products_mri.yaml)
  mri_runs: one fold run per brain of settings/mri_construction.yaml `inputs` (rb_lobo_<ID>/, with its held-out registrations in
  out/heldout_t2/) and rb_final/data (the held-out inputs); FSL (FSLDIR).

What it does, per fold
  1. composes each input's map to the fold's final T2 template (affine + MMORF warp);
  2. warps each input's T2 and brain mask once;
  3. scores it against the template inside the template's mask (r, nmad, Dice, Jacobian percentiles, % outside [0.5, 2]);
  4. compares the held-out brain (and its mirror) with the included inputs; then pools the folds.

You get
  $LINC_NUMBERS/lobo_t2_rebuild.json, and the composed warps and warped images in $LINC_WORK/lobo_t2/<fold>/ (about 6 GB)

    python code/figures/paper_numbers/lobo_t2_rebuild.py [--workers 4] [--plan]

Paper: Methods, "Contribution of each MRI contrast and leave-one-brain-out evaluation"; Results, "Leave-one-brain-out evaluation".
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
from shared import paths, provenance  # noqa: E402
from figures.evaluation import lobo as LB  # noqa: E402

NAME = "lobo_t2_rebuild"
from template.tools import recipe  # noqa: E402

FOLDS = [f"rb_lobo_{S}" for S in recipe()["inputs"]]       # one fold per brain of settings/mri_construction.yaml `inputs`
STEP, IT = "08", "01"                   # the target: step 8, iteration 01 of each fold
KEYS = ("r", "nmad", "dice", "jac_p1", "jac_p99", "jac_outside")
MEAN_KEYS = ("r", "nmad", "dice")
DECIMALS = 5                            # rounding of the per-fold means and differences

# the record's own method text, verbatim (it is part of the record's data)
METHOD = [
 "the held-out brain + its mirror vs the fold's 6 included inputs, ALL solved the same way (T2-only, the fold's step-8 schedule) against",
 "the SAME target (the fold's step 8 iteration 01 T2 template). eval_heldout.py's method, with one deliberate difference stated: every",
 "subject's map is composed as premat (its affine into the fold's standard space) + its raw MMORF warp, WITHOUT the fold's unbiasing",
 "average_invwarp (the stopped folds have no complete iteration_02 average); the composition is identical for held-out and included, so",
 "the comparison stays matched. Metrics (eval_heldout.score_one, inside the target template's brain mask): r, nmad, dice, jac_p1/p99,",
 "jac_outside. Per fold: held-out mean minus included mean (paired within the fold, as eval_heldout insists).",
 "-> paper/numbers/lobo_t2_rebuild.json (a saved script recomputes every value; the 17 Sep LOBO (a) record is quoted as context)",
 "    mri/env/mmorf/bin/python mri/code/eval_lobo_t2.py [compose|score|all]",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4, help="parallel FSL compositions")
    ap.add_argument("--plan", action="store_true", help="print what this script reads, writes and needs; run nothing")
    a = ap.parse_args()
    runs = paths.products("products_mri")["mri_runs"]
    work = paths.work_root() / "lobo_t2"
    if a.plan:
        from figures.evaluation import plan
        return plan.show(NAME, env="settings/env/linc.yml (PY_MRI)", est="about 1 h with 4 workers (CPU; 32 compositions + warps, about 6 GB under $LINC_WORK)",
                         inputs={"fold runs": [runs / f for f in FOLDS], "held-out inputs": runs / "rb_final" / "data"},
                         outputs=[paths.numbers_root() / f"{NAME}.json"] + [work / f for f in FOLDS], needs=["FSLDIR (FSL 6.0.7.4: convertwarp, applywarp)"],
                         cmd="python code/figures/paper_numbers/lobo_t2_rebuild.py [--workers 4]")
    J = LB.jobs(runs, FOLDS)
    with ThreadPoolExecutor(a.workers) as ex:
        for sid, fold, ok in ex.map(lambda j: LB.compose_one(j, runs, work), J):
            print(fold, sid, "composed" if ok else "FAILED", flush=True)
            if not ok:
                sys.exit(f"composition failed: {fold} {sid}")
    res = {"method": METHOD, "folds": []}
    inputs = []
    for fold in FOLDS:
        tgt = runs / fold / "out" / f"nln_step_{STEP}" / f"iteration_{IT}"; stem = f"nln_template_{STEP}_{IT}"
        tp, mp = tgt / f"{stem}_scalar_mod_01_brain.nii.gz", tgt / f"{stem}_scalar_mod_01_brain_mask.nii.gz"
        T = LB.load(tp); M = LB.load(mp) > LB.MASK_THRESHOLD
        od, hd = work / fold, runs / fold / "out" / "heldout_t2"
        inputs += [tp, mp]
        rows = []
        for j in [j for j in J if j["fold"] == fold]:
            jac = hd / f"{j['sid']}_to_template_jac.nii.gz"
            inputs += [j["img"], j["msk"], j["aff"], hd / f"{j['sid']}_to_template_warp.nii.gz", jac]
            r = LB.score_one(od / f"{j['sid']}_T2_to_template.nii.gz", od / f"{j['sid']}_mask_to_template.nii.gz", jac, T, M)
            r.update(subject=j["sid"], group=j["group"]); rows.append(r)
        g = lambda k, grp: float(np.mean([r[k] for r in rows if r["group"] == grp]))  # noqa: E731
        diff = {k: round(g(k, "held-out") - g(k, "included"), DECIMALS) for k in KEYS}
        res["folds"].append(dict(fold=fold, target=f"nln_step_{STEP}/iteration_{IT}", rows=rows, held_out_minus_included=diff,
                                 held_out_mean={k: round(g(k, "held-out"), DECIMALS) for k in MEAN_KEYS},
                                 included_mean={k: round(g(k, "included"), DECIMALS) for k in MEAN_KEYS}))
        print(fold, "held-out", res["folds"][-1]["held_out_mean"], "included", res["folds"][-1]["included_mean"], "diff", diff, flush=True)
    for k in MEAN_KEYS:
        res[f"pooled_mean_of_fold_diffs_{k}"] = round(float(np.mean([f["held_out_minus_included"][k] for f in res["folds"]])), DECIMALS)
    provenance.write_numbers(NAME, res, inputs=inputs, md5=False, cmd=f"python code/figures/paper_numbers/{NAME}.py")


if __name__ == "__main__":
    main()
