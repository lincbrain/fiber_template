#!/usr/bin/env python3
"""Leave one brain out for the MRI template: does a brain the template never saw align as well as the brains it was built from?
(paper: Results "MRI template", out-of-sample check; Methods, "Leave one brain out").

Four folds, one per MRI brain. In each fold the held-out brain + its mirror and the fold's six included inputs are registered the same
way (T2 only, the fold's step-8 schedule) to the same target (the fold's step-8 iteration-01 T2 template); linc/evaluation/lobo.py
composes each map (affine + MMORF warp), warps each input's T2 and brain mask, and scores it against the target inside the target's
brain mask (r, nmad, dice, Jacobian percentiles and % outside [0.5, 2]). Per fold: the per-input rows, the held-out and included means
(r, nmad, dice; 5 decimals) and their difference (all six metrics; 5 decimals). Pooled: the mean of the four per-fold differences.
The composed warps and warped images go to $LINC_WORK/lobo_t2/<fold>/ (about 6 GB). Needs FSL ($FSLDIR).
Inputs (products_mri): mri_runs (rb_lobo_<ID>/ fold runs with their out/heldout_t2/ registrations, rb_final/data for the held-out
inputs).
-> $LINC_NUMBERS/lobo_t2_rebuild.json
    python reproduce/numbers/lobo_t2_rebuild.py [--workers 4]"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from linc import paths, provenance  # noqa: E402
from linc.evaluation import lobo as LB  # noqa: E402

NAME = "lobo_t2_rebuild"
FOLDS = ["rb_lobo_MF283", "rb_lobo_MF284", "rb_lobo_MF285", "rb_lobo_MF286"]
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
        from linc.evaluation import plan
        return plan.show(NAME, env="env/mri.yml (PY_MRI)", est="about 1 h with 4 workers (CPU; 32 compositions + warps, about 6 GB under $LINC_WORK)",
                         inputs={"fold runs": [runs / f for f in FOLDS], "held-out inputs": runs / "rb_final" / "data"},
                         outputs=[paths.numbers_root() / f"{NAME}.json"] + [work / f for f in FOLDS], needs=["FSLDIR (FSL 6.0.7.4: convertwarp, applywarp)"],
                         cmd="python reproduce/numbers/lobo_t2_rebuild.py [--workers 4]")
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
    provenance.write_numbers(NAME, res, inputs=inputs, md5=False, cmd=f"python reproduce/numbers/{NAME}.py")


if __name__ == "__main__":
    main()
