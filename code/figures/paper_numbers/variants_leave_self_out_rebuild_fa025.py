#!/usr/bin/env python3
"""Compare the seven MRI templates built from different channel combinations, without their two in-sample biases (Table 1 and the
supplementary template tables), so that the comparison does not favor a template for having seen the brain it is scored on.

You need (settings/internal/products_mri.yaml)
  crossmodal_cache: every template's warped inputs; template_mri: the template's supports and FA; mri_runs: the four-channel run's
  coverage mask and the evaluators' records; numbers_of_record: the old-metrics records.

What it does
  1. checks that it reproduces the evaluators' in-sample values (to 1e-9) before changing anything;
  2. computes the template-referenced measures (T2 NMI, fraction MAE, tensor overlap) in sample and leaving each brain out of its
     own reference;
  3. computes the pairwise measures (angular error between principal eigenvectors in white matter = FA > 0.25, Pearson r between the
     warped b0 images, Dice coefficient between the warped brain masks) on all pairs and on the pairs of
     distinct brains (a brain against its own mirror is left out);
  4. compares four-channel with T2 + tensor per brain, with an exact paired bootstrap over the brains;
  5. ranks the templates before and after the correction.

You get
  $LINC_NUMBERS/variants_leave_self_out_rebuild_fa025.json

    OMP_NUM_THREADS=4 python code/figures/paper_numbers/variants_leave_self_out_rebuild_fa025.py [--plan]

CPU only, about 30 minutes with 4 threads (paper: Methods, "Contribution of each MRI contrast and leave-one-brain-out evaluation").
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "4"
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
from shared import paths, provenance  # noqa: E402
from figures.evaluation import variants as V  # noqa: E402

NAME = "variants_leave_self_out_rebuild_fa025"
ARMS = [("rb_t2", "T2"), ("rb_micro", "SANDI"), ("rb_dti", "tensor"), ("rb_t2micro", "T2 + SANDI"),
        ("rb_t2dti", "T2 + tensor"), ("rb_dtimicro", "tensor + SANDI"), ("rb_final", "four-channel template")]
A4, AT = "rb_final", "rb_t2dti"               # four-channel template vs T2 + tensor (the paired comparison)
SUPPORT_RUN = "rb_final"                      # the common support = the four-channel run's deliverable core
WM_FA = 0.25                                  # white matter for the fiber angle: final-template FA > 0.25
ABS_TOL = 1e-9                                # reproduction tolerance of the in-sample values
HEADLINE = ["T2_NMI", "FRAC1_MAE", "DTI_OVL", "fiber_angle_deg", "B0_r", "mask_dice"]
MODS = ("T2", "FRAC1", "FRAC2", "DTI", "B0", "MASK")

# the record's own definitions, verbatim (they are part of the record's data)
DEFINITIONS = {
    "inputs": "8 = 4 brains MF283..MF286 + their exact left-right mirrors (suffix f); n = 4 independent brains",
    "source_volumes": "mri/data/_crossmodal/<arm>__<MOD>[_vr]__<id>.nii.gz: each input's canonical payload pushed through the "
                      "arm's composed full warp by eval_crossmodal.py (tensors: vecreg-canonicalised, tag _vr)",
    "in_sample": "the measures of record: eval_oldmetrics.eval_arm (template = mean of all 8 inputs) and eval_crossmodal "
                 "(mean over all 28 pairs); reproduced here to abs_tol before anything else",
    "leave_self_out": "reference for input s of brain b = mean of the 6 inputs of the other 3 brains (s and its mirror "
                      "excluded); built as eval_oldmetrics builds the template: T2 each input scaled to mean 1000 in the "
                      "whole support then plain mean; FRAC plain mean; tensor log-Euclidean mean (eigenvalue floor 1e-6 x "
                      "median MD of MF283/MF283f, as in eval_oldmetrics); a brain and its mirror share one reference",
    "T2_NMI": "FSL-like Parzen NMI (H(A)+H(B))/H(A,B), 64 bins, robust 0.5/99.5 pct rescale, on the whole support "
              "(template/mri/linc_mri_core_t2); _WM on template/mri/linc_mri_wm_fa025",
    "FRAC1_MAE": "mean |soma fraction - reference| on linc_mri_core_micro (paper's 'fraction MAE'); FRAC2 = neurite",
    "DTI_OVL": "Basser-Pajevic tensor overlap with the reference, mean over the whole support; _WM on linc_mri_wm_fa025",
    "fiber_angle_deg": "mean angle between principal eigenvectors of a pair of inputs, in the deliverable core (mmsym_model_dti "
                       "coverage >= 0.999, eroded 6) with final-template FA > 0.25 (the white-matter rule of the 14 Oct 2026 revision)",
    "B0_r": "Pearson r between the warped b0 images of a pair of inputs, in the deliverable core",
    "mask_dice": "Dice between a pair of warped model masks (>= 0.5), whole grid",
    "per_brain_template": "mean of the brain's two inputs (brain and mirror)",
    "distinct24_mean": "mean over the 24 pairs between distinct brains (= mean of the 6 unique brain-pair means, 4 pairs each)",
    "per_brain_distinct": "mean over the 12 distinct-brain pairs that involve the brain (or its mirror); NOT independent across "
                          "brains (each pair enters two brains)",
    "paired_four_channel_vs_t2tensor": "per brain: four-channel (mmsym_model_dti_v) minus T2 + tensor (tbl_t2dti_v); exact "
                                       "bootstrap = all 4^4 = 256 resamples of the 4 brains with replacement, equally weighted, "
                                       "quantiles by numpy linear interpolation; n = 4, so the interval is descriptive",
    "corrected": "leave-self-out for template-referenced measures, distinct-brain pairs for pairwise measures",
    "rankings": "arms ordered best first by the arm mean; 'moved' = arm: [in-sample rank, corrected rank]"}


def main():
    P = paths.products("products_mri")
    root = paths.products_root()
    cache, tpl, runs, num = P["crossmodal_cache"], P["template_mri"], P["mri_runs"], P["numbers_of_record"]
    om_rec = {a: num / f"table1_oldmetrics_{a}_rebuild_vr.json" for a, _ in ARMS}          # in-sample template-referenced values
    cx_rec = {a: runs / f"crossmodal_{a}_rebuild_vr_wmfa040_deliverable.json" for a, _ in ARMS}   # in-sample pairwise values
    if "--plan" in sys.argv[1:]:                                             # print what this script reads and writes; run nothing
        from figures.evaluation import plan
        return plan.show("variants_leave_self_out_rebuild_fa025", env="settings/env/linc.yml (PY_MRI); OMP_NUM_THREADS=4", est="about 30 min on 4 threads (CPU)",
                         inputs={"crossmodal_cache": cache, "template_mri": tpl, "mri_runs": runs, "numbers_of_record": num,
                                 "support run": runs / SUPPORT_RUN, "in-sample records": list(om_rec.values()) + list(cx_rec.values())},
                         outputs=[paths.numbers_root() / "variants_leave_self_out_rebuild_fa025.json"],
                         notes=["the warped inputs of the seven variants (<arm>__<MOD>__<ID>.nii.gz) are read from crossmodal_cache; nothing is registered"])

    sup = {"whole": V.load(tpl / "linc_mri_core_t2.nii.gz") > 0, "micro": V.load(tpl / "linc_mri_core_micro.nii.gz") > 0,
           "wm": V.load(tpl / "linc_mri_wm_fa025.nii.gz") > 0}
    common = V.coverage_core(V.final_iteration(runs / SUPPORT_RUN))
    tfa = np.asanyarray(nib.load(str(tpl / "linc_mri_FA.nii.gz")).dataobj)
    wm_angle = tfa[common] > WM_FA
    out = {"n_brains": 4, "n_inputs": 8, "ids": V.IDS, "abs_tol": ABS_TOL,
           "supports": {"whole": int(sup["whole"].sum()), "micro": int(sup["micro"].sum()), "wm_fa025": int(sup["wm"].sum()),
                        "pairwise_common": int(common.sum()), "fiber_angle_wm": int(wm_angle.sum())},
           "arms": {}, "reproduction": {}, "missing": []}
    inputs = []
    for arm, label in ARMS:
        need = [V.cached(cache, arm, m, s) for m in MODS for s in V.IDS]
        miss = [str(p) for p in need if not p.exists()]
        if miss:
            out["missing"].append({"arm": arm, "n_missing": len(miss), "example": miss[0]}); print("skip", arm, miss[0]); continue
        inputs += need
        t0 = time.time()
        tm = V.template_measures(cache, arm, sup)
        pm = V.pairwise_measures(cache, arm, common, wm_angle)
        print(f"{arm}: {time.time() - t0:.0f} s", flush=True)
        # ---- reproduction of the evaluators' in-sample values ----
        rec_om = json.loads(om_rec[arm].read_text())["data"]["arms"][arm]["per_subject"]
        assert [r["subject"] for r in rec_om] == V.IDS
        dev_om = max(abs(a[k] - b[k]) for a, b in zip(tm["in_sample"], rec_om) for k in a if k != "subject")
        rec_cx = json.loads(cx_rec[arm].read_text())["runs"][arm]
        mine = {m: float(np.mean(list(pm[m].values()))) for m in V.PMEAS}
        theirs = {"fiber_angle_deg": rec_cx["DTI"]["mean_pairwise_angle_deg"], "B0_r": rec_cx["B0"]["pairwise_r_mean"],
                  "mask_dice": rec_cx["MASK"]["pairwise_dice_mean"]}
        dev_cx = {m: abs(mine[m] - theirs[m]) for m in V.PMEAS}
        out["reproduction"][arm] = {"template_measures_record": str(om_rec[arm].relative_to(root)),
                                    "template_measures_max_abs_dev_per_input": dev_om,
                                    "pairwise_record": str(cx_rec[arm].relative_to(root)),
                                    "pairwise_abs_dev_of_28pair_mean": dev_cx}
        print(arm, "reproduction deviation: template-referenced", dev_om, "pairwise", dev_cx, flush=True)
        # the evaluators' fiber angle used FA > 0.4, so it is recorded but not asserted
        assert dev_om < ABS_TOL and max(v for k, v in dev_cx.items() if k != "fiber_angle_deg") < ABS_TOL, (arm, dev_om, dev_cx)
        ins, lso = V.summarize_template(tm["in_sample"]), V.summarize_template(tm["leave_self_out"])
        pw = V.summarize_pairs(pm)
        out["arms"][arm] = {"label": label,
                            "template_referenced": {"in_sample": ins, "leave_self_out": lso,
                                                    "change_lso_minus_in_sample": {m: lso[m]["mean"] - ins[m]["mean"] for m in V.TMEAS}},
                            "pairwise": pw,
                            "pairwise_change_distinct_minus_all28": {m: pw[m]["distinct24_mean"] - pw[m]["all28_mean"] for m in V.PMEAS}}
    arms = list(out["arms"])
    # ---- rankings: in sample vs corrected ----
    rk = {}
    for m in V.TMEAS + V.PMEAS:
        if m in V.TMEAS:
            a = {x: out["arms"][x]["template_referenced"]["in_sample"][m]["mean"] for x in arms}
            b = {x: out["arms"][x]["template_referenced"]["leave_self_out"][m]["mean"] for x in arms}
        else:
            a = {x: out["arms"][x]["pairwise"][m]["all28_mean"] for x in arms}
            b = {x: out["arms"][x]["pairwise"][m]["distinct24_mean"] for x in arms}
        ra, rb = V.ranking(a, m), V.ranking(b, m)
        rk[m] = {"in_sample_best_first": ra, "corrected_best_first": rb, "ranking_changed": ra != rb,
                 "moved": {x: [ra.index(x) + 1, rb.index(x) + 1] for x in arms if ra.index(x) != rb.index(x)}}
    out["rankings"] = rk
    # ---- paired four-channel vs T2 + tensor ----
    if A4 in out["arms"] and AT in out["arms"]:
        pc = {"corrected": {}, "in_sample": {}}
        for m in V.TMEAS:
            for kind, key in (("corrected", "leave_self_out"), ("in_sample", "in_sample")):
                pc[kind][m] = V.paired(out["arms"][A4]["template_referenced"][key][m]["per_brain"],
                                       out["arms"][AT]["template_referenced"][key][m]["per_brain"], m)
        for m in V.PMEAS:
            pc["corrected"][m] = V.paired(out["arms"][A4]["pairwise"][m]["per_brain_distinct"],
                                          out["arms"][AT]["pairwise"][m]["per_brain_distinct"], m)
        out["paired_four_channel_vs_t2tensor"] = pc
    out["headline"] = {x: {**{m: out["arms"][x]["template_referenced"]["leave_self_out"][m]["mean"] for m in HEADLINE if m in V.TMEAS},
                           **{m: out["arms"][x]["pairwise"][m]["distinct24_mean"] for m in HEADLINE if m in V.PMEAS}} for x in arms}
    out["definitions"] = DEFINITIONS
    inputs += [tpl / "linc_mri_core_t2.nii.gz", tpl / "linc_mri_core_micro.nii.gz", tpl / "linc_mri_wm_fa025.nii.gz",
               tpl / "linc_mri_FA.nii.gz"] + sorted(set(om_rec.values()) | set(cx_rec.values()))
    print(provenance.write_numbers(NAME, out, inputs=inputs, md5=False, cmd=f"python code/figures/paper_numbers/{NAME}.py"))


if __name__ == "__main__":
    main()
