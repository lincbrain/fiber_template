#!/usr/bin/env python3
"""The inputs of the precision weights of the blockface template of record (Methods, the blockface template's final average), read from
the measurements only.

    python code/figures/paper_numbers/precision_weights_b1_d2b_s8.py

The published blockface template uses the weights of the second record, from the withheld-slab test made without the withheld tissue;
the first record holds the weights of the template files released on 16 Nov.

You need
  tau2       the variance between brains of photographed tissue, measured on the build's members with MRI
             (settings/internal/products_blockface.yaml `bftemplate_step8_state`: precision_d2bs8/tau2.json, `tau2_used`)
  s2_syn(D)  the synthesis network's prediction error against the depth past the junction plane, and s2_interp, the error of an
             interpolated section (the synthesis record d3_A_sigma2_pred_flat.json)
  s2_syn(D), published  the same error from the withheld-slab test made without the withheld tissue
             (sigma2_leakfree_a19.json, curves.lfr1; an input: the test's registrations are not in this release), and the rule file
             the chain reads (`bf_precision_rule`), which must hold that curve with this tau2 and s2_interp
What it does
  It computes what they imply: the weight of synthesized tissue relative to photographed tissue at 0.5, 2 and 8 mm (1/(tau2 + s2)
  over 1/tau2), and that of an interpolated section (tau2 / (tau2 + s2_interp)). For the published weights the curve is read at 0.5, 1,
  2, 4, 7 and 8 mm, constant below its first and beyond its last depth.
You get
  $LINC_NUMBERS/precision_weights_b1_d2b_s8.json, the 16 Nov rule; $LINC_NUMBERS/precision_weights_b1_d2b_s8_s11wb.json, the
  published weights. The records' `sources` and `units` name the input files as they were read (absolute paths), so the records regenerate
  identically only with LINC_PRODUCTS set to the analysis project.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
from shared import paths, provenance  # noqa: E402

SIGMA2 = "d3_A_sigma2_pred_flat.json"
SIGMA2_PUBLISHED = "sigma2_leakfree_a19.json"


def published_weights(prod, tau2, s2i, tpath, spath):
    """The record of the published weights (precision_weights_b1_d2b_s8_s11wb). The rule file the chain reads must be the 16 Nov rule
    with only s2_syn replaced, by the curve of the withheld-slab test made without the withheld tissue."""
    lpath, rpath = str(prod["numbers_of_record"] / SIGMA2_PUBLISHED), str(prod["bf_precision_rule"])
    c = json.load(open(lpath))["data"]["curves"]["lfr1"]; rule = json.load(open(rpath))
    rs = rule["s2_syn"]
    assert rs["d_mm"] == c["d_mm"] and rs["s2"] == c["s2"] and float(rs["slope_beyond"]) == 0.0, f"{rpath}: s2_syn is not {SIGMA2_PUBLISHED} curves.lfr1"
    assert float(rule["tau2"]) == tau2 and float(rule["s2_interp"]) == s2i, f"{rpath}: tau2 or s2_interp differs from the measurements"
    d, v = np.asarray(c["d_mm"], float), np.asarray(c["s2"], float)
    f = lambda x: float(np.interp(x, d, v))  # noqa: E731
    depths = {"mm0p5": 0.5, "mm1": 1.0, "mm2": 2.0, "mm4": 4.0, "mm7": 7.0, "mm8": 8.0}
    precision = dict(
        tau2=tau2, s2_interp=s2i, s2_syn={k: f(x) for k, x in depths.items()},
        s2_syn_measured_to_mm=float(d[-1]), s2_syn_from_mm=float(d[0]),
        s2_syn_flat_from_mm=float(d[int(np.flatnonzero(v != v[-1])[-1]) + 1]) if (v != v[-1]).any() else float(d[0]),
        s2_syn_curve=("per-depth MSE of the leave-slab-out network's joined prediction against the withheld photographs, on the leak-free "
                      "targets in which the whole registration was made without the withheld tissue (sigma2_leakfree_a19 curves.lfr1; 12 brains), "
                      "adjusted for brain, isotonic over the depths reached by >= 3 brains; constant below the first and beyond the last measured depth"),
        weight_ratio={k: round(tau2 / (tau2 + f(x)), 4) for k, x in (("mm0p5", 0.5), ("mm2", 2.0), ("mm8", 8.0))},
        weight_ratio_interp=round(tau2 / (tau2 + s2i), 4))
    out = dict(precision=precision, rule=rule, rule_file=rpath,
               scope="the end-of-step-8 average (its blockface channels) and every average of step 9 (the start, the three rounds, the final "
                     "average; out_s11w_full); MMORF's steps 1-8, the step-8 member warps and the step-8 b0 channel (weighted by the cost masks) "
                     "and the link stay of record")
    provenance.write_numbers("precision_weights_b1_d2b_s8_s11wb", out, inputs=[tpath, spath, lpath, rpath],
                             cmd="python code/figures/paper_numbers/precision_weights_b1_d2b_s8.py")


def main():
    prod = paths.products("products_blockface")
    tpath = str(prod["bftemplate_step8_state"] / "precision_d2bs8" / "tau2.json")
    spath = str(prod["numbers_of_record"] / SIGMA2)
    t2 = json.load(open(tpath)); t2 = t2.get("data", t2)
    sj = json.load(open(spath)); sj = sj.get("data", sj)
    syn = sj["s2_syn"]
    for k in ("d_mm", "s2", "slope_beyond"):
        assert k in syn, f"{SIGMA2} s2_syn has no '{k}' ({{d_mm, s2, slope_beyond, index}})"
    d, v = [float(x) for x in syn["d_mm"]], [float(x) for x in syn["s2"]]
    assert len(d) == len(v) >= 2 and all(x < y for x, y in zip(d, d[1:])), "s2_syn: d_mm must increase, one s2 per d"
    si = sj["s2_interp"]
    s2i = float(si["value"] if isinstance(si, dict) else si)
    tau2 = float(t2["tau2_used"])
    assert 0 < tau2 < 0.05 and 0 < s2i < 1 and min(v) > 0, (tau2, s2i, min(v))
    f = lambda x: (v[-1] + float(syn["slope_beyond"]) * (x - d[-1])) if x > d[-1] else \
        next(v[i] + (v[i + 1] - v[i]) * (x - d[i]) / (d[i + 1] - d[i]) for i in range(len(d) - 1) if x <= d[i + 1]) if x > d[0] else v[0]  # noqa: E731
    out = dict(tau2=tau2, s2_interp=s2i,
               s2_syn=dict(d_mm=d, s2=v, slope_beyond=float(syn["slope_beyond"]),
                           index=syn.get("index", "mm past the member's photographed end, column-wise (finish_t1: max(-seam_depth, 0))")),
               units="the final caches' grey (tau2 re-measured on the B1 stage-1 build; sigma2 from " + (
                   "the completion's S1 folds)" if os.path.basename(spath) == "s1_sigma2_pred.json" else f"{os.path.basename(spath)})"),
               sources=dict(tau2=f"{tpath} tau2_used", s2_syn=f"{spath} s2_syn", s2_interp=f"{spath} s2_interp"),
               tau2_region=dict(t2["tau2_map_summary"]["any_mri_member_synthetic"],
                                measured_frac=t2["tau2_map_summary"]["any_mri_member_synthetic_measured_frac"]),
               ratios_at={f"{x:g}mm": round((1 / (tau2 + f(x))) / (1 / tau2), 4) for x in (0.5, 2.0, 8.0)},
               interp_ratio=round(tau2 / (tau2 + s2i), 4))
    provenance.write_numbers("precision_weights_b1_d2b_s8", out, inputs=[tpath, spath],
                             cmd="python code/figures/paper_numbers/precision_weights_b1_d2b_s8.py")
    published_weights(prod, tau2, s2i, tpath, spath)


if __name__ == "__main__":
    main()
