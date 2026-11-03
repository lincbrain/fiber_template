#!/usr/bin/env python3
"""The inputs of the precision weights of the blockface template of record (Methods, the blockface template's final average), read from
the measurements only.

    python reproduce/numbers/precision_weights_b1_d2b_s8.py

You need
  tau2       the variance between brains of photographed tissue, measured on the build's members with MRI
             (configs/products_blockface.yaml `bftemplate_step8_state`: precision_d2bs8/tau2.json, `tau2_used`)
  s2_syn(D)  the synthesis network's prediction error against the depth past the junction plane, and s2_interp, the error of an
             interpolated section (the synthesis record d3_A_sigma2_pred_flat.json)
What it does
  It computes what they imply: the weight of synthesized tissue relative to photographed tissue at 0.5, 2 and 8 mm (1/(tau2 + s2)
  over 1/tau2), and that of an interpolated section (tau2 / (tau2 + s2_interp)).
You get
  $LINC_NUMBERS/precision_weights_b1_d2b_s8.json. The record's `sources` and `units` name the input files as they were read
  (absolute paths), so the record regenerates identically only with LINC_PRODUCTS set to the analysis project.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from linc import paths, provenance  # noqa: E402

SIGMA2 = "d3_A_sigma2_pred_flat.json"


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
                             cmd="python reproduce/numbers/precision_weights_b1_d2b_s8.py")


if __name__ == "__main__":
    main()
