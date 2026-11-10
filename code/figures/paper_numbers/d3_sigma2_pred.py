#!/usr/bin/env python3
"""sigma^2_pred(D) of the completion (code/blockface/synthesis/sigma2.py): the records d3_A_sigma2_pred(_flat) (own b0, the leave-slab-out
network d3_lso_A_s0) and d3p_A_sigma2_pred(_flat) (stand-in b0, d3p_lso_A_s0). Not read by the paper's text: the blockface template's
precision weights read d3_A_sigma2_pred_flat (code/figures/paper_numbers/precision_weights_b1_d2b_s8.py), and d3_methods reads s2_interp.
"_flat" = the same curve, constant beyond its last measured depth (the variant the construction used).
From the withheld-slab evaluation of record (products: completion_eval/<run>.json and its saved joined predictions
pred/<run>_<ID>_deployed.nii.gz) and the prepared brains. Depth tables -> $LINC_WORK/completion/<run>_depth_tables.json.
-> $LINC_NUMBERS/{d3_A,d3p_A}_sigma2_pred{,_flat}.json
    python code/figures/paper_numbers/d3_sigma2_pred.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
from shared import paths, provenance  # noqa: E402
from blockface.synthesis import CFG, data, sigma2  # noqa: E402

RUNS = {"d3_A": "d3_lso_A_s0", "d3p_A": "d3p_lso_A_s0"}

# the records' own texts, verbatim (they are part of the records' data)
WHAT = ("sigma^2_pred(D) of the d3 model (canonical run-113 frame, input 0.5 mm -> output 0.25 mm): MSE of its leave-slab-out "
        "deployed completion vs the withheld photographs, by D = mm past the simulated cut along the seam normal, 12 members, "
        "one seed, fitted as depth_mse.py (member-adjusted log MSE, isotonic, WLS slope beyond)")
WHAT_FLAT = "; flat beyond its last measured depth (option b, as design 1c's isbi_v21_A_sigma2_pred_flat_1c)"
UNITS = "cache grey (the batches' unrotated real_bf, blo/bhi) = the grey of the d3 products (export_v3.py)"
INDEX = ("mm caudal of the join (the clean cut face plane), along the seam normal (finish_t1: max(-seam_depth, 0) "
         "of the d3 products)")
INTERP_WHAT = "frame A-P neighbour (0.25 mm) real-vs-real MSE at the first scored depth of the slab, cache grey"
NOTES = {"d3p_A": ("CAVEAT: measured from d3p_lso_A_s0, trained and scored on the DEV pseudo-b0s (canon_d3p_dev: the development S2, "
                   "nonlinear_s2val), not on the PROD pseudo-b0s that d3p_full (the 8's production model) was trained on; the prod "
                   "model has no leave-slab-out run, so this is the closest measurement of the pseudo-input error. Same 12 withheld "
                   "slabs, method and edge3 rule as d3_A_sigma2_pred (own-b0 inputs). For a per-group precision finish: the 8 brains "
                   "without MRI (decision of 5 Oct).")}


def one(name, run, P):
    evp = P["completion_eval"] / f"{run}.json"
    ev = json.load(open(evp))
    assert ev["protocol"] == "lso", "sigma^2_pred comes from the leave-slab-out run"
    root = Path(data.recorded_path(ev["root"]))
    pred_of = lambda S: P["completion_eval"] / "pred" / f"{run}_{S}_deployed.nii.gz"  # noqa: E731
    members = CFG["training"]["members"]
    tabs, lags, nvox = sigma2.depth_tables(ev, lambda S: root / S, pred_of, members)
    curves = {v: sigma2.fit(sigma2.pooled_from(tabs[v])) for v in tabs}
    pools = {v: sigma2.pooled_from(tabs[v]) for v in tabs}
    head = curves["edge3"]
    lagv = [lg["mse"] for lg in lags.values()]
    out = dict(
        what=WHAT, run=run, root=ev["root"], frame=ev.get("frame"), author_spec=ev.get("author_spec"), author_frame=ev.get("author_frame"),
        units=UNITS, headline="edge3",
        curves=curves, pooled={v: [[p["d_mid"], p["mse_adj"], p["n_subj"], p["mse_mean"], p["bias_mean"]] for p in pools[v]] for v in pools},
        s2_syn=dict(d_mm=[float(d) for d, _ in head["isotonic"]], s2=[float(v) for _, v in head["isotonic"]],
                    slope_beyond=float(head["slope_per_mm"]), index=INDEX),
        s2_interp=float(np.mean(lagv)),
        s2_interp_detail=dict(value=float(np.mean(lagv)), n_members=len(lagv), per_member={S: lg["mse"] for S, lg in lags.items()},
                              what=INTERP_WHAT),
        caveats=[f"measured to {head['measured_bins_mm'][1]:g} mm with >= {head['min_subjects']} members; linear beyond", "one seed"])
    if name in NOTES:
        out["note"] = NOTES[name]
    ss = out["s2_syn"]
    assert set(ss) == {"d_mm", "s2", "slope_beyond", "index"} and len(ss["d_mm"]) == len(ss["s2"]) >= 2
    assert all(b > a for a, b in zip(ss["d_mm"], ss["d_mm"][1:])) and isinstance(out["s2_interp"], float)
    work = paths.work_root() / "completion"; work.mkdir(parents=True, exist_ok=True)
    tp = work / f"{run}_depth_tables.json"; tp.write_text(sigma2.tables_json(tabs, lags, nvox))
    preds = [pred_of(S) for S in members]
    provenance.write_numbers(f"{name}_sigma2_pred", out, inputs=[evp, tp] + preds, md5=False,
                             cmd="python code/figures/paper_numbers/d3_sigma2_pred.py")
    flat = json.loads(json.dumps(out)); flat["s2_syn"]["slope_beyond"] = 0.0
    flat["what"] = out["what"] + WHAT_FLAT
    provenance.write_numbers(f"{name}_sigma2_pred_flat", flat, inputs=[tp], md5=False, cmd="python code/figures/paper_numbers/d3_sigma2_pred.py")
    print(f"{name}: edge3 {head['rule']}; s2_interp {out['s2_interp']:.7f}")


def main():
    P = paths.products("products_completion")
    for name, run in RUNS.items():
        one(name, run, P)


if __name__ == "__main__":
    main()
