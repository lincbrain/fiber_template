#!/usr/bin/env python3
"""Does training on stand-in b0 inputs recover the domain-shift gap? (paper:
Supplementary Section S5.2, "Blockface synthesis network", the stand-in network; the paper reads
the --final record, d3p_vs_d3_pseudo_prod.summary.all.{n, mean_gain, mean_d3p_pseudo, mean_d3_pseudo, mean_d3_own,
mean_remaining_gap_to_own, ci95_remaining_gap}).

The decision rule was stated before any number of the stand-in network existed (RULE). On the 12 withheld slabs, paired by brain, with
the same scored regions, targets and joining code on both sides:
  d3p_pseudo   the stand-in leave-slab-out network (d3p_lso_A_s0) fed the stand-in b0: its withheld-slab evaluation of record
               (completion_eval/d3p_lso_A_s0.json, deployed SSIM and l1)
  d3_pseudo    the own-b0 leave-slab-out network (d3_lso_A_s0) fed the stand-in b0     } d3_domain_shift (per_brain.<ID>.pseudo / .own),
  d3_own       the same network fed the brain's own b0 (the ceiling)                   } the record of the paper
Mean SSIM gain d3p - d3 and the remaining gap d3p - own, with paired bootstrap 95 % CIs (10,000 resamples, seed 20261004), for all 12
brains and per species.
-> $LINC_NUMBERS/d3p_vs_d3_pseudo.json: the test of 4 Oct, on the development stand-in b0s; it chose the network that completed
   the 8 brains without MRI
    python code/figures/paper_numbers/d3p_vs_d3_pseudo.py
-> $LINC_NUMBERS/d3p_vs_d3_pseudo_prod.json: the same test on the final stand-in b0s (6 Nov), as the paper reports it. Both sides
   are fed the final stand-ins: d3p_pseudo from the stand-in network trained on them with the same recipe (d3p_lso_prod_A_s0,
   completion_eval/d3p_lso_prod_A_s0.json), d3_pseudo and d3_own from d3_domain_shift_prod (d3_domain_shift.py --final)
    python code/figures/paper_numbers/d3p_vs_d3_pseudo.py --final
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
from shared import paths, provenance  # noqa: E402

# the record's own texts, verbatim (they are part of the record's data)
RULE = ("Use d3p for the 8 if d3p_lso with pseudo inputs beats d3_lso with pseudo inputs on the 12 withheld slabs, paired by brain, by "
        "a mean SSIM gain whose 95 % CI lies above 0. Otherwise: d3_full as is, with the gap stated. Report per animal too "
        "(fascicularis vs rhesus), and the remaining gap to the own-b0 ceiling. One model for all 8 either way. (decision of 4 Oct)")
WHAT = {False: "step 2: d3p_lso (trained on the S2 dev pseudo-b0) vs d3_lso, both fed the pseudo-b0, on the 12 withheld slabs",
        True: ("rerun (6 Nov): d3p_lso_prod (trained on the final stand-in b0s, canon_d3p_prod) vs d3_lso, both fed the final "
               "stand-in b0s, on the 12 withheld slabs")}
NAME = {False: "d3p_vs_d3_pseudo", True: "d3p_vs_d3_pseudo_prod"}
EVAL = {False: "d3p_lso_A_s0.json", True: "d3p_lso_prod_A_s0.json"}      # completion_eval/<EVAL>
SHIFT = {False: "d3_domain_shift.json", True: "d3_domain_shift_prod.json"}  # numbers_of_record/<SHIFT>
DECISION = {True: "d3p (train d3p_full on the PROD pseudo-b0s; complete the 8 with it)", False: "d3_full as is for the 8 (the gap stated)"}


def boot_ci(d, rng, n):
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=10000); ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--final", action="store_true", help="the test on the final stand-in b0s: writes d3p_vs_d3_pseudo_prod")
    a = ap.parse_args()
    P = paths.products("products_completion")
    ep, dp = P["completion_eval"] / EVAL[a.final], P["numbers_of_record"] / SHIFT[a.final]
    ev = json.load(open(ep)); ds = json.load(open(dp))["data"]
    assert ev["protocol"] == "lso" and Path(ev["root"]).name.startswith("canon_d3p"), f"{ep}: not d3p_lso on a d3p root"
    if a.final:
        assert Path(ev["root"]).name == "canon_d3p_prod" and ds["pseudo_dir"] == "nonlinear_s2val_prod", "not the final stand-ins"
    ds = ds["per_brain"]
    rows = {}
    for S in sorted(ds):
        p_new = ev["subjects"][S]["deployed"]["ssim"]; p_old = ds[S]["pseudo"]["ssim"]; own = ds[S]["own"]["ssim"]
        assert ev["subjects"][S]["score_vox"] == ds[S]["score_vox"], f"{S}: scored regions differ"
        rows[S] = dict(animal="fascicularis" if S.startswith("MF") else "rhesus", d3p_pseudo=p_new, d3_pseudo=p_old, d3_own=own,
                       gain=p_new - p_old, remaining_gap_to_own=p_new - own,
                       l1=dict(d3p_pseudo=ev["subjects"][S]["deployed"]["l1"], d3_pseudo=ds[S]["pseudo"]["l1"], d3_own=ds[S]["own"]["l1"]))
    rng = np.random.default_rng(a.seed)
    subs = sorted(rows)
    g = np.array([rows[S]["gain"] for S in subs]); gap = np.array([rows[S]["remaining_gap_to_own"] for S in subs])
    summ = dict(all=dict(n=len(subs), mean_gain=float(g.mean()), ci95_gain=boot_ci(g, rng, a.boot),
                         mean_remaining_gap_to_own=float(gap.mean()), ci95_remaining_gap=boot_ci(gap, rng, a.boot),
                         mean_d3p_pseudo=float(np.mean([rows[S]["d3p_pseudo"] for S in subs])),
                         mean_d3_pseudo=float(np.mean([rows[S]["d3_pseudo"] for S in subs])),
                         mean_d3_own=float(np.mean([rows[S]["d3_own"] for S in subs]))))
    for an in ("fascicularis", "rhesus"):
        ss = [S for S in subs if rows[S]["animal"] == an]
        ga = np.array([rows[S]["gain"] for S in ss]); gp = np.array([rows[S]["remaining_gap_to_own"] for S in ss])
        summ[an] = dict(n=len(ss), brains=ss, mean_gain=float(ga.mean()), mean_remaining_gap_to_own=float(gp.mean()),
                        ci95_gain=boot_ci(ga, rng, a.boot) if len(ss) >= 3 else None)
    use_d3p = summ["all"]["ci95_gain"][0] > 0
    data = dict(what=WHAT[a.final], rule=RULE, decision=DECISION[bool(use_d3p)], use_d3p=bool(use_d3p), per_brain=rows, summary=summ)
    provenance.write_numbers(NAME[a.final], data, inputs=[ep, dp],
                             cmd="python code/figures/paper_numbers/d3p_vs_d3_pseudo.py" + (" --final" if a.final else ""))
    s = summ["all"]
    print(f"gain {s['mean_gain']:+.4f} CI [{s['ci95_gain'][0]:+.4f}, {s['ci95_gain'][1]:+.4f}] | d3p(pseudo) {s['mean_d3p_pseudo']:.4f} "
          f"d3(pseudo) {s['mean_d3_pseudo']:.4f} d3(own) {s['mean_d3_own']:.4f} | remaining gap {s['mean_remaining_gap_to_own']:+.4f}")
    print("decision:", data["decision"])


if __name__ == "__main__":
    main()
