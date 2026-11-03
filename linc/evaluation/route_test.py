"""The comparison of the two ways of aligning a photographed brain to a template, the blockface-based and the MRI-based alignment, on
brains the template being tested never saw (Methods, "Evaluation of the registration pipeline"; Results).

You need
  the comparison's runs (the folders are arguments). A brain that helped build an average cannot test it, and every brain helped
  build the blockface template, so the comparison uses an earlier blockface-only average (design 1c in the records): its six folds
  (each built without one pair of brains with MRI) and its final average with the four test brains it never saw (MF8, MF9, MF6,
  MM14 = MF191, MF196, MF275, MR263).
What it does
  Each brain is aligned to its target's 0.25 mm grid in two ways:
    blockface-based  blockface -> the target in one feature-based registration (linc/registration/direct.py to_blockface_template;
                     warp sigma 1.25, 4-8 mm padding, the brainstem cut flat at -2.0 mm)
    MRI-based        blockface -> own b0 (the registration of record) -> MRI template -> the target (the target's own mapping from
                     the template: the composition of its member warps, inverted onto its 0.5 mm frame); the test brains without
                     MRI go blockface -> MRI template by their cross-modal alignment of record
  Inside the pre-registered mask (the template's core, photographed, and inside both alignments' extent) it measures the
  normalized cross-correlation (NCC) between the aligned blockface images of every two brains aligned to the same target, and the
  Dice coefficient between their brain masks, with the pair as the unit; per brain, edge_mean (regeval.py), the envelope Dice,
  the normalized mutual information and the NCC with the template. Paired values use only rows whose two alignments passed their
  acceptance checks.
You get
  `summary`: the record's summary (its strings kept verbatim).

Ported from the analysis project's bftemplate/cv/route_test_1c.py (its `summary` stage, unchanged).
"""
import json
import os

import numpy as np

PAIRS = {0: ("MF278", "MR251"), 1: ("MF282", "MR243"), 2: ("MR250", "MR256"), 3: ("MR249", "MR253"), 4: ("MR257", "MR261"), 5: ("MR252", "MR254")}
TEST = ["MF191", "MF196", "MF275", "MR263"]
METRICS = ("pair_ncc", "pair_dice", "edge_mean", "env_dice", "nmi", "ncc_template")
TARGETS = [f"pair_{k}" for k in PAIRS] + ["final"]


def summary(out):
    """The route test's record from the per-target evaluation rows (<out>/<target>/rows.json) and links. Returns (record, inputs)."""
    rows, paths = [], []
    for key in TARGETS:
        p = f"{out}/{key}/rows.json"
        if os.path.exists(p):
            rows += json.load(open(p)); paths.append(p)
    gated = lambda r: bool(r["gates"].get("A") and r["gates"].get("B"))
    diffs, diffs_all = {}, {}
    for r in rows:
        for m in METRICS:
            if r["A"].get(m) is not None and r["B"].get(m) is not None:
                d = dict(subject=r["subject"], target=r["target"], A_minus_B=r["A"][m] - r["B"][m], gates_passed=gated(r))
                diffs_all.setdefault(m, []).append(d)
                if gated(r):
                    diffs.setdefault(m, []).append(d)
    pairs = {}                                                             # the pair as the unit: one value per fold
    for k, (S1, S2) in PAIRS.items():
        rr = {r["subject"]: r for r in rows if r["target"] == f"pair_{k}"}
        if S1 in rr and S2 in rr:
            q = {R: dict(ncc=rr[S1][R]["pairwise"][S2]["ncc"], dice=rr[S1][R]["pairwise"][S2]["dice"]) for R in ("A", "B")}
            pairs[f"pair_{k}"] = dict(brains=[S1, S2], A=q["A"], B=q["B"], A_minus_B=dict(ncc=q["A"]["ncc"] - q["B"]["ncc"], dice=q["A"]["dice"] - q["B"]["dice"]),
                                      gates_passed=gated(rr[S1]) and gated(rr[S2]))
    rng = lambda v: dict(n=len(v), min=float(min(v)), median=float(np.median(v)), max=float(max(v))) if v else None
    gp = [v for v in pairs.values() if v["gates_passed"]]
    test = {}
    tr = {r["subject"]: r for r in rows if r["target"] == "final"}
    if tr:
        pw = {R: [dict(a=S, b=S2, **tr[S][R]["pairwise"][S2]) for i, S in enumerate(TEST) for S2 in TEST[i + 1:] if S in tr and S2 in tr] for R in ("A", "B")}
        test = dict(brains=TEST, pairwise=pw, ncc=dict(A=rng([x["ncc"] for x in pw["A"]]), B=rng([x["ncc"] for x in pw["B"]])),
                    dice=dict(A=rng([x["dice"] for x in pw["A"]]), B=rng([x["dice"] for x in pw["B"]])),
                    MF275=dict(A={m: tr["MF275"]["A"][m] for m in METRICS}, B={m: tr["MF275"]["B"][m] for m in METRICS},
                               route_b=tr["MF275"].get("route_b"), gates=tr["MF275"]["gates"]) if "MF275" in tr else None,
                    route_b={S: tr[S].get("route_b") for S in tr}, gates={S: tr[S]["gates"] for S in tr})
    res = dict(n_rows=len(rows), rows=rows, design="descriptive: design 1c's six pair folds (the pair as the unit) + its final average with the "
               "four test brains; route A = direct (1c's R3 settings), route B = R1 + R5 (S2 for the test brains without MRI) + the target's "
               "own mapping from the MRI template (composition of member warps)",
               pairs=pairs, pairs_A_minus_B=dict(ncc=rng([v["A_minus_B"]["ncc"] for v in gp]), dice=rng([v["A_minus_B"]["dice"] for v in gp]),
                                                 n_folds=len(pairs), n_gates_passed=len(gp)),
               test_brains=test, paired_differences=diffs, paired_differences_all_rows=diffs_all,
               gating="paired values use only rows whose route-A regqc gate passed and whose route-B links all passed QC (R1, R5 or S2) "
                      "with the target's composition link not FAIL (outline) and inverted to < 0.1 mm p99; *_all_rows is descriptive",
               links={k: json.load(open(f"{out}/{k}/link/mri-template__to__bftemplate/maps.json")) for k in TARGETS
                      if os.path.exists(f"{out}/{k}/link/mri-template__to__bftemplate/maps.json")},
               lgn_stn="not available: no pair of design 1c has two labeled brains (MR243 pair 1, MR256 pair 2, MR252 pair 5)",
               notes=["5 Oct 2026: (i) the test brains without MRI take their S2 placement as their MRI route; MF275 reported separately; "
                      "(ii) the targets' mapping from the MRI template = the composition of their member warps"])
    return res, paths
