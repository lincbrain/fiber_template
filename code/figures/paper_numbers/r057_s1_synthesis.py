#!/usr/bin/env python3
"""The numbers of the synthesis network's supplement and of the synthesis term of the precision weights, as one record (paper:
Supplementary S1.2-S1.4; the paper reads isbi.*, precision.* and inference.*).

Nothing is computed here that another record already holds; every value is copied unchanged from the record that computed it:
  isbi.*        the synthesis network's validation, from the number records of the network's own paper (Gonzales et al. 2026, ISBI:
                isbi_numbers/{table1,extra,results,data,compute}_v3.json, dated snapshots; brain_labels_v4.json there = this paper's
                brain names for the 12 evaluation brains, checked below)
  precision.*   the precision rule's inputs (numbers_of_record/precision_weights_b1_d2b_s8.json; d3_A_sigma2_pred_flat.json), with
                the synthesis error variance at fixed depths under dot-free keys (the paper's tokens cannot hold a dot), and how the
                tau^2 map was measured (precision_tau2, checked against the rule's text)
  inference.*   the prediction and join parameters of the completed volumes (settings/completion.yaml)
Each input is listed in `sources` with its md5.
-> $LINC_NUMBERS/r057_s1_synthesis.json
    python code/figures/paper_numbers/r057_s1_synthesis.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
from shared import paths, provenance  # noqa: E402

LABELS = REPO / "settings" / "specimen_labels.json"
COMPLETION = REPO / "settings" / "completion.yaml"
BLOCKS = {
    "table1_v3": ["rows", "brains"],
    "extra_v3": ["slab_ssim", "slab_l1", "slab_psnr", "completion_minus", "pix2pix_sharpness", "atlas_coverage_pct", "seen_minus_unseen",
                 "depth", "per_mm", "alignment", "alignment_masked", "alignment_slab", "misregistration", "registration"],
    "results_v3": ["vs_best_fill", "proxy_minus_slab"],
    "data_v3": ["n_brains", "n_mf", "n_mr", "dstar", "missing", "truegap", "training", "k_folds", "voxel_mm", "input_voxel_mm"],
    "compute_v3": ["released", "leave_slab_out", "cross_validation", "completion_per_brain_s", "on_rtx6000ada"],
}
DEPTHS = {"mm0p5": 0.5, "mm1": 1.0, "mm2": 2.0, "mm4": 4.0, "mm7": 7.0, "mm8": 8.0}

# the record's own texts, verbatim (they are part of the record's data)
WHAT = ("Supplementary S1 (synthesis network) and the synthesis term of the precision weights: numbers copied unchanged from their "
        "records of record")
RULE = ("w = 1/(tau2 + sigma2): per brain and voxel, photographed part r[phi/tau2 + (1 - phi)/(tau2 + s2_interp)], synthesized "
        "part (1 - r) m/(tau2 + s2_syn(D)); r = the cross-fade weight, phi = the photographed share, D = mm past the junction "
        "plane, m = 1 for a brain with MRI and 0 for one without (bftemplate/finish_t1.py = linc/construction/blockface_finish.py)")
TAU2_RULE = ("median of the between-brain variance map of photographed tissue where some brain with MRI is synthesized "
             "(bftemplate/d2b/s8/precision_d2bs8/tau2.json tau2_used)")
S2_SYN_CURVE = ("per-depth MSE of the leave-slab-out network's joined prediction against the withheld photographs (12 brains, "
                "voxels >= 3 mm inside the photographed tissue), adjusted for brain, isotonic over the depths reached by >= 3 "
                "brains; constant below the first and beyond the last measured depth")
UNITS = "the photographs' brightness per brain, (B - p1)/(p99 - p1) over its photographed tissue"


def no_host(o):
    """a copied block without the machines' names (the GPU model stays): the release names no host"""
    if isinstance(o, dict):
        return {k: no_host(v) for k, v in o.items() if k != "host"}
    return [no_host(x) for x in o] if isinstance(o, list) else o


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def main():
    P = paths.products("products_completion")
    ISBI, NUM = P["isbi_numbers"], P["numbers_of_record"]
    inputs, sources, isbi = [], {}, {}
    for rec, keys in BLOCKS.items():
        p = ISBI / f"{rec}.json"; d = json.load(open(p)); d = d.get("data", d); inputs.append(p); sources[rec] = dict(path=str(p), md5=md5(p))
        missing = [k for k in keys if k not in d]
        assert not missing, f"{rec}: no {missing}"
        isbi[rec.replace("_v3", "")] = {k: no_host(d[k]) for k in keys}
    # the ISBI's brain names are this paper's (one name per brain across both papers)
    lp = ISBI / "brain_labels_v4.json"; l4 = json.load(open(lp)); l4 = l4.get("data", l4)["label"]; inputs.append(lp)
    lab = json.load(open(LABELS))["labels"]; inputs.append(LABELS)
    assert all(lab[S] == n for S, n in l4.items()) and len(l4) == 12, "ISBI brain names differ from this paper's"
    # the precision rule's inputs and the synthesis error variance at fixed depths
    pw = NUM / "precision_weights_b1_d2b_s8.json"; Pw = json.load(open(pw))["data"]; inputs.append(pw)
    s2p = NUM / "d3_A_sigma2_pred_flat.json"; S2 = json.load(open(s2p))["data"]; inputs.append(s2p)
    for p in (pw, s2p, lp, LABELS):
        sources[p.name.replace(".json", "")] = dict(path=str(p), md5=md5(p))
    syn = Pw["s2_syn"]
    assert syn["d_mm"] == S2["s2_syn"]["d_mm"] and syn["s2"] == S2["s2_syn"]["s2"] and syn["slope_beyond"] == 0.0
    d, v = np.asarray(syn["d_mm"], float), np.asarray(syn["s2"], float)
    f = lambda x: float(np.interp(x, d, v))                              # constant below d[0] and beyond d[-1] (slope 0)  # noqa: E731
    # how the tau^2 map was measured (the rule's text, values checked against it)
    tp = P["precision_tau2"]; tj = json.load(open(tp)); tj = tj.get("data", tj); inputs.append(tp)
    sources["tau2_json"] = dict(path=str(tp), md5=md5(tp))
    rule = tj["rule"]
    assert rule["photographed"] == "p = clip(provenance) * brain > 0.99" and rule["min_members"] == 6 and rule["map"].startswith("Gaussian sigma 2.0 mm")
    assert abs(tj["tau2_used"] - Pw["tau2"]) < 1e-15 and tj["tau2_used_rule"].startswith("median of the map where some member with MRI is synthetic")
    tau2_map = dict(photographed_share_min=0.99, min_brains=rule["min_members"], gauss_sigma_mm=2.0, rule=rule,
                    region_cm3=Pw["tau2_region"]["cm3"], measured_frac=Pw["tau2_region"]["measured_frac"])
    edge = S2["curves"]["edge3"]
    precision = dict(
        rule=RULE, tau2=Pw["tau2"], tau2_rule=TAU2_RULE, tau2_map=tau2_map,
        s2_interp=Pw["s2_interp"], s2_interp_n=S2["s2_interp_detail"]["n_members"],
        s2_syn={k: f(x) for k, x in DEPTHS.items()},
        s2_syn_measured_to_mm=float(d[-1]), s2_syn_from_mm=float(d[0]), s2_syn_min_brains=edge["min_subjects"],
        s2_syn_all_brains_to_mm=edge["all_subjects_to_mm"], s2_syn_curve=S2_SYN_CURVE,
        weight_ratio={"mm0p5": Pw["ratios_at"]["0.5mm"], "mm2": Pw["ratios_at"]["2mm"], "mm8": Pw["ratios_at"]["8mm"]},
        weight_ratio_interp=Pw["interp_ratio"], units=UNITS)
    # prediction and join parameters of the completed volumes (the code that reproduces them bitwise reads this file)
    cfg = yaml.safe_load(open(COMPLETION)); inputs.append(COMPLETION)
    sources["completion_yaml"] = dict(path=str(COMPLETION), md5=md5(COMPLETION))
    inference = dict(window_in_vox=cfg["window"], window_out_vox=2 * cfg["window"], overlap=cfg["overlap"],
                     overlap_pct=round(100 * cfg["overlap"]), blend=cfg["blend"], pava_bins=cfg["pava_bins"],
                     fully_photographed=cfg["fully_photographed"], input_percentiles=cfg["input_percentiles"],
                     stand_in_lo=cfg["stand_in_lo"], valid_trim_mm=cfg["training"]["valid_trim_mm"],
                     ms_ssim_scales=len(cfg["training"]["ms_ssim_weights"]), ms_ssim_weights=cfg["training"]["ms_ssim_weights"])
    data = dict(what=WHAT, isbi=isbi, precision=precision, inference=inference, sources=sources,
                evaluated=[l4[S] for S in sorted(l4, key=lambda s: (l4[s][:2], int(l4[s][2:])))])
    provenance.write_numbers("r057_s1_synthesis", data, inputs=inputs, cmd="python code/figures/paper_numbers/r057_s1_synthesis.py")
    print("tau2", precision["tau2"], "s2_interp", precision["s2_interp"], "s2_syn", {k: round(x, 5) for k, x in precision["s2_syn"].items()})
    print("ratios", precision["weight_ratio"], "interp", precision["weight_ratio_interp"])


if __name__ == "__main__":
    main()
