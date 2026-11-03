#!/usr/bin/env python3
"""The numbers of the template-variant table (paper: Table 1 and Supplementary Tables of all metrics and of white matter).

The seven variants, one recipe, the same eight inputs; per variant, from the evaluators' records:
  new        pairwise r and CV of T2, pairwise r of soma / neurite fraction and of b0, warped-mask Dice, all in the deliverable core;
             fiber angle in the white matter common to every variant (FA > 0.4 in the four-channel template)
  new_wm     the same scalar measures in the deliverable core AND white matter (FA > 0.25)
  old        T2 NMI, fraction MAE, tensor overlap (whole and white matter), means over the eight inputs
  deformation  % of in-mask voxels with Jacobian determinant outside [0.5, 2] (mean over inputs), core voxels, step and iteration
  landmark_spread  median over the midline landmarks other than AC of the spread of the four specimens in the variant's space
plus the channels each variant was built with and the in-sample flags (a measure on a channel the variant was built with).
Numbers only: the rendered tables are made by the table scripts from this record.
Inputs (products_mri): mri_runs (crossmodal_<arm>_rebuild_vr_wmfa040_deliverable.json, crossmodal_<arm>_rebuild_vr_deliverable_wm.json,
<arm>/evaluation_multimodal.json, <arm>/run_spec.json), numbers_of_record (table1_oldmetrics_<arm>_rebuild_vr.json,
mri_template_landmark_spread_rebuild.json).
-> $LINC_NUMBERS/table1_rebuild.json
    python reproduce/numbers/table1_rebuild.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from linc import paths, provenance  # noqa: E402

NAME = "table1_rebuild"
ARMS = [("rb_t2", "T2"), ("rb_micro", "SANDI"), ("rb_dti", "tensor"), ("rb_t2micro", "T2 + SANDI"),
        ("rb_t2dti", "T2 + tensor"), ("rb_dtimicro", "tensor + SANDI"), ("rb_final", "four-channel template")]
# construction channel name -> measured modality
MOD_OF = {"T2": "T2", "T2M": "T2", "T2D": "T2", "FRAC1": "FRAC1", "FRAC1M": "FRAC1", "FRAC2": "FRAC2", "FRAC2M": "FRAC2",
          "DTI": "DTI", "DTIM": "DTI", "DTIV": "DTI", "B0M": "B0"}
NOTE = "new metrics on the deliverable core (whole); fibre angle on the common WM; old metrics whole + WM"   # verbatim (record data)


def built_with(runs, arm):
    """The modalities that drove the variant's registration (weight > 0); B0M enters some variants only as a weight-0 carrier."""
    spec = runs / arm / "run_spec.json"
    if not spec.exists():
        return None
    s = json.loads(spec.read_text())
    mods = {MOD_OF.get(m["name"], m["name"]) for m in s.get("modalities", []) if m.get("lambda_scalar", 1) > 0}
    mods |= {MOD_OF.get(m["name"], m["name"]) for m in s.get("tensor_modalities", []) if m.get("lambda_tensor", 1) > 0}
    return sorted(mods)


def main():
    P = paths.products("products_mri")
    root = paths.products_root()
    runs, num = P["mri_runs"], P["numbers_of_record"]
    rel = lambda p: str(p.relative_to(root))                                    # noqa: E731
    lmp = num / "mri_template_landmark_spread_rebuild.json"
    lm = json.loads(lmp.read_text())["data"] if lmp.exists() else None
    rows, inputs = {}, ([lmp] if lm else [])
    for arm, label in ARMS:
        r = {"label": label, "build": arm, "vecreg_tensors": arm.endswith("_v"), "built_with": built_with(runs, arm), "sources": {}}
        cp = runs / f"crossmodal_{arm}_rebuild_vr_wmfa040_deliverable.json"
        cwp = runs / f"crossmodal_{arm}_rebuild_vr_deliverable_wm.json"
        op = num / f"table1_oldmetrics_{arm}_rebuild_vr.json"
        ev = runs / arm / "evaluation_multimodal.json"
        cm = json.loads(cp.read_text())["runs"].get(arm) if cp.exists() else None
        cw = json.loads(cwp.read_text())["runs"].get(arm) if cwp.exists() else None
        om = json.loads(op.read_text())["data"]["arms"].get(arm) if op.exists() else None
        if cm:
            inputs.append(cp)
            r["new"] = {"T2_r": cm["T2"]["pairwise_r_mean"], "T2_cv": cm["T2"]["cv"],
                        "FRAC1_r": cm["FRAC1"]["pairwise_r_mean"], "FRAC2_r": cm["FRAC2"]["pairwise_r_mean"],
                        "B0_r": cm["B0"]["pairwise_r_mean"], "mask_dice": cm["MASK"]["pairwise_dice_mean"],
                        "fibre_angle_wm_deg": cm["DTI"]["mean_pairwise_angle_deg"]}
            r["sources"]["new"] = rel(cp)
        if cw:
            inputs.append(cwp)
            r["new_wm"] = {"T2_r": cw["T2"]["pairwise_r_mean"], "T2_cv": cw["T2"]["cv"],
                           "FRAC1_r": cw["FRAC1"]["pairwise_r_mean"], "FRAC2_r": cw["FRAC2"]["pairwise_r_mean"],
                           "B0_r": cw["B0"]["pairwise_r_mean"]}
            r["sources"]["new_wm"] = rel(cwp)
        if om:
            inputs.append(op)
            r["old"] = om["mean"]; r["sources"]["old"] = rel(op)
        if ev.exists():
            e = json.loads(ev.read_text()); inputs.append(ev)
            r["deformation"] = {"jacobian_outside_0p5_2_pct": e.get("jacobian_mean_outside"),
                                "core_voxels": e.get("coverage_core", {}).get("voxels"),
                                "step": e.get("step"), "iteration": e.get("iteration")}
        lv = (lm or {}).get("variants", {}).get(arm)
        if lv:
            r["landmark_spread"] = {"median_excl_AC_mm": lv["median_excl_AC_mm"], "AC_mm": lv["AC_mm"],
                                    "n_specimens": lm["baseline_rigid_placement"]["per_point"]["AC"]["n"]}
            r["sources"]["landmarks"] = rel(lmp)
        r["status"] = "evaluated" if (cm and om) else ("built, not evaluated" if ev.exists() else "not yet built")
        ins = set(r["built_with"] or [])
        r["in_sample"] = {"T2": "T2" in ins, "FRAC1": "FRAC1" in ins, "FRAC2": "FRAC2" in ins, "DTI": "DTI" in ins,
                          "B0": False, "MASK": False}
        rows[arm] = r
    out = {"arms": rows, "order": [a for a, _ in ARMS],
           "n_evaluated": sum(r["status"] == "evaluated" for r in rows.values()),
           "note": NOTE,
           "landmark_spread_baseline_rigid_median_excl_AC_mm": lm["baseline_rigid_placement"]["median_excl_AC_mm"] if lm else None}
    provenance.write_numbers(NAME, out, inputs=inputs, md5=False, cmd=f"python reproduce/numbers/{NAME}.py")


if __name__ == "__main__":
    main()
