#!/usr/bin/env python3
"""The blockface template's numbers (Results, the blockface template), read from its saved products only (no template recomputed).

    python code/figures/paper_numbers/bf_template_metrics_d2b.py

You need
  the template's products (settings/internal/products_blockface.yaml `bftemplate_*`; design 2b in the records).
What it does
  Per template (the template of record after step 8, and after step 7 beside it): the members, the grid, the brain, photographed
  and predicted volumes and the predicted fraction (predicted = brain with a photographed fraction < 0.5), the shell rule's
  footprint, the coverage (median over the brain; the core = the part of the brain that every brain covers), the symmetry (max |V -
  mirror(V)| of the b0, grayscale and color channels; the Dice coefficient between the brain mask and its mirror image; the
  grayscale asymmetry before the final symmetrization), and the construction's Jacobian determinants (% of brain voxels with det J
  outside [0.5, 2] per member; folded voxels).
You get
  $LINC_NUMBERS/bf_template_metrics_d2b.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
from shared import paths, provenance  # noqa: E402

# the record's `definition`, verbatim (it is part of the record's data)
DEFINITION = "Design 2b template numbers for the paper (5 Oct 2026): bftemplate/template_numbers.py's bf_template_metrics row\nfor the published 2b templates, from their saved products only (no template recomputed):\n  d2bs8   bftemplate/d2b/final_s8 (step 8, the 2b of record)     d2bA   bftemplate/d2b/final (plan A, for reference)\nper template: members (brains / blockface inputs / b0-only TPL), grid, brain / photographed / predicted volume and the predicted share\n(predicted = brain & photo_share < 0.5, finish_t1's synth_mask), the shell rule's footprint, coverage (median over the brain; the core\n= the share of the brain where EVERY brain overlaps, coverage >= n_brains: predicted and photographed tissue both count), symmetry\n(d2_symmetry.json: max |V - mirror(V)| of b0 / grey / rgb = 0 exactly; brain-mask mirror Dice; grey asymmetry before the final\nsymmetrization) and the construction det J gate (b) record: % of brain voxels with det J outside [0.5, 2] per member (median, max,\nworst) and folded voxels (det J <= 0), on the brain mask, from the gate JSON (step 8: bftemplate/d2b/s8/gate_jac_it03.json, which\nalso carries plan A's step-7 values).\n    python bftemplate/d2b/d2b_template_numbers.py  -> paper/numbers/bf_template_metrics_d2b.json"

# (label, arm, products_blockface key, gate key)
TEMPLATES = [("2b step 8 (of record)", "d2bs8", "bftemplate_step8", "step8"), ("2b plan A", "d2bA", "bftemplate_plan_a", "plan_a")]


GATE = "(b) brain mask; no folds; every member <= max(2 %, plan A step-7 value + 0.5 points); 2b: TPL at the absolute 2 %"


def gate_text(recorded):
    """The det J gate (b) as stated here; the gate JSON's own text says the same after its decision stamp ("(b) <who> <time>: ...")."""
    import re
    assert re.sub(r"^\(b\) [^:]*\d+:\d+x?: ", "(b) ", recorded) == GATE, recorded
    return GATE


def row(label, arm, key, gkey, cfg, prod, gate_path):
    F = prod[key]
    m = json.load(open(F / "metrics.json")); m = m.get("data", m)
    B = np.asanyarray(nib.load(str(F / f"{arm}_brainmask.nii.gz")).dataobj) > 0
    C = np.asanyarray(nib.load(str(F / f"{arm}_coverage.nii.gz")).dataobj).astype(np.float32)
    A = nib.load(str(F / f"{arm}_brainmask.nii.gz")).affine
    x = A[0, 3] + A[0, 0] * np.arange(B.shape[0]); assert np.allclose(x, -x[::-1], atol=1e-3), "grid not centred on x = 0"
    sym = json.load(open(F / "d2_symmetry.json"))["data"]
    g = json.load(open(gate_path)); g = g.get("data", g)
    gs = g["summary"][gkey]["brain"]
    per = {k: (v["step8_brain_pct"] if gkey == "step8" else v["plan_a_brain_pct"]) for k, v in g["members"].items()}
    kinds = list((m.get("members") or {}).values())
    w = json.load(open(F / "d2_member_warps.json"))
    return dict(label=label, arm=arm, folder=cfg[key], n_brains=int(m["n_brains"]), n_blockface_inputs=m["n_inputs"],
                members_mri=kinds.count("mri"), members_without_mri=kinds.count("nomri"),
                b0_only_members=[r["member"] for r in w["data"] if r["member"] == "TPL" or r.get("link")], n_member_warps=len(w["data"]),
                grid_mm=m["grid_mm"], rule=m["rule"], finish_trim_mm_r_i=m.get("trim_mm"), brain_cm3=m["brain_cm3"],
                photographed_cm3=m["real_cm3"], predicted_cm3=m["synth_cm3"], predicted_share_of_brain=m["synth_frac_of_brain"],
                shell_rule=m.get("shell_rule"),
                coverage=dict(median_in_brain=float(np.median(C[B])), core_frac_of_brain=float((C[B] >= m["n_brains"] - 1e-3).mean())),
                symmetry=dict(max_abs_mirror_diff={ch: sym[ch]["max_abs_diff"] for ch in ("b0", "grey", "rgb")},
                              brainmask_mirror_dice=float(2 * (B & np.flip(B, 0)).sum() / (2 * B.sum())),
                              grey_asym_before_symmetrisation=m.get("grey_asym_before")),
                detj_gate_b=dict(gate=gate_text(g["gate"]), verdict=g["verdict"], mask="brain", n_members=len(per),
                                 median_outside_0p5_2_pct=gs["median_outside_pct"], max_outside_0p5_2_pct=gs["max_outside_pct"], worst=gs["worst"],
                                 max_folds_pct=gs["max_folds_pct"],
                                 folded_voxels_total=0 if all((v.get("folds_pct") or 0.0) == 0.0 for v in g["members"].values()) and gkey == "step8"
                                 else (0 if gs["max_folds_pct"] == 0.0 else None),
                                 per_member_outside_pct=per))


def main():
    import yaml
    cfg = yaml.safe_load((Path(__file__).resolve().parents[3] / "settings" / "internal" / "products_blockface.yaml").read_text())
    prod = paths.products("products_blockface")
    gate = prod["bftemplate_step8_state"] / "gate_jac_it03.json"           # step 8's det J record; it also carries plan A's step 7
    rows = [row(label, arm, key, gkey, cfg, prod, gate) for label, arm, key, gkey in TEMPLATES]
    provenance.write_numbers("bf_template_metrics_d2b", dict(templates=rows, definition=DEFINITION),
                             inputs=[prod[t[2]] / "metrics.json" for t in TEMPLATES] + [gate], cmd="python code/figures/paper_numbers/bf_template_metrics_d2b.py",
                             md5=False)


if __name__ == "__main__":
    main()
