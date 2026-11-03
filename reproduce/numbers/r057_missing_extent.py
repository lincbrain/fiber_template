#!/usr/bin/env python3
"""The caudal extent of each sectioned brain that its blockface volume lacks (paper: Supplementary S1, missing extent per brain; the
paper reads per_brain.<ID> and groups.{mri,no_mri,all}).

Per brain, on its section-aligned deploy grid, planes parallel to the photographed sections, 0.25 mm apart (the clean cut face record
of the batch that prepared the brain, cache_<batch>/clean_face.json; the same z_f as the prepared folder's seam.json):
  unphotographed_mm        from the brain's caudal end to its most caudal photographed plane: no photograph at all
  synthesized_mm           from the brain's caudal end to the junction plane (true_gap_mm): the caudal tissue that was synthesized; it
                           adds the incomplete last photographed sections (junction_from_photographed_end_mm) to the unphotographed part
  synthesized_caudal_pct   brain volume caudal of the junction plane / brain volume (clean_face missing_pct), and its volume (ml)
  synthesized_share_pct    all synthesized voxels / brain voxels on the 0.25 mm product grid (d3_predicted_share: also counts the thin
                           rim of brain outside the photographed outline rostral of the junction plane)
The brain's caudal end is its brain mask's: for a brain with MRI the b0 brain mask (minus the MRI tissue that the photographs lack); for
a brain without MRI the template's brain mask carried into the brain by the direct placement, so its extent is an estimate.
Reads d3_methods and d3_predicted_share from $LINC_NUMBERS (run reproduce/numbers/d3_methods.py and d3_predicted_share.py first).
-> $LINC_NUMBERS/r057_missing_extent.json
    python reproduce/numbers/r057_missing_extent.py
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from linc import paths, provenance  # noqa: E402

ISO = 0.25
LABELS = REPO / "docs" / "specimen_labels.json"
ROOTS = ["canon_d3", "canon_d3n_prod"]                              # prepared brains: 13 with MRI (own b0), 8 without (stand-in b0)
KEYS = ("unphotographed_mm", "synthesized_mm", "junction_from_photographed_end_mm", "synthesized_caudal_pct", "synthesized_share_pct")

# the record's own texts, verbatim (they are part of the record's data)
WHAT = "the caudal extent each sectioned brain's blockface volume lacks (mm along the section normal, % of brain)"
DEFINITIONS = dict(
    unphotographed_mm="brain's caudal end to its most caudal photographed plane (no photograph at all)",
    synthesized_mm="brain's caudal end to the junction plane: the synthesized caudal end (unphotographed + the incomplete last "
                   "photographed sections, junction_from_photographed_end_mm)",
    synthesized_caudal_pct="brain volume caudal of the junction plane / brain volume (section-aligned 0.25 mm grid)",
    synthesized_share_pct="all synthesized voxels / brain voxels on the 0.25 mm product grid (d3_predicted_share; includes the "
                          "rim of brain outside the photographed outline rostral of the junction plane)",
    brain_extent="brain mask of the b0 (brains with MRI, minus MRI tissue the photographs lack) or the template's brain mask "
                 "carried into the brain (brains without MRI: an estimate)")


def summ(v):
    v = np.asarray(v, float)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return dict(n=int(v.size), median=float(med), iqr=[float(q1), float(q3)], min=float(v.min()), max=float(v.max()))


def main():
    P = paths.products("products_completion"); D = P["completion_inputs"]; N = paths.numbers_root()
    inputs, canon = [], {}
    for r in ROOTS:
        p = D / r / "canon_root.json"; inputs.append(p)
        for S, v in json.load(open(p))["brains"].items():
            canon[S] = dict(v, root=r)
    lab = json.load(open(LABELS))["labels"]; inputs.append(LABELS)
    pp, mp = N / "d3_predicted_share.json", N / "d3_methods.json"
    share = json.load(open(pp))["data"]["per_brain"]; meth = json.load(open(mp))["data"]["per_brain"]; inputs += [pp, mp]
    assert sorted(canon) == sorted(share) == sorted(meth) and len(canon) == 21
    rows = {}
    for S in sorted(canon, key=lambda s: (lab[s][:2], int(lab[s][2:]))):
        b = canon[S]["batch"]; cfp = D / f"cache_{b}" / "clean_face.json"; cf = json.load(open(cfp))["brains"][S]; inputs.append(cfp)
        seam = json.load(open(D / canon[S]["root"] / S / "seam.json"))["seam"]
        assert seam["z_f"] == cf["z_f"], f"{S}: junction plane differs between the prepared folder and {cfp}"
        assert abs(cf["true_gap_mm"] - meth[S]["true_gap_mm"]) < 1e-9 and abs(cf["missing_ml"] - meth[S]["missing_ml"]) < 1e-9
        unphot = (cf["photographed_end"] - cf["brain_caudal_end"]) * ISO
        assert abs(unphot + cf["z_f_mm_from_photographed_end"] - cf["true_gap_mm"]) < 1e-9
        rows[S] = dict(name=lab[S], species="M. fascicularis" if S.startswith("MF") else "M. mulatta",
                       mri="own b0" if meth[S]["group"] == "mri" else "none (stand-in b0)",
                       unphotographed_mm=round(unphot, 2), synthesized_mm=cf["true_gap_mm"],
                       junction_from_photographed_end_mm=cf["z_f_mm_from_photographed_end"],
                       synthesized_caudal_pct=cf["missing_pct"], synthesized_caudal_ml=cf["missing_ml"],
                       synthesized_share_pct=round(100.0 * share[S]["predicted_share"], 2), batch=b)
    groups = {"mri": [S for S in rows if meth[S]["group"] == "mri"], "no_mri": [S for S in rows if meth[S]["group"] == "no_mri"],
              "all": list(rows)}
    data = dict(what=WHAT, definitions=DEFINITIONS, per_brain=rows,
                groups={g: dict(n=len(ss), brains=[rows[S]["name"] for S in ss], **{k: summ([rows[S][k] for S in ss]) for k in KEYS})
                        for g, ss in groups.items()})
    provenance.write_numbers("r057_missing_extent", data, inputs=inputs, cmd="python reproduce/numbers/r057_missing_extent.py")
    for g, v in data["groups"].items():
        print(g, {k: (round(v[k]["median"], 2), [round(x, 2) for x in (v[k]["min"], v[k]["max"])]) for k in KEYS})


if __name__ == "__main__":
    main()
