#!/usr/bin/env python3
"""The volumes behind Fig. 8's caption sentence on SARM's larger nuclei, and the border check behind "shares the ventral and lateral borders".

You need (settings/internal/products_completion.yaml)
  lgnstn_labels: <ID>/<ID>_<struct>_mri-template_mask_demons.nii.gz, the drawn LGN and STN carried into the template (MM1-MM3);
  lgnstn_slab: SARM_<struct>_mri-template.nii.gz, SARM's nuclei split by hemisphere on the template grid.

What it does
  per nucleus: SARM's volume (mean of the two hemispheres) and the drawn volume (mean over the three brains and both hemispheres), with the
  per-hemisphere and per-brain values (mm3 = voxel count x voxel volume); per brain and hemisphere, SARM minus the drawn label of the 5th
  percentile of z (ventral edge), the 95th percentile of z (dorsal), the 95th percentile of |x| (lateral) and the 5th of |x| (medial), mm,
  and their mean and range over the three brains and both hemispheres.

You get
  $LINC_NUMBERS/lgnstn_label_volumes_h9.json (the labels carried through the registrations with the per-component nonlinear stiffness)

    python code/figures/paper_numbers/lgnstn_label_volumes.py
"""
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
from shared import paths, provenance  # noqa: E402

NAME = "lgnstn_label_volumes_h9"
SUBJ = ["MR243", "MR252", "MR256"]
EDGES = ("ventral", "lateral", "dorsal", "medial")
NOTE = ("mm3 = voxel count x voxel volume on the template grid; edges = 5th/95th percentiles of the label's voxel coordinates "
        "(ventral = p5 of z, dorsal = p95 of z, lateral = p95 of |x|, medial = p5 of |x|); positive differences = SARM's edge higher / more lateral")


def measure(p):
    im = nib.load(str(p)); a = np.asanyarray(im.dataobj) > 0; X = nib.affines.apply_affine(im.affine, np.argwhere(a))
    return dict(mm3=float(a.sum() * abs(np.linalg.det(im.affine[:3, :3]))), ventral=float(np.percentile(X[:, 2], 5)),
                lateral=float(np.percentile(np.abs(X[:, 0]), 95)), dorsal=float(np.percentile(X[:, 2], 95)),
                medial=float(np.percentile(np.abs(X[:, 0]), 5)))


def main():
    P = paths.products("products_completion")
    names = json.load(open(REPO / "settings" / "specimen_labels.json"))["labels"]
    out, inputs = {}, []
    for nuc in ("LGN", "STN"):
        per = {}
        for hemi in ("lh", "rh"):
            st = hemi + nuc
            sp = P["lgnstn_slab"] / f"SARM_{st}_mri-template.nii.gz"; s = measure(sp); inputs.append(sp)
            o = {}
            for S in SUBJ:
                p = P["lgnstn_labels"] / S / f"{S}_{st}_mri-template_mask_demons.nii.gz"; o[names[S]] = measure(p); inputs.append(p)
            per[st] = dict(sarm_mm3=round(s["mm3"], 2), drawn_mm3={k: round(v["mm3"], 2) for k, v in o.items()},
                           drawn_mean_mm3=round(float(np.mean([v["mm3"] for v in o.values()])), 2),
                           **{f"sarm_minus_drawn_{e}_edge_mm": {k: round(s[e] - v[e], 3) for k, v in o.items()} for e in EDGES})
        out[nuc] = dict(sarm_mm3=round(float(np.mean([per[h + nuc]["sarm_mm3"] for h in ("lh", "rh")])), 2),
                        drawn_mm3=round(float(np.mean([v for h in ("lh", "rh") for v in per[h + nuc]["drawn_mm3"].values()])), 2),
                        per_hemisphere=per)
        for e in EDGES:
            v = [x for h in ("lh", "rh") for x in per[h + nuc][f"sarm_minus_drawn_{e}_edge_mm"].values()]
            out[nuc][f"sarm_minus_drawn_{e}_edge_mm_mean_range"] = [round(float(np.mean(v)), 3), round(float(min(v)), 3), round(float(max(v)), 3)]
        print(nuc, "SARM", out[nuc]["sarm_mm3"], "drawn", out[nuc]["drawn_mm3"], flush=True)
    out["note"] = NOTE
    print(provenance.write_numbers(NAME, out, inputs=inputs, md5=False, cmd="python code/figures/paper_numbers/lgnstn_label_volumes.py"))


if __name__ == "__main__":
    main()
