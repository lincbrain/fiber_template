#!/usr/bin/env python3
"""Where each drawn LGN and STN label sits in the template, against SARM's: the numbers behind Fig. 8's sentences on how far the
drawn nuclei lie from SARM's along each axis (descriptive).

You need (settings/internal/products_completion.yaml)
  per label set (code/labels/route.py lgnstn_sets, as code/figures/paper_numbers/lgnstn_label_volumes.py):
  <ID>/<ID>_<struct>_mri-template_mask_demons.nii.gz, the drawn LGN and STN carried into the template (MM1-MM3);
  lgnstn_slab: SARM_<struct>_mri-template.nii.gz, SARM's nuclei split by hemisphere on the template grid.

What it does
  per label (the three brains and SARM, each nucleus and hemisphere): its volume (mm3 = voxel count x voxel volume), and the centroid and
  the 5th and 95th percentiles of its voxels' coordinates along x (as the distance from the midline, |x|, so larger = more lateral), y
  and z; then, per nucleus and hemisphere, SARM minus the mean of the three brains, and each brain's centroid minus SARM's.
  A label that is larger but shares one border shows a centroid difference of about half the extent difference, toward the side where it
  is larger.

You get
  $LINC_NUMBERS/lgnstn_label_extents<tag>b.json, one per set. The values are not rounded: the paper rounds once, when it prints them.

    python code/figures/paper_numbers/lgnstn_label_extents.py
"""
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
from shared import paths, provenance  # noqa: E402
from labels import route as RT  # noqa: E402

NAME = "lgnstn_label_extents"                                       # + the set's tag + "b" (unrounded values)
SUBJ = ["MR243", "MR252", "MR256"]
STRUCTS = ["lhLGN", "rhLGN", "lhSTN", "rhSTN"]
NOTE = "x is reported as lateral distance |x| (positive = lateral); p05/p95 = 5th/95th percentile of the label's voxel coordinates"


def stats(p):
    im = nib.load(str(p)); a = np.asanyarray(im.dataobj) > 0; X = nib.affines.apply_affine(im.affine, np.argwhere(a))
    r = dict(volume_mm3=float(a.sum() * abs(np.linalg.det(im.affine[:3, :3]))))
    for k, v in (("lateral", np.abs(X[:, 0])), ("y", X[:, 1]), ("z", X[:, 2])):
        r[k] = dict(centroid=float(v.mean()), p05=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)))
    return r


def main():
    for tag, labdir, _rmap in RT.lgnstn_sets():
        record(tag, labdir)


def record(tag, labdir):
    P = paths.products("products_completion")
    names = json.load(open(REPO / "settings" / "specimen_labels.json"))["labels"]
    rows, summ, inputs = {}, {}, []
    for st in STRUCTS:
        for S in SUBJ:
            p = labdir / S / f"{S}_{st}_mri-template_mask_demons.nii.gz"; rows[f"{names[S]}_{st}"] = stats(p); inputs.append(p)
        sp = P["lgnstn_slab"] / f"SARM_{st}_mri-template.nii.gz"; rows[f"SARM_{st}"] = stats(sp); inputs.append(sp)
        ours = [rows[f"{names[S]}_{st}"] for S in SUBJ]; s = rows[f"SARM_{st}"]
        d = dict(volume_ours_mean_mm3=float(np.mean([o["volume_mm3"] for o in ours])), volume_sarm_mm3=s["volume_mm3"])
        for k in ("lateral", "y", "z"):
            d[k] = {q: float(s[k][q] - np.mean([o[k][q] for o in ours])) for q in ("centroid", "p05", "p95")}
            d[k]["ours_each_centroid_minus_sarm"] = [float(o[k]["centroid"] - s[k]["centroid"]) for o in ours]
        summ[st] = d
        print(st, "volume ours", round(d["volume_ours_mean_mm3"], 2), "SARM", round(d["volume_sarm_mm3"], 2),
              {k: round(d[k]["centroid"], 3) for k in ("lateral", "y", "z")}, flush=True)
    out = dict(rows=rows, sarm_minus_ours_mean=summ, note=NOTE)
    print(provenance.write_numbers(NAME + tag + "b", out, inputs=inputs, md5=False, cmd="python code/figures/paper_numbers/lgnstn_label_extents.py"))


if __name__ == "__main__":
    main()
