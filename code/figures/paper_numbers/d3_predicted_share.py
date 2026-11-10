#!/usr/bin/env python3
"""How much of each brain the caudal completion predicted, and where that tissue lies, as median and IQR over the brains (paper:
Results, caudal completion; the paper reads groups.{mri,no_mri}.predicted_share.{median,iqr} and groups.all.caudal_share.median).

From the completed volumes of record (21 brains), with d3_methods' definitions, checked against the d3_methods record:
  predicted share   predicted voxels / brain voxels on the 0.25 mm product grid; predicted = <ID>_synth_mask (the brain minus the
                    photographed tissue rostral of the clean cut face); brain = the completion's brain mask
  caudal share      predicted voxels beyond the clean cut face (seam depth < 0) / predicted voxels: the caudal end, including the
                    incomplete last photographed sections; the rest is the rostral rim (seam depth >= 0: brain voxels outside the
                    photographed outline), with the fraction of rim voxels within 1 mm (3-D) of photographed tissue
Groups: the 13 brains with MRI (own b0), the 8 without (stand-in b0), all 21.
-> $LINC_NUMBERS/d3_predicted_share.json
    python code/figures/paper_numbers/d3_predicted_share.py
"""
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
from shared import paths, provenance  # noqa: E402

RUNS = {"mri": "d3_full_A_s0", "no_mri": "d3p_full_A_s0"}

# the record's own texts, verbatim (they are part of the record's data)
DEFINITIONS = dict(predicted_share="predicted voxels / brain voxels (the completion's brain mask) on the 0.25 mm product grid: the "
                                   "share of the brain's volume that was predicted",
                   caudal_share="predicted voxels beyond the clean cut face / predicted voxels (the caudal end, incl. the "
                                "incomplete last photographed sections)",
                   rim="predicted voxels rostral of the clean cut face (brain outside the photographed outline)")


def summ(v):
    v = np.asarray(v, float)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return dict(n=int(v.size), median=float(med), iqr=[float(q1), float(q3)], min=float(v.min()), max=float(v.max()))


def main():
    P = paths.products("products_completion")
    OUT, CACHE = P["completed"], P["completed_masks"]
    methods = P["numbers_of_record"] / "d3_methods.json"
    inputs = [methods]
    rec = json.load(open(methods))["data"]["per_brain"]
    rows = {}
    for tag, run in RUNS.items():
        sp = OUT / f"summary_{run}.json"
        inputs.append(sp)
        for S in sorted(json.load(open(sp))["members"]):
            ps = dict(syn=OUT / S / f"{S}_synth_mask.nii.gz", brain=CACHE / S / "deploy/brain.nii.gz",
                      seam=OUT / S / f"{S}_seam_depth.nii.gz", real=CACHE / S / "deploy/real_mask.nii.gz")
            inputs += list(ps.values())
            sm = nib.load(ps["syn"]); A = sm.affine
            syn = np.asanyarray(sm.dataobj) > 0
            brain = np.asanyarray(nib.load(ps["brain"]).dataobj) > 0
            sdp = np.asanyarray(nib.load(ps["seam"]).dataobj)
            real = np.asanyarray(nib.load(ps["real"]).dataobj) > 0
            assert syn.shape == brain.shape == sdp.shape == real.shape
            assert not (syn & ~brain).any(), f"{S}: predicted voxels outside the brain"
            n_syn, n_brain = int(syn.sum()), int(brain.sum())
            rim = syn & (sdp >= 0)
            vox_mm = float(abs(np.linalg.det(A[:3, :3])) ** (1 / 3))
            d = ndi.distance_transform_edt(~real)[rim] * vox_mm if rim.any() else np.zeros(0)
            r = dict(group=tag, predicted_share=n_syn / n_brain, caudal_share=1.0 - rim.sum() / max(n_syn, 1),
                     rim_share=float(rim.sum() / max(n_syn, 1)), rim_within_1mm=float((d <= 1.0).mean()) if d.size else None,
                     predicted_ml=n_syn * vox_mm ** 3 / 1000.0, brain_ml=n_brain * vox_mm ** 3 / 1000.0)
            m = rec[S]
            assert abs(r["predicted_share"] - m["synth_frac"]) < 1e-4, f"{S}: predicted share {r['predicted_share']} vs record {m['synth_frac']}"
            assert abs(r["rim_share"] - m["rim"]["rostral_share_of_synth"]) < 1e-3, f"{S}: rim share differs from the record"
            rows[S] = r
    groups = {"mri": [S for S in rows if rows[S]["group"] == "mri"], "no_mri": [S for S in rows if rows[S]["group"] == "no_mri"],
              "all": sorted(rows)}
    out = dict(
        definitions=DEFINITIONS,
        per_brain=rows,
        groups={g: dict(n=len(ss), brains=ss,
                        predicted_share=summ([rows[S]["predicted_share"] for S in ss]),
                        predicted_pct=summ([100 * rows[S]["predicted_share"] for S in ss]),
                        caudal_share=summ([rows[S]["caudal_share"] for S in ss]),
                        caudal_pct=summ([100 * rows[S]["caudal_share"] for S in ss]),
                        rim_within_1mm=summ([rows[S]["rim_within_1mm"] for S in ss if rows[S]["rim_within_1mm"] is not None]))
                for g, ss in groups.items()})
    provenance.write_numbers("d3_predicted_share", out, inputs=inputs, md5=False, cmd="python code/figures/paper_numbers/d3_predicted_share.py")
    for g, v in out["groups"].items():
        p, c = v["predicted_pct"], v["caudal_pct"]
        print(f"{g:6s} n {v['n']:2d}: predicted {p['median']:.1f}% [IQR {p['iqr'][0]:.1f}-{p['iqr'][1]:.1f}] of the brain; caudal "
              f"{c['median']:.1f}% [IQR {c['iqr'][0]:.1f}-{c['iqr'][1]:.1f}] of the predicted tissue")


if __name__ == "__main__":
    main()
