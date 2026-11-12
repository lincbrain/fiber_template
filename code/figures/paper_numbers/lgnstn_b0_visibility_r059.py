#!/usr/bin/env python3
"""Are the LGN and STN visible on each brain's own b0, the precondition for outlining them on the MRI? (paper: Supplementary S3)

For every brain and hemisphere-structure, the label carried to the brain's own b0 (histology -> blockface -> own b0, the R1 of record):
the observer's labels for MM1-MM3 (products push_current/<ID>/<ID>_<struct>_b0-native_mask_demons.nii.gz) and the network's for every
histology brain with MRI (products lgnstn_model_labels/<ID>/...), on that b0 (products_blockface data: sub-<ID>/mri/<ID>_mri_b0_native.nii.gz,
the R1 target grid):
  contrast   (mean inside - mean in the ring 1-3 voxels outside) / mean ring;  CNR = (mean inside - mean ring) / sd ring
  null       the same label shape moved to 200 random positions inside the brain (b0 > 10 % of its 99th percentile, shape fully inside),
             the same measures; the percentile of the true |CNR| among them (a visible nucleus stands out: percentile near 100).
The random positions come from one generator seeded 0, drawn in the record's order (sets, brains, structures).
Ported from the analysis project's labels/r059/visibility_r059.py; its description is kept verbatim as METHOD (part of the record).
-> $LINC_NUMBERS/lgnstn_b0_visibility_r059_h9.json
    python code/figures/paper_numbers/lgnstn_b0_visibility_r059.py          (settings/env/linc.yml, PY_MRI)"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "4"                                           # CPU only, at most 4 threads

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
from shared import paths, provenance  # noqa: E402

NAME = {"MR243": "MM1", "MR252": "MM2", "MR256": "MM3", "MF278": "MF5", "MF275": "MF6"}
STRUCTS = ["lhLGN", "lhSTN", "rhLGN", "rhSTN"]
SETS = {"observer": ("push_current", ["MR243", "MR252", "MR256"]),
        "model": ("lgnstn_model_labels", ["MR243", "MR252", "MR256", "MF278", "MF275"])}
METHOD = [
    "For every brain and hemisphere-structure, the label carried to the brain's own b0 (histology -> blockface -> own b0, the R1 of record; the",
    "observer's labels for MM1-MM3: labels/push_current/<S>/<S>_<st>_b0-native_mask_demons.nii.gz; the model's for every histology brain with",
    'MRI: labels/nnunet/deploy_r059/<S>/...) on that b0 (data/sub-<S>/mri/<S>_mri_b0_native.nii.gz, the R1 target grid):',
    '  contrast   (mean inside - mean in the ring 1-3 voxels outside) / mean ring;  CNR = (mean inside - mean ring) / sd ring',
    '  null       the same label shape moved to 200 random positions inside the brain (b0 > 10 % of its 99th percentile, shape fully inside), the',
    '             same measures; |CNR| percentile of the true position among them (a visible nucleus stands out: percentile near 100)',
    '-> paper/numbers/lgnstn_b0_visibility_r059.json',
    '    mri/env/mmorf/bin/python labels/r059/visibility_r059.py']


def measure(b0, lab):
    ring = ndi.binary_dilation(lab, iterations=3) & ~ndi.binary_dilation(lab, iterations=1)
    i, r = b0[lab], b0[ring]
    return float((i.mean() - r.mean()) / max(r.mean(), 1e-6)), float((i.mean() - r.mean()) / max(r.std(), 1e-6))


def main():
    P = paths.products("products_mri"); data = paths.products("products_blockface")["data"]
    rng = np.random.default_rng(0)
    out = dict(method=METHOD, sets={}); inputs = []
    for setname, (key, subs) in SETS.items():
        res = {}
        for S in subs:
            bp = data / f"sub-{S}" / "mri" / f"{S}_mri_b0_native.nii.gz"; inputs.append(bp)
            b0i = nib.load(str(bp)); b0 = np.asanyarray(b0i.dataobj).astype(np.float64)
            brain = b0 > 0.1 * np.percentile(b0[b0 > 0], 99)
            res[S] = {}
            for st in STRUCTS:
                p = P[key] / S / f"{S}_{st}_b0-native_mask_demons.nii.gz"
                if not p.exists():
                    continue
                inputs.append(p)
                lab = np.asanyarray(nib.load(str(p)).dataobj) > 0
                if lab.sum() < 3:
                    continue
                c, cnr = measure(b0, lab)
                ij = np.argwhere(lab); ctr = ij.mean(0).round().astype(int); rel = ij - ctr
                cand = np.argwhere(ndi.binary_erosion(brain, iterations=4))
                null = []
                for k in rng.choice(len(cand), size=min(2000, len(cand)), replace=False):
                    q = rel + cand[k]
                    if (q < 0).any() or (q >= np.array(b0.shape)).any():
                        continue
                    m = np.zeros(b0.shape, bool); m[q[:, 0], q[:, 1], q[:, 2]] = True
                    if not brain[m].all():
                        continue
                    null.append(abs(measure(b0, m)[1]))
                    if len(null) >= 200:
                        break
                pct = float((np.array(null) < abs(cnr)).mean() * 100) if null else None
                res[S][st] = dict(voxels=int(lab.sum()), voxel_mm=[round(float(z), 3) for z in b0i.header.get_zooms()[:3]],
                                  contrast=round(c, 4), cnr=round(cnr, 3), null_n=len(null),
                                  null_abs_cnr_median=round(float(np.median(null)), 3) if null else None, abs_cnr_percentile_in_null=pct)
                print(setname, NAME[S], st, res[S][st], flush=True)
        out["sets"][setname] = res
    print(provenance.write_numbers("lgnstn_b0_visibility_r059_h9", out, inputs=inputs, md5=False,
                                   cmd="python code/figures/paper_numbers/lgnstn_b0_visibility_r059.py"))


if __name__ == "__main__":
    main()
