#!/usr/bin/env python3
"""Between-brain agreement of the LGN and STN in the MRI template: the observer's labels (MM1-MM3) and the 2D network's labels (every
histology brain with sections through the nuclei), side by side (paper: Supplementary S3; the network rows of Table S6).

Inputs, all on the MRI template grid (0.25 mm, RAS):
  observer   products push_current/<ID>/<ID>_<struct>_mri-template_mask_demons.nii.gz (the labels of record; labels_overlap_current)
  network    products lgnstn_model_labels/<ID>/<ID>_<struct>_mri-template_mask_demons.nii.gz (the network's labels, interpolated and
             carried by the rule of record: R1 of record + R5; MF8 through its blockface-based chain)
Per pair of brains and hemisphere-structure (lhLGN, lhSTN, rhLGN, rhSTN): Dice, the symmetric average surface distance (ASSD, mm), the
centroid distance (mm) and its x / y / z components (b - a). Sets: observer_MM (3 pairs), model_MM (3 pairs), model_with_MRI (MM1-MM3,
MF5, MF6: 10 pairs), model_MF8 (MF8 against each of the five), model_vs_observer (per brain, MM1-MM3). Summary per set and nucleus
(hemispheres pooled): median and IQR of centroid distance, Dice and ASSD; a row needs both labels non-empty (else counted missing).
Ported from the analysis project's labels/r059/model_agreement_r059.py; its description is kept verbatim as METHOD (part of the record).
-> $LINC_NUMBERS/lgnstn_model_agreement_r059.json
    python reproduce/numbers/lgnstn_model_agreement_r059.py          (env/linc.yml, PY_MRI)"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "4"                                           # CPU only, at most 4 threads

import itertools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from linc import paths, provenance  # noqa: E402

NAME = {"MR243": "MM1", "MR252": "MM2", "MR256": "MM3", "MF278": "MF5", "MF275": "MF6", "MF191": "MF8"}
STAIN = {"MR243": "LY", "MR252": "FS", "MR256": "LY", "MF278": "LY", "MF275": "FR", "MF191": "FR"}
STRUCTS = ["lhLGN", "lhSTN", "rhLGN", "rhSTN"]
MM = ["MR243", "MR252", "MR256"]; WITH_MRI = MM + ["MF278", "MF275"]
METHOD = [
    '(Dataset532 fold `all`; every histology brain with sections through the nuclei), side by side. Labels on the template grid (0.25 mm, RAS):',
    '  observer  labels/push_current/<S>/<S>_<st>_mri-template_mask_demons.nii.gz (the labels of record, labels_overlap_current)',
    '  model     labels/nnunet/deploy_r059/<S>/<S>_<st>_mri-template_mask_demons.nii.gz (labels/r059/model_deploy_r059.py, histo_interp, the carry',
    '            of record: route B through the R1 of record + R5; MF8 through its 2b chain, labels/r059/carry_mf191_r059.py)',
    'Per pair of brains and hemisphere-structure (lhLGN, lhSTN, rhLGN, rhSTN): Dice, the symmetric average surface distance (ASSD, mm), the',
    'centroid distance (mm) and its x / y / z components (b - a). Sets:',
    "  observer_MM        the observer's three brains (3 pairs), as in the paper's Table 3 (recomputed here, same rule)",
    '  model_MM           the model on the same three brains (3 pairs)',
    '  model_with_MRI     the model on every brain with its own MRI (MM1-MM3, MF5, MF6: 10 pairs; MF5/MF6 are fascicularis, MM rhesus)',
    '  model_MF8          MF8 (no MRI; blockface-based alignment) against each of the five (5 pairs), its own row',
    "  model_vs_observer  per brain (MM1-MM3): the model's label against the observer's label of the same brain (template space)",
    'Summary per set and nucleus (both hemispheres pooled): median and IQR of the centroid distance, Dice and ASSD, number of rows; a row needs',
    'both labels non-empty (an empty label is counted as missing). Stains: MM1 MM3 MF5 Lucifer yellow, MM2 fluorescein, MF6 MF8 Fluoro-Ruby; the',
    "model's STN is unvalidated on fluorescein (0 in its leave-one-subject-out test) and on Fluoro-Ruby (no reference).",
    '-> paper/numbers/lgnstn_model_agreement_r059.json',
    '    mri/env/mmorf/bin/python labels/r059/model_agreement_r059.py']
INPUTS = []


def load(root, S, st):
    p = Path(root) / S / f"{S}_{st}_mri-template_mask_demons.nii.gz"
    if not p.exists():
        return None, None
    INPUTS.append(p)
    im = nib.load(str(p)); m = np.asanyarray(im.dataobj) > 0
    return (m if m.any() else None), im.affine


def metrics(a, b, A):
    vx = float(abs(A[0, 0]))
    d = 2 * (a & b).sum() / (a.sum() + b.sum())
    sa, sb = a & ~ndi.binary_erosion(a), b & ~ndi.binary_erosion(b)
    lo = np.maximum(np.minimum(np.argwhere(a).min(0), np.argwhere(b).min(0)) - 5, 0)
    hi = np.minimum(np.maximum(np.argwhere(a).max(0), np.argwhere(b).max(0)) + 6, a.shape)
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    da = ndi.distance_transform_edt(~sb[sl])[sa[sl]]; db = ndi.distance_transform_edt(~sa[sl])[sb[sl]]
    assd = float(np.concatenate([da, db]).mean() * vx)
    ca = nib.affines.apply_affine(A, np.argwhere(a).mean(0)); cb = nib.affines.apply_affine(A, np.argwhere(b).mean(0))
    dv = cb - ca
    return dict(dice=round(float(d), 4), assd_mm=round(assd, 3), centroid_mm=round(float(np.linalg.norm(dv)), 3),
                dx=round(float(dv[0]), 3), dy=round(float(dv[1]), 3), dz=round(float(dv[2]), 3), vol_a_mm3=round(float(a.sum() * vx ** 3), 2),
                vol_b_mm3=round(float(b.sum() * vx ** 3), 2))


def pairs(root_a, root_b, plist):
    rows = []
    for a, b in plist:
        for st in STRUCTS:
            ma, A = load(root_a, a, st); mb, _ = load(root_b, b, st)
            r = dict(a=NAME[a], b=NAME[b], struct=st)
            if ma is None or mb is None:
                r["missing"] = [NAME[s] for s, m in ((a, ma), (b, mb)) if m is None]
            else:
                r.update(metrics(ma, mb, A))
            rows.append(r)
    return rows


def summary(rows):
    out = {}
    for nuc in ("LGN", "STN"):
        rr = [r for r in rows if r["struct"].endswith(nuc) and "centroid_mm" in r]
        if not rr:
            out[nuc] = dict(n=0, missing=sum(1 for r in rows if r["struct"].endswith(nuc))); continue
        q = lambda k: [round(float(x), 3) for x in np.percentile([r[k] for r in rr], [25, 50, 75])]  # noqa: E731
        out[nuc] = dict(n=len(rr), missing=sum(1 for r in rows if r["struct"].endswith(nuc) and "missing" in r),
                        centroid_mm_q25_50_75=q("centroid_mm"), dice_q25_50_75=q("dice"), assd_mm_q25_50_75=q("assd_mm"))
    return out


def main():
    P = paths.products("products_mri")
    OBS, MOD = P["push_current"], P["lgnstn_model_labels"]
    sets = dict(observer_MM=pairs(OBS, OBS, itertools.combinations(MM, 2)),
                model_MM=pairs(MOD, MOD, itertools.combinations(MM, 2)),
                model_with_MRI=pairs(MOD, MOD, itertools.combinations(WITH_MRI, 2)),
                model_MF8=pairs(MOD, MOD, [(S, "MF191") for S in WITH_MRI]),
                model_vs_observer=pairs(OBS, MOD, [(S, S) for S in MM]))
    out = dict(method=METHOD, names=NAME, stains=STAIN, sets={k: dict(summary=summary(v), rows=v) for k, v in sets.items()})
    for k, v in out["sets"].items():
        print(k, json.dumps(v["summary"]), flush=True)
    print(provenance.write_numbers("lgnstn_model_agreement_r059", out, inputs=sorted(set(INPUTS)), md5=False,
                                   cmd="python reproduce/numbers/lgnstn_model_agreement_r059.py"))


if __name__ == "__main__":
    main()
