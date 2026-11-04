#!/usr/bin/env python3
"""The LGN / STN networks' predictions against the observer's outlines on held-out sections, leave one subject out
(paper: Supplementary S3, Table S6's network rows; fold 0 = MM1, fold 1 = MM2, fold 2 = MM3).

Inputs, read as data (products_mri.yaml `lgnstn_nnunet`; no network is run and no weights are in the repository): each fold's
predictions on its held-out sections (nnUNet_results/Dataset532_LGNSTNnegdup/<trainer>/fold_<k>/validation/<case>.png), the
observer's references (nnUNet_raw/Dataset532_LGNSTNnegdup/labelsTr/<case>.png) and the fold assignment folds_loso_532.json.
Per held-out case (a whole-section piece on the blockface piece grid, halved as in training: 0.08 mm pixels) and nucleus (1 = LGN,
2 = STN; both hemispheres in one class): Dice; per reference component (8-connected) matched to the nearest predicted component within
2 mm: the centroid distance and the symmetric average surface distance (ASSD) of the matched pair. Pooled per fold: Dice over all cases
(sum of overlaps), median ASSD and centroid distance, missed and false-positive components.
Ported from the analysis project's labels/r059/loso_metrics_r059.py; its description is kept verbatim as METHOD (part of the record).
-> $LINC_NUMBERS/lgnstn_model_loso_r059.json
    python reproduce/numbers/lgnstn_model_loso_r059.py          (env/mri.yml, PY_MRI)"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from linc import paths, provenance  # noqa: E402

DATASET = "Dataset532_LGNSTNnegdup"
TRAINER = "nnUNetTrainer_100epochs__nnUNetPlans_small__2d"
STAIN = {"MR243": "LY", "MR252": "FS", "MR256": "LY"}
NAME = {"MR243": "MM1", "MR252": "MM2", "MR256": "MM3"}
PX_MM = 2 * 0.04                      # halved store level 0 of the MM blockface stores (the stores' level 1 = 0.08 mm, the "_L2" grid of the label files)
MATCH_MM = 2.0
METHOD = [
    "subject out (fold 0 = MR243 / MM1, fold 1 = MR252 / MM2, fold 2 = MR256 / MM3; labels/nnunet/folds_loso_532.json), from the folds'",
    'own validation predictions (nnUNet_results/.../fold_<k>/validation/<case>.png) against the references (nnUNet_raw/.../labelsTr).',
    'Per held-out case (a whole-section piece in the piece space TIRL registers, halved as in training) and nucleus (1 = LGN, 2 = STN; both',
    'hemispheres in one class): Dice; per nucleus component (8-connected; one per hemisphere) matched to the nearest predicted component (centroids within 2 mm):',
    'the centroid distance and the symmetric average surface distance (ASSD) of the matched pair; distances in mm on',
    'the blockface piece grid (the canvas is the blockface piece at store level 0, 2 x 40 um for the MM brains after the halving). Pooled per',
    "fold: Dice over all the fold's cases (sum of overlaps), the median case ASSD and component centroid distance, missed components (no",
    'predicted component within 2 mm) and false-positive components (predicted, no reference within 2 mm).',
    'Stains: MM1 and MM3 Lucifer yellow, MM2 fluorescein; the training subjects of each fold are the two others. No FR (Fluoro-Ruby)',
    "histology has a reference (MF6, MF8 are FR): the model's performance on FR is not measured here.",
    '-> paper/numbers/lgnstn_model_loso_r059.json',
    '    mri/env/mmorf/bin/python labels/r059/loso_metrics_r059.py']


def surf(m):
    return m & ~ndi.binary_erosion(m)


def assd(a, b):
    sa, sb = surf(a), surf(b)
    if not sa.any() or not sb.any():
        return None
    da = ndi.distance_transform_edt(~sb)[sa]; db = ndi.distance_transform_edt(~sa)[sb]
    return float(np.concatenate([da, db]).mean() * PX_MM)


def comps(m):
    lab, n = ndi.label(m, structure=np.ones((3, 3)))
    if not n:
        return []
    return [(np.array(c), lab == i + 1) for i, c in enumerate(ndi.center_of_mass(m, lab, range(1, n + 1)))]


def main():
    NN = paths.products("products_mri")["lgnstn_nnunet"]
    RES, REF = NN / "nnUNet_results" / DATASET / TRAINER, NN / "nnUNet_raw" / DATASET / "labelsTr"
    FOLDS = json.loads((NN / "folds_loso_532.json").read_text())
    out = dict(method=METHOD, model="Dataset532_LGNSTNnegdup nnUNetTrainer_100epochs nnUNetPlans_small 2d", px_mm=PX_MM, folds={})
    for k, S in FOLDS.items():
        vd = RES / f"fold_{k}" / "validation"
        cases = sorted(p.name for p in vd.glob(f"{S}_*.png"))
        rec = dict(subject=S, name=NAME[S], stain=STAIN[S], trained_on=[x for x in FOLDS.values() if x != S], cases=len(cases), structures={})
        for v, st in ((1, "LGN"), (2, "STN")):
            tp = fp = fn = 0; rows = []; cdist = []; missed = 0; fpc = 0; fp_px = []
            for c in cases:
                p = np.asarray(Image.open(vd / c)) == v
                r = np.asarray(Image.open(REF / c)) == v
                tp += int((p & r).sum()); fp += int((p & ~r).sum()); fn += int((~p & r).sum())
                if not (p.any() or r.any()):
                    continue
                d = 2 * (p & r).sum() / max(p.sum() + r.sum(), 1)
                cr, cp = comps(r), comps(p)
                ca = []
                for x, xm in cr:
                    dd = [(np.linalg.norm(x - y) * PX_MM, ym) for y, ym in cp]
                    dd = min(dd, key=lambda t: t[0]) if dd else None
                    if dd is None or dd[0] > MATCH_MM:
                        missed += 1
                    else:
                        cdist.append(dd[0]); ca.append(assd(dd[1], xm))
                for y, ym in cp:
                    if not cr or min(np.linalg.norm(x - y) * PX_MM for x, _ in cr) > MATCH_MM:
                        fpc += 1; fp_px.append(int(ym.sum()))
                rows.append(dict(case=c[:-4], dice=round(float(d), 4), assd_mm_matched=[round(z, 3) for z in ca if z is not None],
                                 ref_px=int(r.sum()), pred_px=int(p.sum())))
            a = [z for x in rows for z in x["assd_mm_matched"]]
            rec["structures"][st] = dict(pooled_dice=round(2 * tp / max(2 * tp + fp + fn, 1), 4), cases_scored=len(rows),
                                         assd_mm_median=round(float(np.median(a)), 3) if a else None,
                                         centroid_mm_median=round(float(np.median(cdist)), 3) if cdist else None,
                                         centroid_mm_max=round(float(np.max(cdist)), 3) if cdist else None,
                                         components_matched=len(cdist), components_missed=missed, false_positive_components=fpc,
                                         false_positive_component_mm2=[round(x * PX_MM ** 2, 3) for x in fp_px], cases_detail=rows)
        out["folds"][NAME[S]] = rec
        print(NAME[S], S, {st: {kk: vv for kk, vv in x.items() if kk != "cases_detail"} for st, x in rec["structures"].items()}, flush=True)
    print(provenance.write_numbers("lgnstn_model_loso_r059", out, inputs=[NN / "folds_loso_532.json"], md5=False,
                                   cmd="python reproduce/numbers/lgnstn_model_loso_r059.py"))


if __name__ == "__main__":
    main()
