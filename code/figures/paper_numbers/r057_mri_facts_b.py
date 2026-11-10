#!/usr/bin/env python3
"""Acquisition and processing facts of the four MRI-template brains (MF1-MF4 = MF283-MF286), as they stand in the data this work used
(paper: Methods, MRI acquisition and preprocessing).

  dwi_025   the 0.25 mm diffusion series: $LINC_RAW/macaque_mri/Preprocessing/<ID>/CS_DSI/dwi_orig.bvals (volumes, b < 200 s/mm^2 = b0
            volumes, b max) and dwi_realori.JSON (EchoTime, RepetitionTime; s -> ms)
  micro     the 0.5 mm multi-shell series the microstructure model was fitted to: $LINC_RAW/microstructure/data/<ID>/dmri.dt2/
            dwi_orig.bvals (b rounded to 100: shells and their volumes) and dwi_realori.JSON (TE, delta, Delta, TR)
  msme      $LINC_RAW/microstructure/data/<ID>/log.txt, the msme_process lines (TR, number of echoes, first TE = echo spacing)
  tensor    the tensor of record (products tensor_d9: <ID>_existing/run.json): volumes acquired / used, b-values used, gradient check
  sandi     the delivered fractions before normalization (products mri_inputs: sub-<ID>/micro/frac{1,2}_brain.nii.gz) inside the micro
            hand mask (sub-<ID>/masks/micro_brain_mask.nii.gz): mean and 5th/95th percentiles of frac1 + frac2, the share of voxels
            with frac1 > 1 or frac2 > 1, the maxima
Ported from the analysis project's paper/scripts/r057_mri_facts_b.py (r057_mri_facts with the source series described as in their
source paper, Gong et al. 2026, instead of by a folder name; same values); its description is kept verbatim as METHOD (part of the
record), with the raw-data prefix written as its placeholder ($LINC_RAW/macaque_mri; settings/internal/path_placeholders.yaml).
-> $LINC_NUMBERS/r057_mri_facts_b.json
    python code/figures/paper_numbers/r057_mri_facts_b.py          (settings/env/linc.yml, PY_MRI)"""
import json
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
from shared import paths, provenance  # noqa: E402

SUBJ = ["MF283", "MF284", "MF285", "MF286"]
B0 = 200
METHOD = [
    'as they stand in the data this work used (no raw ParaVision read; mri_acq_params_r057 covers the sectioned brains).',
    '  dwi_025   the 0.25 mm diffusion series of record ($LINC_RAW/macaque_mri/Preprocessing/<S>/CS_DSI): dwi_orig.bvals',
    '            (volumes, b < 200 s/mm^2 = b0 volumes, b max) and dwi_realori.JSON (EchoTime, RepetitionTime; s -> ms).',
    '  micro     the 0.5 mm multi-shell series SANDIdot was fitted to (preprocess_mri MICRO_RAW: Gong et al. 2026, multi-shell dMRI, <S>/dmri.dt2):',
    '            dwi_orig.bvals (b rounded to 100: shells and their volumes) and dwi_realori.JSON (TE, delta, Delta, TR).',
    '  msme      Gong et al. 2026, multi-slice multi-echo spin-echo images, <S>/log.txt, the msme_process lines (TR, number of echoes, first TE = echo spacing).',
    '  tensor    the tensor of record (D9, mri/tensor_refit_d9/<S>_existing/run.json): volumes acquired / used, b-values used, gradcheck.',
    '  sandi     the delivered SANDIdot fractions before normalisation (mri/data/sub-<S>/micro/frac{1,2}_brain.nii.gz; frac1 = soma, frac2 =',
    '            neurite) inside the micro hand mask (masks/micro_brain_mask): mean and 5th/95th percentiles of frac1 + frac2, the share of',
    '            voxels with frac1 > 1 or frac2 > 1, the maxima. preprocess_mri.normalize_fracs_and_weights then clips each to [0, 1] and',
    '            divides by their sum (the `_norm` files of record).',
    '-> paper/numbers/r057_mri_facts_b.json',
    '    mri/env/mmorf/bin/python paper/scripts/r057_mri_facts_b.py']


def main():
    raw, P = paths.raw_root(), paths.products("products_mri")
    out = dict(method=METHOD, brains={})
    ins = []
    for S in SUBJ:
        r = {}
        p, tg = raw / "macaque_mri" / "Preprocessing" / S / "CS_DSI", raw / "microstructure" / "data" / S
        b = np.loadtxt(p / "dwi_orig.bvals").ravel(); j = json.loads((p / "dwi_realori.JSON").read_text())
        r["dwi_025"] = dict(n_volumes=int(b.size), n_b0=int((b < B0).sum()), n_dw=int((b >= B0).sum()), b_max=int(round(b.max())),
                            te_ms=round(j["EchoTime"] * 1000, 2), tr_ms=round(j["RepetitionTime"] * 1000, 1))
        b = np.loadtxt(tg / "dmri.dt2" / "dwi_orig.bvals").ravel(); j = json.loads((tg / "dmri.dt2" / "dwi_realori.JSON").read_text())
        u, c = np.unique(np.round(b, -2).astype(int), return_counts=True)
        r["micro"] = dict(n_volumes=int(b.size), n_b0=int((b < B0).sum()), shells={str(k): int(v) for k, v in zip(u, c) if k >= B0},
                          n_shells=int((u >= B0).sum()), b_max=int(u.max()), te_ms=round(j["EchoTime"] * 1000, 2),
                          delta_ms=round(j["SmallDelta"] * 1000, 2), Delta_ms=round(j["BigDelta"] * 1000, 2), tr_ms=round(j["RepetitionTime"] * 1000, 1))
        log = open(tg / "log.txt", errors="ignore").read()
        g = lambda pat: (re.findall(pat, log) or [None])[0]  # noqa: E731
        r["msme"] = dict(tr_ms=float(g(r"The TR: ([\d.]+)")), n_echoes=int(g(r"The number of echos: (\d+)")),
                         echo_spacing_ms=round(float(g(r"echo spacing between echos: ([\d.]+)")) * 1000, 2))
        d9 = json.loads((P["tensor_d9"] / f"{S}_existing" / "run.json").read_text())
        r["tensor"] = dict(n_acquired=d9["n_acquired"], n_used=d9["n_used"], b_max=d9["bmax"], bvalues_used=d9["bvalues_used"],
                           n_bvalues_dw=len([x for x in d9["bvalues_used"] if x > 0]), gradcheck=d9["gradcheck_applied"], source=d9["source"])
        md = P["mri_inputs"] / f"sub-{S}"
        m = np.asanyarray(nib.load(str(md / "masks" / "micro_brain_mask.nii.gz")).dataobj) > 0
        f1 = np.asanyarray(nib.load(str(md / "micro" / "frac1_brain.nii.gz")).dataobj).astype(np.float64)[m]
        f2 = np.asanyarray(nib.load(str(md / "micro" / "frac2_brain.nii.gz")).dataobj).astype(np.float64)[m]
        s = f1 + f2
        r["sandi_raw"] = dict(n_voxels=int(m.sum()), sum_mean=round(float(s.mean()), 4), sum_p5=round(float(np.percentile(s, 5)), 4),
                              sum_p95=round(float(np.percentile(s, 95)), 4), share_any_gt1=round(float(((f1 > 1) | (f2 > 1)).mean()), 5),
                              share_sum_gt1=round(float((s > 1).mean()), 5), frac1_max=round(float(f1.max()), 3), frac2_max=round(float(f2.max()), 3))
        ins += [p / "dwi_orig.bvals", p / "dwi_realori.JSON", tg / "dmri.dt2" / "dwi_orig.bvals", tg / "dmri.dt2" / "dwi_realori.JSON",
                tg / "log.txt", P["tensor_d9"] / f"{S}_existing" / "run.json", md / "micro" / "frac1_brain.nii.gz", md / "micro" / "frac2_brain.nii.gz"]
        out["brains"][S] = r
        print(S, json.dumps(r), flush=True)
    print(provenance.write_numbers("r057_mri_facts_b", out, ins, md5=False, cmd="python code/figures/paper_numbers/r057_mri_facts_b.py"))


if __name__ == "__main__":
    main()
