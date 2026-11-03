#!/usr/bin/env python3
"""The tracer labels' volumes in the MRI template, per case and class, under both label rules, with the fraction inside the
template's white matter and, for the bundles, inside the SARM internal capsule (Results, tracer labels; Supplementary Table S4).

    python reproduce/numbers/tracer_label_volumes_r058d.py [--labels <products key>] [--name <record>]

You need
  labels   the final tracer labels in the template (configs/products_blockface.yaml `tracer_template_r058d`, the set of record for
           Fig. 9): <S>/<S>_<class>_mri-template_density.nii.gz, the transformed density (0-1; the fraction of annotated source
           tissue in the 0.25 mm voxel)
  masks    on the template's 0.25 mm grid: white matter = linc_mri_wm_fa025 (template FA > 0.25 inside the template brain mask,
           asserted); internal capsule = SARM level 6 label 50 (ic, NMT v2 SARM key) transformed into the template by the
           registration of record (`atlas_nmt`/SARM_level6_in_template.nii.gz), also + 0.5 mm (2 dilations), as Fig. 9 (c) draws it
What it does
  Rules: footprint = density >= 0.1 (what Fig. 9 shows; the _footprint files); solid = density > 0.5 (the _mri-template files);
  both recomputed from the density and asserted equal to the files. Groups as Fig. 9 draws them: bundles =
  Fiber_{light,moderate,dense}_bundle + Bundle (the "Fibers" strokes left out); terminals = Terminal_{light,moderate,dense}.
You get
  $LINC_NUMBERS/tracer_label_volumes_r058d.json: volumes in mm^3 (voxels x 0.015625) and fractions of the label's own voxels.

Ported from the analysis project's labels/tracer_label_volumes_r058.py (run there as --labels labels/tracer_template_r058d
--name tracer_label_volumes_r058d).
"""
import argparse
import json
import sys
from glob import glob
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from linc import paths, provenance  # noqa: E402

NAMES = Path(__file__).resolve().parents[2] / "docs" / "specimen_labels.json"
IC_LABEL = 50
CASES = ["MR243", "MR252", "MR256", "MF191"]
RULES = {"footprint": lambda d: d >= 0.1, "solid": lambda d: d > 0.5}
RULE_TEXT = {"footprint": "density >= 0.1 (what Fig. 9 shows)", "solid": "density > 0.5"}
BUNDLES = ("Fiber_light_bundle", "Fiber_moderate_bundle", "Fiber_dense_bundle", "Bundle")
TERMINALS = ("Terminal_light", "Terminal_moderate", "Terminal_dense")


def load(p):
    return np.asanyarray(nib.load(str(p)).dataobj)


def stats(m, wm, ic, ic05, vox):
    n = int(m.sum())
    r = dict(voxels=n, volume_mm3=round(n * vox, 3), share_in_wm=round(float((m & wm).sum() / n), 4) if n else None)
    r.update(share_in_ic=round(float((m & ic).sum() / n), 4) if n else None,
             share_in_ic_plus_0p5mm=round(float((m & ic05).sum() / n), 4) if n else None)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="tracer_template_r058d", help="label folder: a key of configs/products_blockface.yaml")
    ap.add_argument("--name", default="tracer_label_volumes_r058d", help="record name ($LINC_NUMBERS/<name>.json)")
    a = ap.parse_args()
    P = paths.products("products_blockface")
    LAB, T = P[a.labels], P["template_mri"]
    WM, FA, BM = T / "linc_mri_wm_fa025.nii.gz", T / "linc_mri_FA.nii.gz", T / "linc_mri_brainmask.nii.gz"
    SARM6 = P["atlas_nmt"] / "SARM_level6_in_template.nii.gz"
    ref = nib.load(str(WM)); vox = float(np.prod(np.abs(np.diag(ref.affine)[:3])))
    wm = load(WM) > 0
    assert np.array_equal(wm, (load(FA) > 0.25) & (load(BM) > 0)), "wm mask != FA > 0.25 in the brain"
    sarm = load(SARM6); ic = sarm == IC_LABEL; ic05 = ndi.binary_dilation(ic, iterations=2)
    names = json.loads(NAMES.read_text())["labels"]
    rec = dict(rules=RULE_TEXT, voxel_mm3=vox, white_matter="template FA > 0.25 in the template brain (linc_mri_wm_fa025)",
               internal_capsule=f"SARM level 6 label {IC_LABEL} (ic) carried into the template; + 0.5 mm = 2 binary dilations (Fig. 9 c)",
               template_scale=dict(wm_volume_mm3=round(float(wm.sum() * vox), 1), ic_volume_mm3=round(float(ic.sum() * vox), 1),
                                   ic_plus_0p5mm_volume_mm3=round(float(ic05.sum() * vox), 1)),
               groups=dict(bundles=list(BUNDLES), terminals=list(TERMINALS), left_out_of_the_figure=["Fibers"]), cases={})
    ins = [WM, FA, BM, SARM6, NAMES]
    for S in CASES:
        files = sorted(glob(str(LAB / S / f"{S}_*_mri-template_density.nii.gz")))
        dens = {Path(f).name[len(S) + 1:-len("_mri-template_density.nii.gz")]: f for f in files}
        case = dict(label=names.get(S, S), classes={}, groups={})
        G = {g: {r: np.zeros(ic.shape, bool) for r in RULES} for g in ("bundles", "terminals")}
        for c, f in dens.items():
            D = load(f); ins.append(Path(f))
            fp, so = load(f.replace("_density", "_footprint")) > 0, load(f.replace("_density", "")) > 0
            assert np.array_equal(fp, D >= 0.1) and np.array_equal(so, D > 0.5), f"{f}: rule files differ from the density"
            case["classes"][c] = {}
            for r, rule in RULES.items():
                m = rule(D); case["classes"][c][r] = stats(m, wm, ic, ic05, vox)
                for g, members in (("bundles", BUNDLES), ("terminals", TERMINALS)):
                    if c in members:
                        G[g][r] |= m
        for g in G:
            if any(c in dens for c in (BUNDLES if g == "bundles" else TERMINALS)):
                case["groups"][g] = {r: stats(G[g][r], wm, ic, ic05, vox) for r in RULES}
        rec["cases"][S] = case
        print(S, {g: {r: (v["volume_mm3"], v["share_in_wm"], v["share_in_ic"]) for r, v in case["groups"][g].items()}
                  for g in case["groups"]})
    rec["labels"] = str(LAB.relative_to(paths.products_root()))       # the label folder, relative to the products (as in the record)
    provenance.write_numbers(a.name, rec, inputs=ins, md5=False,
                             cmd=" ".join(["python reproduce/numbers/tracer_label_volumes_r058d.py"] + sys.argv[1:]))


if __name__ == "__main__":
    main()
