#!/usr/bin/env python3
"""Where the tracer labels lie in the template after their transformation: in the brain, in white matter, how far from it
(Supplementary S2), as a check that the transformed labels follow the anatomy.

You need (configs/products_mri.yaml, configs/products_blockface.yaml)
  tracer_sections_v2: the charted sections transformed without interpolation; tracer_template_r058d: the earlier interpolated labels;
  tracer_template_v2 and tracer_template_v2c: the interpolated labels and the connected pathways of record; template_mri: the brain
  mask and the white matter (FA > 0.25).

What it does
  for every case, group and label source, under both rules (density > 0.5, density >= 0.1): the fraction outside the brain, the
  fractions in white matter and other tissue, and for bundles outside white matter the distance to it.

You get
  $LINC_NUMBERS/tracer_placement_v2c.json

    python reproduce/numbers/tracer_placement_v2c.py
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from linc import paths, provenance  # noqa: E402

NAME = "tracer_placement_v2c"
CASES = ["MR243", "MR252", "MR256", "MF191"]
GROUPS = {"Bundles": ["Fiber_light_bundle", "Fiber_moderate_bundle", "Fiber_dense_bundle", "Bundle"],
          "Terminals": ["Terminal_light", "Terminal_moderate", "Terminal_dense"]}
TV = 0.25
# the record's method text, fixed (it was this script's docstring when the record was made; the docstring may change)
METHOD = [
    'Where the transformed tracer labels lie in the template (paper: Results, tracer labels; Supplementary S2): per case and group (bundles,',
    'terminals), for the section labels (no interpolation between sections; products_mri `tracer_sections_v2`), the interpolated labels of',
    'the earlier chain (products_blockface `tracer_template_r058d`, per-class densities combined by their maximum, as the classes overlap), the',
    'interpolated labels of record (products_mri `tracer_template_v2`) and the connected pathways (products_mri `tracer_template_v2c`, bundles',
    'only; a 0/1 label, so its solid and footprint rules coincide), under both rules (solid: density > 0.5; footprint: density >= 0.1):',
    '  fraction outside the template brain mask; tissue breakdown of the label voxels: white matter (template FA > 0.25), other brain tissue',
    '  (brain mask minus white matter: gray matter, but also the ventricles, which the template has no mask for), outside the brain;',
    '  for bundle voxels outside white matter: the distance to the nearest white-matter voxel (mm; median, p90, fraction within one voxel =',
    '  0.25 mm, the partial-volume rim, and fraction beyond 0.5 mm = misplacement).',
    'Report only (no visual verdict).',
]


def breakdown(m, bundles, B, W, DW):
    n = int(m.sum())
    if n == 0:
        return dict(voxels=0)
    out = dict(voxels=n, volume_mm3=round(n * TV ** 3, 3), outside_brain=round(float((m & ~B).sum() / n), 4),
               tissue=dict(white_matter=round(float((m & W).sum() / n), 4), other_brain=round(float((m & B & ~W).sum() / n), 4),
                           outside_brain=round(float((m & ~B).sum() / n), 4)))
    if bundles:
        d = DW[m & ~W]
        out["outside_wm_distance_mm"] = (dict(voxels=int(d.size), median=round(float(np.median(d)), 3), p90=round(float(np.percentile(d, 90)), 3),
                                              share_within_one_voxel=round(float((d <= TV + 1e-6).mean()), 4), share_beyond_0p5mm=round(float((d > 0.5).mean()), 4))
                                         if d.size else dict(voxels=0))
    return out


def density(SRC, S, g, which):
    root, pat, kind = SRC[which]
    if kind == "group":
        p = root / S / pat.format(S=S, g=g)
        return np.clip(np.asarray(nib.load(str(p)).dataobj, np.float32), 0, 1) if p.exists() else None, [p] if p.exists() else []
    U = None; used = []
    for c in GROUPS[g]:
        p = root / S / pat.format(S=S, c=c)
        if p.exists():
            d = np.asarray(nib.load(str(p)).dataobj, np.float32); U = d if U is None else np.maximum(U, d); used.append(p)   # max over classes (they overlap)
    return (np.clip(U, 0, 1) if U is not None else None), used


def main():
    P1, P2 = paths.products("products_blockface"), paths.products("products_mri"); tpl = P2["template_mri"]
    SRC = {"sections": (P2["tracer_sections_v2"], "{S}_{g}_mri-template_sections_density.nii.gz", "group"),
           "interpolated_record": (P1["tracer_template_r058d"], "{S}_{c}_mri-template_density.nii.gz", "classes"),
           "interpolated_v2": (P2["tracer_template_v2"], "{S}_{g}_mri-template_density.nii.gz", "group"),
           "connected_v2c": (P2["tracer_template_v2c"], "{S}_{g}_mri-template_solid_v2c.nii.gz", "group")}   # a 0/1 mask (bundles only)
    B = np.asanyarray(nib.load(str(tpl / "linc_mri_brainmask.nii.gz")).dataobj) > 0
    W = np.asanyarray(nib.load(str(tpl / "linc_mri_wm_fa025.nii.gz")).dataobj) > 0
    DW = ndi.distance_transform_edt(~W) * TV
    rec = dict(method=METHOD, rules=dict(solid="density > 0.5", footprint="density >= 0.1", white_matter="template FA > 0.25",
               other_brain="template brain mask minus white matter (gray matter and ventricles; no CSF mask in the template)", voxel_mm=TV), cases={})
    inputs = [tpl / "linc_mri_brainmask.nii.gz", tpl / "linc_mri_wm_fa025.nii.gz"]
    for S in CASES:
        rec["cases"][S] = {}
        for g in GROUPS:
            rec["cases"][S][g] = {}
            for which in SRC:
                U, used = density(SRC, S, g, which)
                if U is None:
                    continue
                inputs += used
                rec["cases"][S][g][which] = dict(solid=breakdown(U > 0.5, g == "Bundles", B, W, DW), footprint=breakdown(U >= 0.1, g == "Bundles", B, W, DW))
                r = rec["cases"][S][g][which]["solid"]
                print(S, g, which, "solid: outside %.3f WM %.3f other %.3f" % (r.get("outside_brain", 0), r.get("tissue", {}).get("white_matter", 0), r.get("tissue", {}).get("other_brain", 0)), flush=True)
    print(provenance.write_numbers(NAME, rec, inputs=inputs, md5=False, cmd="python reproduce/numbers/tracer_placement_v2c.py"))


if __name__ == "__main__":
    main()
