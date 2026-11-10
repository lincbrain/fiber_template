#!/usr/bin/env python3
"""Join each tracer case's bundle label into one connected pathway, marking the joins as inferred (Fig. 10; Supplementary S2).

You need (settings/internal/products_mri.yaml)
  tracer_template_v2: each case's interpolated bundle density and its connected solid in the template; template_mri: the brain mask
  and the white matter (FA > 0.25).

What it does, per case
  1. splits the connected solid into its connected components and keeps the largest as the pathway;
  2. joins every other component along the cheapest path inside the brain (cheap where the density is high, cheaper in white
     matter), the cheapest component first, the path widened to a 1 mm tube; drops only specks smaller than 8 voxels;
  3. marks the join voxels that were not in the solid as inferred.

You get
  $LINC_WORK/tracer_v2c/<ID>/<ID>_Bundles_mri-template_connected.nii.gz (uint8: 1 = charted, 2 = inferred), _solid_v2c.nii.gz
  (the pathway, 0/1), _bridges.nii.gz (each join's component number), <ID>_connect_v2c.json; and
  $LINC_NUMBERS/tracer_connected_v2c.json (per case: the components, every join with its length, the inferred fraction, the longest join)

    python code/figures/paper_numbers/tracer_connected_v2c.py [--cases MR243 ...]
"""
import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
from shared import paths, provenance  # noqa: E402
from labels.connect import CASES, GROUP, connect  # noqa: E402

NAME = "tracer_connected_v2c"
# the record's method text, fixed (it was this script's docstring when the record was made; the docstring may change)
METHOD = [
    'One connected pathway per tracer case (paper: Results, tracer labels; Fig. 10; Supplementary S2): the interpolated bundle label of each',
    'case joined into a single component, the joins marked as inferred.',
    '',
    'Inputs (products_mri `tracer_template_v2`): <S>/<S>_Bundles_mri-template_density.nii.gz (the transformed density of the interpolated bundle',
    'label, 0-1 on the 0.25 mm template grid) and <S>_Bundles_mri-template_solid2.nii.gz (its connected solid: density > 0.5 grown through',
    'connected voxels of density >= 0.25); the template brain mask and white matter (template_mri: linc_mri_brainmask, linc_mri_wm_fa025).',
    'Rule: the 26-connected components of the connected solid, largest first; every other component is joined to the pathway along the cheapest',
    'path inside the template brain mask (voxel cost 1 / (density + EPS), halved in white matter and inside the footprint, density >= 0.1;',
    '6-connected graph; multi-source Dijkstra from the current pathway, the cheapest component first, so chains of components connect through each',
    'other), the path dilated to a tube of TUBE_R voxels; components smaller than MIN_VOX voxels are dropped (listed), no component is dropped for',
    'distance; every join is recorded with its length and voxel count and flagged when longer than LONG_MM. Join voxels that were not in',
    'the solid are inferred: label 2 in the written label (1 = the solid).',
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="+", default=list(CASES)); ap.add_argument("--labels", default="tracer_template_v2", help="products_mri key of the interpolated labels")
    a = ap.parse_args()
    P = paths.products("products_mri"); lab_dir, tpl = P[a.labels], P["template_mri"]; out_dir = paths.work_root() / "tracer_v2c"
    rec = dict(method=METHOD, cases={}); inputs = []
    for S in a.cases:
        rec["cases"][S] = connect(S, lab_dir / S, tpl, out_dir / S)
        inputs += [lab_dir / S / f"{S}_{GROUP}_mri-template_density.nii.gz", lab_dir / S / f"{S}_{GROUP}_mri-template_solid2.nii.gz"]
    inputs += [tpl / "linc_mri_brainmask.nii.gz", tpl / "linc_mri_wm_fa025.nii.gz"]
    print(provenance.write_numbers(NAME, rec, inputs=inputs, md5=False, cmd=" ".join(["python code/figures/paper_numbers/tracer_connected_v2c.py"] + sys.argv[1:])))


if __name__ == "__main__":
    main()
