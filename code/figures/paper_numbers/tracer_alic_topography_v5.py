#!/usr/bin/env python3
"""Where the tracer bundles cross the anterior limb of the internal capsule, on the charted sections Fig. 10 draws.

You need (settings/internal/products_mri.yaml, settings/internal/products_blockface.yaml)
  tracer_sections_v2: the per-section sheets carried into the template without interpolation (<S>_Bundles_mri-template_sections_density,
  <S>_Injection_site_mri-template_sections_density: each charted section's carried fraction where it lands);
  atlas_nmt: the SARM level-6 atlas transformed to the template (its internal capsule).

What it does
  tracer_alic_topography_r058d.py's measure, method unchanged, on those sections at Fig. 10's slabs y = 16, 18 and 20 mm: each case's
  position inside the capsule (dorsal-ventral and medial-lateral, over the whole limb and per coronal plane) under the footprint
  (carried fraction >= 0.1, what Fig. 10 shows) and solid (> 0.5) rules; the Bundles group as one label, and the injection site under
  the same rules. No "connected" rule (Fig. 10 draws no pathway).

You get
  $LINC_NUMBERS/tracer_alic_topography_v5<tag>.json (the labels carried through the registrations with the per-component nonlinear stiffness)

    python code/figures/paper_numbers/tracer_alic_topography_v5.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from shared.paths import template_chain_tag  # noqa: E402
TAG = template_chain_tag()                    # (<tag> = settings/labels.yaml template_chain_tag)
BASE = HERE / "tracer_alic_topography_r058d.py"
src = BASE.read_text()
SUBS = [
    ('RULE_TEXT = {"footprint": "density >= 0.1 (what Fig. 9 shows)", "solid": "density > 0.5"}',
     'RULE_TEXT = {"footprint": "carried fraction >= 0.1, each charted section where it lands (what Fig. 10 shows)", "solid": "carried fraction > 0.5"}'),
    ('Y_RANGE, PLANES, HALF = (20.0, 24.0), (20.0, 22.0, 24.0), 0.75', 'Y_RANGE, PLANES, HALF = (16.0, 20.0), (16.0, 18.0, 20.0), 0.75'),
    ('    ap.add_argument("--labels", default="tracer_template_r058d", help="label folder: a key of configs/products_blockface.yaml")\n'
     '    ap.add_argument("--name", default="tracer_alic_topography_r058d", help="record name ($LINC_NUMBERS/<name>.json)")\n'
     '    a = ap.parse_args()\n'
     '    P = paths.products("products_blockface")\n'
     '    LAB, SARM6, RUN = P[a.labels], P["atlas_nmt"] / "SARM_level6_in_template.nii.gz", P["atlas_nmt"] / "run"\n',
     '    ap.add_argument("--labels", default="tracer_sections_v2", help="label folder: a key of configs/products_mri.yaml")\n'
     '    ap.add_argument("--name", default="tracer_alic_topography_v5' + TAG + '", help="record name ($LINC_NUMBERS/<name>.json)")\n'
     '    a = ap.parse_args()\n'
     '    P = paths.products("products_blockface"); P2 = paths.products("products_mri")\n'
     '    LAB, SARM6, RUN = P2[a.labels], P["atlas_nmt"] / "SARM_level6_in_template.nii.gz", P["atlas_nmt"] / "run"\n'),
    ('        files = [f for c in BUNDLES for f in glob(str(LAB / S / f"{S}_{c}_mri-template_density.nii.gz"))]',
     '        files = [str(LAB / S / f"{S}_Bundles_mri-template_sections_density.nii.gz")]          # the charted sections, the group as one label'),
    ('    f = LAB / S / f"{S}_Injection_site_mri-template_density.nii.gz"',
     '    f = LAB / S / f"{S}_Injection_site_mri-template_sections_density.nii.gz"'),
    ('cmd=" ".join(["python code/figures/paper_numbers/tracer_alic_topography_r058d.py"] + sys.argv[1:]))',
     'cmd=" ".join(["python code/figures/paper_numbers/tracer_alic_topography_v5.py"] + sys.argv[1:]))'),
]
for a, b in SUBS:
    assert src.count(a) == 1, a[:70]
    src = src.replace(a, b)
exec(compile(src, str(BASE), "exec"), {"__name__": "__main__", "__file__": str(BASE)})
