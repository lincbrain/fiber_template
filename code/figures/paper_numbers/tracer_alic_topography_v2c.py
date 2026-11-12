#!/usr/bin/env python3
"""Where the tracer bundles cross the anterior limb of the internal capsule (Supplementary S2).

You need (settings/internal/products_mri.yaml, settings/internal/products_blockface.yaml)
  tracer_template_v2: the interpolated labels; tracer_template_v2c: the connected pathways (or --connected-from work);
  atlas_nmt: the SARM level-6 atlas transformed to the template (its internal capsule).

What it does
  measures each case's position inside the capsule (dorsal-ventral and medial-lateral, over the whole limb and per coronal plane)
  under the footprint and solid rules and, for the bundles, the connected pathway; the injection site under the first two rules.

You get
  $LINC_NUMBERS/tracer_alic_topography_v2c_h9.json

    python code/figures/paper_numbers/tracer_alic_topography_v2c.py [--labels tracer_template_v2] [--connected-from products|work]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / "tracer_alic_topography_r058d.py"
src = BASE.read_text()
CONNECTED_FROM = "work" if "--connected-from" in sys.argv and sys.argv[sys.argv.index("--connected-from") + 1] == "work" else "products"
if "--connected-from" in sys.argv:
    i = sys.argv.index("--connected-from"); del sys.argv[i:i + 2]
SUBS = [
    ('RULE_TEXT = {"footprint": "density >= 0.1 (what Fig. 9 shows)", "solid": "density > 0.5"}',
     'RULE_TEXT = {"footprint": "density >= 0.1 (what the tracer figure shows)", "solid": "density > 0.5", "connected": "the connected pathway: density > 0.5 seeds grown through >= 0.25 (26-connected) plus the inferred joins that connect every component into one pathway (tracer_connected_v2c; alic only)"}'),
    ('    ap.add_argument("--labels", default="tracer_template_r058d", help="label folder: a key of configs/products_blockface.yaml")\n'
     '    ap.add_argument("--name", default="tracer_alic_topography_r058d", help="record name ($LINC_NUMBERS/<name>.json)")\n'
     '    a = ap.parse_args()\n'
     '    P = paths.products("products_blockface")\n'
     '    LAB, SARM6, RUN = P[a.labels], P["atlas_nmt"] / "SARM_level6_in_template.nii.gz", P["atlas_nmt"] / "run"\n',
     '    ap.add_argument("--labels", default="tracer_template_v2", help="label folder: a key of configs/products_mri.yaml")\n'
     '    ap.add_argument("--name", default="tracer_alic_topography_v2c_h9", help="record name ($LINC_NUMBERS/<name>.json)")\n'
     '    a = ap.parse_args()\n'
     '    P = paths.products("products_blockface"); P2 = paths.products("products_mri")\n'
     '    LAB, SARM6, RUN = P2[a.labels], P["atlas_nmt"] / "SARM_level6_in_template.nii.gz", P["atlas_nmt"] / "run"\n'
     f'    V2C = (paths.work_root() / "tracer_v2c") if "{CONNECTED_FROM}" == "work" else P2["tracer_template_v2c"]\n'),
    ('        files = [f for c in BUNDLES for f in glob(str(LAB / S / f"{S}_{c}_mri-template_density.nii.gz"))]',
     '        files = [str(LAB / S / f"{S}_Bundles_mri-template_density.nii.gz")]                    # v2: the group carried as one label'),
    ('''            case["alic"][r] = {cn: dict(range=position(m, cap, A, SIDE[S], Y_RANGE),
                                        per_plane={str(y): position(m, cap, A, SIDE[S], (y - HALF, y + HALF)) for y in PLANES})
                               for cn, cap in caps.items()}
''',
     '''            case["alic"][r] = {cn: dict(range=position(m, cap, A, SIDE[S], Y_RANGE),
                                        per_plane={str(y): position(m, cap, A, SIDE[S], (y - HALF, y + HALF)) for y in PLANES})
                               for cn, cap in caps.items()}
        p2 = V2C / S / f"{S}_Bundles_mri-template_solid_v2c.nii.gz"; m2 = load(p2) > 0; ins.append(p2)
        case["alic"]["connected"] = {cn: dict(range=position(m2, cap, A, SIDE[S], Y_RANGE),
                                              per_plane={str(y): position(m2, cap, A, SIDE[S], (y - HALF, y + HALF)) for y in PLANES})
                                     for cn, cap in caps.items()}
'''),
    ('cmd=" ".join(["python code/figures/paper_numbers/tracer_alic_topography_r058d.py"] + sys.argv[1:]))',
     'cmd=" ".join(["python code/figures/paper_numbers/tracer_alic_topography_v2c.py"] + sys.argv[1:]))'),
]
for a, b in SUBS:
    assert src.count(a) == 1, a[:70]
    src = src.replace(a, b)
exec(compile(src, str(BASE), "exec"), {"__name__": "__main__", "__file__": str(BASE)})
