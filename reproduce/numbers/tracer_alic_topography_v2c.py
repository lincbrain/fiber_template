#!/usr/bin/env python3
"""The tracer bundles' position in the anterior limb of the internal capsule with the connected pathways (paper: Results, tracer labels;
Supplementary S2): tracer_alic_topography_r058d.py on the interpolated labels of record (products_mri `tracer_template_v2`: the Bundles
group carried as one label) under the footprint and solid rules, plus the "connected" rule = the connected pathway (products_mri
`tracer_template_v2c`, or the one tracer_connected_v2c.py regenerates under $LINC_WORK/tracer_v2c with --connected-from work); the
injection site keeps the two rules.
-> $LINC_NUMBERS/tracer_alic_topography_v2c.json
    python reproduce/numbers/tracer_alic_topography_v2c.py [--labels tracer_template_v2] [--connected-from products|work]
Ported from the analysis project's labels/tracer_v2/alic_topography_v2c.py (a wrapper of the same shape over its r058 script).
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
     'RULE_TEXT = {"footprint": "density >= 0.1 (what Fig. 9 shows)", "solid": "density > 0.5", "connected": "the connected pathway: density > 0.5 seeds grown through >= 0.25 (26-connected) plus the inferred joins that connect every piece into one pathway (tracer_connected_v2c; alic only)"}'),
    ('    ap.add_argument("--labels", default="tracer_template_r058d", help="label folder: a key of configs/products_blockface.yaml")\n'
     '    ap.add_argument("--name", default="tracer_alic_topography_r058d", help="record name ($LINC_NUMBERS/<name>.json)")\n'
     '    a = ap.parse_args()\n'
     '    P = paths.products("products_blockface")\n'
     '    LAB, SARM6, RUN = P[a.labels], P["atlas_nmt"] / "SARM_level6_in_template.nii.gz", P["atlas_nmt"] / "run"\n',
     '    ap.add_argument("--labels", default="tracer_template_v2", help="label folder: a key of configs/products_mri.yaml")\n'
     '    ap.add_argument("--name", default="tracer_alic_topography_v2c", help="record name ($LINC_NUMBERS/<name>.json)")\n'
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
    ('cmd=" ".join(["python reproduce/numbers/tracer_alic_topography_r058d.py"] + sys.argv[1:]))',
     'cmd=" ".join(["python reproduce/numbers/tracer_alic_topography_v2c.py"] + sys.argv[1:]))'),
]
for a, b in SUBS:
    assert src.count(a) == 1, a[:70]
    src = src.replace(a, b)
exec(compile(src, str(BASE), "exec"), {"__name__": "__main__", "__file__": str(BASE)})
