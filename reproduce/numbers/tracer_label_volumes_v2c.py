#!/usr/bin/env python3
"""The tracer labels' volumes in the MRI template, including the connected pathways (Supplementary S2).

You need (configs/products_mri.yaml)
  tracer_template_v2: the interpolated labels (bundle, terminal-field and injection-site groups with their density classes);
  tracer_template_v2c: the connected pathways (or --connected-from work: the ones tracer_connected_v2c.py writes in $LINC_WORK).

What it does
  measures every label's volume under three rules: footprint (density >= 0.1), solid (density > 0.5) and, for the bundles, the
  connected pathway with its inferred fraction; groups are measured on the group's own density (the union of the classes is kept beside).

You get
  $LINC_NUMBERS/tracer_label_volumes_v2c.json

    python reproduce/numbers/tracer_label_volumes_v2c.py [--labels tracer_template_v2] [--connected-from products|work]
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / "tracer_label_volumes_r058d.py"
src = BASE.read_text()
CONNECTED_FROM = "work" if "--connected-from" in sys.argv and sys.argv[sys.argv.index("--connected-from") + 1] == "work" else "products"
if "--connected-from" in sys.argv:
    i = sys.argv.index("--connected-from"); del sys.argv[i:i + 2]
SUBS = [
    ('internal_capsule=f"SARM level 6 label {IC_LABEL} (ic) carried into the template; + 0.5 mm = 2 binary dilations (Fig. 9 c)"',
     'internal_capsule=f"SARM level 6 label {IC_LABEL} (ic) transformed to the template; + 0.5 mm = 2 binary dilations (the capsule panel of the tracer figure)"'),
    ('RULE_TEXT = {"footprint": "density >= 0.1 (what Fig. 9 shows)", "solid": "density > 0.5"}',
     'RULE_TEXT = {"footprint": "density >= 0.1 (what the tracer figure shows)", "solid": "density > 0.5", "connected": "the connected pathway: density > 0.5 seeds grown through >= 0.25 (26-connected) plus the inferred joins that connect every component into one pathway (tracer_connected_v2c; bundles; terminals keep the grown solid)"}'),
    ('    ap.add_argument("--labels", default="tracer_template_r058d", help="label folder: a key of configs/products_blockface.yaml")\n'
     '    ap.add_argument("--name", default="tracer_label_volumes_r058d", help="record name ($LINC_NUMBERS/<name>.json)")\n'
     '    a = ap.parse_args()\n'
     '    P = paths.products("products_blockface")\n'
     '    LAB, T = P[a.labels], P["template_mri"]\n',
     '    ap.add_argument("--labels", default="tracer_template_v2", help="label folder: a key of configs/products_mri.yaml")\n'
     '    ap.add_argument("--name", default="tracer_label_volumes_v2c", help="record name ($LINC_NUMBERS/<name>.json)")\n'
     '    a = ap.parse_args()\n'
     '    P = paths.products("products_mri"); P1 = paths.products("products_blockface")\n'
     '    LAB, T = P[a.labels], P1["template_mri"]\n'
     f'    V2C = (paths.work_root() / "tracer_v2c") if "{CONNECTED_FROM}" == "work" else P["tracer_template_v2c"]\n'),
    ('    SARM6 = P["atlas_nmt"] / "SARM_level6_in_template.nii.gz"', '    SARM6 = P1["atlas_nmt"] / "SARM_level6_in_template.nii.gz"'),
    ('        dens = {Path(f).name[len(S) + 1:-len("_mri-template_density.nii.gz")]: f for f in files}\n',
     '        dens = {Path(f).name[len(S) + 1:-len("_mri-template_density.nii.gz")]: f for f in files}\n'
     '        gdens = {g: dens.pop(g) for g in ("Bundles", "Terminals") if g in dens}                     # v2: the groups carried as one label\n'),
    ('''        for g in G:
            if any(c in dens for c in (BUNDLES if g == "bundles" else TERMINALS)):
                case["groups"][g] = {r: stats(G[g][r], wm, ic, ic05, vox) for r in RULES}''',
     '''        for g, gn in (("bundles", "Bundles"), ("terminals", "Terminals")):
            if gn in gdens:
                D = load(gdens[gn]); ins.append(Path(gdens[gn]))
                masks = {r: rule(D) for r, rule in RULES.items()}
                assert np.array_equal(load(gdens[gn].replace("_density", "")) > 0, masks["solid"]), f"{gn}: the solid file differs from the density"
                s2 = (V2C / S / f"{S}_Bundles_mri-template_solid_v2c.nii.gz") if gn == "Bundles" else Path(gdens[gn].replace("_density", "_solid2"))   # terminals: v2's connected solid
                ins.append(s2); masks["connected"] = load(s2) > 0
                case["groups"][g] = {r: stats(m, wm, ic, ic05, vox) for r, m in masks.items()}
                if gn == "Bundles":
                    cj = json.loads((V2C / S / f"{S}_connect_v2c.json").read_text())["final"]; ins.append(V2C / S / f"{S}_connect_v2c.json")
                    case["groups"][g]["connected"].update(inferred_voxels=cj["inferred_voxels"], inferred_mm3=cj["inferred_mm3"], inferred_share=cj["inferred_share"], pieces=cj["pieces"],
                                                           pieces_attached=cj["pieces_attached"], pieces_dropped=cj["pieces_dropped"], longest_bridge_mm=cj["longest_bridge_mm"], bridges_longer_than_10mm=cj["bridges_longer_than_flag"])
                case.setdefault("groups_union_of_grades", {})[g] = {r: stats(G[g][r], wm, ic, ic05, vox) for r in RULES}   # the record's way, for comparison'''),
    ('cmd=" ".join(["python reproduce/numbers/tracer_label_volumes_r058d.py"] + sys.argv[1:]))',
     'cmd=" ".join(["python reproduce/numbers/tracer_label_volumes_v2c.py"] + sys.argv[1:]))'),
]
for a, b in SUBS:
    assert src.count(a) == 1, a[:70]
    src = src.replace(a, b)
exec(compile(src, str(BASE), "exec"), {"__name__": "__main__", "__file__": str(BASE)})
