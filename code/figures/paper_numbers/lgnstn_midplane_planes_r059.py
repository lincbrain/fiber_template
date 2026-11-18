#!/usr/bin/env python3
"""MM3's left-right offset of the LGN and STN, stage by stage, against two symmetry planes side by side (paper: Supplementary S3).

For every 3D stage (bf_3d, b0, template) and both label sources (observer, network), the left-right asymmetry (asym = dL - dR, > 0: the
left nucleus lies farther from the midline) against the plane fitted near the nuclei ("local", within 20 mm) and against the plane
fitted to the whole brain mask ("global"), each in template-equivalent mm (asym_frac x the template-stage dL + dR of the same brain and
nucleus; the template's planes are x = 0 for both fits). Also the change at each registration step (R1: bf_3d -> b0; R5: b0 -> template)
under each plane. Read from the record of record lgnstn_midplane_r059 (products numbers_of_record; the plane fits themselves are that
record's, made in the analysis project by labels/r059/lgnstn_midplane_r059.py), no refit.
Ported from the analysis project's labels/r059/lgnstn_midplane_planes_r059.py; its definitions text is kept verbatim (part of the record).
-> $LINC_NUMBERS/lgnstn_midplane_planes_r059_h9r17.json
    python code/figures/paper_numbers/lgnstn_midplane_planes_r059.py          (any env; json only)"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
from shared import paths, provenance  # noqa: E402

DEFINITIONS = (
    'For every 3D stage (bf_3d, b0, template) and both label sources, the left-right asymmetry (asym = dL - dR, > 0: the left nucleus lies\n'
    'farther from the midline) against the plane fitted near the nuclei ("local", within 20 mm) and against the plane fitted to the whole\n'
    'brain mask ("global"), each in template-equivalent mm (asym_frac x the template-stage dL + dR of the same brain and nucleus, the\n'
    "template's planes being x = 0 for both fits). Also the change at each registration step (R1: bf_3d -> b0; R5: b0 -> template) under each\n"
    'plane. bf_sections and chart were measured against the local plane only and are not repeated here. Read from the record; no refit.')


def main():
    src = paths.products("products_mri")["numbers_of_record"] / "lgnstn_midplane_r059_h9r17.json"
    D = json.loads(src.read_text())["data"]
    out = dict(definitions=DEFINITIONS, cases={}, planes=D["planes"])
    for name, nucs in D["cases"].items():
        out["cases"][name] = {}
        for nuc, who_d in nucs.items():
            out["cases"][name][nuc] = {}
            for who, st in who_d.items():
                span = st["template"]["dL"] + st["template"]["dR"]
                r = {}
                for plane in ("local", "global"):
                    v = {s: round((st[s]["asym_frac"] if plane == "local" else st[s]["global_plane"]["asym_frac"]) * span, 3)
                         for s in ("bf_3d", "b0", "template")}
                    v["R1_change"] = round(v["b0"] - v["bf_3d"], 3); v["R5_change"] = round(v["template"] - v["b0"], 3)
                    r[plane] = v
                out["cases"][name][nuc][who] = r
                print(name, nuc, who, r)
    print(provenance.write_numbers("lgnstn_midplane_planes_r059_h9r17", out, inputs=[src],
                                   cmd="python code/figures/paper_numbers/lgnstn_midplane_planes_r059.py"))


if __name__ == "__main__":
    main()
