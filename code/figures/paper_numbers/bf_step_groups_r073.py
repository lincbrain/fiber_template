#!/usr/bin/env python3
"""The blockface section step by groups of brains (paper: Methods, "Sectioning and blockface photography").

    python code/figures/paper_numbers/bf_step_groups_r073.py

You need
  settings/brains/<ID>.yaml (the nominal step, reconstruction_spacing_um vz, else vz; the step as used, vz) and
  settings/specimen_labels.json.
What it does
  It groups the 21 sectioned brains by the nominal step, the photographing interval of the protocol for 50 um sections:
  50 um = every section, 400 um = every 8th, >= 600 um = coarser (MF8 and MF9 at 600 um, every 12th; MF10 at 810 um).
  Per group: the brains, their number, and the range of the nominal and of the used (calibrated) step.
You get
  $LINC_NUMBERS/bf_step_groups_r073.json.

Ported from the analysis project's paper/r073_l1/bf_step_groups_r073.py, which read the same two values per brain from
s1_perbrain_r057 (itself read from these configs).
"""
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
from shared import provenance  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
CFG = REPO / "settings" / "brains"
LABELS = REPO / "settings" / "specimen_labels.json"
SECTION_UM = 50.0


def group(nominal):
    if nominal == SECTION_UM:
        return "every_section"
    if nominal == 8 * SECTION_UM:
        return "every_8th"
    if nominal >= 600.0:
        return "coarser"
    raise ValueError(nominal)


def main():
    lab = json.loads(LABELS.read_text())["labels"]
    g, ins = {}, [LABELS]
    for p in sorted(CFG.glob("M*.yaml")):
        c = yaml.safe_load(p.read_text()); S = c.get("subject", p.stem)
        if S not in lab or "vz" not in c:
            continue
        rs = c.get("reconstruction_spacing_um") or dict(vx=c["vx"], vz=c["vz"]); ins.append(p)
        g.setdefault(group(float(rs["vz"])), []).append((lab[S], S, float(rs["vz"]), float(c["vz"])))
    out = {}
    for k, rows in g.items():
        rows.sort(key=lambda r: (r[0][:2], int(r[0][2:])))
        out[k] = dict(n=len(rows), brains=[r[0] for r in rows], ids=[r[1] for r in rows],
                      nominal_um_min=min(r[2] for r in rows), nominal_um_max=max(r[2] for r in rows),
                      step_um_min=min(r[3] for r in rows), step_um_max=max(r[3] for r in rows),
                      per_brain={r[0]: dict(nominal_um=r[2], step_um=r[3]) for r in rows})
    assert sum(v["n"] for v in out.values()) == 21, sum(v["n"] for v in out.values())
    rec = dict(groups=out, section_um=SECTION_UM,
               rule="by the nominal step: 50 um = every section, 400 um = every 8th, >= 600 um = coarser (MF8, MF9 600 um = every 12th; MF10 810 um)")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_brain"} for k, v in out.items()}, indent=1))
    provenance.write_numbers("bf_step_groups_r073", rec, inputs=ins, md5=False, cmd="python code/figures/paper_numbers/bf_step_groups_r073.py")


if __name__ == "__main__":
    main()
