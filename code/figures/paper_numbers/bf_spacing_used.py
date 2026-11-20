#!/usr/bin/env python3
"""The blockface spacing as used by the reconstructions, per brain and as ranges over the 21 photographed brains
(Methods, "Sectioning and blockface photography").

    python code/figures/paper_numbers/bf_spacing_used.py

You need
  each brain's 02_blockface.ome.zarr and settings/brains/<ID>.yaml; the cohort record of 28 Sep 2026
  (tests/reference/bf_cohort.json) for the nominal values.
What it does
  It reads the photograph pixel size and the section step from each reconstruction's own store header (level 0: z = rows = the
  in-plane pixel, y = sections = the section step; um), checks them against the brain's config (vx, vz), and sets them beside the
  nominal values recorded before the spacing calibration of 2 Oct 2026.
You get
  $LINC_NUMBERS/bf_spacing_used.json.
"""
import json
import sys
from pathlib import Path

import yaml
import zarr

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
from shared import paths, provenance  # noqa: E402


def main():
    prod = paths.products("products_blockface")
    cohort = REPO / "tests" / "reference" / "bf_cohort.json"        # the cohort record of 28 Sep, shipped
    nominal = {r["subject"]: (float(r["vx_um"]), float(r["vz_um"])) for r in json.loads(cohort.read_text())["data"]["rows"]}
    per, ins = {}, []
    for S in sorted(nominal):
        c = yaml.safe_load((REPO / "settings" / "brains" / f"{S}.yaml").read_text())
        store = prod["data"] / f"sub-{S}" / "blockface" / "02_blockface.ome.zarr"
        g = zarr.open_group(str(store), mode="r")
        sc = g.attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][1]["scale"]   # [c, z, y, x] um
        vx, vz = float(sc[1]), abs(float(sc[2]))
        assert abs(float(sc[3]) - vx) < 1e-6, (S, sc)
        assert abs(vx - float(c["vx"])) < 1e-3 and abs(vz - float(c["vz"])) < 1e-3, (S, vx, vz, c["vx"], c["vz"])
        per[S] = dict(vx_um=round(vx, 4), vz_um=round(vz, 4), nominal_vx_um=nominal[S][0], nominal_vz_um=nominal[S][1])
        ins.append(store / "zarr.json")
    vxs = [p["vx_um"] for p in per.values()]; vzs = [p["vz_um"] for p in per.values()]
    data = dict(n=len(per), vx_um=dict(range=[min(vxs), max(vxs)]), vz_um=dict(range=[min(vzs), max(vzs)]),
                nominal=dict(vx_um_range=[min(v[0] for v in nominal.values()), max(v[0] for v in nominal.values())],
                             vz_um_range=[min(v[1] for v in nominal.values()), max(v[1] for v in nominal.values())]),
                changed=sorted(S for S, p in per.items() if (p["vx_um"], p["vz_um"]) != (p["nominal_vx_um"], p["nominal_vz_um"])),
                per_brain=per)
    provenance.write_numbers("bf_spacing_used", data, inputs=ins + [cohort], cmd="python code/figures/paper_numbers/bf_spacing_used.py")


if __name__ == "__main__":
    main()
