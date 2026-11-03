#!/usr/bin/env python3
"""The tracer labels carried into the MRI template, all four tracer cases in one summary (paper: Results, tracer labels in the
template; Fig. 10). Combines two records of the analysis (products: numbers_of_record):
  MR243 / MR252 / MR256  route B (blockface -> own MRI -> template): tracer_template_current.json
  MF191                  route A (blockface template member warp + the blockface-template -> MRI-template link):
                         tracer_template_current_mf191.json
Summary: n_cases, n_brains, brains, route, n_sections (per case and total), injection_site_recorded, classes.
-> $LINC_NUMBERS/tracer_template_current_all.json
    python reproduce/numbers/tracer_template_current_all.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from linc import paths, provenance  # noqa: E402

MF8 = "MF191"                                                      # LINC ID of MF8, the route-A case
# the record's own route text for MF8, verbatim (it is part of the record's data)
ROUTE_MF8 = "A (2b member: d2bs8 member warp + the link d2bs8_s5sym)"


def main():
    P = paths.products("products_mri")
    fb = P["numbers_of_record"] / "tracer_template_current.json"
    fa = P["numbers_of_record"] / "tracer_template_current_mf191.json"
    B = json.loads(fb.read_text())["data"]["summary"]; A = json.loads(fa.read_text())["data"]; ca = A["case"]
    nsec = dict(B["n_sections"]); nsec[MF8] = int(ca["n_sections"])
    route = dict(B["route"]); route[MF8] = ROUTE_MF8
    inj = dict(B["injection_site_recorded"])
    inj[MF8] = bool(ca.get("injection_site_recorded", "Injection_site" in ca.get("classes", [])))
    cls = dict(B["classes"]); cls[MF8] = ca.get("classes")
    root = paths.products_root()
    out = dict(summary=dict(n_cases=len(nsec), n_brains=len(nsec), brains=list(nsec), route=route, n_sections=nsec,
                            n_sections_total=int(sum(nsec.values())), injection_site_recorded=inj, classes=cls),
               sources={"route_B": str(fb.relative_to(root)), "MF191": str(fa.relative_to(root))})
    provenance.write_numbers("tracer_template_current_all", out, inputs=[fb, fa], md5=False,
                             cmd="python reproduce/numbers/tracer_template_current_all.py")


if __name__ == "__main__":
    main()
