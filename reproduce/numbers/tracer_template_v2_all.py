#!/usr/bin/env python3
"""The four tracer cases in the template, in one summary (Fig. 10; Supplementary S2).

You need (configs/products_mri.yaml)
  numbers_of_record: the two records of the transformation of the labels of record (tracer_template_v2, tracer_template_v2_mf191) and the earlier
  summary (tracer_template_r058d_all, for its audit block).

What it does
  combines the two records: the cases, brains, alignments (blockface-based or MRI-based), sections per case, whether an injection site was charted, the classes;
  carries over the audit of the two re-matched sections unchanged.

You get
  $LINC_NUMBERS/tracer_template_v2_all.json

    python reproduce/numbers/tracer_template_v2_all.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from linc import paths, provenance  # noqa: E402

NAME = "tracer_template_v2_all"
MF8 = "MF191"                                                      # LINC ID of MF8, the route-A case
ROUTE_MF8 = "A (2b member: d2bs8 member warp + the link d2bs8_s5sym)"   # the record's own route text for MF8, verbatim
LABEL_SOURCE = ("labels/tracer_v2 (histo_interp_v2.py on the sheets of record labels/push_tracer_r058d: the grades of a group interpolated as one label, "
                "per-hemisphere pairing, ends extended to mid-gap), transformed by the registrations of record")
V2 = dict(interpolation="labels/histo_interp_v2.py", carries=["labels/tracer_v2/carry_v2_B.py", "labels/tracer_v2/carry_v2_A.py"],
          solid_rules=["density > 0.5 (the _mri-template files)", "connected: > 0.5 seeds grown through >= 0.25 (the _solid2 files)"],
          predecessor="tracer_template_r058d_all")


def neutral(block):
    """The audit block of record with its decision note in neutral words (the record names the day's decision maker)."""
    block = dict(block)
    block["rule"] = block["rule"].replace("(found despite", "(detected despite")
    block["rule"] = re.sub(r"; [^;]*: move only where the tissue outline agrees$", "; decided 30 Oct 2026: move only where the tissue outline agrees", block["rule"])
    return block


def main():
    P = paths.products("products_mri"); N = P["numbers_of_record"]
    fb, fa, fp = N / "tracer_template_v2.json", N / "tracer_template_v2_mf191.json", N / "tracer_template_r058d_all.json"
    B = json.loads(fb.read_text())["data"]["summary"]; A = json.loads(fa.read_text())["data"]; ca = A["case"]
    nsec = dict(B["n_sections"]); nsec[MF8] = int(ca["n_sections"])
    route = dict(B["route"]); route[MF8] = ROUTE_MF8
    inj = dict(B["injection_site_recorded"]); inj[MF8] = bool(ca.get("injection_site_recorded", "Injection_site" in ca.get("classes", [])))
    cls = dict(B["classes"]); cls[MF8] = ca.get("classes")
    root = paths.products_root()
    out = dict(summary=dict(n_cases=len(nsec), n_brains=len(nsec), brains=list(nsec), route=route, n_sections=nsec,
                            n_sections_total=int(sum(nsec.values())), injection_site_recorded=inj, classes=cls),
               sources={"route_B": str(fb.relative_to(root)), "MF191": str(fa.relative_to(root))},
               audit_rematch=neutral(json.loads(fp.read_text())["data"]["audit_rematch"]),
               label_sources={S: LABEL_SOURCE for S in ("MR243", "MR252", "MR256", "MF191")}, v2=V2)
    print(provenance.write_numbers(NAME, out, inputs=[fb, fa, fp], md5=False, cmd="python reproduce/numbers/tracer_template_v2_all.py"))


if __name__ == "__main__":
    main()
