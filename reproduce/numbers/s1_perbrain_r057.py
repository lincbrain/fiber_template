#!/usr/bin/env python3
"""The per-brain table of the blockface brains (Supplementary S1.5, the blockface columns).

    python reproduce/numbers/s1_perbrain_r057.py

You need
  the regenerated records in $LINC_NUMBERS (run bf_cohort, bf_camera_exif and bf_spacing_used first), configs/subjects/<ID>.yaml and
  docs/specimen_labels.json.
What it does
  It reads, per sectioned brain:
  photographed_sections, interpolated_sections, missing_intervals
      bf_cohort.rows (native_slices; interpolated_sections; missing_intervals)
  camera
      bf_camera_exif.by_camera (model, or "no EXIF (<frame size>)")
  pixel_um, section_step_um (the calibrated spacing of record)
      bf_spacing_used.per_brain, asserted equal to configs/subjects/<ID>.yaml vx, vz
  nominal_pixel_um, nominal_section_step_um (the spacing each volume was reconstructed at, before calibration)
      configs/subjects/<ID>.yaml reconstruction_spacing_um (else vx, vz); bf_spacing_used's "nominal" fields are not used: for MF8
      and MF9 they hold the calibrated step of 26 Sep (750 / 870 um), not 600 um
  name
      docs/specimen_labels.json (the paper's names)
You get
  $LINC_NUMBERS/s1_perbrain_r057.json.

Ported from the analysis project's paper/r057_l1/s1_perbrain_r057.py, which read the same three records (the fields read here are
identical in the regenerated records).
"""
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from linc import paths, provenance  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CFG = REPO / "configs" / "subjects"
LABELS = REPO / "docs" / "specimen_labels.json"
NOTE = ("pixel_um / section_step_um: the spacing of record (each brain calibrated against its own MRI where it has one, "
        "bf_spacing_used); interpolated_sections: sections filled between photographed neighbors (bf_cohort)")


def main():
    num = paths.numbers_root()
    rec = lambda n: json.loads((num / f"{n}.json").read_text())["data"]  # noqa: E731
    co = {r["subject"]: r for r in rec("bf_cohort")["rows"]}
    cam = {S: c for c, Ss in rec("bf_camera_exif")["by_camera"].items() for S in Ss}
    sp = rec("bf_spacing_used")["per_brain"]; lab = json.loads(LABELS.read_text())["labels"]
    assert set(co) == set(cam) == set(sp) and len(co) == 21
    short = lambda c: c.replace("NIKON CORPORATION ", "").replace("NIKON ", "Nikon ")  # noqa: E731
    per = {}
    for S in sorted(co, key=lambda s: (lab[s][:2], int(lab[s][2:]))):
        c = yaml.safe_load((CFG / f"{S}.yaml").read_text()); rs = c.get("reconstruction_spacing_um") or dict(vx=c["vx"], vz=c["vz"])
        assert float(c["vx"]) == sp[S]["vx_um"] and float(c["vz"]) == sp[S]["vz_um"], f"{S}: config spacing differs from bf_spacing_used"
        per[S] = dict(name=lab[S], photographed_sections=co[S]["native_slices"], interpolated_sections=co[S]["interpolated_sections"],
                      missing_intervals=co[S]["missing_intervals"], camera=short(cam[S]), pixel_um=sp[S]["vx_um"],
                      section_step_um=sp[S]["vz_um"], nominal_pixel_um=float(rs["vx"]), nominal_section_step_um=float(rs["vz"]))
    out = dict(per_brain=per, cameras=sorted({v["camera"] for v in per.values()}), note=NOTE)
    for S, v in per.items():
        print(S, v)
    provenance.write_numbers("s1_perbrain_r057", out, inputs=[num / f"{n}.json" for n in ("bf_cohort", "bf_camera_exif", "bf_spacing_used")]
                             + [LABELS] + sorted(CFG.glob("M*.yaml")), md5=False, cmd="python reproduce/numbers/s1_perbrain_r057.py")


if __name__ == "__main__":
    main()
