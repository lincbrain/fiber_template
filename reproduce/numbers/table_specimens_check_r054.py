#!/usr/bin/env python3
"""Check of the MRI, Histology, Tracer and LGN/STN cells of Table 1 (the specimen table; paper: Methods, "Photographed brains and
histology"), and the answer to "for MF5-MF7, 'b0 0.5 mm, T2': is the T2 also 0.5 mm?".
  MRI         for the 13 photographed brains with MRI (bf_cohort rows with an MRI tier): the voxel size (mm, NIfTI header) of the b0
              image used (products: curated_mri/<ID>/dwi_b0.nii.gz) and, where present, of the T2-weighted image (the multi-echo
              spin-echo series msme_echo_denoised_debiased.nii.gz and its raw echo msme_denoised_las_echo.nii.gz), next to bf_cohort's
              mri_voxel_mm, t2_available and tier. MF1-MF4 are not re-read here.
  Histology   brains with registered histology pieces (registration_qc_histo2bf_20261001, family histo2bf).
  Tracer      cases in tracer_template_current_all (summary.brains).
  LGN/STN     subjects in lgnstn_geometry_current.
The four records are read from the numbers of record (products: numbers_of_record).
-> $LINC_NUMBERS/table_specimens_check_r054.json
    python reproduce/numbers/table_specimens_check_r054.py"""
import json
import sys
from pathlib import Path

import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from linc import paths, provenance  # noqa: E402

RECORDS = ("bf_cohort", "registration_qc_histo2bf_20261001", "tracer_template_current_all", "lgnstn_geometry_current")
B0, T2, T2_RAW = "dwi_b0.nii.gz", "msme_echo_denoised_debiased.nii.gz", "msme_denoised_las_echo.nii.gz"   # per-brain MRI files

# the record's own method text, verbatim (it is part of the record's data)
METHOD = [
 "question \"for MF5-MF7, 'b0 0.5 mm, T2': is the T2 also 0.5 mm?\".",
 "  MRI         for the 13 photographed brains with MRI: the voxel size (mm, from the NIfTI header) of the b0 image used",
 "              (data/_lcn_src/data/curated/<S>/dwi_b0.nii.gz) and, where present, of the T2-weighted image (the multi-echo spin-echo",
 "              series, msme_echo_denoised_debiased.nii.gz, and its raw echo msme_denoised_las_echo.nii.gz); bf_cohort's mri_voxel_mm and",
 "              t2_available for comparison. MF1-MF4 (Gong et al. 2026 acquisitions) are not re-read here.",
 "  Histology   brains with registered histology pieces (registration_qc_histo2bf_20261001, family histo2bf, latest rows).",
 "  Tracer      cases in tracer_template_current_all (summary.brains).",
 "  LGN/STN     subjects in lgnstn_geometry_current.",
 "-> paper/numbers/table_specimens_check_r054.json",
 "    mri/env/mmorf/bin/python paper/scripts/table_specimens_check_r054.py"
]


def vox(p):
    """Voxel size (mm, 4 decimals) from the NIfTI header, or None when the file is absent."""
    return [round(float(z), 4) for z in nib.load(str(p)).header.get_zooms()[:3]] if p.exists() else None


def main():
    P = paths.products("products_mri")
    NUM, CUR = P["numbers_of_record"], P["curated_mri"]
    J = lambda n: json.loads((NUM / f"{n}.json").read_text())["data"]  # noqa: E731
    coh = J("bf_cohort")["rows"]
    mri = [r["subject"] for r in coh if r.get("mri_tier")]
    out = dict(method=METHOD, mri={}); files = []
    for r in coh:
        S = r["subject"]
        if S not in mri:
            continue
        b0, t2, t2raw = CUR / S / B0, CUR / S / T2, CUR / S / T2_RAW
        files += [p for p in (b0, t2, t2raw) if p.exists()]
        out["mri"][S] = dict(b0_voxel_mm=vox(b0), t2_voxel_mm=vox(t2), t2_raw_echo_voxel_mm=vox(t2raw),
                             cohort_mri_voxel_mm=r.get("mri_voxel_mm"), cohort_t2_available=r.get("t2_available"), cohort_tier=r.get("mri_tier"))
    qc = json.loads((NUM / "registration_qc_histo2bf_20261001.json").read_text())
    rows = qc.get("data", qc)
    rows = rows.get("rows", rows) if isinstance(rows, dict) else rows
    out["histology_subjects"] = sorted({x["subject"] for x in rows if isinstance(x, dict) and x.get("family") == "histo2bf"})
    out["tracer_cases"] = J("tracer_template_current_all")["summary"]["brains"]
    out["lgnstn_subjects"] = J("lgnstn_geometry_current")["subjects"]
    out["t2_brains"] = sorted(S for S, v in out["mri"].items() if v["t2_voxel_mm"])
    out["t2_voxel_mm_of_t2_brains"] = sorted({tuple(v["t2_voxel_mm"]) for v in out["mri"].values() if v["t2_voxel_mm"]})
    provenance.write_numbers("table_specimens_check_r054", out, inputs=[NUM / f"{n}.json" for n in RECORDS] + files, md5=False,
                             cmd="python reproduce/numbers/table_specimens_check_r054.py")


if __name__ == "__main__":
    main()
