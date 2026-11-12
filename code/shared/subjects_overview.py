#!/usr/bin/env python3
"""One row per brain: the manual inputs in settings/brains/<ID>.yaml and, with --data, which hand-made files its folder holds.

    python code/shared/subjects_overview.py [--data $LINC_DATA] [--out file.csv]

You need PyYAML; with --data, a folder of brains, sub-<ID>/ (docs/tutorials/00_start_here.md).
What it does: reads every settings/brains/<ID>.yaml except TEMPLATE.yaml. It is a view: the yaml files stay the only input, nothing
in the pipeline reads this table, so a change goes in the yaml file.
You get CSV, on stdout or in --out. An optional setting a file leaves out shows its default. `jumps` and `missing` count the entries
of those lists (the caudal cutoff is one of the `missing` entries, and is also given by itself).

The in_folder_ columns report the brain's own folder under --data and nothing else (docs/tutorials/00_start_here.md gives the locations):
  in_folder_edited_mask, _edited_tubes, _regions, _section_map   yes or no
  in_folder_mri_masks     those present of dwi, t2, micro, final (mri/masks/<name>_brain_mask.nii.gz) and of_record (the b0 mask the
                          yaml file names under registration_to_own_mri), joined by +, or none
  in_folder_match_table   edited, table (match_table.txt), proposed or none; when there are several, the first of that order, the
                          one the pipeline uses (code/histology/match.py, find_table)
They are empty when --data has no folder for the brain. The paper's brains keep their match tables, regions and section maps under
$LINC_PRODUCTS/labels/, so `no` or `none` there does not mean the paper has none.
"""
import argparse
import csv
import sys
from pathlib import Path

import yaml

SUBJECTS = Path(__file__).resolve().parents[2] / "settings" / "brains"
COLUMNS = ["subject", "vx", "vz", "idx_first", "idx_last", "rotate180", "ccs", "cce", "native_end_crop_mm", "jumps", "missing",
           "caudal_cutoff"]
FOLDER_COLUMNS = ["in_folder_edited_mask", "in_folder_edited_tubes", "in_folder_mri_masks", "in_folder_regions",
                  "in_folder_section_map", "in_folder_match_table"]
MRI_MASKS = ("dwi", "t2", "micro", "final")
MATCH_TABLES = (("edited", "match_table_edited.txt"), ("table", "match_table.txt"), ("proposed", "match_table_proposed.txt"))


def settings(subject, y):
    first, last = y.get("idx_range") or ("", "")
    missing = y.get("missing") or []
    return dict(subject=subject, vx=y.get("vx", ""), vz=y.get("vz", ""), idx_first=first, idx_last=last,
                rotate180=str(y.get("rotate180", False)).lower(), ccs=y.get("ccs", 0), cce=y.get("cce", 0),
                native_end_crop_mm=y.get("native_end_crop_mm", 0.0), jumps=len(y.get("jumps") or []), missing=len(missing),
                caudal_cutoff=min((before for before, after in missing if after < 0), default=""))


def in_folder(folder, subject, y):
    if not folder.is_dir():
        return dict.fromkeys(FOLDER_COLUMNS, "")
    yes_no = lambda p: "yes" if (folder / p).exists() else "no"  # noqa: E731
    mri = [m for m in MRI_MASKS if (folder / "mri" / "masks" / f"{m}_brain_mask.nii.gz").exists()]
    of_record = (y.get("registration_to_own_mri") or {}).get("b0_mask")
    if of_record and (folder / "mri" / of_record).exists():
        mri.append("of_record")
    table = next((name for name, f in MATCH_TABLES if (folder / "histology" / f).exists()), "none")
    return dict(in_folder_edited_mask=yes_no(f"blockface/curated/{subject}_mask_edited.nii.gz"),
                in_folder_edited_tubes=yes_no(f"blockface/curated/{subject}_holes_edited.nii.gz"),
                in_folder_mri_masks="+".join(mri) or "none", in_folder_regions=yes_no("histology/regions.nii.gz"),
                in_folder_section_map=yes_no("histology/section_map.tsv"), in_folder_match_table=table)


def main():
    ap = argparse.ArgumentParser(description="One row per brain: its manual inputs, as CSV (see the top of this file).")
    ap.add_argument("--data", type=Path, help="a folder of brains, sub-<ID>/, to report which hand-made files each one holds")
    ap.add_argument("--out", type=Path, help="write the CSV here (default: stdout)")
    a = ap.parse_args()
    rows = []
    for path in sorted(SUBJECTS.glob("*.yaml")):
        if path.stem == "TEMPLATE":
            continue
        y = yaml.safe_load(path.read_text())
        row = settings(path.stem, y)
        if a.data:
            row.update(in_folder(a.data / f"sub-{path.stem}", path.stem, y))
        rows.append(row)
    out = open(a.out, "w", newline="") if a.out else sys.stdout
    w = csv.DictWriter(out, COLUMNS + FOLDER_COLUMNS if a.data else COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    if a.out:
        out.close()
    if a.data:
        print("The in_folder_ columns report each brain's own folder only: the paper's brains keep their match tables, regions and "
              "section maps under $LINC_PRODUCTS/labels/.", file=sys.stderr)


if __name__ == "__main__":
    main()
