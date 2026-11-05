"""tools/subjects_overview.py: one row per brain from the yaml files, and what each brain's folder holds (CPU, no data, seconds).

    python -m pytest tests/test_subjects_overview.py      or      python tests/run.py tests/test_subjects_overview.py

You need
  PyYAML.
What it does
  It runs the tool on the repository's configs/subjects/ and on a folder of empty files made on the fly.
You get
  pytest's pass / fail per test.
"""
import csv
import io
import subprocess
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "subjects_overview.py"


def overview(*args):
    r = subprocess.run([sys.executable, str(TOOL), *map(str, args)], capture_output=True, text=True, check=True)
    return {row["subject"]: row for row in csv.DictReader(io.StringIO(r.stdout))}


def test_one_row_per_brain():
    rows = overview()
    assert len(rows) == 21 and "TEMPLATE" not in rows
    r = rows["MR252"]
    assert (r["vx"], r["vz"], r["idx_first"], r["idx_last"], r["rotate180"]) == ("20.0", "45.0", "8527", "9966", "false")
    assert (r["jumps"], r["missing"], r["caudal_cutoff"], r["native_end_crop_mm"]) == ("13", "22", "1436", "0.0")
    assert rows["MF204"]["native_end_crop_mm"] == "1.62" and rows["MF191"]["caudal_cutoff"] == ""
    assert not any(k.startswith("in_folder_") for k in r)


def test_what_the_folder_holds(tmp_path):
    def files(*paths):
        for p in paths:
            f = tmp_path / "sub-MR252" / p
            f.parent.mkdir(parents=True, exist_ok=True)
            f.touch()
        return overview("--data", tmp_path)

    rows = files("blockface/curated/MR252_mask_edited.nii.gz", "mri/masks/t2_brain_mask.nii.gz", "mri/masks/dwi_brain_mask.nii.gz")
    r = rows["MR252"]
    assert (r["in_folder_edited_mask"], r["in_folder_edited_tubes"]) == ("yes", "no")
    assert (r["in_folder_mri_masks"], r["in_folder_regions"], r["in_folder_match_table"]) == ("dwi+t2", "no", "none")
    assert rows["MF191"]["in_folder_match_table"] == "" and rows["MF191"]["in_folder_edited_mask"] == ""      # no folder: not looked at
    assert files("histology/match_table.txt")["MR252"]["in_folder_match_table"] == "table"
    assert files("histology/match_table_proposed.txt")["MR252"]["in_folder_match_table"] == "proposed"
    assert files("histology/match_table_edited.txt")["MR252"]["in_folder_match_table"] == "edited"
    r = files("mri/MR252_mri_mask_r1pair_reg_2026-10-03.nii.gz", "histology/section_map.tsv")["MR252"]
    assert r["in_folder_mri_masks"] == "dwi+t2+of_record" and r["in_folder_section_map"] == "yes"


def test_writes_the_file_it_is_given(tmp_path):
    subprocess.run([sys.executable, str(TOOL), "--out", str(tmp_path / "o.csv")], check=True, capture_output=True)
    assert (tmp_path / "o.csv").read_text().startswith("subject,vx,vz,")
