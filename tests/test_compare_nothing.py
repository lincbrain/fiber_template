"""A step's check says "nothing to compare with" when the data folder holds none of the paper's files (CPU, no data: toy folders).

    python -m pytest tests/test_compare_nothing.py      or      python tests/run.py tests/test_compare_nothing.py

You need
  settings/env/linc.yml.
What it does
  It runs the checks of tests/compare/ on toy folders in which the paper's side is absent (a data folder of inputs only), and checks
  that each exits 3 with a line "nothing to compare with" instead of reporting a difference; that a real difference still exits 1 and
  a match 0; and that run.py's status reads a check's exit 3 as passed, "ran to its end; nothing to compare with".
You get
  pytest's pass / fail per test.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
C = REPO / "tests" / "compare"


def check(*args):
    r = subprocess.run([sys.executable, *map(str, args)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    return r.returncode, r.stdout + r.stderr


def img(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.full((4, 4, 4), v, np.float32), np.eye(4)), str(p))


def test_nifti(tmp_path):
    mine, paper = tmp_path / "mine", tmp_path / "paper"
    img(mine / "a.nii.gz", 1)
    rc, out = check(C / "compare_nifti.py", mine, paper, "a.nii.gz")
    assert rc == 3 and "nothing to compare with" in out, out
    img(paper / "a.nii.gz", 1)
    assert check(C / "compare_nifti.py", mine, paper, "a.nii.gz")[0] == 0
    img(paper / "a.nii.gz", 2)
    rc, out = check(C / "compare_nifti.py", mine, paper, "a.nii.gz")
    assert rc == 1 and "differs" in out


def test_registration_reconstruction_histology_mri_build(tmp_path):
    mine, paper = tmp_path / "mine", tmp_path / "absent"
    mine.mkdir()
    img(mine / "a.nii.gz", 1)
    for args in ((C / "compare_registration.py", "entry", mine, paper),
                 (C / "compare_registration.py", "image", mine / "a.nii.gz", paper / "a.nii.gz"),
                 (C / "compare_reconstruction.py", "MR252", mine, paper),
                 (C / "compare_histology.py", mine, paper),
                 (C / "compare_mri_build.py", "run", mine, paper)):
        rc, out = check(*args)
        assert rc == 3 and "nothing to compare with" in out, (args[0].name, rc, out[-400:])


def test_run_reads_exit_3_as_passed(tmp_path):
    spec = importlib.util.spec_from_file_location("linc_run", str(REPO / "run.py"))
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    run.OUT = tmp_path
    rows = {}
    for name, ck in (("nothing", "3"), ("same", "0"), ("other", "1")):
        d = tmp_path / "steps" / name
        d.mkdir(parents=True)
        (d / "launched").write_text("host 1000 -1\n"); (d / "run.exit").write_text("0\n"); (d / "check.exit").write_text(ck + "\n")
        (d / "end").write_text("1100\n")
        (d / "check.log").write_text({"3": "MR252: nothing to compare with: the data folder holds no reconstruction of the paper's",
                                      "0": "12 of 12 items identical", "1": "3 of 12 items identical"}[ck] + "\n")
    steps = [dict(name=n, check="x", hours=0, needs=[]) for n in ("nothing", "same", "other")]
    rows = run.snapshot(steps)
    assert rows["nothing"]["state"] == "passed" and rows["nothing"]["evidence"].startswith("ran to its end; ") \
        and "nothing to compare with" in rows["nothing"]["evidence"], rows["nothing"]
    assert rows["same"]["state"] == "passed" and rows["other"]["state"] == "differs"
