"""The tools never write into the data folder (CPU, no data: empty toy files).

    python -m pytest tests/test_data_folder.py      or      python tests/run.py tests/test_data_folder.py

You need
  settings/env/linc.yml.
What it does
  LINC_DATA is always a working folder (settings/roots.sh makes it under outputs/; the data folder is in LINC_DATA_TREE). It checks
  that code/shared/paths.py data_root() stops, with one line, when LINC_DATA is a data folder (a folder that holds a file
  settings/files.tsv lists, the test code/shared/link_data.py makes), and lets a working folder or an empty temporary folder through;
  and that a new brain's MRI preparation run with LINC_DATA at a data folder stops before it writes anything there, and the blockface
  sanity check refuses a data folder given as --data.
You get
  pytest's pass / fail per test.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
from shared import paths  # noqa: E402


def data_folder(tmp_path):
    """A toy data folder: the first file settings/files.tsv lists, empty."""
    with open(REPO / "settings" / "files.tsv") as f:
        head = f.readline().rstrip("\n").split("\t"); row = f.readline().rstrip("\n").split("\t")
    p = tmp_path / "data_folder" / row[head.index("new_path")]
    p.parent.mkdir(parents=True); p.write_bytes(b"")
    return tmp_path / "data_folder"


def listing(folder):
    return sorted(str(p.relative_to(folder)) for p in folder.rglob("*"))


def with_linc_data(value, fn):
    old = os.environ.get("LINC_DATA"); os.environ["LINC_DATA"] = str(value)
    try:
        return fn()
    finally:
        if old is None:
            os.environ.pop("LINC_DATA", None)
        else:
            os.environ["LINC_DATA"] = old


def test_data_root_stops_at_the_data_folder(tmp_path):
    d = data_folder(tmp_path)
    assert paths.is_data_folder(d)
    try:
        with_linc_data(d, paths.data_root)
        raise AssertionError("data_root() must stop when LINC_DATA is the data folder")
    except SystemExit as e:
        msg = str(e)
        assert "\n" not in msg and "LINC_DATA is the data folder" in msg and str(d) in msg
        assert "source settings/roots.sh" in msg and "LINC_DATA_TREE" in msg
    try:                                                                           # and so does every path built on it
        with_linc_data(d, lambda: paths.subject_dir("MF278"))
        raise AssertionError("subject_dir() must stop too")
    except SystemExit:
        pass


def test_working_and_temporary_folders_pass(tmp_path):
    w = tmp_path / "outputs" / "local_data" / "data"; (w / "sub-MR000").mkdir(parents=True)
    assert not paths.is_data_folder(w)
    assert with_linc_data(w, paths.data_root) == w
    assert with_linc_data(w, lambda: paths.subject_dir("MR000")) == w / "sub-MR000"
    e = tmp_path / "empty"; e.mkdir()                                              # the tests' temporary folders
    assert with_linc_data(e, paths.data_root) == e
    assert with_linc_data(tmp_path / "not_yet", paths.data_root) == tmp_path / "not_yet"   # a folder made later


def test_mri_prep_with_linc_data_at_the_data_folder_writes_nothing(tmp_path):
    d = data_folder(tmp_path); before = listing(d)
    img = tmp_path / "MF278_b0.nii.gz"; img.write_bytes(b"")
    env = dict(os.environ, LINC_DATA=str(d), LINC_WORK=str(tmp_path / "work"), LINC_TEMPLATE=str(d / "template"), PYTHONNOUSERSITE="1")
    r = subprocess.run([sys.executable, str(REPO / "code" / "mri" / "linc-mri"), "prep", "--subject", "MF278", "--image", str(img)],
                       capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode != 0, r.stdout + r.stderr
    assert "LINC_DATA is the data folder" in r.stdout + r.stderr, r.stdout + r.stderr
    assert listing(d) == before, "the data folder changed"


def test_sanity_refuses_the_data_folder(tmp_path):
    d = data_folder(tmp_path)
    r = subprocess.run([sys.executable, str(REPO / "code" / "blockface" / "sanity.py"), "MR000", "--data", str(d)], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONNOUSERSITE="1"), timeout=300)
    assert r.returncode != 0 and "is the data folder" in r.stderr, r.stderr
