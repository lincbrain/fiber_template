"""Tests of outputs/brains/, each brain's results in the data folder's layout (code/shared/results.py, linc-results; CPU, no data).

    python -m pytest tests/test_results_view.py      or      python tests/run.py tests/test_results_view.py

You need
  any Python 3.
What it does
  It checks that every name of settings/results.tsv is a name the data folder gives a brain's finished file (settings/files.tsv, under
  brains/sub-<ID>/outputs), and that its run files use only the working folders' variables; then, on a made-up working folder and data
  folder whose brain folder is a link to a second location, that a file the run made appears under its data-folder name and the
  paper's file (a link, through the links folder, into the data folder) does not, nor a file inside a linked folder, that a rebuilt
  volume wins over the paper's link, with the brain's inputs and the places the run wrote for it; that a second run changes nothing, that several at once end in the same
  view, that a link whose source is gone goes, that nothing of the user's and nothing in the data folder is touched; and that the
  automatic update at the end of a tool prints nothing and keeps its exit status.
You get
  pytest's pass / fail per test.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
from shared import results  # noqa: E402


def test_every_name_is_a_data_folder_name():
    rows = [r.split("\t") for r in (REPO / "settings" / "files.tsv").read_text().splitlines()]
    col = rows[0].index("new_path")
    given = set()
    for r in rows[1:]:
        m = re.match(r"brains/sub-([^/]+)/outputs/(.+)$", r[col])
        if m:
            given.add(m.group(2).replace(m.group(1), "<ID>"))
    given |= {f"<ID>_{s}_to_template.nii.gz" for s in ("b0", "blockface", "t2", "micro")}   # in the data folder, but made by its packaging
    names = [r["name"] for r in results.table()]                                               # (not in files.tsv): linc-results --transforms
    assert len(names) == len(set(names)), "a name twice"
    assert not [n for n in names if n not in given], [n for n in names if n not in given]
    for r in results.table():
        for c in r["run_file"]:
            assert set(re.findall(r"\$\{?([A-Z_]+)", c)) <= {"LINC_DATA", "LINC_REBUILD_DATA", "LINC_MRI_DATA", "LINC_WORK"}, c
            assert c.startswith("$") and "<ID>" in c, c


def toy(tmp_path):
    """a data folder whose brain folder is a link to a second location (as a tester's is), and a working folder whose blockface holds
    one file of the run's own and two links to the paper's files (through the links folder), a rebuild folder with a rebuilt volume,
    a labels folder that is a link to elsewhere, and the completion's file in the work folder"""
    second = tmp_path / "lab_data_folder" / "brains" / "sub-MR999"
    (second / "inputs").mkdir(parents=True)
    (second / "outputs").mkdir()
    for n in ("MR999_brain_mask.nii.gz", "MR999_blockface_L2.nii.gz"):
        (second / "outputs" / n).write_text("paper")
    tree = tmp_path / "data_folder"
    (tree / "brains").mkdir(parents=True)
    (tree / "brains" / "sub-MR999").symlink_to(second)
    out = tmp_path / "repo" / "outputs"
    bf = out / "local_data" / "data" / "sub-MR999" / "blockface"
    bf.mkdir(parents=True)
    (bf / "MR999_holes.nii.gz").write_text("mine")
    (out / "in").mkdir()
    for code_name, paper_name in (("MR999_mask.nii.gz", "MR999_brain_mask.nii.gz"), ("MR999_blockface_L2.nii.gz", "MR999_blockface_L2.nii.gz")):
        (out / "in" / code_name).symlink_to(tree / "brains" / "sub-MR999" / "outputs" / paper_name)    # the paper's, through the links folder
        (bf / code_name).symlink_to(out / "in" / code_name)
    rb = out / "local_rebuild" / "data" / "sub-MR999" / "blockface"
    rb.mkdir(parents=True)
    (rb / "MR999_blockface_L2.nii.gz").write_text("rebuilt")
    elsewhere = tmp_path / "elsewhere" / "tracer"                                   # a real file, but inside a linked folder
    elsewhere.mkdir(parents=True)
    (elsewhere / "MR999_Bundles_bfnative_L2_demons.nii.gz").write_text("not the run's")
    (out / "local_data" / "data" / "sub-MR999" / "labels").mkdir()
    (out / "local_data" / "data" / "sub-MR999" / "labels" / "blockface").symlink_to(elsewhere.parent)
    c = out / "work" / "complete" / "MR999"
    c.mkdir(parents=True)
    (c / "MR999_bf_completed.nii.gz").write_text("completed")
    return tree, out


def state(folder):
    return sorted((str(p.relative_to(folder)), os.readlink(str(p)) if p.is_symlink() else None, os.lstat(str(p)).st_ino)
                  for p in folder.rglob("*"))


def links(folder):
    return [x[:2] for x in state(folder)]


def links_of(st):
    return [x[:2] for x in st]


def test_the_view_shows_the_runs_own_files_under_the_data_folders_names(tmp_path):
    tree, out = toy(tmp_path)
    before = state(tmp_path / "lab_data_folder")
    env = {"LINC_WORK": str(out / "work")}
    done = results.update(outputs=out, tree=tree, env=env)
    v = out / "brains" / "sub-MR999"
    assert done == [("MR999", 3, 3, True)], done
    assert (v / "outputs" / "MR999_tubes.nii.gz").read_text() == "mine"
    assert os.readlink(str(v / "outputs" / "MR999_blockface_L2.nii.gz")) == str(out / "local_rebuild" / "data" / "sub-MR999" / "blockface" / "MR999_blockface_L2.nii.gz")
    assert (v / "outputs" / "MR999_completed.nii.gz").read_text() == "completed"
    assert not os.path.lexists(str(v / "outputs" / "MR999_brain_mask.nii.gz"))     # the paper's file is not the run's
    assert not os.path.lexists(str(v / "outputs" / "MR999_tracer_bundles_blockface.nii.gz"))   # nor one inside a linked folder
    assert os.readlink(str(v / "inputs")) == str(tree / "brains" / "sub-MR999" / "inputs")
    assert sorted(os.listdir(str(v / "intermediate"))) == ["local_data", "local_rebuild", "work_complete"]
    assert state(tmp_path / "lab_data_folder") == before                           # nothing written into the data folder

    mine = v / "notes.txt"                                                         # something of the user's in the view
    mine.write_text("mine")
    first = state(out / "brains")
    assert results.update(outputs=out, tree=tree, env=env) == done                 # a second run changes nothing
    assert state(out / "brains") == first

    import shutil
    shutil.rmtree(str(out / "brains"))                                             # made again by several at once: the same view
    procs = [subprocess.Popen([sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); from shared import results; "
                               "results.update(outputs=sys.argv[2], tree=sys.argv[3], env={'LINC_WORK': sys.argv[4]})",
                               str(REPO / "code"), str(out), str(tree), str(out / "work")]) for _ in range(4)]
    assert [p.wait() for p in procs] == [0] * 4
    assert links(out / "brains") == [x for x in links_of(first) if x[0] != "sub-MR999/notes.txt"]
    mine.write_text("mine")

    (out / "work" / "complete" / "MR999" / "MR999_bf_completed.nii.gz").unlink()   # a source gone: its link goes
    (out / "local_rebuild" / "data" / "sub-MR999" / "blockface" / "MR999_blockface_L2.nii.gz").unlink()   # and the paper's link is not
    assert results.update(outputs=out, tree=tree, env=env) == [("MR999", 1, 3, True)]                    # shown in its place
    assert not os.path.lexists(str(v / "outputs" / "MR999_completed.nii.gz"))
    assert not os.path.lexists(str(v / "outputs" / "MR999_blockface_L2.nii.gz"))
    assert mine.read_text() == "mine" and state(tmp_path / "lab_data_folder") == before


def test_linc_results_needs_the_data_folder(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "LINC_DATA_TREE"}
    r = subprocess.run([sys.executable, str(REPO / "code" / "shared" / "linc-results")], env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, universal_newlines=True, timeout=60)
    assert r.returncode != 0 and "source settings/roots.sh" in r.stderr and r.stderr.count("\n") == 1, r.stderr


def test_the_update_at_a_tools_end_is_quiet_and_keeps_its_exit_status(tmp_path):
    view = REPO / "outputs" / "brains"
    had = view.exists()
    code = ("import sys; sys.path.insert(0, sys.argv[1]); from shared import paths; sys.exit(3)")
    for extra in ({"LINC_DATA_TREE": str(tmp_path / "missing")},                  # a data folder that is not there
                  {"LINC_DATA_TREE": str(tmp_path), "LINC_DATA": str(tmp_path)},  # a working folder not of this clone: nothing done
                  {"LINC_DATA_TREE": str(tmp_path), "LINC_RESULTS_VIEW": "0"}):    # turned off
        env = dict(os.environ, PYTHONNOUSERSITE="1", **extra)
        r = subprocess.run([sys.executable, "-c", code, str(REPO / "code")], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, timeout=60)
        assert r.returncode == 3 and r.stdout == "" and r.stderr == "", (extra, r.stderr)
    assert view.exists() == had                                                    # the tests' temporary folders never reach the view

    tree, out = toy(tmp_path)                                                      # a clone of its own: the view is brought up to date
    repo = out.parent
    for rel in ("code/shared/__init__.py", "code/shared/paths.py", "code/shared/results.py", "settings/results.tsv"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text((REPO / rel).read_text())
    (out / "brains").mkdir()
    env = dict(os.environ, PYTHONNOUSERSITE="1", LINC_DATA_TREE=str(tree), LINC_DATA=str(out / "local_data" / "data"), LINC_WORK=str(out / "work"))
    env.pop("LINC_RESULTS_VIEW", None)
    r = subprocess.run([sys.executable, "-c", code, str(repo / "code")], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True, timeout=60)
    assert r.returncode == 3 and r.stdout == "" and r.stderr == "", r.stderr
    assert (out / "brains" / "sub-MR999" / "outputs" / "MR999_tubes.nii.gz").read_text() == "mine"
