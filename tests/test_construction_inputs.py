"""Tests of where a construction run takes its inputs from (CPU, no data: toy folders).

    python -m pytest tests/test_construction_inputs.py      or      python tests/run.py tests/test_construction_inputs.py

You need
  settings/env/linc.yml.
What it does
  It checks that code/template/construction_inputs.py points <work>/prep/<ID> at the paper's inputs when the data folder holds them,
  at the build's own prepare outputs when it does not, leaves alone a link that already leads to construction inputs, replaces a link
  that leads to a folder without them, and stops with one message that names both places when neither holds them; that the two
  construction steps of steps.yaml go through it; and that the comparison says so when the data folder has no run to compare with.
You get
  pytest's pass / fail per test.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code" / "template"))
import construction_inputs as CI  # noqa: E402


def folder(root, name, full=True):
    d = Path(root) / name
    d.mkdir(parents=True)
    for n in (CI.NEEDS if full else CI.NEEDS[:1]):
        (d / n).write_text("x")
    return d


def run(root, ids=("MF283",)):
    said = []
    used = CI.point(str(Path(root) / "work"), str(Path(root) / "paper" / "sub-<ID>"), str(Path(root) / "own" / "<ID>"), ids, say=said.append)
    return used, said


def test_the_papers_inputs_when_the_data_folder_holds_them():
    with tempfile.TemporaryDirectory() as t:
        p = folder(t, "paper/sub-MF283"); folder(t, "own/MF283")
        used, said = run(t)
        assert used["MF283"] == p and (Path(t) / "work/prep/MF283").resolve() == p.resolve() and "the paper's inputs" in said[0]


def test_own_prepare_outputs_when_the_data_folder_does_not():
    with tempfile.TemporaryDirectory() as t:
        folder(t, "paper/sub-MF283", full=False)                      # the folder exists, as in an inputs-only data folder, without the files
        o = folder(t, "own/MF283")
        used, said = run(t)
        assert used["MF283"] == o and (Path(t) / "work/prep/MF283").resolve() == o.resolve() and "own prepare outputs" in said[0]


def test_a_link_that_leads_to_inputs_is_left_alone():
    with tempfile.TemporaryDirectory() as t:
        folder(t, "paper/sub-MF283"); mine = folder(t, "mine")
        (Path(t) / "work/prep").mkdir(parents=True); (Path(t) / "work/prep/MF283").symlink_to(mine)
        used, said = run(t)
        assert (Path(t) / "work/prep/MF283").resolve() == mine.resolve() and "kept as it is" in said[0]


def test_a_link_that_leads_nowhere_useful_is_replaced():
    with tempfile.TemporaryDirectory() as t:
        empty = folder(t, "paper/sub-MF283", full=False); o = folder(t, "own/MF283")
        (Path(t) / "work/prep").mkdir(parents=True); (Path(t) / "work/prep/MF283").symlink_to(empty)   # what an earlier run of the step left
        run(t)
        assert (Path(t) / "work/prep/MF283").resolve() == o.resolve()


def test_stops_with_both_places_named_when_neither_holds_them():
    with tempfile.TemporaryDirectory() as t:
        folder(t, "paper/sub-MF283"); folder(t, "own/MF283")
        try:
            run(t, ids=("MF283", "MF284"))
        except SystemExit as e:
            msg = str(e)
        else:
            raise AssertionError("did not stop")
        assert "MF284" in msg and "paper/sub-MF284" in msg and "own/MF284" in msg and "prepare-MF284" in msg
        assert not (Path(t) / "work/prep/MF284").exists()


def test_the_construction_steps_go_through_it():
    text = (REPO / "steps.yaml").read_text()
    for name in ("mri-assemble", "mri-construction-full"):
        block = text.split(f"  - name: {name}\n")[1].split("\n  - name: ")[0]
        assert "code/template/construction_inputs.py" in block and "ln -sfn" not in block, name


def test_the_comparison_says_when_there_is_no_run_to_compare_with():
    with tempfile.TemporaryDirectory() as t:
        (Path(t) / "mine").mkdir()
        r = subprocess.run([sys.executable, str(REPO / "tests/compare/compare_mri_build.py"), "run", str(Path(t) / "mine"), str(Path(t) / "absent"),
                            "--parts", "config", "--min", "1"], capture_output=True, text=True)
        assert r.returncode == 3 and "nothing to compare with" in (r.stdout + r.stderr)


def _runner():
    import importlib.util
    spec = importlib.util.spec_from_file_location("linc_run_plan", REPO / "run.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_a_step_run_again_is_told_to_replace_what_it_left():
    import types
    R = _runner()
    with tempfile.TemporaryDirectory() as t:
        R.OUT = Path(t)
        R.sh = lambda host, command, timeout=40: types.SimpleNamespace(returncode=0, stderr="")
        s = dict(name="toy", title="a toy step", run="echo build --step toy", check="", again="--force")
        R.launch(s, "here", -1)
        first = (Path(t) / "steps/toy/step.sh").read_text()
        R.launch(s, "here", -1)
        second = (Path(t) / "steps/toy/step.sh").read_text()
        assert "echo build --step toy )" in first and "--force" not in first
        assert "echo build --step toy --force )" in second and (Path(t) / "steps/toy/previous/step.sh").read_text() == first
        s2 = dict(s, name="plain", again="")
        R.launch(s2, "here", -1); R.launch(s2, "here", -1)
        assert "--force" not in (Path(t) / "steps/plain/step.sh").read_text()


def test_the_mri_build_steps_replace_on_a_rerun():
    R = _runner()
    steps, _ = R.load_plan()
    by = {s["name"]: s for s in steps}
    for base in ("mri-stage", "mri-tensor", "mri-nmt", "mri-frames", "mri-prepare"):
        assert by[f"{base}-MF283"]["again"] == "--force", base
    assert by["mri-assemble"]["run"].rstrip().endswith("--step assemble --force")            # seconds, its own run folder: always replaced
    assert "--force" not in by["mri-construction-full"]["run"]                                 # days on a card: never replaced on its own
