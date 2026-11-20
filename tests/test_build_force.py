"""Tests of --force in the template builds: a step whose own command refuses existing files is told to replace them (CPU, no data:
toy steps and toy folders).

    python -m pytest tests/test_build_force.py      or      python tests/run.py tests/test_build_force.py

You need
  settings/env/linc.yml.
What it does
  It checks that the driver refuses a step whose outputs exist and that --force adds the step's force_argv to its command; that the
  MRI build gives one to each step whose command refuses existing files; that the staging, told to overwrite, removes what an earlier
  run left (a run that stopped partway included) and nothing else: not the file a link leads to, not the masks, not the landmarks;
  that prepare removes its output folder only when told to; and that a registration's existing entry is refused before any work
  and, with overwrite, replaced after a backup.
You get
  pytest's pass / fail per test.
"""
import contextlib
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO / "code" / "template"))


@contextlib.contextmanager
def environ(**kw):
    """set these variables for the block, then restore the environment"""
    old = {k: os.environ.get(k) for k in kw}
    os.environ.update({k: str(v) for k, v in kw.items()})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def stopped(fn, *a, **kw):
    """the message a call stops with (SystemExit), or None when it runs through"""
    try:
        fn(*a, **kw)
    except SystemExit as e:
        return str(e)
    return None


def test_force_reaches_the_steps_command(tmp_path):
    import _driver as D
    out, script = tmp_path / "out.txt", tmp_path / "step.py"
    script.write_text("import sys\nopen(sys.argv[1], 'w').write(' '.join(sys.argv[2:]))\n")
    step = D.Step("toy", "toy", "linc", [script, out, "made"], outputs=[out], force_argv=["--overwrite"])
    work = tmp_path / "work"
    D.run_step(1, step, work)
    assert out.read_text() == "made"                               # nothing existed: the command as it is
    msg = stopped(D.run_step, 1, step, work)
    assert msg and "outputs exist" in msg and "--force" in msg     # refused, and the message names the way out
    D.run_step(1, step, work, force=True)
    assert out.read_text() == "made --overwrite"                   # --force: the command is told to replace its files
    script.write_text("import sys\nsys.exit('the reason')\n")
    msg = stopped(D.run_step, 1, step, work, force=True)
    assert msg and "stopped" in msg and "toy" in msg               # a command that stops ends the step with one plain line
    out.unlink()
    msg = stopped(D.run_step, 1, step, work)                       # stopped without --force: the line says what --force would pass
    assert msg and "--force here passes it --overwrite" in msg


def test_the_mri_builds_refusing_steps_take_force(tmp_path):
    with environ(LINC_DATA=tmp_path / "data", LINC_WORK=tmp_path / "work"):
        import build_mri_channels as B
        from shared import paths
        steps = B.make_steps(tmp_path / "work" / "build_mri_channels", paths)
    told = {s.name.split("-")[0] for s in steps if s.force_argv == ["--overwrite"]}
    assert told == {"stage", "t2reg", "microreg", "prepare", "links"}, told
    stage = next(s for s in steps if s.name.startswith("stage-"))
    assert stage.command()[-1] != "--overwrite" and stage.command(force=True)[-1] == "--overwrite"
    assert "--overwrite" not in stage.command(plan=True)           # the plan shows the command as it runs without --force


def test_staging_again_removes_what_an_earlier_run_left_and_nothing_else(tmp_path):
    from mri import mri_prep as P
    root = tmp_path / "sub-X" / "mri"
    b = P.Brain("X", mri_root=root, xfm_root=tmp_path / "sub-X" / "xfm")
    for k in ("dwi_mask", "t2_mask", "micro_mask"):                # the inputs
        b(k).parent.mkdir(parents=True, exist_ok=True)
        b(k).write_text("mask")
    (root / "landmarks").mkdir()
    (root / "landmarks" / "canonical.json").write_text("{}")       # a later step's file
    elsewhere = tmp_path / "elsewhere.nii.gz"
    elsewhere.write_text("someone's file")
    b("b0").parent.mkdir(parents=True)                             # what a run that stopped partway leaves
    b("b0").write_text("left behind")
    os.symlink(str(elsewhere), str(root / "dwi" / "b0_brain.nii.gz"))
    (root / "t2").mkdir()
    (root / "t2" / "msme.nii.gz").write_text("left behind")
    pre = tmp_path / "preprocessed"
    (pre / "dwi").mkdir(parents=True)
    for name in ("dwi_eddycorrect.nii.gz", "dwi_eddycorrect.ecclog", "dwi_orig.bvecs", "dwi_orig.bvals", "brain_mask.nii.gz"):
        (pre / "dwi" / name).write_text(name)

    found = {}

    def raw(bb, *a):                                               # stands in for the FSL and ANTs work: notes what it finds, writes the b0
        found["b0"] = bb("b0").exists()
        found["t2"] = (bb.mri_root / "t2").exists()
        bb("b0").parent.mkdir(parents=True, exist_ok=True)
        bb("b0").write_text("staged")

    real = {n: getattr(P, n) for n in ("stage_raw", "stage_dwi", "stage_t2", "stage_micro")}
    P.stage_raw, P.stage_dwi, P.stage_t2, P.stage_micro = raw, (lambda bb: None), (lambda bb: None), (lambda bb, *a: None)
    try:
        msg = stopped(P.stage, b, pre)
        assert msg and "--overwrite" in msg and b("b0").read_text() == "left behind"       # refused, nothing touched
        P.stage(b, pre, overwrite=True)
    finally:
        for n, f in real.items():
            setattr(P, n, f)
    assert found == dict(b0=False, t2=False)                       # the earlier run's files were gone before it wrote
    assert b("b0").read_text() == "staged" and b("dwi_eddy").read_text() == "dwi_eddycorrect.nii.gz"
    assert not os.path.lexists(str(root / "dwi" / "b0_brain.nii.gz")) and elsewhere.read_text() == "someone's file"   # the link, not its file
    assert b("dwi_mask").read_text() == "mask" and (root / "landmarks" / "canonical.json").exists()

    os.symlink(str(pre), str(tmp_path / "linked"))                 # a folder that is a link is not the step's to remove
    assert stopped(P.remove_own, tmp_path / "linked") and (pre / "dwi" / "brain_mask.nii.gz").exists()


def test_prepare_removes_its_folder_only_when_told_to(tmp_path):
    from mri import mri_prep as P
    b = P.Brain("X", mri_root=tmp_path / "mri", xfm_root=tmp_path / "xfm")
    out = tmp_path / "prep" / "X"
    out.mkdir(parents=True)
    (out / "t2.nii.gz").write_text("an earlier run's")
    msg = stopped(P.prepare, b, tmp_path / "mask.nii.gz", tmp_path / "tensor.nii.gz", out)
    assert msg and "--overwrite" in msg and (out / "t2.nii.gz").exists()
    try:                                                           # told to: the folder goes first; this toy brain has no images, so it stops there
        P.prepare(b, tmp_path / "mask.nii.gz", tmp_path / "tensor.nii.gz", out, overwrite=True)
    except Exception:                                              # noqa: BLE001
        pass
    assert not (out / "t2.nii.gz").exists()


def test_a_registrations_entry_is_refused_first_and_replaced_after_a_backup(tmp_path):
    import nibabel as nib
    import numpy as np
    with environ(LINC_WORK=tmp_path / "work", LINC_DATA=tmp_path / "data"):
        from mri import mri_register as R
        from mri.mri_prep import Brain
        b = Brain("X", mri_root=tmp_path / "mri", xfm_root=tmp_path / "xfm")
        b("b0").parent.mkdir(parents=True)
        nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), np.eye(4)), str(b("b0")))
        warp = tmp_path / "warp.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((4, 4, 4, 1, 3), np.float32), np.eye(4)), str(warp))
        R._free(b, "t2-native", "b0-native", False)                # no entry yet: nothing to refuse
        first = R._entry(b, "t2-native", warp, "toy", 1.25)
        msg = stopped(R._free, b, "t2-native", "b0-native", False)
        assert msg and "--overwrite" in msg                        # refused before any registration runs
        R._free(b, "t2-native", "b0-native", True)
        again = R._entry(b, "t2-native", warp, "toy", 1.25, overwrite=True)
        assert Path(again.dir) == Path(first.dir) and (Path(again.dir) / "meta.json").exists()
        kept = list((tmp_path / "work" / "xfm_backup").rglob("meta.json"))
        assert len(kept) == 1                                      # the old entry is kept aside
        (Path(again.dir) / "meta.json").unlink()                   # what a write that stopped partway leaves: a folder, no meta.json
        msg = stopped(R._free, b, "t2-native", "b0-native", False)
        assert msg and "not finished" in msg and "--overwrite" in msg
        R._free(b, "t2-native", "b0-native", True)                 # removed, so the next entry can be written
        assert not Path(again.dir).exists()
        assert (Path(R._entry(b, "t2-native", warp, "toy", 1.25, overwrite=True).dir) / "meta.json").exists()


def test_plan_gives_the_rerun_words_to_the_steps_that_refuse():
    """steps.yaml: a step of the MRI build whose own command refuses what an earlier run left has `again`, the words run.py adds
    when the step is run again (--force, --redo, or after a failure); without them the step stops on its own outputs. The words
    are added at the end of the step's command, so the command must end with the tool that takes them."""
    import yaml
    steps = {s["name"]: s for s in yaml.safe_load((REPO / "steps.yaml").read_text())["steps"]}
    for name in ("mri-stage", "mri-tensor", "mri-nmt", "mri-frames", "mri-prepare", "mri-channels"):
        assert steps[name].get("again") == "--force", name
        assert "build_mri_channels.py" in steps[name]["run"].split("&&")[-1], name
    assert steps["mri-links"].get("again") == "--overwrite"
    assert "mri.mri_register links" in steps["mri-links"]["run"].split("&&")[-1]
