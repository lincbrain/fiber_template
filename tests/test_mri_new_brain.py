"""code/mri/linc-mri prep's pieces and the MRI channels build driver, without data (CPU, seconds): staging one volume of a 4-D image under its
header, the mask checks, the modality table against settings/mri_construction.yaml, the next steps prep prints (both routes, each with
its tutorial step, and every option of their commands known to linc-register), and build_mri_channels --plan (every step listed,
nothing run, nothing written).
    python -m pytest tests/test_mri_new_brain.py      or      python tests/test_mri_new_brain.py"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
from mri import mri_new_brain as NB  # noqa: E402
from template.tools import recipe  # noqa: E402


def _img(path, shape, affine=None, dtype=np.float32, seed=0):
    rng = np.random.default_rng(seed)
    nib.save(nib.Nifti1Image(rng.random(shape).astype(dtype), np.eye(4) if affine is None else affine), str(path))
    return path


def test_stage_takes_one_volume_and_keeps_the_header():
    with tempfile.TemporaryDirectory() as d:
        A = np.diag([0.25, 0.25, 0.25, 1.0]); A[:3, 3] = (-3.0, 2.0, 1.5)
        src = _img(Path(d) / "series.nii.gz", (8, 9, 10, 5), A)
        rec = NB.stage(src, Path(d) / "out" / "b0.nii.gz", volume=3)
        out = nib.load(str(Path(d) / "out" / "b0.nii.gz"))
        assert out.shape == (8, 9, 10) and rec["shape"] == [8, 9, 10] and rec["volume"] == 3
        assert np.allclose(out.affine, A) and rec["voxel_mm"] == [0.25, 0.25, 0.25]
        assert np.array_equal(np.asanyarray(out.dataobj), np.asanyarray(nib.load(str(src)).dataobj)[..., 3])
        three = _img(Path(d) / "three.nii.gz", (4, 4, 4))
        assert NB.stage(three, Path(d) / "three_out.nii.gz")["volume"] is None
        try:
            NB.stage(_img(Path(d) / "two.nii.gz", (4, 4)), Path(d) / "two_out.nii.gz")
            raise AssertionError("a 2-D image must be refused")
        except ValueError:
            pass


def test_check_mask_grid_and_values():
    with tempfile.TemporaryDirectory() as d:
        img = _img(Path(d) / "img.nii.gz", (6, 6, 6))
        ok = Path(d) / "ok.nii.gz"; nib.save(nib.Nifti1Image((np.arange(216).reshape(6, 6, 6) % 2).astype(np.uint8), np.eye(4)), str(ok))
        NB.check_mask(ok, img)
        for name, arr, aff in (("grid", np.ones((6, 6, 5), np.uint8), np.eye(4)), ("affine", np.ones((6, 6, 6), np.uint8), np.diag([2.0, 1, 1, 1])),
                               ("values", np.full((6, 6, 6), 2, np.uint8), np.eye(4))):
            p = Path(d) / f"{name}.nii.gz"; nib.save(nib.Nifti1Image(arr, aff), str(p))
            try:
                NB.check_mask(p, img)
                raise AssertionError(f"{name}: must be refused")
            except ValueError:
                pass


def test_modality_table_matches_the_recipe_layout():
    L = recipe()["prep"]["layout"]
    for mod, M in NB.MODALITY.items():
        for key in ("staged", "mask", "n4", "out"):
            assert M[key] in L, f"{mod}.{key} = {M[key]} is not a prep.layout key"
        assert M["bias"] is None or M["bias"] in L
        assert M["brain"] in L or M["brain"].endswith(".nii.gz")
    for key in ("final_mask", "sarm_in_subject", "nmt_affine", "dwi_eddy", "dwi_ecclog", "dwi_bvecs", "dwi_bvals", "dwi_fit_mask"):
        assert key in L, f"the build driver's input {key} is missing from prep.layout"


def test_next_steps_name_both_routes_and_their_tutorials():
    import shlex
    from template.link import b0_to_template
    b0, mask = "/d/sub-X/mri/dwi/b0_brain_n4.nii.gz", "/d/sub-X/mri/masks/dwi_brain_mask.nii.gz"
    nxt = NB.next_steps("X", b0, mask)
    assert [r["route"] for r in nxt] == ["mri-template", "blockface"]
    assert nxt[0]["tutorial"] == "docs/tutorials/01_mri.md, Step B2" and nxt[1]["tutorial"] == "docs/tutorials/02_blockface.md, Step 3"
    for r in nxt:                                            # each tutorial step exists under that heading
        doc, step = r["tutorial"].split(", ")
        assert f"Step {step.split()[1]}." in (REPO / doc).read_text(), r["tutorial"]
    (to_template,), (to_b0, then) = nxt[0]["commands"], nxt[1]["commands"]
    mri_help = subprocess.run([sys.executable, str(REPO / "code" / "blockface" / "registration" / "linc-register"), "mri", "-h"],
                              capture_output=True, text=True, env=dict(os.environ, PYTHONNOUSERSITE="1")).stdout
    w = shlex.split(to_b0)
    assert w[1:3] == ["code/blockface/registration/linc-register", "mri"] and w[w.index("--place-b0") + 1:w.index("--place-b0") + 3] == [b0, mask]
    assert all(o in mri_help for o in w if o.startswith("--")), "an option of the mri command that linc-register mri does not know"
    for c, frame in ((to_template, True), (then, False)):    # parsed by mri-template's own parser: the b0, its mask, the frame or none
        w = shlex.split(c)
        assert w[1:3] == ["code/blockface/registration/linc-register", "mri-template"]
        a = b0_to_template.parser().parse_args(w[3:])
        assert (a.subject, a.b0, a.mask) == ("X", b0, mask) and a.template_b0.endswith("linc_mri_b0.nii.gz")
        assert (a.frame == "$LINC_TEMPLATE/frames/mri_template_0p5.nii.gz") if frame else a.frame is None
        assert "--place-b0" not in w                         # mri-template's frame goes by --frame
    text = NB.next_text(nxt)
    assert text.count("docs/tutorials/") == 2 and text.count("linc-register") == 3 and text.startswith("  next: ")
    t2 = NB.next_steps("X", "/d/t2.nii.gz", "/d/m.nii.gz", "t2")
    assert t2[0]["commands"] == [] and "b0" in NB.next_text(t2)


def test_build_mri_channels_plan_lists_every_step_and_runs_nothing():
    with tempfile.TemporaryDirectory() as d:
        env = dict(os.environ, LINC_DATA=f"{d}/data", LINC_WORK=f"{d}/work", LINC_PRODUCTS=d, LINC_TEMPLATE=d, PYTHONNOUSERSITE="1")
        env.pop("PY_MRI", None)
        out = subprocess.run([sys.executable, str(REPO / "code" / "template" / "build_mri_channels.py"), "--plan"],
                             capture_output=True, text=True, env=env, cwd=str(REPO))
        assert out.returncode == 0, out.stderr[-800:]
        ids = recipe()["inputs"]
        for S in ids:
            for st in ("tensor", "landmarks", "frames", "prepare"):
                assert f"{st}-{S}  [mri]" in out.stdout, f"{st}-{S} missing from the plan"
        for st in ("assemble  [mri]", "mmorf  [driver, GPU]", "channels  [mri]"):
            assert st in out.stdout
        assert out.stdout.count("\n $ ") == 0 and "$PY_MRI" in out.stdout            # the mri python is named, not resolved, when PY_MRI is unset
        assert not Path(f"{d}/work").exists(), "--plan must write nothing"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
