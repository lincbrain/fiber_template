"""code/template/blockface_step9.py refine's transform entry, without data (CPU, seconds): a toy brain's own entry bf-native ->
bf-template (an FSL warp, then a linear part; the warp's source grid kept inside the entry, as in the data folder) and a field on
the template's grid. The refined entry must pull as the field, then the own entry; it must keep its source grid inside itself, so
that it does not depend on the work folder; and written into the brain's own xfm/, it must replace the own entry with a backup.
    python -m pytest tests/test_refine_entry.py      or      python tests/run.py tests/test_refine_entry.py"""
import json
import os
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
from shared import xfm  # noqa: E402
from template import blockface_step9 as S9  # noqa: E402

S = "TOY1"
VOX = 0.5


def _grid(shape, origin):
    A = np.diag([VOX, VOX, VOX, 1.0]); A[:3, 3] = origin
    return A


def _setup(tmp_path, inside=True):
    """A data folder with the toy brain's own entry, and a work folder; returns (own entry, template grid image, field file)."""
    data, work = tmp_path / "data", tmp_path / "work"
    os.environ["LINC_DATA"], os.environ["LINC_WORK"] = str(data), str(work)
    os.environ.pop("LINC_RECORDS", None); os.environ.pop("LINC_DATA_TREE", None)
    src = tmp_path / "in"; src.mkdir()
    shp = (12, 14, 10)
    At = _grid(shp, (-3.0, -3.5, -2.5))                                            # the template's grid
    tpl = src / "template.nii.gz"; nib.save(nib.Nifti1Image(np.zeros(shp, np.float32), At), str(tpl))
    sg = src / "src_grid.nii.gz"; nib.save(nib.Nifti1Image(np.zeros((16, 16, 12), np.float32), _grid((16, 16, 12), (-4.0, -4.0, -3.0))), str(sg))
    rng = np.random.default_rng(0)
    w = src / f"{S}_warp.nii.gz"; nib.save(nib.Nifti1Image(rng.normal(0, 0.3, shp + (3,)).astype(np.float32), At), str(w))
    T = np.eye(4); T[:3, :3] += rng.normal(0, 0.02, (3, 3)); T[:3, 3] = (0.4, -0.3, 0.2)
    aff = src / f"{S}_rigid_pull_ras.txt"; np.savetxt(str(aff), T)
    own = xfm.register("bf-native", "bf-template", [("warp", "fsl_rel", w), ("affine", "ras_pull", aff)], subject=S,
                       dst_grid=str(tpl), src_grid=str(sg), engine="toy", root=data / f"sub-{S}" / "xfm")
    if inside:                                                                     # the data folder's form: the grid inside the entry
        shutil.copy2(sg, own.dir / "src_grid.nii.gz")
        m = json.loads((own.dir / "meta.json").read_text()); m["src_grid"] = "src_grid.nii.gz"
        (own.dir / "meta.json").write_text(json.dumps(m, indent=1))
        sg.unlink()
    u = np.zeros(shp + (3,), np.float32); u[..., :] = (0.3, -0.2, 0.1)              # a constant field: sampled exactly inside the grid
    un = src / f"{S}_u.nii.gz"; nib.save(nib.Nifti1Image(u, At), str(un))
    return xfm.Entry(str(own.dir)), tpl, un, At, shp


def _points(At, shp, n=500):
    rng = np.random.default_rng(1)
    ijk = rng.uniform(2, np.array(shp) - 3, (n, 3))                                # inside, away from the edge
    return ijk @ At[:3, :3].T + At[:3, 3]


def test_refined_entry_is_the_field_then_the_own_entry(tmp_path):
    own, tpl, un, At, shp = _setup(tmp_path)
    P = _points(At, shp); ref = own.pull(P + np.array([0.3, -0.2, 0.1]))
    root = Path(os.environ["LINC_DATA"]) / f"sub-{S}" / "xfm_step9"
    made = S9.refined_entry(S, un, str(tpl), root, tmp_path / "work" / "refine")
    e = xfm.Entry(str(made.dir))
    assert [c["convention"] for c in e.meta["components"]] == ["ras_disp", "fsl_rel", "ras_pull"]
    assert np.abs(e.pull(P) - ref).max() < 1e-4
    assert e.meta["src_grid"] == "src_grid.nii.gz" and (e.dir / "src_grid.nii.gz").exists()
    shutil.rmtree(tmp_path / "work" / "refine")                                     # the work folder can go
    assert np.abs(xfm.Entry(str(made.dir)).pull(P) - ref).max() < 1e-4
    assert xfm.Entry(str(own.dir)).meta["engine"] == "toy"                         # the own entry is left as it was


def test_an_own_entry_with_an_outside_grid_keeps_its_path(tmp_path):
    own, tpl, un, At, shp = _setup(tmp_path, inside=False)
    made = S9.refined_entry(S, un, str(tpl), Path(os.environ["LINC_DATA"]) / f"sub-{S}" / "xfm_step9", tmp_path / "work" / "refine")
    assert xfm.Entry(str(made.dir)).meta["src_grid"] == own.meta["src_grid"]


def test_written_into_the_own_registry_it_replaces_the_own_entry(tmp_path):
    own, tpl, un, At, shp = _setup(tmp_path)
    P = _points(At, shp); ref = own.pull(P + np.array([0.3, -0.2, 0.1]))
    made = S9.refined_entry(S, un, str(tpl), Path(os.environ["LINC_DATA"]) / f"sub-{S}" / "xfm", tmp_path / "work" / "refine")
    assert made.dir == own.dir and np.abs(xfm.Entry(str(made.dir)).pull(P) - ref).max() < 1e-4
    backups = list((Path(os.environ["LINC_WORK"]) / "xfm_backup").rglob("meta.json"))
    assert len(backups) == 1 and json.loads(backups[0].read_text())["engine"] == "toy"


if __name__ == "__main__":
    import tempfile
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            with tempfile.TemporaryDirectory() as d:
                f(Path(d)); print("PASS", name)
