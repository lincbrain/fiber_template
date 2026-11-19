"""code/shared/xfm.py's pointwise inverse of a warp, without data (CPU, seconds). The fixed-point iteration x <- y - d(x) solves a
gentle warp but not one that stretches strongly: there it leaves residuals of millimeters. The inverse must then solve those points
by Newton steps, and must leave every point the iteration solved exactly as the iteration gave it.
    python -m pytest tests/test_xfm_inverse.py      or      python tests/run.py tests/test_xfm_inverse.py"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
from shared import xfm  # noqa: E402

VOX, N = 0.5, 48
C = np.array([12.0, 12.0, 12.0])                     # center of the bump, mm
SIGMA = 3.0


def _warp(tmp_path, a):
    """A displacement d(x) = a (x - c) exp(-|x - c|^2 / 2 sigma^2) on a 24 mm cube: x + d(x) stretches by 1 + a at the center.
    It is one-to-one for a < 2.2 (its radial derivative 1 + a exp(-s)(1 - 2 s) is smallest at s = 1.5), but the iteration
    x <- y - d(x) only converges for a < 1."""
    A = np.diag([VOX, VOX, VOX, 1.0])
    g = np.stack(np.meshgrid(*[np.arange(N) * VOX] * 3, indexing="ij"), -1) - C
    d = a * g * np.exp(-(g ** 2).sum(-1, keepdims=True) / (2 * SIGMA ** 2))
    f = tmp_path / f"warp_{a}.nii.gz"
    nib.save(nib.Nifti1Image(d.astype(np.float32), A), str(f))
    return xfm.Warp(f, "ras_disp")


def _points():
    rng = np.random.default_rng(0)
    return C + rng.uniform(-6, 6, size=(4000, 3))


def _fixed_point(w, Y, iters=30, tol=1e-3):
    """The iteration alone, as the inverse ran before the Newton steps were added."""
    x = Y.copy()
    for _ in range(iters):
        xn = Y - w.disp(x)
        step = np.abs(xn - x).max()
        x = xn
        if step < tol:
            break
    return x


def test_strong_warp_is_solved_by_newton_steps(tmp_path):
    w = _warp(tmp_path, 1.5)
    X = _points()
    Y = w.pull(X)                                     # so that X is the answer
    old = _fixed_point(w, Y)
    assert np.linalg.norm(w.pull(old) - Y, axis=1).max() > 0.5, "the toy warp should defeat the iteration alone"
    inv = w.inverse()
    got = inv.pull(Y.copy())
    assert inv.last_newton > 0
    assert inv.last_residual[1] < 1e-2, inv.last_residual
    assert np.linalg.norm(w.pull(got) - Y, axis=1).max() < 1e-2
    solved = np.linalg.norm(w.pull(old) - Y, axis=1) > 1e-2
    assert np.linalg.norm(got[solved] - X[solved], axis=1).max() < 1e-3      # the points Newton took over land on the answer


def test_gentle_warp_is_left_to_the_iteration(tmp_path):
    w = _warp(tmp_path, 0.3)
    Y = w.pull(_points())
    inv = w.inverse()
    got = inv.pull(Y.copy())
    assert inv.last_newton == 0
    assert np.array_equal(got, _fixed_point(w, Y))    # bit for bit what the iteration gives


def test_solved_points_keep_their_value(tmp_path):
    """In a warp that needs Newton steps somewhere, the points the iteration solved are not touched."""
    w = _warp(tmp_path, 1.5)
    Y = w.pull(_points())
    old = _fixed_point(w, Y)
    fine = np.linalg.norm(w.pull(old) - Y, axis=1) <= 1e-2
    got = w.inverse().pull(Y.copy())
    assert fine.any() and (~fine).any()
    assert np.array_equal(got[fine], old[fine])


def test_newton_can_be_switched_off(tmp_path):
    w = _warp(tmp_path, 1.5)
    Y = w.pull(_points())
    inv = xfm._InverseWarp(w, 30, 1e-3, newton_above=None)
    assert np.array_equal(inv.pull(Y.copy()), _fixed_point(w, Y)) and inv.last_newton == 0
