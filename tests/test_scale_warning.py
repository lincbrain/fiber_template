"""The registration to the brain's own b0 warns when its 9-dof stage stretches or compresses the stack beyond the QC gate (CPU, no data).

    python -m pytest tests/test_scale_warning.py      or      python tests/run.py tests/test_scale_warning.py

You need
  settings/env/linc.yml.
What it does
  It checks code/blockface/registration/to_own_mri.py scale_warning: no warning for scales inside settings/registration.yaml qc's
  [scale_lo, scale_hi]; one WARNING line, naming the scales and the range, for a stack stretched along one axis (as one cut short of the
  brain came out on 19 Nov 2026: 1.117) and for one compressed (0.853, as one paper brain's registration of record, which stays usable:
  a warning, never a stop); and that the linear stage returns it.
You get
  pytest's pass / fail per test.
"""
import inspect

import numpy as np
import yaml

from blockface.registration import to_own_mri
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def rot(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def T(scales, deg=20.0):
    M = np.eye(4); M[:3, :3] = rot(deg) @ np.diag(scales); M[:3, 3] = [3.0, -2.0, 5.0]
    return M


def test_inside_the_gate_no_warning():
    assert to_own_mri.scale_warning("X", T([1.02, 0.97, 1.05])) is None


def test_outside_the_gate_one_warning_line():
    g = yaml.safe_load((REPO / "settings" / "registration.yaml").read_text())["qc"]
    for sc, bad in (([1.04, 1.117, 1.04], "1.117"), ([0.853, 0.915, 0.969], "0.853")):
        w = to_own_mri.scale_warning("X", T(sc))
        assert w and "WARNING" in w and bad in w and f"[{g['scale_lo']}, {g['scale_hi']}]" in w and "linc-register qc" in w, w


def test_the_linear_stage_returns_it():
    assert "scale_warning=warning" in inspect.getsource(to_own_mri.linear)
