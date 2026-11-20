"""The grid of the blockface template in the MRI template's frame (code/template/blockface_step9.py mri_grid; CPU, no data).

    python -m pytest tests/test_in_mri_grid.py      or      python tests/run.py tests/test_in_mri_grid.py

You need
  settings/env/linc.yml.
What it does
  It checks that in-mri without --res keeps the MRI template's own grid (shape and affine unchanged, so its five files stay byte for
  byte), and that --res 0.15 gives the 0.15 mm grid over the same outer box that the published 0.15 mm files are on: 427 x 534 x 320,
  origin (-31.95, -30.10, -7.55) mm, symmetric about x = 0.
You get
  pytest's pass / fail per test.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from template.blockface_step9 import mri_grid  # noqa: E402

B0 = np.array([[0.25, 0, 0, -31.875], [0, 0.25, 0, -30.0], [0, 0, 0.25, -7.5], [0, 0, 0, 1.0]])   # the MRI template's grid


def test_default_is_the_mri_templates_grid():
    shp, A = mri_grid((256, 320, 192), B0)
    assert shp == (256, 320, 192) and A is B0


def test_015_grid_over_the_same_box():
    shp, A = mri_grid((256, 320, 192), B0, 0.15)
    assert shp == (427, 534, 320)
    assert np.allclose(A, [[0.15, 0, 0, -31.95], [0, 0.15, 0, -30.1], [0, 0, 0.15, -7.55], [0, 0, 0, 1]])
    lo = A[:3, 3] - 0.075; hi = A[:3, 3] + (np.array(shp) - 1) * 0.15 + 0.075
    assert np.all(lo <= [-32, -30.125, -7.625]) and np.all(hi >= [32, 49.875, 40.375])            # covers the 0.25 mm box
    assert np.all(hi - lo - [64, 80, 48] < 0.15 + 1e-9)                                            # by less than one voxel
    assert abs(A[0, 3] + (shp[0] - 1) * 0.15 / 2) < 1e-9                                           # symmetric about x = 0
