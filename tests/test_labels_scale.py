"""The section scale of the label interpolation (code/labels/interpolate.py section_scales): each section is divided by the 99.5th
percentile of its tissue; a scale more than settings/labels.yaml interpolate.scale_max_over_median times the median of the brain's
sections takes that median. A section over the limit takes the median; a section under it, and every section of a brain with no
section over it, keeps exactly the scale it had before the rule.
    python -m pytest tests/test_labels_scale.py      or      python tests/run.py tests/test_labels_scale.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))


def _sections(levels, seed=0):
    """toy sections on a 40 x 30 grid: tissue in a disc with RGB values around `level`, background 0"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:40, :30]
    disc = (yy - 20) ** 2 + (xx - 15) ** 2 < 12 ** 2
    sec = {}
    for i, lv in zip(range(0, 2 * len(levels), 2), levels):
        a = np.zeros((40, 30, 3), np.float32)
        a[disc] = (lv * (0.5 + 0.5 * rng.random((int(disc.sum()), 3)))).astype(np.float32)
        sec[i] = a
    sec[2 * len(levels)] = np.zeros((40, 30, 3), np.float32)           # a slice with no tissue: scale 1.0
    return sec, sorted(sec)


def _before(sec, idxs):
    """the scale as computed before the rule (interpolate.py, until 17 Nov 2026)"""
    return {i: max(np.percentile(sec[i][sec[i].max(-1) > 0], 99.5), 1.0) if (sec[i].max(-1) > 0).any() else 1.0 for i in idxs}


def test_the_setting_is_read():
    from labels.labels import config
    assert config()["interpolate"]["scale_max_over_median"] == 3.5


def test_no_limit_is_the_rule_before():
    from labels.interpolate import section_scales
    sec, idxs = _sections([120, 130, 140, 900])
    new, old = section_scales(sec, idxs, None), _before(sec, idxs)
    assert list(new) == list(old) and all(type(new[i]) is type(old[i]) and new[i] == old[i] for i in idxs)


def test_a_section_over_the_limit_takes_the_median():
    from labels.interpolate import section_scales
    sec, idxs = _sections([120, 130, 140, 150, 160, 2000])
    old = _before(sec, idxs)
    med = np.median(np.array([old[i] for i in idxs], dtype=np.float64))
    new = section_scales(sec, idxs, 3.5)
    over = [i for i in idxs if old[i] > 3.5 * med]
    assert over == [10]                                                 # the bright one only
    assert type(new[10]) is type(old[10]) and new[10] == type(old[10])(med)    # the median, in the type of the section's own scale
    assert (sec[10] / new[10]).dtype == (sec[10] / old[10]).dtype == np.float32
    for i in idxs:
        if i not in over:                                               # every other section keeps its scale, bit for bit
            assert type(new[i]) is type(old[i]) and new[i] == old[i]


def test_a_section_under_the_limit_is_untouched():
    from labels.interpolate import section_scales
    sec, idxs = _sections([120, 130, 140, 150, 400])                  # bright, but under 3.5 x the median
    old = _before(sec, idxs)
    med = float(np.median([old[i] for i in idxs]))
    assert old[8] > 2 * med and old[8] <= 3.5 * med
    new = section_scales(sec, idxs, 3.5)
    assert all(type(new[i]) is type(old[i]) and new[i] == old[i] for i in idxs)
    a = (sec[8] / new[8]).astype(np.float32); b = (sec[8] / old[8]).astype(np.float32)
    assert a.tobytes() == b.tobytes()                                   # the divided section, byte for byte
