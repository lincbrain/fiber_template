"""The blockface masks' post-processing is one rule per task, whichever engine segments (CPU, no data, seconds).

    python -m pytest tests/test_blockface_rules.py      or      python tests/test_blockface_rules.py

You need
  env/deep.yml (torch; linc/blockface/mask_rules.py needs only numpy and scipy); $LINC_MODELS for the last check.
What it does
  linc-reconstruct applies the pipeline's rules (linc/blockface/mask_rules.py) once, for both engines, and runs linc-segment with
  --postprocess none; each linc-segment model's spec.json holds the same rule for use outside the pipeline. The tests check that the
  two copies agree, that the tube rule's parameters in configs/blockface.yaml are the shipped spec's, and (with $LINC_MODELS set)
  that the shipped specs name those rules.
You get
  pytest's pass / fail per test.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from linc.blockface import mask_rules  # noqa: E402
from linc.segment.predict import postprocess_mask  # noqa: E402

try:
    import pytest
    skip = pytest.skip
except ImportError:                                                     # plain-script run
    class Skip(Exception):
        pass

    def skip(why):
        raise Skip(why)

VS = yaml.safe_load((REPO / "configs" / "blockface.yaml").read_text())["volume_segmentation"]
TUBES = {"kind": "tubes_left_hemisphere", "left_tol_vox": float(VS["tubes_left_tolerance_vox"]), "min_vox": int(VS["tubes_min_voxels"])}


def _blobs(rng, shape, n, r):
    """n random balls of radius up to r, some touching, some not: several components, holes and specks."""
    zz = np.indices(shape)
    m = np.zeros(shape, bool)
    for _ in range(n):
        c = rng.uniform(0, 1, len(shape)) * np.array(shape); rr = rng.uniform(0.5, r)
        m |= sum((zz[k] - c[k]) ** 2 for k in range(len(shape))) <= rr ** 2
    return m


def test_largest_cc_fill_same_rule():
    rng = np.random.default_rng(2026)
    for _ in range(20):
        m = _blobs(rng, (32, 28, 24), 12, 6)
        m[rng.integers(0, 32), rng.integers(0, 28), rng.integers(0, 24)] = False   # a one-voxel hole somewhere
        a, _ = mask_rules.largest_cc_fill(m)
        b = postprocess_mask(m.astype(np.uint8), {"kind": "largest_cc_fill"}, 2)
        assert a.dtype == np.uint8 and (a == (b > 0)).all()


def test_tube_rule_same_rule():
    rng = np.random.default_rng(2026)
    for _ in range(20):
        brain = _blobs(rng, (40, 30, 30), 6, 14)
        P = _blobs(rng, (40, 30, 30), 25, 2.5) | (rng.random((40, 30, 30)) < 0.002)   # tubes, some on the right, specks
        keep, rdrop, specks, ntubes = mask_rules.tube_rule(P, brain, TUBES["left_tol_vox"], TUBES["min_vox"])
        b = postprocess_mask(P.astype(np.uint8), TUBES, 2, brain.astype(np.uint8))
        assert (keep == (b > 0)).all()
        assert ntubes == ndi.label(keep, structure=np.ones((3, 3, 3)))[1]                # every kept component is one tube


def test_fill_photo_mask_same_rule():
    rng = np.random.default_rng(2026)
    m = _blobs(rng, (60, 80), 15, 12)
    m[20:24, 30:34] = False                                                      # interior holes
    a = mask_rules.fill_photo_mask(m)
    assert a.dtype == np.uint8 and (a == ndi.binary_fill_holes(m)).all()
    try:
        b = postprocess_mask(m.astype(np.uint8), {"kind": "fill_holes"}, 2)
    except ValueError:
        skip("linc.segment has no fill_holes post-processing yet (L4 adds it after the models wrap-up's last launch)")
    assert (a == (b > 0)).all()


def test_shipped_specs_name_the_rules():
    """With $LINC_MODELS: each blockface model's shipped spec.json names the pipeline's rule (bf_holes: the same parameters)."""
    root = os.environ.get("LINC_MODELS")
    if not root:
        skip("LINC_MODELS not set")
    models = yaml.safe_load((REPO / "configs" / "blockface.yaml").read_text())["segmentation"]["models"]
    want = {"photos": {"kind": "fill_holes"}, "mask": {"kind": "largest_cc_fill"}, "tubes": TUBES}
    found = 0
    for task, name in models.items():
        f = Path(root) / "segment" / name / "spec.json"
        if not f.exists():
            continue
        pp = json.loads(f.read_text())["inference"]["postprocess"]
        pp = {"kind": pp} if isinstance(pp, str) else pp
        assert pp.get("kind") == want[task]["kind"], (name, pp)
        for k in ("left_tol_vox", "min_vox"):
            if k in want[task]:
                assert float(pp[k]) == float(want[task][k]), (name, k, pp[k], want[task][k])
        found += 1
    if not found:
        skip(f"no blockface model under {root}/segment")


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name)
            except Exception as e:                                               # noqa: BLE001
                tag = "SKIP" if type(e).__name__ in ("Skip", "Skipped") else "FAIL"
                fails += tag == "FAIL"; print(tag, name, e)
    sys.exit(1 if fails else 0)
