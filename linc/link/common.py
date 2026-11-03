"""Shared helpers of linc/link: the config (configs/link.yaml), input paths of record, chunked point maps, the frame bridge."""
import json
from pathlib import Path

import numpy as np

from .. import paths

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "link.yaml"


def config():
    import yaml
    return yaml.safe_load(CONFIG.read_text())


def product(rel, **fmt):
    """An input of record: configs/link.yaml `inputs` value (relative to $LINC_PRODUCTS), formatted with fmt."""
    rel = str(rel).format(**fmt)
    p = Path(rel)
    return p if p.is_absolute() else paths.products_root() / p


def inputs():
    return config()["inputs"]


def chunked(f, P, n=2_000_000):
    """f applied to the rows of P, n at a time."""
    return np.concatenate([np.asarray(f(P[k:k + n])) for k in range(0, len(P), n)])


def apply_affine(M, P):
    return P @ M[:3, :3].T + M[:3, 3]


def bridge(path=None):
    """Rigid world map run-113 frame -> canonical (MRI template) frame, 4x4 RAS."""
    p = Path(path) if path else product(inputs()["bridge"])
    return np.array(json.loads(p.read_text())["rigid_world_run113_to_canonical"])
