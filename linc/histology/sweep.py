"""The histology parameters with one setting of the registration test applied (paper: Supplementary S4, Table S7).

configs/histology_sweep.yaml gives each setting (C0-C6, C8, C9) as dotted keys into configs/histology.yaml; every key must exist.
The result goes to `cli/linc-histology register --config`:

    python -m linc.histology.sweep C2 > histology_C2.yaml
    cli/linc-histology register --config histology_C2.yaml --pieces <piece> --out <folder> ...

The test's scoring (the hidden-tube offset and tissue Dice per piece, then the fixed decision rule) is not part of this release;
its results are the reference records tirl_sweep_r058 and tirl_sweep_r058b (configs/numbers_exceptions.yaml).
"""
import argparse
import copy
import sys
from pathlib import Path

import yaml

from . import match as M

SWEEP = Path(__file__).resolve().parents[2] / "configs" / "histology_sweep.yaml"


def settings():
    return yaml.safe_load(SWEEP.read_text())


def apply(setting, cfg=None):
    """A copy of the parameters (configs/histology.yaml, or `cfg`) with `setting`'s values set."""
    cfg = copy.deepcopy(cfg or M.config())
    for dotted, v in (settings()[setting] or {}).items():
        d, ks = cfg, dotted.split(".")
        for k in ks[:-1]:
            d = d[k]
        if ks[-1] not in d:
            raise KeyError(f"{setting}: {dotted} is not in the parameters")
        d[ks[-1]] = v
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("setting", choices=list(settings()))
    ap.add_argument("--config", help="parameters to start from (default: configs/histology.yaml)")
    a = ap.parse_args()
    yaml.safe_dump(apply(a.setting, M.config(a.config)), sys.stdout, sort_keys=False, default_flow_style=None)


if __name__ == "__main__":
    main()
