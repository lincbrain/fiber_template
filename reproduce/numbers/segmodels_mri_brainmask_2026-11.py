#!/usr/bin/env python3
"""The record segmodels_mri_brainmask_2026-11 (Supplementary S5, Table S8): reproduce/numbers/segmodels.py for the model mri_brainmask."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segmodels import main  # noqa: E402

if __name__ == "__main__":
    main("mri_brainmask")
