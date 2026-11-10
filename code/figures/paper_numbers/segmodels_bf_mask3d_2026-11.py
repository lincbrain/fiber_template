#!/usr/bin/env python3
"""The record segmodels_bf_mask3d_2026-11 (Supplementary S5, Table S8): code/figures/paper_numbers/segmodels.py for the model bf_mask3d."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segmodels import main  # noqa: E402

if __name__ == "__main__":
    main("bf_mask3d")
