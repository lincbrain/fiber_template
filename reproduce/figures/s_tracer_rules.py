#!/usr/bin/env python3
"""The paper's supplementary figure "Two rules for drawing a label": per case, all bundle classes (light bundles included; terminals
translucent) drawn where the carried density is at least 0.1 and where it exceeds 0.5, in the lateral and dorsal projections.

    python reproduce/figures/s_tracer_rules.py [--out figures/s_tracer_rules.png]

Runs reproduce/figures/tracer.py --figure rules (one code path for both tracer figures)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracer  # noqa: E402

if __name__ == "__main__":
    tracer.main(["--figure", "rules"] + sys.argv[1:])
