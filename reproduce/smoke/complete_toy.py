#!/usr/bin/env python3
"""Smoke test of cli/linc-complete on toy data (env/deep.yml; CPU; no release data, no weights; under a minute).

Writes a toy prepared brain (tests/test_completion.py toy_brain: a ball, photographed rostral of a cut) and a tiny untrained network
checkpoint in the layout of record ({"G": weights, "cfg": configuration}), runs the command line and checks its product set, the
summary row, the provenance file, and that a second run refuses to overwrite. The synthesized tissue is meaningless (random weights);
this checks the installation, not the networks.

    python reproduce/smoke/complete_toy.py [--keep DIR]
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tests")]
from linc.completion import network  # noqa: E402
from test_completion import toy_brain  # noqa: E402

CLI = [sys.executable, str(REPO / "cli" / "linc-complete")]
PRODUCTS = ("bf_completed", "bf_completed_rgb", "bf_synth", "bf_real", "seam_ramp", "seam_ramp_real", "seam_depth", "photo_fraction",
            "provenance", "synth_mask", "b0_input_0p5")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--keep", type=Path); a = ap.parse_args()
    T = a.keep or Path(tempfile.mkdtemp(prefix="linc_complete_smoke_"))
    toy_brain(T / "brains" / "TOY1")
    torch.manual_seed(0)
    cfg = dict(features=[4, 4, 8, 8, 8, 4], head=4, in_channels=1)
    (T / "net").mkdir(parents=True)
    torch.save(dict(G=network.build(cfg).state_dict(), cfg=cfg, epoch=0), T / "net" / "last.pt")
    args = ["--brain", T / "brains" / "TOY1", "--out", T / "out", "--model", T / "net", "--device", "cpu"]
    r = subprocess.run(CLI + [str(x) for x in args], capture_output=True, text=True)
    assert r.returncode == 0, f"linc-complete failed\n{r.stdout}\n{r.stderr}"
    for nm in PRODUCTS:
        assert (T / "out" / "TOY1" / f"TOY1_{nm}.nii.gz").exists(), nm
    assert (T / "out" / "TOY1" / "TOY1_chroma.json").exists()
    summ = json.loads((T / "out" / "summary_net.json").read_text())
    assert "TOY1" in summ["members"] and summ["members"]["TOY1"]["z_f"] == 12, summ
    r2 = subprocess.run(CLI + [str(x) for x in args], capture_output=True, text=True)
    assert r2.returncode != 0 and "never overwritten" in (r2.stdout + r2.stderr), "a second run must refuse to overwrite"
    print(f"linc-complete smoke: OK ({len(PRODUCTS)} volumes + chroma + summary; refuses to overwrite) in {T}")


if __name__ == "__main__":
    main()
