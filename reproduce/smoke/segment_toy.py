#!/usr/bin/env python3
"""Smoke test of cli/linc-segment on toy data (env/deep.yml; CPU; no release data, no weights; seconds).

Writes two tiny untrained model folders (model.pt + spec.json, the shipped layout) and toy inputs, runs the command line on them
and checks what it writes: shapes, dtype, the input's grid, nnU-Net file naming, native-size photographs, every post-processing
choice. The masks' content is meaningless (random weights); this checks the installation, not the models.

    python reproduce/smoke/segment_toy.py [--keep DIR]
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from linc.segment.net import UNet  # noqa: E402

CLI = [sys.executable, str(REPO / "cli" / "linc-segment"), "predict"]


def model(d, dim, cin, **kw):
    torch.manual_seed(0)
    net = UNet(dim, cin, 2, features=(4, 8, 16), deep_supervision=True)
    spec = dict(task="toy", fold="all", net=net.to_spec(), patch=[32] * dim, classes={"0": "background", "1": "fg"}, channels=cin,
                normalization="zscore", spacing=None, target_spacing=None, whole_image=False, native_downsample=None,
                inference=dict(step=0.5, mirror=True, postprocess=None))
    spec.update(kw)
    d.mkdir(parents=True)
    torch.save(net.state_dict(), d / "model.pt"); (d / "spec.json").write_text(json.dumps(spec))
    return d


def cli(*args):
    r = subprocess.run(CLI + [str(a) for a in args], capture_output=True, text=True)
    assert r.returncode == 0, f"linc-segment {' '.join(map(str, args))}\n{r.stdout}\n{r.stderr}"
    return r.stdout


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--keep", type=Path); a = ap.parse_args()
    T = a.keep or Path(tempfile.mkdtemp(prefix="linc_segment_smoke_"))
    rng = np.random.default_rng(0)
    # 3D: one case, two channels, resampled to the model's spacing and back; the mask comes back on the input's grid
    m3 = model(T / "m3", 3, 2, target_spacing=[1.0, 1.0, 1.0])
    aff = np.diag([0.5, 0.6, 0.5, 1.0]); aff[:3, 3] = [-10, 4, 2]
    for k in range(2):
        nib.save(nib.Nifti1Image(rng.random((40, 34, 30)).astype(np.float32), aff), T / f"vol_{k:04d}.nii.gz")
    cli("--model", m3, "--out", T / "vol_mask.nii.gz", T / "vol_0000.nii.gz", T / "vol_0001.nii.gz", "--device", "cpu")
    o = nib.load(T / "vol_mask.nii.gz")
    assert o.shape == (40, 34, 30) and np.allclose(o.affine, aff) and o.get_data_dtype() == np.uint8
    for pp in ("none", "largest_component", "largest_cc_fill", "fill_holes"):
        cli("--model", m3, "--out", T / f"vol_mask_{pp}.nii.gz", T / "vol_0000.nii.gz", T / "vol_0001.nii.gz", "--device", "cpu",
            "--postprocess", pp, "--mirror", "off")
        assert set(np.unique(np.asanyarray(nib.load(T / f"vol_mask_{pp}.nii.gz").dataobj))) <= {0, 1}, pp
    # 2D: a folder of native-size RGB photographs, nnU-Net naming, reduced 4 x 4 for the network, masks back at native size
    m2 = model(T / "m2", 2, 3, whole_image=True, native_downsample=4)
    (T / "photos").mkdir()
    for k in range(2):
        Image.fromarray((rng.random((100, 132, 3)) * 255).astype(np.uint8)).save(T / "photos" / f"SLICE_{k:04d}_0000.png")
    cli("--model", m2, "--in", T / "photos", "--out", T / "photo_masks", "--native", "--device", "cpu")
    for k in range(2):
        m = np.asarray(Image.open(T / "photo_masks" / f"SLICE_{k:04d}.png"))
        assert m.shape == (100, 132) and m.dtype == np.uint8, m.shape
    print(f"linc-segment smoke: OK (3D single case + 4 post-processing choices, 2D native folder) in {T}")


if __name__ == "__main__":
    main()
