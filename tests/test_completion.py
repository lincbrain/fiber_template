"""Smoke tests for linc.completion on a toy prepared brain (CPU, no data, ~1 min):  python -m pytest tests/test_completion.py
Network sizes of record, the paired augmentation's shapes, the isotonic map, the join (photographs kept beyond the cross-fade,
prediction caudal of the cut) and the export's product set."""
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from linc.completion import CFG, augment, deploy, export, losses, network  # noqa: E402


def test_network_of_record():
    net = network.build(dict(features=CFG["features"], head=CFG["head"]))
    assert sum(p.numel() for p in net.parameters()) == 5_810_945          # d3_methods: models.*.parameters
    net.eval()
    with torch.no_grad():
        y = net(torch.rand(1, 1, 32, 32, 32))
    assert tuple(y.shape) == (1, 1, 64, 64, 64) and 0 <= float(y.min()) and float(y.max()) <= 1


def test_augment_and_loss():
    torch.manual_seed(0)
    x, y, m = torch.rand(2, 1, 32, 32, 32), torch.rand(2, 1, 64, 64, 64), (torch.rand(2, 1, 64, 64, 64) > 0.3).float()
    xa, ya, ma = augment.two_grid_augment(x, y, m)
    assert xa.shape == x.shape and ya.shape == y.shape and set(torch.unique(ma).tolist()) <= {0.0, 1.0}
    loss = losses.masked_l1(ya, ya, ma) + losses.masked_grad_l1(ya, ya, ma)
    assert float(loss) == 0.0


def test_pava_monotone():
    rng = np.random.default_rng(0)
    x = rng.random(5000); y = np.sqrt(x) + 0.05 * rng.standard_normal(5000)
    cx, cy = deploy.pava_fit(x, y)
    assert np.all(np.diff(cy) >= 0) and abs(np.interp(0.5, cx, cy) - np.sqrt(0.5)) < 0.03


def toy_brain(D, n=16, zf=12, dstar=4):
    """a prepared folder: coarse n^3, fine (2n)^3; planes = the fine grid's axis 1 (y_dep = k_1); photographs from plane zf - 4"""
    N = 2 * n
    (D / "coarse").mkdir(parents=True); (D / "fine").mkdir()
    Ac = np.diag([0.5, 0.5, 0.5, 1.0]); Af = np.diag([0.25, 0.25, 0.25, 1.0])
    k = np.indices((N, N, N)); r = np.sqrt(((k - (N - 1) / 2.0) ** 2).sum(0))
    brain = r < N / 2.0 - 1
    real = brain & (k[1] >= zf - 4)
    grey = (0.3 + 0.4 * np.sin(k[0] / 3.0) * np.cos(k[2] / 4.0) ** 2).astype(np.float32) * real
    b0 = np.asarray(brain[::2, ::2, ::2], np.float32) * 0.8
    sv = lambda a, A, p: nib.save(nib.Nifti1Image(a, A), str(D / p))  # noqa: E731
    sv(b0, Ac, "coarse/in.nii.gz"); sv(brain[::2, ::2, ::2].astype(np.uint8), Ac, "coarse/brain.nii.gz")
    sv(grey, Af, "fine/bf.nii.gz"); sv(real.astype(np.uint8), Af, "fine/real_mask.nii.gz"); sv(real.astype(np.uint8), Af, "fine/valid.nii.gz")
    sv(real.astype(np.float32), Af, "fine/photo_fraction.nii.gz"); sv(brain.astype(np.uint8), Af, "fine/brain.nii.gz")
    sv(np.stack([grey * 1.05, grey, grey * 0.95], -1).astype(np.float32), Af, "fine/bf_rgb.nii.gz")
    json.dump(dict(subject=D.name, mlo=0.0, mhi=1.0, coarse_box=dict(lo=[0, 0, 0], shape=[n] * 3),
                   seam=dict(z_f=zf, dstar_planes=dstar, y_dep=dict(y0=0.0, g=[0.0, 1.0, 0.0]))), open(D / "seam.json", "w"))
    return grey, real, brain


def test_join_and_export(tmp_path):
    D = tmp_path / "TOY1"
    grey, real, brain = toy_brain(D)
    rg = deploy.regions(str(D), "full")
    assert rg["cut"] == 12 and rg["retained"].sum() == (real & (np.indices(real.shape)[1] >= 12)).sum() > 0
    raw = np.clip(grey + 0.1, 0, 1).astype(np.float32) * brain                # a prediction: the photographs, brighter
    synth, w, comp = deploy.join(raw, grey, rg)
    far = rg["retained"] & (rg["depth"] <= -deploy.FADE_MM)                   # >= 2 mm rostral of the cut: photographs only
    assert np.array_equal(comp[far], grey[far])
    caudal = brain & (rg["depth"] > 0)
    assert np.array_equal(comp[caudal], synth[caudal]) and np.all(w[caudal] == 0)

    class Up(torch.nn.Module):                                               # a stand-in network: x2 nearest upsampling
        def forward(self, x):
            return torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
    row, normal = export.export_brain(Up(), str(D), "TOY1", str(tmp_path / "out"), device="cpu", cache_out=str(tmp_path / "cache"))
    for nm in ("bf_completed", "bf_completed_rgb", "bf_synth", "bf_real", "seam_ramp", "seam_depth", "synth_mask", "provenance",
               "photo_fraction", "b0_input_0p5"):
        assert (tmp_path / "out" / "TOY1" / f"TOY1_{nm}.nii.gz").exists(), nm
    assert row["z_f"] == 12 and row["provenance_le_photo_fraction"] and row["seam_depth_nonneg_exactly_rostral"]
    assert (tmp_path / "cache" / "TOY1" / "deploy" / "real_mask.nii.gz").exists() and normal["seam_normal_world"] == [0.0, 1.0, 0.0]
