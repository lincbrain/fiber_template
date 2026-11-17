"""Smoke tests for shared.segment on toy data (CPU, no data, seconds):  python -m pytest tests/test_segment.py
The network of record's size, nnU-Net file naming in the folder API, a model folder round trip (state_dict + spec.json -> masks in
the input's grid, NIfTI and PNG, native-size photographs), and the post-processing rules."""
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from shared.segment import api, predict  # noqa: E402
from shared.segment.net import UNet  # noqa: E402


def test_network_of_record():
    net = UNet(3, 1, 2)                                                   # the 3D models (mri_brainmask: one channel)
    assert sum(p.numel() for p in net.parameters()) == 31_195_594
    net.eval()
    with torch.no_grad():
        y = net(torch.rand(1, 1, 64, 64, 64))                              # five stride-2 stages: 2^3 voxels at the bottom
    assert tuple(y.shape) == (1, 2, 64, 64, 64)


def _model(tmp, dim, cin, **kw):
    torch.manual_seed(0)
    net = UNet(dim, cin, 2, features=(4, 8, 16), deep_supervision=True)
    spec = dict(task="toy", fold="all", net=net.to_spec(), patch=[32] * dim, classes={"0": "background", "1": "fg"}, channels=cin,
                normalization="zscore", spacing=None, target_spacing=None, whole_image=False, native_downsample=None,
                inference=dict(step=0.5, mirror=False, postprocess=None))
    spec.update(kw)
    d = tmp / f"model{dim}d"; d.mkdir()
    torch.save(net.state_dict(), d / "model.pt"); (d / "spec.json").write_text(json.dumps(spec))
    return d


def test_cases_in(tmp_path):
    for n in ("a_0000.nii.gz", "a_0001.nii.gz", "b_0000.nii.gz", "notes.txt"):
        (tmp_path / n).write_bytes(b"")
    c = api.cases_in(tmp_path)
    assert list(c) == ["a", "b"] and [p.name for p in c["a"][1]] == ["a_0000.nii.gz", "a_0001.nii.gz"] and c["b"][0] == ".nii.gz"
    (tmp_path / "c_0001.nii.gz").write_bytes(b"")
    with pytest.raises(ValueError):
        api.cases_in(tmp_path)                                            # channel 0000 missing


def test_folder_nifti_3d(tmp_path):
    m = _model(tmp_path, 3, 2, target_spacing=[1.0, 1.0, 1.0])
    ind = tmp_path / "in"; ind.mkdir()
    aff = np.diag([0.5, 0.5, 0.5, 1.0])
    for k in range(2):
        nib.save(nib.Nifti1Image(np.random.rand(40, 36, 30).astype(np.float32), aff), ind / f"case_{k:04d}.nii.gz")
    took = api.predict_folder(m, ind, tmp_path / "out", device="cpu")
    out = nib.load(tmp_path / "out" / "case.nii.gz")
    assert list(took) == ["case"] and out.shape == (40, 36, 30) and np.allclose(out.affine, aff)
    assert out.get_data_dtype() == np.uint8 and set(np.unique(np.asanyarray(out.dataobj))) <= {0, 1}
    with pytest.raises(ValueError):                                       # three channels given to a two-channel model
        nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), np.float32), aff), ind / "case_0002.nii.gz")
        api.predict_folder(m, ind, tmp_path / "out2", device="cpu")


def test_folder_png_native(tmp_path):
    from PIL import Image
    m = _model(tmp_path, 2, 3, whole_image=True, native_downsample=4)
    ind = tmp_path / "in"; ind.mkdir()
    Image.fromarray((np.random.rand(100, 132, 3) * 255).astype(np.uint8)).save(ind / "SLICE_0001_0000.png")
    api.predict_folder(m, ind, tmp_path / "out", native=True, device="cpu")
    a = np.asarray(Image.open(tmp_path / "out" / "SLICE_0001.png"))
    assert a.shape == (100, 132) and a.dtype == np.uint8


def test_postprocess():
    m = np.zeros((20, 20, 20), np.uint8); m[2:10, 2:10, 2:10] = 1; m[5, 5, 5] = 0; m[15:17, 15:17, 15:17] = 1
    f = predict.postprocess_mask(m, {"kind": "largest_cc_fill"}, 2)
    assert f.sum() == 512 and f[5, 5, 5] == 1 and f[16, 16, 16] == 0
    brain = np.zeros_like(m); brain[:, 2:18, 2:18] = 1
    t = np.zeros_like(m); t[3:5, 5:7, 5:7] = 1; t[15:17, 5:7, 5:7] = 1; t[3, 12, 12] = 1   # left tube, right tube, a speck
    k = predict.postprocess_mask(t, {"kind": "tubes_left_hemisphere", "left_tol_vox": 3.0, "min_vox": 3}, 2, brain)
    assert k[3:5, 5:7, 5:7].all() and k[15:17].sum() == 0 and k[3, 12, 12] == 0


def test_fill_holes():
    """the photograph rule (code/blockface/mask_rules.py fill_photo_mask): enclosed holes filled, nothing else changed"""
    from scipy.ndimage import binary_fill_holes
    m = np.zeros((60, 80), np.uint8); m[10:50, 10:70] = 1; m[20:30, 20:30] = 0; m[0:5, 0:5] = 1
    f = predict.postprocess_mask(m, {"kind": "fill_holes"}, 2)
    assert f.dtype == np.uint8 and np.array_equal(f, binary_fill_holes(m > 0).astype(np.uint8)) and f[25, 25] == 1 and f[2, 2] == 1


def test_augment_keeps_label_on_image():
    """pure crop + mirroring: every sample's label stays its own image's threshold (an earlier version mirrored the channel and batch dims)"""
    from shared.segment import augment
    for dim, B, ctx, patch in ((3, 2, [24, 20, 28], [16, 12, 20]), (2, 4, [40, 36], [32, 28])):
        y = torch.zeros([B] + ctx, dtype=torch.long)
        for b in range(B):
            y[(b,) + tuple(slice(2 + 3 * b, 2 + 3 * b + s // 2) for s in ctx)] = 1
        x = y[:, None].float().repeat(1, 3, *([1] * dim)) * torch.arange(1, 4).view([1, 3] + [1] * dim)
        for it in range(50):
            P = dict(augment.DEFAULTS, p_rot=0.0, p_scale=0.0)
            xo, yo = augment.spatial(x.clone(), y.clone(), patch, torch.Generator().manual_seed(it), P)
            for b in range(B):
                for c in range(3):
                    assert torch.equal((xo[b, c] > 0.5 * (c + 1)).long(), yo[b])


def test_ds_targets_centered():
    """the coarse deep-supervision targets keep the label's centroid (block-center convention) within 0.05 voxel to 1/8"""
    from shared.segment import losses
    g = torch.meshgrid(*[torch.arange(128.)] * 3, indexing="ij")              # the 3D patch; an off-grid ellipsoid
    y = ((((g[0] - 61.3) / 30) ** 2 + ((g[1] - 66.8) / 22) ** 2 + ((g[2] - 58.1) / 35) ** 2) < 1).long()[None]

    def com(w):
        gg = torch.meshgrid(*[torch.arange(s, dtype=torch.float64) for s in w.shape], indexing="ij")
        return torch.stack([(gi * w).sum() / w.sum() for gi in gg])
    c0 = com(y[0].double())
    for f in (2, 4, 8):
        t = losses.ds_target(y, 2, [128 // f] * 3)[0, 1].double()           # legacy nearest: +0.5 / +1.5 / +3.5
        assert (((com(t) + 0.5) * f - 0.5) - c0).abs().max() < 0.05


def _com(w):
    w = torch.as_tensor(w, dtype=torch.float64)
    g = torch.meshgrid(*[torch.arange(s, dtype=torch.float64) for s in w.shape], indexing="ij")
    return torch.stack([(gi * w).sum() / w.sum() for gi in g])


def test_lowres_keeps_content_in_place():
    """the low-resolution augmentation (block means down, linear up): mean centroid shift over positions < 0.05 voxel, each < 0.25
    (a mode="nearest" reduction moves the image +0.5-0.85 voxel toward high indices, its label unmoved)"""
    from shared.segment import augment
    for dim in (3, 2):
        S = [96] + [32] * (dim - 1)
        g = torch.meshgrid(*[torch.arange(s, dtype=torch.float32) for s in S], indexing="ij")
        cs = [[24 + 2.087 * i] + [10 + (0.531 * i * (a + 1)) % 12 for a in range(dim - 1)] for i in range(24)]
        x = torch.stack([torch.exp(-sum((gi - ci) ** 2 for gi, ci in zip(g, c)) / 18.0) for c in cs])[:, None]
        for z in (0.5, 0.7, 0.9):
            d = torch.stack([_com(augment.lowres(xi[None], z)[0, 0]) - _com(xi[0]) for xi in x])
            assert d.mean(0).abs().max() < 0.05 and d.abs().max() < 0.25


def test_label_resample_centered():
    """labels: nearest-exact, block mean >= 0.5 at even integer reductions: the centroid stays put at x 2, near-1 and up-sampling
    factors, and a thin tube keeps its volume when up-sampled"""
    from shared.segment import io
    S = [120, 90, 100]
    g = np.meshgrid(*[np.arange(s, dtype=np.float64) for s in S], indexing="ij")
    lab = ((((g[0] - 57.3) / 30) ** 2 + ((g[1] - 44.8) / 22) ** 2 + ((g[2] - 51.1) / 35) ** 2) < 1).astype(np.uint8)
    c0 = _com(lab)
    for shape in ([60, 45, 50], [118, 88, 98], [168, 126, 140]):
        k = torch.tensor([s / t for s, t in zip(S, shape)], dtype=torch.float64)
        new = io.resample_label(lab, shape)
        assert new.dtype == np.uint8 and list(new.shape) == shape
        assert (((_com(new) + 0.5) * k - 0.5) - c0).abs().max() < (0.05 if shape[0] == 60 else 0.08) * float(k.max())
    T, shape = [60, 52, 56], [81, 70, 76]
    g = np.meshgrid(*[np.arange(s, dtype=np.float64) for s in T], indexing="ij")
    p = np.stack(g, -1) - np.array(T) / 2; d = np.array([0.8, 0.45, 0.4]); d /= np.linalg.norm(d)
    tube = (np.linalg.norm(p - (p @ d)[..., None] * d, axis=-1) < 1.6).astype(np.uint8)
    r = io.resample_label(tube, shape).sum() * np.prod([s / t for s, t in zip(T, shape)]) / tube.sum()
    assert 0.97 < r < 1.03


def test_inference_mirroring_equivariant():
    """sliding_window with every flip = without, for a pointwise (flip-equivariant) network: the flips are on the spatial axes"""
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Conv3d(3, 8, 1), torch.nn.ReLU(), torch.nn.Conv3d(8, 2, 1)).eval()
    net.n_classes = 2
    img = np.random.default_rng(0).standard_normal((3, 40, 36, 44)).astype(np.float32)
    a = predict.sliding_window(net, img, [24, 24, 24], "cpu", 0.5, ())
    b = predict.sliding_window(net, img, [24, 24, 24], "cpu", 0.5, (0, 1, 2))
    assert np.abs(a - b).max() < 1e-5
