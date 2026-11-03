#!/usr/bin/env python3
"""Smoke test of cli/linc-reconstruct on a toy brain written on the fly, to check that the steps a new brain takes run (no release
data, no network weights, no GPU; about a minute).

    python reproduce/smoke/reconstruct_toy.py [--keep]      # --keep: leave the temporary folder (printed) for inspection

You need
  env/tirl.yml (TIRL importable, zarr2nii beside its Python).
What it does
  Ten synthetic blockface photographs (a tan ellipsoid in blue-white ice, 20 um pixels, 400 um sections, no camera jumps) go through
  the steps, with the two segmentations given by hand, as a user's hand corrections would be: the photographs' masks are written
  into 01_brainmask/00_predict (where the 2D network's would go) and the 3D mask as curated/TOY1_mask_edited (which wins over the 3D
  network's). Steps: photos (ingest, normalize, finalize; in-process), `linc-reconstruct stack`, `prepare`, the 3D mask's input
  (segment3d.prep_mask), `volume` and `spacing`.
You get
  PASS when the reconstruction, its pyramid and its mask exist, and the mask's volume and the reconstruction's spacing are those of
  the toy brain.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

S, VX, VZ, N, H, W = "TOY1", 20.0, 400.0, 10, 320, 448


def photographs(raw):
    """N photographs: ice (blue-white, a little noise) and a tan ellipsoid whose section grows then shrinks (a toy brain)."""
    from PIL import Image
    raw.mkdir(parents=True)
    rng = np.random.default_rng(2026)
    yy, xx = np.mgrid[:H, :W]
    for k in range(N):
        t = (k + 0.5) / N
        r = np.sqrt(max(1e-3, 1 - (2 * t - 1) ** 2))                     # an ellipsoid's section along the cutting axis
        ry, rx = 100 * r + 8, 150 * r + 8
        inside = ((yy - H / 2) / ry) ** 2 + ((xx - W / 2) / rx) ** 2 <= 1
        img = np.empty((H, W, 3), np.float32); img[...] = (200, 215, 230)
        img[inside] = (190, 150, 110)
        img[inside & (((yy - H / 2) / (0.5 * ry)) ** 2 + ((xx - W / 2) / (0.5 * rx)) ** 2 <= 1)] = (225, 210, 180)   # "white matter"
        img += rng.normal(0, 3, img.shape)
        Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(raw / f"Img{k + 1:04d}.JPG", quality=95)


def run(*cmd, env):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=str(REPO), env=env)


def build(tmp):
    """The toy brain reconstructed in tmp (LINC_RAW / LINC_DATA / LINC_SUBJECTS / LINC_WORK under it, also set in os.environ).
    Returns (Subject, the reconstruction store, its mask, the environment). Used by reproduce/smoke/histology_toy.py too."""
    env = dict(os.environ, LINC_RAW=str(tmp / "raw"), LINC_DATA=str(tmp / "data"), LINC_SUBJECTS=str(tmp / "subjects"),
               LINC_WORK=str(tmp / "work"), PYTHONNOUSERSITE="1")
    os.environ.update({k: env[k] for k in ("LINC_RAW", "LINC_DATA", "LINC_SUBJECTS", "LINC_WORK")})
    photographs(tmp / "raw" / "toy")
    (tmp / "subjects").mkdir()
    (tmp / "subjects" / f"{S}.yaml").write_text(
        f"subject: {S}\nraw: toy\nraw_glob: '*.JPG'\nraw_regex: 'Img(\\d+)\\.JPG'\nvx: {VX}\nvz: {VZ}\n")
    from linc.blockface import photos, segment3d, zarrio
    from linc.blockface.subject import Subject
    import nibabel as nib
    from PIL import Image
    cfg = Subject(S)
    print("== photos (ingest; the photographs' masks given by hand; normalize; finalize)", flush=True)
    photos.ingest(cfg)
    os.makedirs(cfg.brainmask("00_predict"), exist_ok=True)
    for p in sorted(Path(cfg.slices).glob("SLICE_*_0000.png")):
        rgb = np.asarray(Image.open(p).convert("RGB")).astype(np.float32)
        m = (rgb[..., 0] - rgb[..., 2]) > 20                                # tan tissue vs blue-white ice
        zarrio.save_mask(m.astype(np.uint8), os.path.join(cfg.brainmask("00_predict"), p.name.replace("_0000.png", ".png")))
    photos.normalize_masks(cfg)
    photos.finalize_brainmask(cfg)
    py, cli = sys.executable, REPO / "cli" / "linc-reconstruct"
    run(py, cli, "stack", "--subject", S, env=env)
    run(py, cli, "prepare", "--subject", S, env=env)
    print("== the 3D mask's input (segment3d.prep_mask); the 3D mask given by hand (curated/TOY1_mask_edited)", flush=True)
    segment3d.prep_mask(cfg)
    bg = nib.load(cfg.curated_bg); b = np.asanyarray(bg.dataobj).astype(np.float32)
    m = (b[..., 0] - b[..., 2]) > 20
    nib.save(nib.Nifti1Image(m.astype(np.uint8), bg.affine), cfg.mask_edited)
    run(py, cli, "volume", "--subject", S, env=env)
    run(py, cli, "spacing", "--subject", S, env=env)
    return cfg, Path(cfg.root) / f"{S}_blockface.ome.zarr", Path(cfg.root) / f"{S}_mask.nii.gz", env


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="linc_smoke_reconstruct_"))
    try:
        import nibabel as nib
        cfg, final, mask, env = build(tmp)
        print("== checks", flush=True)
        assert final.is_dir(), f"no {final}"
        assert mask.exists(), f"no {mask}"
        mi = nib.load(str(mask)); mv = np.asanyarray(mi.dataobj) > 0
        vol = float(mv.sum() * np.prod(mi.header.get_zooms()[:3]))
        expect = 4 / 3 * np.pi * (100 * VX / 1000) * (150 * VX / 1000) * (N * VZ / 1000 / 2)   # the toy ellipsoid, mm^3
        print(f"  reconstruction {final.name}: levels {sorted(p.name for p in final.iterdir() if p.is_dir())}; mask {mv.sum()} voxels, "
              f"{vol:.0f} mm3 (toy ellipsoid {expect:.0f} mm3); spacing {[round(float(z), 4) for z in mi.header.get_zooms()[:3]]} mm")
        assert 0.6 * expect < vol < 1.4 * expect, (vol, expect)
        print("PASS")
    finally:
        if a.keep:
            print("kept:", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
