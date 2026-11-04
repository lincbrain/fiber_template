#!/usr/bin/env python3
"""Smoke test of cli/linc-histology on toy data written on the fly, to check that every step runs (no release data, no network
weights, no GPU; a few minutes).

    python reproduce/smoke/histology_toy.py [--keep]

You need
  env/tirl.yml with TIRL and zarr2nii.
What it does
  The toy brain of reproduce/smoke/reconstruct_toy.py is reconstructed first (its blockface store and mask); then one synthetic
  histology section (a stained ellipse on a white slide, JPEG 2000) is matched to the store's middle section by a one-row match
  table, the regions are the mask split into left and right halves, and `linc-histology all` runs every step: match, pieces, masks,
  prepare, register (TIRL, one whole-section component, a lighter schedule of the same four stages: the smoke test checks that the
  steps run, not the registration of record) and rgb.
You get
  PASS when every step's product exists, with its provenance; --keep leaves the temporary folder (printed).
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
REPO = H.parents[1]
sys.path.insert(0, str(H)); sys.path.insert(0, str(REPO))
import reconstruct_toy as R  # noqa: E402

SECTION = 5                     # the histology section number; the match table sends it to photograph 5 (store slice 4)


def section_jp2(path, shape=(768, 1024)):
    """A stained toy section: white slide, a purple ellipse (the tissue) with a paler core, JPEG 2000 with 4 resolution levels."""
    import glymur
    hh, ww = shape
    yy, xx = np.mgrid[:hh, :ww]
    e = ((yy - hh / 2) / (0.36 * hh)) ** 2 + ((xx - ww / 2) / (0.4 * ww)) ** 2
    img = np.empty((hh, ww, 3), np.float32); img[...] = (245, 245, 245)
    img[e <= 1] = (150, 90, 160); img[e <= 0.25] = (200, 160, 205)
    img += np.random.default_rng(2026).normal(0, 2, img.shape)
    glymur.Jp2k(str(path), data=np.clip(img, 0, 255).astype(np.uint8), numres=4)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="linc_smoke_histology_"))
    try:
        import nibabel as nib
        cfg, store, mask, env = R.build(tmp)
        print("== toy histology: one section, the match table, the regions", flush=True)
        hd = tmp / "histology"; hd.mkdir()
        section_jp2(hd / f"{R.S}_s{SECTION:04d}_nDF.jp2")
        table = tmp / "match_table.csv"                    # section n -> photograph 1 * n + 0 (whole); no hemisphere / cerebellum pieces
        table.write_text("lower,upper,m_whole,a_whole,m_left,a_left,m_right,a_right,m_cere,a_cere\n0,inf,1,0,0,-1,0,-1,0,-1\n")
        mi = nib.load(str(mask)); m = np.asanyarray(mi.dataobj) > 0
        xw = (mi.affine @ np.c_[np.argwhere(m), np.ones(m.sum())].T)[0]
        reg = np.zeros(m.shape, np.uint8); idx = np.argwhere(m)
        reg[tuple(idx.T)] = np.where(xw < np.median(xw), 1, 2)          # 1 left, 2 right (RAS world x)
        regions = tmp / f"{R.S}_regions.nii.gz"; nib.save(nib.Nifti1Image(reg, mi.affine), str(regions))
        out = tmp / "histology_out"
        # a lighter TIRL schedule for the smoke only (the same four stages, fewer orientations, scales and iterations); the
        # parameters of record are configs/histology.yaml's
        import yaml
        hc = yaml.safe_load((REPO / "configs" / "histology.yaml").read_text())
        rp = hc["tirl"]["config"]["regparams"]
        rp["rotsearch"]["coarse"] = 30
        rp["rigid"].update(scaling=[4, 2], smoothing=[0, 0])
        rp["affine"].update(scaling=[4, 2], smoothing=[0, 0])
        rp["nonlinear"].update(scaling=[4, 2], smoothing=[0, 0], maxiter=[10, 10])
        smoke_cfg = tmp / "histology_smoke.yaml"; smoke_cfg.write_text(yaml.safe_dump(hc, sort_keys=False))
        cmd = [sys.executable, str(REPO / "cli" / "linc-histology"), "all", "--config", str(smoke_cfg), "--subject", R.S, "--store", str(store), "--histology", str(hd),
               "--table", str(table), "--regions", str(regions), "--idx0", "1", "--out", str(out),
               "--threads", str(min(12, os.cpu_count() or 1))]
        print("  $", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=str(REPO), env=env)
        print("== checks", flush=True)
        for d in ("matched", "partition", "pieces", "tissue_masks", "moving", "fixed", "tirl"):
            files = [p for p in (out / d).rglob("*") if p.is_file()]
            assert files, f"{out / d}: empty"
            print(f"  {d}: {len(files)} files")
        rgb = list((out / "tirl").rglob("moving4_nonlinear_rgb.png"))
        assert rgb, "no registered RGB section"
        assert all((out / d / "provenance.json").exists() for d in ("matched", "partition", "pieces")), "a step without provenance"
        print("PASS")
    finally:
        if a.keep:
            print("kept:", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
