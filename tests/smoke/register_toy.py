#!/usr/bin/env python3
"""Smoke test of code/blockface/registration/linc-register on toy volumes written on the fly, to check the acceptance checks and the b0 step (no release data,
no network weights, no GPU; seconds).

    python tests/smoke/register_toy.py

You need
  settings/env/linc.yml; FSL (FSLDIR) for the b0 step, which is skipped without it.
What it does
  qc         `linc-register qc` on a good toy registration (a b0-like ellipsoid and the same, moved 0.25 mm, a zero warp, an
             identity linear part), then on a bad one (the linear part rotated 45 degrees about z)
  place-b0   the b0 step of `linc-register mri --place-b0` (a rigid FLIRT registration onto the template grid, 6 degrees of
             freedom) on the toy b0 and its mask; the deformable stage needs the anatomix environment and a GPU, and is not part
             of the smoke test
You get
  PASS when the good registration passes (the JSON row and PNG written, pass recorded in a toy registry entry), the bad one fails
  with the rotation named, and the b0 step writes its outputs.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))


def toy(tmp):
    """A b0-like toy brain at 0.5 mm: an ellipsoid with a darker core; its mask; the same moved by 0.25 mm; a zero warp; identity."""
    A = np.diag([0.5, 0.5, 0.5, 1.0]); A[:3, 3] = [-16, -18, -14]
    zz = np.stack(np.meshgrid(*[np.arange(n) for n in (64, 72, 56)], indexing="ij"), -1) * 0.5 + A[:3, 3]

    def brain(shift):
        p = zz - np.array([shift, 0, 0])
        e = (p[..., 0] / 12) ** 2 + (p[..., 1] / 15) ** 2 + (p[..., 2] / 10) ** 2
        img = np.where(e <= 1, 1000.0, 0.0) - np.where(e <= 0.3, 400.0, 0.0) + 200.0 * np.sin(p[..., 1] / 2.0) * (e <= 1)
        return img.astype(np.float32), (e <= 1).astype(np.uint8)
    F, Fm = brain(0.0); M, Mm = brain(0.25)
    out = {}
    for name, arr in (("fixed", F), ("fixed_mask", Fm), ("moved", M), ("moved_mask", Mm)):
        out[name] = tmp / f"{name}.nii.gz"; nib.save(nib.Nifti1Image(arr, A), out[name])
    out["warp"] = tmp / "warp.nii.gz"; nib.save(nib.Nifti1Image(np.zeros(F.shape + (1, 3), np.float32), A), out["warp"])
    out["mat"] = tmp / "linear_world.txt"; np.savetxt(out["mat"], np.eye(4))
    R = np.eye(4); c, s = np.cos(np.radians(45)), np.sin(np.radians(45)); R[:2, :2] = [[c, -s], [s, c]]
    out["mat_bad"] = tmp / "linear_world_rot45.txt"; np.savetxt(out["mat_bad"], R)
    return out


def qc(t, mat, name, entry=None):
    cmd = [sys.executable, str(REPO / "code" / "blockface" / "registration" / "linc-register"), "qc", "--preset", "b02template", "--fixed", t["fixed"], "--moved", t["moved"],
           "--fixed-mask", t["fixed_mask"], "--moved-mask", t["moved_mask"], "--warp", t["warp"], "--mat", mat, "--mat-kind", "world",
           "--outline-target", t["fixed_mask"], "--out", t["out"], "--name", name] + (["--entry", entry] if entry else [])
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env=dict(os.environ, PYTHONNOUSERSITE="1"))
    print(r.stdout + r.stderr)
    r.check_returncode()
    return json.loads((Path(t["out"]) / f"{name}.json").read_text())


def main():
    tmp = Path(tempfile.mkdtemp(prefix="linc_smoke_register_"))
    try:
        t = toy(tmp); t["out"] = tmp / "qc"
        entry = tmp / "xfm" / "toy-native__to__toy-template"; entry.mkdir(parents=True)
        (entry / "meta.json").write_text(json.dumps({"src": "toy-native", "dst": "toy-template", "qc": {"status": None}}))
        print("== qc: a good registration", flush=True)
        row = qc(t, t["mat"], "good", entry)
        assert row["gate_pass"] and row["outline"]["status"] == "PASS", row["gate_fail_reasons"]
        assert (Path(t["out"]) / "good.png").exists()
        assert json.loads((entry / "meta.json").read_text())["qc"]["status"] == "pass"
        print("== qc: a bad registration (the linear part rotated 45 degrees)", flush=True)
        row = qc(t, t["mat_bad"], "bad")
        assert not row["gate_pass"] and any("rotation" in r for r in row["gate_fail_reasons"]), row["gate_fail_reasons"]
        if os.environ.get("FSLDIR"):
            print("== place-b0 (FLIRT 6 dof onto a toy frame)", flush=True)
            from blockface.registration import to_own_mri
            r = to_own_mri.place_b0(str(t["moved"]), str(t["moved_mask"]), str(t["fixed"]), str(tmp / "mri"), "TOY", search_deg=10)
            assert all(Path(p).exists() for p in r.values()), r
            print("  placed:", {k: Path(v).name for k, v in r.items()})
        else:
            print("== place-b0: SKIP (FSLDIR not set)")
        print("PASS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
