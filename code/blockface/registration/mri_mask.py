"""A brain mask for a new brain's b0, made by the MRI brain-mask network, so `linc-register mri --place-b0 B0_NATIVE` needs no
manually made mask.

You need
  the b0 image, the network's weights ($LINC_MODELS/segment/mri_brainmask; settings/models/README.md) and $LINC_SEGMENT_PYTHON
  (settings/env/deep.yml's Python, which has torch).
What it does
  segment_b0(b0_native, out_mask) runs `linc-segment predict --model mri_brainmask --out <out_mask> <b0_native> --postprocess none`
  (the model ships with no post-processing) and checks the weights against the model folder's SHA256SUMS when it has one. The
  paper's brains keep their masks of record (settings/brains/<ID>.yaml); there is one MRI brain-mask network, so there is no
  engine choice here (unlike the blockface steps).
You get
  the mask on the b0's own grid (uint8 0/1) and a provenance file <out_mask without .nii.gz>.json beside it (the model folder, the
  SHA256 of model.pt, spec.json and the input, the command, the time).
"""
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TASK = "mri_brainmask"


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def model_dir(model=TASK):
    """a model folder as given, or a task name -> $LINC_MODELS/segment/<task> (as code/shared/segment/api.py resolves it)"""
    if os.path.isdir(model):
        return Path(model)
    root = os.environ.get("LINC_MODELS")
    if not root:
        raise RuntimeError("set LINC_MODELS (the folder holding segment/mri_brainmask/; settings/models/README.md)")
    d = Path(root) / "segment" / model
    if not (d / "model.pt").exists() or not (d / "spec.json").exists():
        raise RuntimeError(f"{d}: no model.pt / spec.json (the MRI brain-mask weights; settings/models/README.md)")
    return d


def check_weights(d):
    """SHA256 of model.pt and spec.json; against SHA256SUMS when the folder has one (a mismatch stops here)."""
    got = {f: _sha256(d / f) for f in ("model.pt", "spec.json")}
    sums = d / "SHA256SUMS"
    if sums.exists():
        want = {ln.split()[1].lstrip("*"): ln.split()[0] for ln in sums.read_text().splitlines() if ln.strip()}
        bad = [f for f in got if want.get(f) != got[f]]
        if bad:
            raise RuntimeError(f"{d}: {bad} do not match SHA256SUMS")
    return got, sums.exists()


def segment_b0(b0_native, out_mask, device="cuda", model=TASK):
    d = model_dir(model)
    shas, checked = check_weights(d)
    py = os.environ.get("LINC_SEGMENT_PYTHON") or sys.executable
    if not os.path.exists(py):
        raise RuntimeError(f"LINC_SEGMENT_PYTHON={py} not found (settings/env/deep.yml's python)")
    os.makedirs(os.path.dirname(os.path.abspath(out_mask)), exist_ok=True)
    cmd = [py, str(REPO / "code" / "shared" / "linc-segment"), "predict", "--model", str(d), "--out", str(out_mask), str(b0_native),
           "--postprocess", "none", "--device", device]
    print("  $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=dict(os.environ, PYTHONNOUSERSITE="1"))
    prov = dict(made_by="code/blockface/registration/mri_mask.py segment_b0 (linc-register mri --place-b0 without MASK_NATIVE)",
                model=TASK, model_dir=str(d), model_sha256=shas["model.pt"], spec_sha256=shas["spec.json"],
                checked_against_SHA256SUMS=checked, postprocess="none (the model ships with none)",
                input=str(b0_native), input_sha256=_sha256(b0_native), output=str(out_mask), command=cmd,
                time=datetime.datetime.now().astimezone().isoformat(timespec="seconds"))
    stem = str(out_mask)[:-7] if str(out_mask).endswith(".nii.gz") else os.path.splitext(str(out_mask))[0]
    with open(stem + ".json", "w") as f:
        json.dump(prov, f, indent=1)
    return str(out_mask)
