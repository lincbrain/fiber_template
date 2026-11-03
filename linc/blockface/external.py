"""The external tools of the blockface reconstruction, each behind a small wrapper that checks its version first.

You need
  linc-segment          `cli/linc-segment predict` (linc/segment), run with env/deep.yml's Python (it has torch) named by
                        LINC_SEGMENT_PYTHON; weights under $LINC_MODELS/segment/<task> (models/segmentation.md). The engine of all
                        three masks (configs/blockface.yaml `segmentation.engine`)
  nnU-Net v2 2.5.1      only for the engine `nnunet`: the networks the paper's masks of record were made with
                        (models/blockface_segmentation.md; weights under $LINC_MODELS/nnUNet_results)
  TIRL 3.6.3 (or 3.6.2) with tirlscripts-mgh-linc 1.0.1 (env/tirl.yml): the camera-jump registrations; the reconstruction also
                        imports `tirl` to read them back (jumps.py)
  nifti-zarr 1.0.0rc6   zarr2nii, nii2zarr (env/tirl.yml)
  Each tool is taken from the folder its variable names (NNUNET_BIN, TIRL_BIN, ZARR2NII_BIN), else beside the running Python, else
  from PATH (beside the Python comes first because FSL puts its own older `tirl` on PATH).
What it does
  segment, predict_2d, predict_3d   the mask networks
  tirl_h2b                          one camera-jump registration (`linc.h2b`, run by path: tirl_script)
  check_tirl_import                 tirl importable here, at the same version
  zarr2nii, nii2zarr                the conversions
You get
  each tool's outputs where the caller asks; a RuntimeError naming the tool when it is missing or at another version.

The commands are those of the analysis project's code/run.py (`_nnunet`, `_nnunet3d`, `_export`) and code/recon.py (`tirl_jumps`).
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

VERSIONS = {"nnunetv2": "2.5.1", "tirl": "3.6.2", "tirlscripts-mgh-linc": "1.0.1", "nifti-zarr": "1.0.0rc6"}
# also accepted: TIRL 3.6.3, the source the install builds (env/tirl.yml): its code equals 3.6.2's except the version and license
# strings, and a registration of record's schedule gives bit-identical images in both (env/tirl.yml's header)
ALSO_ACCEPTED = {"tirl": ("3.6.3",)}
_checked = set()


def _bin(tool, env_var):
    """The tool from env_var's folder, else beside the running interpreter (the environment this step runs in, e.g. env/tirl.yml's
    bin with tirl and zarr2nii, so `$PY_TIRL cli/linc-reconstruct ...` needs no PATH change), else PATH. Beside the interpreter comes
    before PATH: FSL's setup puts its own, older `tirl` (2.2.1) on PATH."""
    d = os.environ.get(env_var)
    here = Path(sys.executable).parent / tool
    p = str(Path(d) / tool) if d else (str(here) if os.access(here, os.X_OK) else shutil.which(tool))
    if not p or not os.path.exists(p):
        raise RuntimeError(f"{tool} not found: put it on PATH or set {env_var} to its bin folder")
    return p


def _check(tool_path, packages):
    """The packages' versions, read by the Python interpreter next to the tool, must equal VERSIONS."""
    key = (tool_path, packages)
    if key in _checked:
        return
    py = str(Path(tool_path).parent / "python")
    code = "import importlib.metadata as m, sys; print(' '.join(m.version(p) for p in sys.argv[1:]))"
    got = subprocess.run([py, "-c", code, *packages], capture_output=True, text=True, check=True).stdout.split()
    want = [VERSIONS[p] for p in packages]
    if any(g != w and g not in ALSO_ACCEPTED.get(p, ()) for p, g, w in zip(packages, got, want)) or len(got) != len(want):
        raise RuntimeError(f"{tool_path}: {dict(zip(packages, got))}, the paper used {dict(zip(packages, want))}")
    _checked.add(key)


def _run(cmd, env=None):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, env=env)


# ------------------------------------------------------------------------------------------------ linc-segment
def segment(model, inp, out, native=False, postprocess="spec"):
    """`linc-segment predict --model <model> --in <inp> --out <out>`: the same folders as nnUNetv2_predict (a case is <case>_<cccc><ext>;
    an RGB PNG is one file), the masks written as <out>/<case><ext>, uint8 0/1, in the input's grid. native: photographs at native size
    (reduced 4 x 4 to the model's level, the mask returned at native size). postprocess "none" leaves the post-processing to the
    pipeline (linc/blockface/mask_rules.py: photos.normalize_masks, segment3d.post_mask, segment3d.post_holes), as for the nnU-Net
    path; cli/linc-reconstruct passes "none" for all three tasks, so each rule is applied once. Device: $LINC_SEGMENT_DEVICE
    (default cuda)."""
    py = os.environ.get("LINC_SEGMENT_PYTHON") or sys.executable
    if not os.path.exists(py):
        raise RuntimeError(f"LINC_SEGMENT_PYTHON={py} not found (env/deep.yml's python)")
    cmd = [py, str(Path(__file__).resolve().parents[2] / "cli" / "linc-segment"), "predict", "--model", model, "--in", inp, "--out", out,
           "--postprocess", postprocess, "--device", os.environ.get("LINC_SEGMENT_DEVICE", "cuda")] + (["--native"] if native else [])
    _run(cmd, dict(os.environ, PYTHONNOUSERSITE="1"))


# ------------------------------------------------------------------------------------------------ nnU-Net
def _nnunet_env(models_root, work, compile_off=False):
    """nnU-Net reads its weights from $LINC_MODELS/nnUNet_results; raw/preprocessed are unused at inference (set so it does not
    warn). compile_off: nnUNet_compile=f, as the 3D runs of record set it."""
    env = dict(os.environ)
    env.update(nnUNet_results=str(Path(models_root) / "nnUNet_results"), nnUNet_raw=str(Path(work) / "nnUNet_raw"),
               nnUNet_preprocessed=str(Path(work) / "nnUNet_preprocessed"))
    if compile_off:
        env["nnUNet_compile"] = "f"
    return env


def models_root():
    d = os.environ.get("LINC_MODELS")
    if not d:
        raise RuntimeError("set LINC_MODELS (the folder holding nnUNet_results/; see models/blockface_segmentation.md)")
    return d


def predict_2d(p, inp, predict_out, final_out, work):
    """The photo brain masks: nnUNetv2_predict + nnUNetv2_apply_postprocessing (code/run.py `_nnunet`, Dataset500).
    p = configs/blockface.yaml `photo_segmentation`."""
    exe = _bin("nnUNetv2_predict", "NNUNET_BIN"); _check(exe, ("nnunetv2",))
    post = str(Path(exe).parent / "nnUNetv2_apply_postprocessing")
    env = _nnunet_env(models_root(), work)
    pp_dir = Path(env["nnUNet_results"]) / p["dataset"] / f"{p['trainer']}__{p['plans']}__{p['configuration']}" / p["postprocessing"]
    _run([exe, "-d", p["dataset"], "-i", inp, "-o", predict_out, "-f", *str(p["folds"]).split(), "-tr", p["trainer"], "-c",
          p["configuration"], "-p", p["plans"]], env)
    _run([post, "-i", predict_out, "-o", final_out, "-pp_pkl_file", pp_dir / "postprocessing.pkl", "-np", "8", "-plans_json",
          pp_dir / "plans.json"], env)


def predict_3d(p, dataset, inp, out, work):
    """A 3D network on the RGB channels in `inp` (code/run.py `_nnunet3d`). p = configs/blockface.yaml `volume_segmentation`."""
    exe = _bin("nnUNetv2_predict", "NNUNET_BIN"); _check(exe, ("nnunetv2",))
    _run([exe, "-i", inp, "-o", out, "-d", dataset, "-c", p["configuration"], "-f", *str(p["folds"]).split(), "-tr", p["trainer"],
          "-p", p["plans"], "-chk", p["checkpoint"]], _nnunet_env(models_root(), work, compile_off=True))


# ------------------------------------------------------------------------------------------------ TIRL
H2B_MODULE = "tirlscripts.mgh.linc.histology_to_block"      # the script `tirl linc.h2b` names


def tirl_script(exe, module=H2B_MODULE):
    """The path of a TIRL script module in the environment of `exe`. The `tirl` launcher runs a script as `<its python> <script>
    <args>` with the script's folder appended to PATH; tirl_h2b does the same itself, so neither `tirl register` (whose registry,
    ~/.config/tirlscripts.yml, every TIRL environment of the account shares) nor the registry file is needed, and the script's own
    exit status is kept (the launcher drops it)."""
    py = str(Path(exe).parent / "python")
    r = subprocess.run([py, "-c", "import importlib.util, sys; print(importlib.util.find_spec(sys.argv[1]).origin)", module],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(r.stdout.strip()):
        raise RuntimeError(f"{module} not found by {py}: install tirlscripts-mgh-linc (env/tirl.yml)")
    return r.stdout.strip()


def tirl_h2b(config_file):
    """`tirl linc.h2b --config <file> --verbose` (code/recon.py `tirl_jumps`), run as the launcher runs it (tirl_script)."""
    exe = _bin("tirl", "TIRL_BIN"); _check(exe, ("tirl", "tirlscripts-mgh-linc"))
    script = tirl_script(exe)
    _run([str(Path(exe).parent / "python"), script, "--config", config_file, "--verbose"],
         dict(os.environ, PATH=os.environ.get("PATH", "") + ":" + os.path.dirname(script)))


def check_tirl_import():
    """The reconstruction reads the jump registrations back with tirl's own loader: tirl must be importable here, same version."""
    import importlib.metadata as m
    v = m.version("tirl")
    if v != VERSIONS["tirl"] and v not in ALSO_ACCEPTED["tirl"]:
        raise RuntimeError(f"tirl {v} in this environment; the paper used {VERSIONS['tirl']}")


# ------------------------------------------------------------------------------------------------ zarr2nii
def zarr2nii(level, zarr_path, nii_path):
    """`zarr2nii --level <level> <zarr> <nii>` (code/run.py `_export`)."""
    exe = _bin("zarr2nii", "ZARR2NII_BIN"); _check(exe, ("nifti-zarr",))
    _run([exe, "--level", level, zarr_path, nii_path])


def nii2zarr(nii_path, zarr_path):
    """`nii2zarr <nii> <zarr> --chunk 1024 --levels 1 --fill 0 --no-time --no-pyramid-axis z --zarr-version 3` (code/run.py stage 2b)."""
    exe = _bin("nii2zarr", "ZARR2NII_BIN"); _check(exe, ("nifti-zarr",))
    _run([exe, nii_path, zarr_path, "--chunk", "1024", "--levels", "1", "--fill", "0", "--no-time", "--no-pyramid-axis", "z",
          "--zarr-version", "3"])
