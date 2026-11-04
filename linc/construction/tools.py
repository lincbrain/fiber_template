"""The construction recipe (configs/mri_construction.yaml) and the external programs the MRI construction calls.

External programs are found from the environment, never from a hard-coded path, and their versions are checked before use
(the versions the paper's template was built with, CONVENTIONS.md "External tools"):

    FSLDIR      FSL 6.0.7.4             ($FSLDIR/etc/fslversion)
    MMORF       MMORF 1.0.1             (`mmorf --version`; set MMORF to the executable, default: `mmorf` on PATH)
    ANTSPATH    ANTs 2.5.4              (antsRegistration --version; default: ANTs on PATH)
    MRTRIX      MRtrix3 3.0.2           (directory holding dwigradcheck; default: on PATH; the tensor refit and mri_raw need it)
    JULIA       Julia 1.11.4            (the executable; default: `julia` on PATH; mri_raw's Bruker conversion and SANDI fit)
    TORTOISE    TORTOISE V4.1.0         (the folder holding TORTOISEProcess; default: on PATH; mri_raw's dt2 session; it prints
                                         its version when it runs, and mri_raw checks that line)

LINC_SKIP_VERSION_CHECK=1 turns a version mismatch into a warning (the result is then not the paper's).
"""
import functools
import os
import shutil
import subprocess
import warnings
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
RECIPE = REPO / "configs" / "mri_construction.yaml"


@functools.lru_cache(maxsize=None)
def recipe():
    """configs/mri_construction.yaml as a dict."""
    return yaml.safe_load(RECIPE.read_text())


def _mismatch(tool, want, got):
    msg = f"{tool}: version {got!r}, the paper used {want!r}"
    if os.environ.get("LINC_SKIP_VERSION_CHECK") == "1":
        warnings.warn(msg + " (LINC_SKIP_VERSION_CHECK=1: continuing)")
    else:
        raise RuntimeError(msg + " (set LINC_SKIP_VERSION_CHECK=1 to run anyway)")


def _which(name, env_dir=None):
    if env_dir and os.environ.get(env_dir):
        p = Path(os.environ[env_dir]) / name
        if p.exists():
            return str(p)
    p = shutil.which(name)
    if not p:
        raise RuntimeError(f"{name} not found (set {env_dir or 'PATH'}; see linc/construction/tools.py)")
    return p


@functools.lru_cache(maxsize=None)
def fsldir():
    d = os.environ.get("FSLDIR")
    if not d:
        raise RuntimeError("set FSLDIR (FSL 6.0.7.4, see CONVENTIONS.md)")
    want = recipe()["tools"]["fsl"]
    vf = Path(d) / "etc" / "fslversion"
    got = vf.read_text().split(":")[0].strip() if vf.exists() else "unknown"
    if got != want:
        _mismatch("FSL", want, got)
    return Path(d)


def fsl(tool):
    """Absolute path of an FSL program ($FSLDIR/bin/<tool>), after the FSL version check."""
    return str(fsldir() / "bin" / tool)


def fsl_env():
    d = fsldir()
    return dict(os.environ, FSLDIR=str(d), FSLOUTPUTTYPE="NIFTI_GZ", PATH=f"{d}/bin:" + os.environ.get("PATH", ""))


@functools.lru_cache(maxsize=None)
def ants(tool):
    """Absolute path of an ANTs program ($ANTSPATH or PATH), after the ANTs version check."""
    want = recipe()["tools"]["ants"]
    p = _which(tool, "ANTSPATH")
    reg = str(Path(p).with_name("antsRegistration"))
    out = subprocess.run([reg, "--version"], capture_output=True, text=True).stdout
    got = out.split("Version:")[1].split()[0] if "Version:" in out else "unknown"
    if got != want:
        _mismatch("ANTs", want, got)
    return p


@functools.lru_cache(maxsize=None)
def mmorf():
    """The MMORF executable ($MMORF or `mmorf` on PATH), after the version check. Only the construction driver runs it."""
    want = recipe()["tools"]["mmorf"]
    p = os.environ.get("MMORF") or _which("mmorf")
    out = subprocess.run([p, "--version"], capture_output=True, text=True)
    txt = out.stdout + out.stderr
    got = txt.split("version", 1)[1].split()[0].lstrip("v") if "version" in txt else "unknown"
    if got != want:
        _mismatch("MMORF", want, got)
    return p


@functools.lru_cache(maxsize=None)
def mrtrix(tool):
    """Absolute path of an MRtrix3 program ($MRTRIX or PATH), after the version check."""
    want = recipe()["tools"]["mrtrix"]
    p = _which(tool, "MRTRIX")
    out = subprocess.run([str(Path(p).with_name("mrinfo")), "-version"], capture_output=True, text=True).stdout
    got = out.split("mrinfo", 1)[1].split()[0] if "mrinfo" in out else "unknown"
    if not got.startswith(want):
        _mismatch("MRtrix3", want, got)
    return p


@functools.lru_cache(maxsize=None)
def julia():
    """The Julia executable ($JULIA or `julia` on PATH), after the version check."""
    want = recipe()["tools"]["julia"]
    p = os.environ.get("JULIA") or _which("julia")
    out = subprocess.run([p, "--version"], capture_output=True, text=True).stdout
    got = out.split("version", 1)[1].split()[0] if "version" in out else "unknown"
    if got != want:
        _mismatch("Julia", want, got)
    return p


def tortoise():
    """TORTOISEProcess ($TORTOISE or PATH). Its version is checked from the line it prints when it runs (check_tortoise)."""
    return _which("TORTOISEProcess", "TORTOISE")


def check_tortoise(log_text):
    want = recipe()["tools"]["tortoise"]
    line = next((l for l in log_text.splitlines() if l.startswith("TORTOISE version:")), "")
    got = line.split(":", 1)[1].strip().replace("TORTOISE ", "") if line else "unknown"
    if got != want:
        _mismatch("TORTOISE", want, got)
    return got


def run(cmd, env=None, log=None):
    """Run a command (list), raising on failure; with `log`, append the command and its output to that file."""
    cmd = [str(c) for c in cmd]
    if log is None:
        subprocess.run(cmd, check=True, env=env)
        return
    with open(log, "a") as fh:
        fh.write("$ " + " ".join(cmd) + "\n"); fh.flush()
        subprocess.run(cmd, check=True, env=env, stdout=fh, stderr=subprocess.STDOUT)


def sh(cmd):
    """Run a shell command line with the FSL environment; return stdout, raise with stderr on failure."""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=fsl_env())
    if r.returncode:
        raise RuntimeError(f"failed: {cmd}\n{r.stderr[-2000:]}")
    return r.stdout
