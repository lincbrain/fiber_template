"""ANTs as an external program, version-checked (settings/link.yaml `atlas.ants.version`).

The executable is found on $PATH or in $ANTSPATH (the usual ANTs variable). Used only for the affine start of the atlas mappings
(analysis project: mri/code/preprocess_mri.py affine_init, ANTs 2.5.4).
"""
import os
import re
import shutil
import subprocess
from pathlib import Path


def executable(name="antsRegistration"):
    for d in ([os.environ["ANTSPATH"]] if os.environ.get("ANTSPATH") else []):
        p = Path(d) / name
        if p.exists():
            return str(p)
    p = shutil.which(name)
    if not p:
        raise RuntimeError(f"{name} not found: put ANTs on PATH or set ANTSPATH (see settings/env/README.txt)")
    return p


def version(exe=None):
    out = subprocess.run([exe or executable(), "--version"], capture_output=True, text=True).stdout
    m = re.search(r"ANTs Version:\s*v?([0-9][0-9.]*)", out)
    return m.group(1) if m else None


def check(required=None, exe=None):
    """Raise unless the ANTs found is the version of record (set LINC_ANTS_ANY_VERSION=1 to only warn)."""
    from template.link.common import config
    required = required or config()["atlas"]["ants"]["version"]
    v = version(exe)
    if v != required:
        msg = f"ANTs {v} found ({exe or executable()}), the paper used {required}"
        if os.environ.get("LINC_ANTS_ANY_VERSION"):
            print(f"[ants] WARNING: {msg}", flush=True)
        else:
            raise RuntimeError(msg + " (set LINC_ANTS_ANY_VERSION=1 to run anyway)")
    return v


def affine_init(fixed, moving, fmask, mmask, prefix):
    """Rigid + affine ITK transform moving -> fixed (mutual information, masks on both sides, centre-of-mass start).
    Writes <prefix>0GenericAffine.mat and <prefix>warped.nii.gz; returns the .mat path."""
    from template.link.common import config
    c = config()["atlas"]["ants"]
    exe = executable(); check(exe=exe)
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    s = c["schedule"]
    sched = ["-c", s["c"], "-f", s["f"], "-s", s["s"]]
    metric = c["metric"].format(fixed=fixed, moving=moving)
    cmd = [exe, "-d", "3", "--float", "0", "-v", "0",
           "-o", f"[{prefix},{prefix}warped.nii.gz]",
           "--interpolation", "Linear",
           "--winsorize-image-intensities", c["winsorize"],
           "--use-histogram-matching", str(c["histogram_matching"]),
           "--initial-moving-transform", f"[{fixed},{moving},1]",
           "-x", f"[{fmask},{mmask}]",
           "-t", f"Rigid[{c['step']}]", "-m", metric, *sched,
           "-t", f"Affine[{c['step']}]", "-m", metric, *sched]
    subprocess.run(cmd, check=True)
    mat = f"{prefix}0GenericAffine.mat"
    if not os.path.exists(mat):
        raise RuntimeError(f"affine init failed, no {mat}")
    return mat
