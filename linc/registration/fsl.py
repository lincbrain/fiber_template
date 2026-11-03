"""FSL FLIRT, called as an external program, and the conversions between FLIRT matrices and world transforms.

FSL is not redistributed: set FSLDIR to an FSL 6.0.7.4 installation (the version used for the paper; `check_version` warns on any
other). World transforms are 4 x 4 RAS matrices, moving world -> fixed world (mm).

Ported from the analysis project: `fsl_to_world` = code/register_mri.py `_fsl_to_world` (= code/regqc.py `fsl_to_world`),
`world_to_fsl` = featurereg/r1_redo_v3.py `fsl_mat_from_world` (= bftemplate/direct/register_direct.py `_world_to_fsl`),
`decompose` and `euler_xyz` = code/regqc.py; unchanged.
"""
import os
import subprocess
import warnings

import numpy as np

FSL_VERSION = "6.0.7.4"


def fsldir():
    d = os.environ.get("FSLDIR")
    if not d:
        raise RuntimeError("set FSLDIR to an FSL installation (the paper used FSL 6.0.7.4)")
    return d


def check_version():
    """The installed FSL's version string; warns when it is not the paper's."""
    p = os.path.join(fsldir(), "etc", "fslversion")
    v = open(p).read().split(":")[0].strip() if os.path.exists(p) else "unknown"
    if v != FSL_VERSION:
        warnings.warn(f"FSL {v} at {fsldir()}; the paper used FSL {FSL_VERSION}")
    return v


def run(*cmd):
    """Run an FSL program (flirt, ...) with FSLDIR's binaries first on PATH and NIfTI-gz output; returns stdout."""
    d = fsldir()
    env = dict(os.environ, FSLDIR=d, FSLOUTPUTTYPE="NIFTI_GZ", PATH=f"{d}/share/fsl/bin:{d}/bin:" + os.environ.get("PATH", ""))
    r = subprocess.run([str(c) for c in cmd], env=env, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{' '.join(map(str, cmd))}:\n{(r.stdout + r.stderr)[-1500:]}")
    return r.stdout


def _fsl_scaled_voxel(img):
    """FSL's 'scaled voxel' coordinates of an image (x flipped when the sform determinant is positive)."""
    z = img.header.get_zooms()[:3]; n = img.shape[:3]
    S = np.diag([z[0], z[1], z[2], 1.0])
    if np.linalg.det(img.affine[:3, :3]) > 0:
        S[0, 0] = -z[0]; S[0, 3] = (n[0] - 1) * z[0]
    return S


def fsl_to_world(mat, mov, fix):
    """RAS -> RAS transform (moving world -> fixed world) implied by a FLIRT `-omat` matrix for (moving, fixed) images."""
    return fix.affine @ np.linalg.inv(_fsl_scaled_voxel(fix)) @ mat @ _fsl_scaled_voxel(mov) @ np.linalg.inv(mov.affine)


def world_to_fsl(T, mov, fix):
    """The FLIRT matrix (an -init) that realises the world transform T (moving world -> fixed world); inverse of fsl_to_world."""
    return _fsl_scaled_voxel(fix) @ np.linalg.inv(fix.affine) @ T @ mov.affine @ np.linalg.inv(_fsl_scaled_voxel(mov))


def euler_xyz(R):
    """R = Rz @ Ry @ Rx; returns (rx, ry, rz) in degrees."""
    ry = np.degrees(np.arcsin(np.clip(-R[2, 0], -1, 1)))
    rx = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    rz = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return float(rx), float(ry), float(rz)


def decompose(W, allow_180=False):
    """Polar decomposition of the 3x3 linear part: nearest proper rotation + principal scales. With allow_180, the 180-degree rotation
    about x, y or z that leaves the smallest residual rotation is absorbed (and named) before the angles are reported."""
    A = np.asarray(W, float)[:3, :3]
    U, s, Vt = np.linalg.svd(A)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    stretch = R.T @ A
    out = dict(scales=[float(x) for x in np.sort(np.linalg.eigvalsh((stretch + stretch.T) / 2))],
               det=float(np.linalg.det(A)), reflection=bool(np.linalg.det(A) < 0))
    best = ("none", R)
    if allow_180:
        cands = {"none": np.eye(3), "x180": np.diag([1., -1, -1]), "y180": np.diag([-1., 1, -1]),
                 "z180": np.diag([-1., -1, 1])}
        ang = lambda M: np.degrees(np.arccos(np.clip((np.trace(M) - 1) / 2, -1, 1)))  # noqa: E731
        best = min(((k, R @ F) for k, F in cands.items()), key=lambda kv: ang(kv[1]))
    sc = np.asarray(out["scales"], float)
    out["anisotropy"] = float(sc.max() / sc.min()); out["iso_scale"] = float(np.prod(sc) ** (1 / 3))
    out["absorbed_180"] = best[0]
    out["rot_xyz_deg"] = list(euler_xyz(best[1]))
    out["rot_total_deg"] = float(np.degrees(np.arccos(np.clip((np.trace(best[1]) - 1) / 2, -1, 1))))
    out["translation_mm"] = [float(x) for x in np.asarray(W, float)[:3, 3]]
    return out
