"""Voxelwise between-input variability of an MRI template variant (paper: Results "MRI template", the variability figure).

Per voxel, the mean over the eight inputs (four brains + mirrors) of each input's deviation from the variant's own reconstructed
template (the mean of the inputs):
  T2       mean_s |z_s - z_template|, z = z-score inside the core (offset and scale removed), for each input and for the template
  FRAC1/2  mean_s |f_s - f_template| in percentage points (soma / neurite fraction; equal, since soma + neurite = 1)
  DTI      mean_s angle(V1_s, V1_template) in degrees, V1_template = principal eigenvector of the log-Euclidean mean tensor
Inputs are the cached warped inputs of linc/evaluation/variants.py; maps are zero outside the core.
"""
from pathlib import Path

import nibabel as nib
import numpy as np

from .variants import EIG_FLOOR_REL, IDS, cached, expm, load, logm, mats

METRICS = [("T2", "T2 |Δz|"), ("FRAC1", "soma |Δ| (pp)"), ("FRAC2", "neurite |Δ| (pp)"), ("DTI", "V1 angle (°)")]


def principal(M):
    return np.linalg.eigh(M)[1][..., :, -1]


def arm_maps(cache, arm, core, ref, out_dir=None):
    """{metric: deviation map} for one variant inside `core` (boolean volume). `ref` is the template image whose affine the maps take;
    with `out_dir`, each map is also written as <out_dir>/<arm>__<metric>.nii.gz (float32)."""
    f = lambda m, s: cached(cache, arm, m, s)                                    # noqa: E731
    out = {}
    z = lambda v: (v - v[core].mean()) / v[core].std()                          # noqa: E731
    t2 = [z(load(f("T2", s))) for s in IDS]
    tz = z(np.mean([load(f("T2", s)) for s in IDS], 0))
    out["T2"] = np.mean([np.abs(v - tz) for v in t2], 0)
    for m in ("FRAC1", "FRAC2"):
        vs = [load(f(m, s)) for s in IDS]
        t = np.mean(vs, 0)
        out[m] = 100 * np.mean([np.abs(v - t) for v in vs], 0)
    Ts = [mats(load(f("DTI", s))[core]) for s in IDS]
    md = np.median([np.trace(M, axis1=1, axis2=2).mean() / 3 for M in Ts[:2]])
    vt = principal(expm(np.mean([logm(M, EIG_FLOOR_REL * md) for M in Ts], 0)))
    ang = np.mean([np.degrees(np.arccos(np.clip(np.abs(np.einsum("ni,ni->n", principal(M), vt)), 0, 1))) for M in Ts], 0)
    a = np.zeros(core.shape); a[core] = ang
    out["DTI"] = a
    for k in out:
        out[k] = np.where(core, out[k], 0.0)
        if out_dir is not None:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(out[k].astype(np.float32), ref.affine), str(Path(out_dir) / f"{arm}__{k}.nii.gz"))
    return out


def summary(maps, core, wm):
    """{metric: {"core_mean", "wm_mean"}} of one variant's maps."""
    return {k: {"core_mean": float(v[core].mean()), "wm_mean": float(v[wm].mean())} for k, v in maps.items()}
