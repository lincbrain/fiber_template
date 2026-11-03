"""Direct placement checked against the route through the brain's own MRI (paper: Methods and Results, direct placement).

For each brain with MRI, its direct placement onto the MRI template's b0 (linc/registration/direct_placement.py) and the path through its
own MRI (blockface -> own b0, the registration of record (R1); then own b0 -> MRI template (R5)) both pull template points inside the
brain (the direct placement's moved mask, every 1 mm) back into blockface native space; their distance (mm) is the direct placement's
error relative to that path. Per brain: median / p95 / p99 / max mm, the linear scales and their anisotropy, the gates. The brains used
to choose the smoothing are summarized apart from the others (held out); the expected error for the brains without MRI comes from the
held-out brains only.

Ported unchanged from the analysis project's featurereg/d2_s2_validate.py (`one`, `summary`, `r5_guard`); folders are arguments.
"""
import json
import os

import numpy as np
import nibabel as nib

from linc import xfm
from linc.registration import xfmtools as X


def r5_guard(entry_dir, recipe):
    """4 Oct 2026: the R5 reruns overwrite the DEFAULT entries (data/sub-<S>/xfm), which until then mix old R5s and a stray; refuse an
    entry whose config.recipe is not the required one (e.g. "D22"), so no table or gate ever composes with a superseded R5."""
    if not recipe:
        return
    import ast
    m = json.load(open(f"{entry_dir}/meta.json")); c = m.get("config") or {}
    if isinstance(c, str):
        try:
            c = ast.literal_eval(c)
        except Exception:
            c = {}
    got = c.get("recipe") if isinstance(c, dict) else None
    if got != recipe:
        raise SystemExit(f"R5 entry {entry_dir}: config.recipe {got!r} != required {recipe!r} (not registered yet?): refused")


def one(S, d, r1_entry, r5_entry, r1_variant=None, r1_qc=None, recipe=""):
    """S2 against R1 then R5 for one brain with MRI: d = its direct-placement products (<ID>_d2s2.json, <ID>_linear_world.txt,
    <ID>_warp.nii.gz, <ID>_mask_in_template.nii.gz); r1_entry / r5_entry = transform folders (linc/xfm.py Entry)."""
    if not os.path.exists(f"{d}/{S}_d2s2.json"):
        return dict(subject=S, missing="no finished S2 product")
    info = json.load(open(f"{d}/{S}_d2s2.json"))
    M = np.loadtxt(f"{d}/{S}_linear_world.txt")
    df, da = X.read_fireants_field(f"{d}/{S}_warp.nii.gz")
    s2 = X.Map(L=np.linalg.inv(M), d=df, d_affine=da)
    r1 = xfm.Entry(r1_entry)
    r5 = xfm.Entry(r5_entry); r5_guard(r5.dir, recipe)
    mi = nib.load(f"{d}/{S}_mask_in_template.nii.gz"); m = np.asanyarray(mi.dataobj) > 0
    ijk = np.argwhere(m[::4, ::4, ::4]) * 4                                      # every 1 mm on the 0.25 mm grid
    P = ijk @ mi.affine[:3, :3].T + mi.affine[:3, 3]
    ys2 = s2(P)
    yref = r1.pull(r5.pull(P))
    dist = np.linalg.norm(ys2 - yref, axis=1)
    sc = info.get("linear_dof9", {}).get("scales") or []
    def fit_aniso(y):   # the affine that best maps the template points onto their pulls (least squares): max/min singular value
        A = np.linalg.lstsq(np.c_[P, np.ones(len(P))], y, rcond=None)[0][:3].T
        sv = np.linalg.svd(A, compute_uv=False); return round(float(sv.max() / sv.min()), 3)
    return dict(subject=S, n_points=int(len(P)), median_mm=round(float(np.median(dist)), 3), p95_mm=round(float(np.percentile(dist, 95)), 3), p99_mm=round(float(np.percentile(dist, 99)), 3),
                max_mm=round(float(dist.max()), 3), linear_used=info.get("linear_used"), scales_dof9=sc,
                anisotropy_ap_over_mean_lr_si=(round(sc[1] / ((sc[0] + sc[2]) / 2), 3) if len(sc) == 3 else None),
                anisotropy_regqc_max_over_min=(round(max(sc) / min(sc), 3) if len(sc) == 3 else None),   # the regqc gate's number
                anisotropy_fit_s2=fit_aniso(ys2), anisotropy_fit_r1_r5=fit_aniso(yref),                  # same measure on both paths
                outline_reasons=(info.get("outline") or {}).get("reasons"),
                gate_pass=info.get("gate_pass"), ncc=info.get("ncc"), outline=(info.get("outline") or {}).get("status"),
                r1_variant=r1_variant, r1_qc=r1_qc, r5_qc=r5.qc_status(), r5_entry=str(r5.dir))


def summary(rows):
    ok = [r for r in rows if "median_mm" in r]
    if not ok:
        return {}
    med = [r["median_mm"] for r in ok]; p95 = [r["p95_mm"] for r in ok]
    return dict(n=len(ok), median_of_medians_mm=round(float(np.median(med)), 3), max_median_mm=round(float(max(med)), 3),
                median_p95_mm=round(float(np.median(p95)), 3), max_p95_mm=round(float(max(p95)), 3))
