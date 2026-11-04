"""The canonical frame of the MRI template: every brain placed rigidly in the NMT v2 stereotaxic frame, on one grid where the left-right
mirror is exact (paper: Methods, "Symmetric groupwise construction").

Per brain, one rigid map T (subject world RAS -> canonical world RAS), `write_frames_nmt`:
  x  the brain's own mid-sagittal plane, fitted with a free orientation by maximizing the in-mask correlation of the b0 with its
     reflection (`symmetry_plane`); the plane lands on world x = 0, so the mirror stays exact;
  y  NMT's +y pulled into subject space through the ROTATION of the subject -> NMT affine (polar decomposition: scale and shear are
     dropped, ex-vivo brains keep their size);
  t  the AC lands on NMT's own AC in y and z; x is set by the plane.
The subject -> NMT affine and the AC/PC come from registering NMT_v2.0_sym to the brain's b0 and carrying the SARM level-6 labels
(`landmarks_from_labels` takes the carried label volume; the registration itself is linc.registration's).

The canonical grid (configs/mri_construction.yaml `canonical`) is NMT_v2.0_sym's own lattice shifted by whole voxels, 256 x 320 x 192 at
0.25 mm, with the flip axis (nx - 1) / 2 = 127.5 at world x = 0: np.flip(axis=0) is the anatomical mirror, with no interpolation.
Mirroring a tensor negates its xy and xz components (FSL order xx, xy, xz, yy, yz, zz).

Ported from the analysis project's mri/code/canonical.py (final path: NMT frame, `write_frames_nmt`); the AC-PC frame it superseded is
not ported.
"""
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from .tools import recipe


def _c():
    return recipe()["canonical"]


def box():
    return tuple(int(x) for x in _c()["box"])


# --------------------------------------------------------------------------- grid

def canonical_affine():
    """Affine of the canonical grid: RAS, isotropic, NMT-anchored origin."""
    c = _c()
    A = np.eye(4)
    A[0, 0] = A[1, 1] = A[2, 2] = float(c["vx_mm"])
    A[:3, 3] = np.asarray(c["origin_mm"], float)
    return A


def canonical_ref(path):
    """Write an empty volume on the canonical grid (the -r reference of every resampling into it)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.zeros(box(), np.float32), canonical_affine()), str(path))
    return str(path)


def canon_ac_voxel():
    """Voxel index of the canonical AC; x comes out at exactly the flip axis (nx - 1) / 2."""
    c = _c()
    return tuple((np.asarray(c["ac_target_mm"], float) - np.asarray(c["origin_mm"], float)) / float(c["vx_mm"]))


def itk_from_ras(T, dst):
    """Write T (RAS, subject -> canonical) as an ITK .mat for antsApplyTransforms: ITK stores the fixed (canonical) -> moving
    (subject) point map in LPS, i.e. inv(T) conjugated by D = diag(-1, -1, 1)."""
    from scipy.io import savemat
    D = np.diag([-1.0, -1.0, 1.0])
    Ti = np.linalg.inv(np.asarray(T, float))
    A = D @ Ti[:3, :3] @ D
    t = D @ Ti[:3, 3]
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    savemat(str(dst), {"AffineTransform_double_3_3": np.r_[A.ravel(), t].reshape(12, 1),
                       "fixed": np.zeros((3, 1))}, format="4")
    return str(dst)


def fsl_mm(img):
    """FSL's scaled-voxel coordinates of an image (x flipped when the voxel -> world determinant is positive)."""
    z = img.header.get_zooms()[:3]
    Fm = np.diag([z[0], z[1], z[2], 1.0])
    if np.linalg.det(img.affine[:3, :3]) > 0:
        Fm[0, 0], Fm[0, 3] = -z[0], (img.shape[0] - 1) * z[0]
    return Fm


def flirt_matrix(T, native, ref):
    """T (RAS, subject -> canonical) as a FLIRT matrix between the native image and the canonical grid (for vecreg -t):
    M = F_ref A_ref^-1 T A_native F_native^-1."""
    return fsl_mm(ref) @ np.linalg.inv(ref.affine) @ np.asarray(T, float) @ native.affine @ np.linalg.inv(fsl_mm(native))


# --------------------------------------------------------------------------- mirror

MIRROR_RULES = {"scalar": None, "vector": [0], "tensor": [1, 2]}    # components negated under R = diag(-1, 1, 1)
_INTENT_KIND = {1007: "vector", 1005: "tensor", 1004: "vector"}


def channel_kind(arr, intent=None):
    """scalar / vector / tensor; refuses to guess when the layout is ambiguous (stacked label maps, ITK 5-D warps)."""
    if intent is not None and int(intent) in _INTENT_KIND:
        return _INTENT_KIND[int(intent)]
    if arr.ndim == 3:
        return "scalar"
    if arr.ndim == 4 and arr.shape[-1] in (3, 6) and np.issubdtype(arr.dtype, np.floating):
        return "vector" if arr.shape[-1] == 3 else "tensor"
    raise ValueError(f"cannot infer channel kind for shape {arr.shape} dtype {arr.dtype}: pass kind= explicitly")


def mirror_array(arr, kind=None):
    """Exact left-right mirror on the canonical grid, with the channel's reorientation (vectors: -x; tensors: -xy, -xz)."""
    kind = kind or channel_kind(arr)
    comps = MIRROR_RULES.get(kind)
    if comps and arr.ndim < 4:
        raise ValueError(f"kind={kind!r} needs a channel axis, got a {arr.ndim}-D array")
    if comps and arr.shape[-1] <= max(comps):
        raise ValueError(f"kind={kind!r} needs >{max(comps)} components, got {arr.shape[-1]}")
    out = np.flip(arr, axis=0).copy()
    if comps:
        out = out.astype(np.result_type(out.dtype, np.float32), copy=False)
        for c in comps:
            out[..., c] *= -1.0
    return out


def mirror_volume(src, dst, kind=None):
    n = nib.load(str(src))
    arr = np.asanyarray(n.dataobj)
    out = mirror_array(arr, kind or channel_kind(arr, n.header.get("intent_code")))
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(out, n.affine, n.header), str(dst))
    return dst


def renormalise_compositional(out, names=("frac1.nii.gz", "frac2.nii.gz"), mask_name="micro_mask.nii.gz"):
    """Restore sum-to-1 of the SANDI fractions after resampling (a boundary correction: the interior sums to 1 already). Voxels whose
    sum is at or below the floor are set to 0."""
    floor = float(_c()["compositional_floor"])
    out = Path(out)
    paths = [out / n for n in names]
    if not all(p.exists() for p in paths):
        return out
    imgs = [nib.load(str(p)) for p in paths]
    arrs = [np.asanyarray(i.dataobj, dtype=np.float64) for i in imgs]
    m = np.asanyarray(nib.load(str(out / mask_name)).dataobj) > 0.5
    tot = sum(arrs)
    ok = m & (tot > floor)
    for a in arrs:
        a[ok] /= tot[ok]
        a[~ok] = 0.0
    for a, i, p in zip(arrs, imgs, paths):
        nib.save(nib.Nifti1Image(a.astype(np.float32), i.affine, i.header), str(p))
    return out


def extract(img, m):
    """Brain extraction + normalization of an anatomical image on the canonical grid: clip to the 1-99 percentiles inside the mask,
    rescale to (norm_eps, norm_max], 0 outside."""
    c = _c()
    nmax, eps = float(c["norm_max"]), float(c["norm_eps"])
    d = np.asanyarray(img.dataobj, dtype=np.float64)
    vmin, vmax = np.percentile(d[m], 1), np.percentile(d[m], 99)
    d = np.clip((d - vmin) / (vmax - vmin), 0, 1) * (nmax - eps) + eps
    return np.where(m, d, 0.0).astype(np.float32)


# --------------------------------------------------------------------------- landmarks and the mid-sagittal plane

def mid_sagittal_plane(img_path, mask_path):
    """Voxel-axis mid-sagittal column x = c (half-voxel steps) maximizing the in-mask correlation of the image with its reflection.
    Returns (c, r); only used to restrict the AC/PC centroids to the midline."""
    d = np.asanyarray(nib.load(str(img_path)).dataobj).astype(np.float32)
    m = np.asanyarray(nib.load(str(mask_path)).dataobj) > 0
    nx = d.shape[0]
    idx = np.argwhere(m)[:, 0]
    c0 = (idx.min() + idx.max()) / 2.0
    best = None
    for c2 in range(int((c0 - 6) * 2), int((c0 + 6) * 2) + 1):
        c = c2 / 2.0
        sh = int(round(2 * c)) - (nx - 1)
        f = np.roll(np.flip(d, axis=0), sh, axis=0)
        mm = m & np.roll(np.flip(m, axis=0), sh, axis=0)
        if mm.sum() < 5000:
            continue
        a, b = d[mm], f[mm]
        if a.std() == 0 or b.std() == 0:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if best is None or r > best[1]:
            best = (c, r)
    return best


def landmarks_from_labels(sarm_in_subject, b0_n4_brain, dwi_mask, out_json=None, subject=None):
    """AC and PC in subject world coordinates: centroids of the SARM level-6 commissure labels carried onto the b0 grid (nearest
    neighbor), restricted to the voxels near the mid-sagittal column. Writes landmarks.json when out_json is given."""
    c = _c()
    lab = nib.load(str(sarm_in_subject))
    d = np.asanyarray(lab.dataobj)
    plane = mid_sagittal_plane(b0_n4_brain, dwi_mask)
    res = {"subject": subject, "space": "dwi 0.25 mm", "mid_sagittal_plane_voxel_x": plane[0], "mid_sagittal_mirror_r": plane[1],
           "landmarks": {}}
    for name, val in c["sarm_labels"].items():
        idx = np.argwhere(d == val)
        if len(idx) == 0:
            res["landmarks"][name] = None
            continue
        mid = idx[np.abs(idx[:, 0] - plane[0]) <= c["landmark_midline_vox"]]
        use = mid if len(mid) >= c["landmark_midline_min_vox"] else idx
        com = use.mean(0)
        w = (lab.affine @ np.append(com, 1.0))[:3]
        res["landmarks"][name] = {"voxel": com.tolist(), "world": w.tolist(),
                                  "n_voxels_total": int(len(idx)), "n_voxels_midline": int(len(mid))}
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(res, indent=1))
    return res


def _reflect_matrix(n, d):
    """4 x 4 world reflection about the plane {x : n.x = d}."""
    n = np.asarray(n, float); n = n / np.linalg.norm(n)
    M = np.eye(4)
    M[:3, :3] = np.eye(3) - 2.0 * np.outer(n, n)
    M[:3, 3] = 2.0 * d * n
    return M


def symmetry_plane(img_path, mask_path, n0, d0):
    """The mid-sagittal plane with a free orientation: maximize the in-mask correlation of the image with its reflection about the
    candidate plane (normal angles theta, phi and offset d), on a copy resampled to plane_step_mm. Nelder-Mead is restarted from its
    own optimum until the score stops improving (a single pass stops at non-stationary points). n0, d0 only start the search."""
    from scipy.ndimage import affine_transform, zoom
    from scipy.optimize import minimize
    c = _c()
    step_mm, max_restarts, tol = float(c["plane_step_mm"]), int(c["plane_max_restarts"]), float(c["plane_tol"])
    n = nib.load(str(img_path))
    img = np.asanyarray(n.dataobj).astype(np.float32)
    msk = (np.asanyarray(nib.load(str(mask_path)).dataobj) > 0)
    zoomf = float(n.header.get_zooms()[0]) / step_mm
    if abs(zoomf - 1.0) > 1e-3:
        img = zoom(img, zoomf, order=1)
        msk = zoom(msk.astype(np.float32), zoomf, order=0) > 0.5
    A = n.affine.copy()
    A[:3, :3] = A[:3, :3] / zoomf
    Ainv = np.linalg.inv(A)

    def score(par):
        theta, phi, d = par
        nn = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])
        V = Ainv @ _reflect_matrix(nn, d) @ A
        ref = affine_transform(img, V[:3, :3], offset=V[:3, 3], order=1, output_shape=img.shape)
        rm = affine_transform(msk.astype(np.float32), V[:3, :3], offset=V[:3, 3], order=0, output_shape=img.shape) > 0.5
        mm = msk & rm
        if mm.sum() < 5000:
            return 1.0
        a, b = img[mm], ref[mm]
        if a.std() == 0 or b.std() == 0:
            return 1.0
        return -float(np.corrcoef(a, b)[0, 1])

    n0 = np.asarray(n0, float); n0 /= np.linalg.norm(n0)
    p0 = [np.arctan2(n0[1], n0[0]), np.arcsin(np.clip(n0[2], -1, 1)), float(d0)]
    p, best, n_it = list(p0), None, 0
    for n_it in range(1, max_restarts + 1):
        r = minimize(score, p, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 400})
        p = list(r.x)
        if best is not None and (-r.fun) - best < tol:
            best = max(best, -r.fun)
            break
        best = -r.fun if best is None else max(best, -r.fun)
    th, ph, d = p
    nn = np.array([np.cos(th) * np.cos(ph), np.sin(th) * np.cos(ph), np.sin(ph)])
    return {"normal": nn.tolist(), "offset": float(d), "mirror_r": float(best), "init_mirror_r": float(-score(p0)),
            "n_restarts": int(n_it), "converged": bool(n_it < max_restarts)}


def _affine_ras(affine_mat):
    """The 4 x 4 subject -> NMT affine in RAS from an ITK/LPS .mat (fixed = subject, moving = NMT): y = A(x - c) + c + t."""
    from scipy.io import loadmat
    m = loadmat(str(affine_mat))
    k = [x for x in m if x.startswith("AffineTransform")][0]
    v = m[k].ravel()
    A, t, c = v[:9].reshape(3, 3), v[9:12], m["fixed"].ravel()
    D = np.diag([-1.0, -1.0, 1.0])
    M = np.eye(4)
    M[:3, :3] = D @ A @ D
    M[:3, 3] = D @ (A @ (-c) + c + t)
    return M


def plane_from_nmt_affine(affine_mat, anchor=None):
    """NMT's mid-sagittal plane (world x = 0) pulled into subject world (inverse of the subject -> NMT affine), fitted by SVD over a
    grid of plane points; with `anchor` the offset is re-anchored to pass through it. Starting point of `symmetry_plane`."""
    nmt_to_subject = np.linalg.inv(_affine_ras(affine_mat))
    g = np.linspace(-25, 25, 11)
    pts = np.array([(nmt_to_subject @ np.array([0.0, y, z, 1.0]))[:3] for y in g for z in g])
    ctr = pts.mean(0)
    _, _, Vt = np.linalg.svd(pts - ctr)
    n = Vt[-1]
    if n[0] < 0:
        n = -n
    d0 = float(n @ ctr)
    if anchor is not None:
        d0 = float(n @ np.asarray(anchor, float))
    return n.tolist(), d0


def nmt_rotation(affine_mat):
    """Rotation (polar part) of the subject -> NMT affine, RAS."""
    from scipy.linalg import polar
    R, _ = polar(_affine_ras(affine_mat)[:3, :3])
    if np.linalg.det(R) < 0:
        raise ValueError(f"{affine_mat}: reflection in the subject->NMT rotation (det < 0)")
    return R


def canonical_frame_nmt(R_nmt, normal, anchor_subject, anchor_canonical):
    """Rigid subject -> canonical: x = the brain's own mid-sagittal normal, y = NMT's +y pulled into subject space and made
    orthogonal to x, z = x cross y; translation puts anchor_subject at anchor_canonical."""
    nx = np.asarray(normal, float); nx /= np.linalg.norm(nx)
    if nx[0] < 0:
        nx = -nx
    y = np.asarray(R_nmt, float).T @ np.array([0.0, 1.0, 0.0])
    y = y - np.dot(y, nx) * nx
    y /= np.linalg.norm(y)
    z = np.cross(nx, y); z /= np.linalg.norm(z)
    R = np.vstack([nx, y, z])
    if np.linalg.det(R) < 0:
        raise ValueError("canonical_frame_nmt produced a left-handed frame")
    T = np.eye(4); T[:3, :3] = R
    T[:3, 3] = np.asarray(anchor_canonical, float) - R @ np.asarray(anchor_subject, float)
    return T


def write_frames_nmt(b0_n4_brain, dwi_mask, landmarks_json, nmt_affine, out_json, subject=None, refine=True):
    """The brain's rigid map into the canonical frame -> canonical.json (T_subject_to_canonical, RAS 4 x 4).

    b0_n4_brain, dwi_mask: the brain's b0 (N4, brain-extracted) and its dwi brain mask; landmarks_json: AC/PC (`landmarks_from_labels`);
    nmt_affine: the subject -> NMT ITK affine (fixed = subject, moving = NMT)."""
    c = _c()
    lm = json.loads(Path(landmarks_json).read_text())
    ac = np.array(lm["landmarks"]["AC"]["world"])
    pc = np.array(lm["landmarks"]["PC"]["world"])
    n0, d0 = plane_from_nmt_affine(nmt_affine, anchor=(ac + pc) / 2.0)
    plane = {"normal": list(n0), "offset": float(d0), "source": "atlas-affine", "mirror_r": None}
    if refine:
        r = symmetry_plane(b0_n4_brain, dwi_mask, n0, d0)
        nn = np.array(r["normal"])
        dev = max(abs(float(nn @ ac) - r["offset"]), abs(float(nn @ pc) - r["offset"]))
        if dev <= float(c["plane_max_landmark_dev_mm"]):
            plane = {**r, "source": "symmetry-refined", "max_landmark_dev_mm": dev}
        else:
            plane["rejected_refit"] = {**r, "max_landmark_dev_mm": dev}
    R_nmt = nmt_rotation(nmt_affine)
    T = canonical_frame_nmt(R_nmt, plane["normal"], ac, c["ac_target_mm"])
    p_on_plane = np.array(plane["normal"], float) * plane["offset"]            # the plane must land on world x = 0
    T[0, 3] -= float((T @ np.append(p_on_plane, 1.0))[0])
    ac_c = (T @ np.append(ac, 1.0))[:3]
    pc_c = (T @ np.append(pc, 1.0))[:3]
    out = {"subject": subject, "frame": "NMT_v2.0_sym stereotaxic",
           "AC_world": ac.tolist(), "PC_world": pc.tolist(), "acpc_mm": float(np.linalg.norm(ac - pc)),
           "mid_sagittal_plane": plane, "T_subject_to_canonical": T.tolist(),
           "AC_canonical": ac_c.tolist(), "PC_canonical": pc_c.tolist(),
           "AC_x_residual_mm": float(abs(ac_c[0])), "PC_x_residual_mm": float(abs(pc_c[0])),
           "rotation_deg": float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1))))}
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(out, indent=1))
    return out


# --------------------------------------------------------------------------- integer crop (no resampling)

def box_origin(mask_path, box_vox):
    """Integer start index centering `box_vox` on the mask's bounding box; (start, (lo, hi), fits)."""
    d = np.asanyarray(nib.load(str(mask_path)).dataobj) > 0
    idx = np.argwhere(d)
    lo, hi = idx.min(0), idx.max(0) + 1
    centre = (lo + hi) / 2.0
    start = np.rint(centre - np.asarray(box_vox) / 2.0).astype(int)
    return start, (lo, hi), bool(((hi - lo) <= np.asarray(box_vox)).all())


def crop_to_box(src, dst, start, box_vox, dtype=None):
    """Integer crop/pad onto `box_vox` from voxel `start`, origin shifted to match: world coordinates are preserved exactly."""
    n = nib.load(str(src))
    arr = np.asanyarray(n.dataobj)
    bx, start = np.asarray(box_vox), np.asarray(start)
    out = np.zeros(tuple(bx) + arr.shape[3:], dtype=arr.dtype if dtype is None else dtype)
    src_lo = np.maximum(start, 0)
    src_hi = np.minimum(start + bx, arr.shape[:3])
    if (src_hi <= src_lo).any():
        raise ValueError(f"box does not intersect {src}")
    dst_lo = src_lo - start
    dst_hi = dst_lo + (src_hi - src_lo)
    out[dst_lo[0]:dst_hi[0], dst_lo[1]:dst_hi[1], dst_lo[2]:dst_hi[2]] = \
        arr[src_lo[0]:src_hi[0], src_lo[1]:src_hi[1], src_lo[2]:src_hi[2]]
    aff = n.affine.copy()
    aff[:3, 3] = n.affine[:3, :3] @ start + n.affine[:3, 3]
    hdr = n.header.copy()
    hdr.set_data_shape(out.shape)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(out, aff, hdr)
    img.set_sform(aff, code=int(n.header["sform_code"]) or 1)
    img.set_qform(aff, code=int(n.header["qform_code"]) or 1)
    nib.save(img, str(dst))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lm = sub.add_parser("landmarks", help="AC/PC from SARM labels carried onto the b0 grid -> landmarks.json")
    lm.add_argument("--sarm", required=True); lm.add_argument("--b0", required=True); lm.add_argument("--mask", required=True)
    lm.add_argument("--out", required=True); lm.add_argument("--subject")
    fr = sub.add_parser("frames", help="rigid subject -> canonical (NMT frame) -> canonical.json")
    fr.add_argument("--b0", required=True, help="dwi/b0_brain_n4"); fr.add_argument("--mask", required=True, help="masks/dwi_brain_mask")
    fr.add_argument("--landmarks", required=True); fr.add_argument("--nmt-affine", required=True, help="subject -> NMT ITK .mat")
    fr.add_argument("--out", required=True); fr.add_argument("--subject"); fr.add_argument("--no-refine", dest="refine", action="store_false")
    a = ap.parse_args(argv)
    if a.cmd == "landmarks":
        print(json.dumps(landmarks_from_labels(a.sarm, a.b0, a.mask, a.out, a.subject), indent=1))
    else:
        print(json.dumps(write_frames_nmt(a.b0, a.mask, a.landmarks, a.nmt_affine, a.out, a.subject, a.refine), indent=1))


if __name__ == "__main__":
    main()
