#!/usr/bin/env python3
"""Pull maps between world spaces, composed and applied with one interpolation; linear registration with symmetry-preserving
parametrisations.

A transform from space SRC to space TGT is stored as a PULL map evaluated on the TGT grid,

        src_world = L @ (x + d(x))          x = TGT world point (RAS mm)

with L a 4x4 RAS world matrix and d a displacement field in RAS mm sampled on a named TGT grid (d may be absent). Chains are
composed as world maps and applied ONCE (one interpolation of the source image, linear or nearest). FireANTs/ANTs fields are x/y
LPS-signed on disk: `read_fireants_field` converts them to RAS on load.

On disk (`save_map` / `load_map`): <name>_L.txt (4x4 RAS world), optional <name>_d.nii.gz (RAS mm on its grid) and maps.json.

Linear registration with SYMMETRY-PRESERVING parametrisations (for template<->template steps whose two frames both have their
mid-sagittal plane at x = 0):
    'pitch_yz'   rotation about x + (ty, tz)                                   3 dof
    'sym_affine' diag(sx) (+) a general 2x2 on (y, z) + (ty, tz)               7 dof
                 -- the most general affine that commutes with the x-flip
plus ordinary 'rigid' (6), 'similarity' (7), 'affine' (12). Cost = NCC (sign selectable) on masked, downsampled images, Powell.

Ported unchanged from the analysis project's featurereg/xfmtools.py (only this docstring is new).
"""
import json
import os

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from scipy import optimize

LPS = np.array([-1.0, -1.0, 1.0], np.float32)


# ------------------------------------------------------------------------------------------ fields
def read_fireants_field(path):
    """FireANTs saved warp (X,Y,Z,1,3) or (X,Y,Z,3), x/y LPS-signed -> RAS displacement (X,Y,Z,3), affine."""
    img = nib.load(path)
    W = np.squeeze(np.asanyarray(img.dataobj)).astype(np.float32)
    return W * LPS, img.affine


def grid_world(shape, affine, z0=0, z1=None):
    z1 = shape[2] if z1 is None else z1
    ii, jj, kk = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(z0, z1), indexing="ij")
    P = np.stack([ii, jj, kk, np.ones_like(ii)], -1).reshape(-1, 4).astype(np.float64)
    return (P @ affine.T)[:, :3]


def sample(vol, affine, pts, order=1, cval=0.0):
    """Sample a 3D (or 3D x C) volume at world points pts (N,3)."""
    inv = np.linalg.inv(affine)
    v = pts @ inv[:3, :3].T + inv[:3, 3]
    if vol.ndim == 3:
        return ndi.map_coordinates(vol, v.T, order=order, mode="constant", cval=cval)
    return np.stack([ndi.map_coordinates(vol[..., c], v.T, order=order, mode="constant", cval=cval)
                     for c in range(vol.shape[-1])], -1)


class Map:
    """src_world = L @ (x + d(x)); d is (field RAS mm, field_affine) or None."""

    def __init__(self, L=None, d=None, d_affine=None, name=""):
        self.L = np.eye(4) if L is None else np.asarray(L, float)
        self.d, self.da, self.name = d, d_affine, name

    def __call__(self, x):
        y = x
        if self.d is not None:
            y = x + sample(self.d, self.da, x, order=1)
        return y @ self.L[:3, :3].T + self.L[:3, 3]


def chain(*maps):
    """chain(m_tgt, m_mid, ...) -> function x_tgt -> x_src applying m_tgt first (tgt->mid), then m_mid ..."""
    def f(x):
        for m in maps:
            x = m(x)
        return x
    return f


def apply_once(src_path_or_arr, src_affine, tgt_shape, tgt_affine, world_map, order=1, chunk=16, cval=0.0):
    """Pull a source image onto a target grid through a composed world map: ONE interpolation."""
    if isinstance(src_path_or_arr, str):
        im = nib.load(src_path_or_arr)
        src = np.asanyarray(im.dataobj).astype(np.float32)
        src_affine = im.affine
    else:
        src = src_path_or_arr
    src = np.squeeze(src)
    out = np.zeros(tuple(tgt_shape[:3]) + src.shape[3:], np.float32)
    for z0 in range(0, tgt_shape[2], chunk):
        z1 = min(tgt_shape[2], z0 + chunk)
        pts = world_map(grid_world(tgt_shape, tgt_affine, z0, z1))
        v = sample(src, src_affine, pts, order=order, cval=cval)
        out[:, :, z0:z1] = v.reshape((tgt_shape[0], tgt_shape[1], z1 - z0) + src.shape[3:])
    return out


def invert_field(d, d_affine, L, out_shape, out_affine, iters=30):
    """Inverse of the map  src = L(x + d(x))  (x on the TGT side), sampled on a SRC-side grid (out_*).
    For y on the out grid, x solves L(x + d(x)) = y  <=>  x = z - d(x) with z = inv(L) y (fixed-point
    iteration). Returned in the same convention: Map(L'=inv(L), d') with x = L'(y + d'(y)),
    d' = L_lin (x - z)."""
    Li = np.linalg.inv(L)
    y = grid_world(out_shape, out_affine)
    z = y @ Li[:3, :3].T + Li[:3, 3]
    x = z.copy()
    for _ in range(iters):
        x = z - sample(d, d_affine, x, order=1)
    e = (x - z).reshape(tuple(out_shape[:3]) + (3,))
    dp = (e.reshape(-1, 3) @ np.asarray(L, float)[:3, :3].T).reshape(e.shape).astype(np.float32)
    return Map(L=Li, d=dp, d_affine=out_affine, name="inverse")


def jacobian_det(d, vox):
    """det J of x -> x + d(x) for a RAS displacement field."""
    J = np.empty(d.shape[:3] + (3, 3), np.float32)
    for i in range(3):
        g = np.gradient(d[..., i], *vox)
        for j in range(3):
            J[..., i, j] = g[j]
        J[..., i, i] += 1
    return np.linalg.det(J)


# ------------------------------------------------------------------------------------------ linear
def _rot(ax, th):
    c, s = np.cos(th), np.sin(th)
    R = np.eye(3)
    i, j = [(1, 2), (0, 2), (0, 1)][ax]
    R[i, i] = c; R[j, j] = c; R[i, j] = -s; R[j, i] = s
    if ax == 1:
        R = R.T                                   # right-handed about y
    return R


def params_to_matrix(p, kind, centre):
    """World matrix (moving -> fixed) about `centre` (fixed world) for each parametrisation."""
    M = np.eye(4)
    if kind == "pitch_yz":
        A = _rot(0, p[0]); t = np.array([0.0, p[1], p[2]])
    elif kind == "sym_affine":
        A = np.array([[p[0], 0, 0], [0, p[1], p[2]], [0, p[3], p[4]]]); t = np.array([0.0, p[5], p[6]])
    elif kind in ("rigid", "similarity"):
        A = _rot(2, p[2]) @ _rot(1, p[1]) @ _rot(0, p[0]); t = np.array(p[3:6])
        if kind == "similarity":
            A = A * p[6]
    elif kind == "affine":
        A = np.array(p[:9]).reshape(3, 3); t = np.array(p[9:12])
    else:
        raise ValueError(kind)
    M[:3, :3] = A
    M[:3, 3] = t + centre - A @ centre
    return M


def init_params(kind):
    return {"pitch_yz": [0, 0, 0], "sym_affine": [1, 1, 0, 0, 1, 0, 0], "rigid": [0] * 6,
            "similarity": [0] * 6 + [1], "affine": list(np.eye(3).ravel()) + [0, 0, 0]}[kind]


def matrix_to_params(M, kind, centre):
    """Inverse of params_to_matrix for warm starts between parametrisations (pitch_yz -> sym_affine)."""
    A = M[:3, :3]; t = M[:3, 3] - centre + A @ centre
    if kind == "sym_affine":
        return [A[0, 0], A[1, 1], A[1, 2], A[2, 1], A[2, 2], t[1], t[2]]
    if kind == "affine":
        return list(A.ravel()) + list(t)
    raise ValueError(kind)


def downsample(img_arr, affine, factor):
    if factor == 1:
        return img_arr, affine
    a = ndi.zoom(ndi.gaussian_filter(img_arr, 0.5 * factor), 1.0 / factor, order=1)
    A = affine.copy(); A[:3, :3] = affine[:3, :3] * factor
    A[:3, 3] = affine[:3, 3] + affine[:3, :3] @ np.full(3, (factor - 1) / 2.0)
    return a.astype(np.float32), A


def linear_register(fixed, fixed_aff, fmask, moving, moving_aff, mmask, kind, *, init=None,
                    sign=+1, factor=2, centre=None, grid_search=None, maxiter=4000, log=print):
    """Find M (moving world -> fixed world) maximising sign*NCC in the fixed mask (dilated moving
    footprint). Returns (M, info). `grid_search`: {param_index: array of values} tried before Powell."""
    F, Fa = downsample(fixed * (fmask > 0), fixed_aff, factor)
    Fm = downsample((fmask > 0).astype(np.float32), fixed_aff, factor)[0] > 0.5
    Mv, Ma = downsample(moving * (mmask > 0), moving_aff, factor)
    pts = grid_world(F.shape, Fa)[Fm.ravel()]
    fv = F.ravel()[Fm.ravel()].astype(np.float64)
    fv = (fv - fv.mean()) / (fv.std() + 1e-9)
    if centre is None:
        centre = pts.mean(0)

    def cost(p):
        M = params_to_matrix(p, kind, centre)
        Mi = np.linalg.inv(M)
        q = pts @ Mi[:3, :3].T + Mi[:3, 3]
        mv = sample(Mv, Ma, q, order=1).astype(np.float64)
        mv = (mv - mv.mean()) / (mv.std() + 1e-9)
        return -sign * float((fv * mv).mean())

    p0 = np.array(init if init is not None else init_params(kind), float)
    if grid_search:
        best = (cost(p0), p0.copy())
        idx = list(grid_search)
        import itertools
        for vals in itertools.product(*[grid_search[i] for i in idx]):
            p = p0.copy()
            for i, v in zip(idx, vals):
                p[i] = v
            c = cost(p)
            if c < best[0]:
                best = (c, p)
        p0 = best[1]
        log(f"  [linear {kind}] grid search best cost {best[0]:.4f} at {np.round(p0, 4).tolist()}")
    r = optimize.minimize(cost, p0, method="Powell", options=dict(maxiter=maxiter, xtol=1e-4, ftol=1e-6))
    M = params_to_matrix(r.x, kind, centre)
    log(f"  [linear {kind}] NCC {-r.fun * sign:+.4f} after {r.nfev} evals; params {np.round(r.x, 5).tolist()}")
    return M, dict(kind=kind, params=[float(x) for x in r.x], ncc=float(-r.fun * sign), nfev=int(r.nfev),
                   centre=[float(c) for c in centre], factor=factor)


# ------------------------------------------------------------------------------------------ writing
def save_map(outdir, name, m, grid_img=None, meta=None):
    """Write a Map as <name>_L.txt (4x4 RAS world), optional <name>_d.nii.gz (RAS mm on its grid),
    and extend meta.json."""
    os.makedirs(outdir, exist_ok=True)
    np.savetxt(f"{outdir}/{name}_L.txt", m.L, fmt="%.10f")
    ent = dict(L=f"{name}_L.txt", convention="src_world = L @ (x + d(x)), x = tgt world RAS mm; d RAS mm")
    if m.d is not None:
        nib.save(nib.Nifti1Image(m.d.astype(np.float32), m.da), f"{outdir}/{name}_d.nii.gz")
        ent["d"] = f"{name}_d.nii.gz"
    mp = f"{outdir}/maps.json"          # NOT meta.json: linc/xfm.py scans meta.json files as registry entries
    M = json.load(open(mp)) if os.path.exists(mp) else {}
    M.setdefault("maps", {})[name] = ent
    if meta:
        M.update(meta)
    json.dump(M, open(mp, "w"), indent=1)


def load_map(outdir, name):
    L = np.loadtxt(f"{outdir}/{name}_L.txt")
    dp = f"{outdir}/{name}_d.nii.gz"
    if os.path.exists(dp):
        im = nib.load(dp)
        return Map(L=L, d=np.asanyarray(im.dataobj).astype(np.float32), d_affine=im.affine, name=name)
    return Map(L=L, name=name)


def world_to_itk(M):
    """RAS world matrix (moving -> fixed) -> ITK AffineTransform text (fixed -> moving, LPS)."""
    Li = np.diag([-1.0, -1.0, 1.0, 1.0])
    P = Li @ np.linalg.inv(M) @ Li
    A = P[:3, :3]; t = P[:3, 3]
    return ("#Insight Transform File V1.0\n#Transform 0\nTransform: AffineTransform_double_3_3\n"
            f"Parameters: {' '.join('%.10f' % v for v in A.ravel())} {' '.join('%.10f' % v for v in t)}\n"
            "FixedParameters: 0 0 0\n")
