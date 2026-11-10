#!/usr/bin/env python3
"""Transform registry: named spaces, compose, apply once.

Layout. Every registration writes one directory
    $LINC_DATA/sub-<S>/xfm/<src>__to__<dst>/      (subject transforms)
    $LINC_TEMPLATE/xfm/<src>__to__<dst>/          (template-level: template<->template, atlases)
holding the transform files and a `meta.json`. `<src>__to__<dst>` moves IMAGES from src to dst.
Space names containing ':' (histo-piece:<p>, atlas:<name>) use '-' in the directory name; the true
names live in meta.json.

meta.json (written by `register()`):
    {"src": "bf-native", "dst": "b0-native", "subject": "MR252",
     "engine": "featurereg anatomix+FireANTs", "config": {...},
     "dst_grid": "<nifti defining the grid the transform resamples onto>",
     "src_grid": "<nifti in the src space; required by FSL conventions>",
     "components": [   # PULL order: a dst-space world point goes through these first -> last
        {"kind": "warp",   "file": "warp.nii.gz", "convention": "itk_lps_disp"},
        {"kind": "affine", "file": "affine.txt",  "convention": "itk"}],
     "files": {"warp.nii.gz": "<md5>", ...}, "created": ..., "qc": {...}}

Conventions (all describe a PULL = dst world -> src world, which is what resampling needs):
  affine  itk        ITK/ANTs text or binary .mat, LPS, fixed(dst) -> moving(src) point map
          fsl        FLIRT .mat (src fsl-mm -> dst fsl-mm, forward); needs src_grid + dst_grid
          ras_pull   4x4 text, dst RAS -> src RAS
          ras_push   4x4 text, src RAS -> dst RAS
  warp    itk_lps_disp  ITK/ANTs/FireANTs displacement field (LPS mm) on its own grid: x_src = x + L d(x)
          ras_disp      the same with RAS components (what `compose --format ras` writes)
          fsl_rel       FSL relative warp (applywarp --rel; MMORF, convertwarp --relout); needs src_grid
          fsl_abs       FSL absolute warp; needs src_grid
featurereg (psoct-mri-reg) = [warp itk_lps_disp, affine itk] (antsApplyTransforms -t warp -t affine).
MMORF `*_warp_resampled_unbiased_full` = [warp fsl_rel] with src_grid = the subject input.

Validated 2026-09-18 (session A) against the reference tools on real transforms: itk_lps_disp
(featurereg MR243 warp) vs antsApplyTransforms r = 1.0000000 (max rel diff 1e-5); warp + centred ITK
affine vs ANTs r = 1.0000000; fsl_rel (MMORF mmsym_model_dti MF283 full warp) vs applywarp --rel
r = 1.0000000; FLIRT .mat vs flirt -applyxfm r = 0.9999999997 when not downsampling; `compose` ->
ITK field -> antsApplyTransforms reproduces applywarp at r = 0.999999999999.
apply() point-samples (linear or nearest), with no anti-aliasing: FLIRT averages when the output grid
is coarser, so downsample-by->1.5x products should be smoothed first or made on a finer grid.

Every entry is also invertible on the fly: affines analytically, each warp component by fixed-point
iteration at the query points (residual reported), so inverse chains never need a stored field.
Registration QC (§0.12, §3.13): `meta["qc"]["status"]` is set by code/regqc.py; chains warn (or with
strict=True refuse) when a link has no passing QC.

CLI:
    xfm.py spaces
    xfm.py list [--subject S]
    xfm.py chain --from bf-native --to mri-template --subject MR252
    xfm.py apply --from bf-native --to mri-template --subject MR252 --ref grid.nii.gz in.nii.gz out.nii.gz [--nearest]
    xfm.py compose --from ... --to ... --subject S --ref grid.nii.gz out_warp.nii.gz [--format itk|ras]
    xfm.py register --src A --dst B [--subject S] --component warp:itk_lps_disp:w.nii.gz \
                    --component affine:itk:a.txt --dst-grid g.nii.gz [--src-grid s.nii.gz] --engine ...
    xfm.py check [--subject S]        # md5 of every registered file vs meta
"""
import datetime as _dt
import hashlib
import json
import os
import shutil
import sys
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates

from . import paths
from . import provenance as timing


def DATA():
    return paths.data_root()


def GLOBAL_ROOTS():
    """Template-level transform directories inside the released template ($LINC_TEMPLATE/xfm)."""
    return [paths.template_root() / "xfm"]


LPS = np.diag([-1.0, -1.0, 1.0, 1.0])
CHUNK = 4_000_000                         # target points per resampling slab

# Named spaces (§0.6). subject=True: the space exists once per subject.
SPACES = {
    "bf-native":     dict(subject=True,  desc="blockface reconstruction grid (native in-plane, native z; OME-Zarr pyramid level)"),
    "bf-iso":        dict(subject=True,  desc="blockface on the isotropic grid made by register_mri.prep_bf (0.15 mm)"),
    "b0-native":     dict(subject=True,  desc="the subject's b0 MRI as acquired (data/_mri_src/<S>/)"),
    "t2-native":     dict(subject=True,  desc="an MRI subject's MSME T2 as acquired (MF283-286: mri/data/sub-<S>/t2/)"),
    "micro-native":  dict(subject=True,  desc="an MRI subject's microstructure session as acquired (mri/data/sub-<S>/micro/)"),
    "mri-canonical": dict(subject=True,  desc="an MRI subject (MF283-286) rigidly placed in the canonical NMT frame (§3.2)"),
    "mri-run113":    dict(subject=False, desc="run-113 frame, BF registration frame (§3.3)",
                          grid="frames/mri_run113_0p5mm.nii.gz"),
    "mri-template":  dict(subject=False, desc="the shipped MRI template (canonical grid 256x320x192 @0.25 mm, §3.1/3.2)",
                          grid="mri/linc_mri_T2.nii.gz"),
    "bf-template":   dict(subject=False, desc="the shipped BF template (run-113 frame, §5.5 T1)"),
    "histo-piece":   dict(subject=True,  desc="one histology piece, histo-piece:<piece>"),
    "atlas":         dict(subject=False, desc="an external atlas/template, atlas:<name>"),
}


def _base(space):
    return space.split(":", 1)[0]


def is_subject_space(space):
    b = _base(space)
    if b not in SPACES:
        raise ValueError(f"unknown space '{space}'; known: {', '.join(SPACES)} (add it to xfm.SPACES)")
    return SPACES[b]["subject"]


def dirname(src, dst):
    return f"{src.replace(':', '-')}__to__{dst.replace(':', '-')}"


def xfm_dir(src, dst, subject=None, root=None):
    if root is not None:
        return Path(root) / dirname(src, dst)
    if subject:
        return DATA() / f"sub-{subject}" / "xfm" / dirname(src, dst)
    return GLOBAL_ROOTS()[0] / dirname(src, dst)


def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- readers

def _grid(path):
    img = nib.load(str(path))
    return img.shape[:3], img.affine.astype(np.float64), np.asarray(img.header.get_zooms()[:3], float)


def read_itk_affine(path):
    """ITK/ANTs affine (text or binary .mat) -> 4x4 RAS pull matrix (fixed/dst RAS -> moving/src RAS)."""
    path = str(path)
    params = fixed = None
    with open(path, "rb") as fh:
        head = fh.read(64)
    if head.startswith(b"#Insight") or head.lstrip().startswith(b"Transform"):
        kind = None
        for line in open(path):
            if line.startswith("Transform:"):
                kind = line.split(":", 1)[1].strip()
            elif line.startswith("Parameters:"):
                params = np.array(line.split(":", 1)[1].split(), float)
            elif line.startswith("FixedParameters:"):
                fixed = np.array(line.split(":", 1)[1].split(), float)
        if kind and not kind.startswith(("AffineTransform", "MatrixOffsetTransformBase")) or params is None \
                or len(params) != 12:
            return _itk_affine_via_sitk(path)
    else:
        from scipy.io import loadmat
        m = loadmat(path)
        key = next(k for k in m if k.startswith(("AffineTransform", "MatrixOffsetTransformBase")))
        params = np.asarray(m[key], float).ravel()
        fixed = np.asarray(m["fixed"], float).ravel()
    M = params[:9].reshape(3, 3)
    t = params[9:12]
    c = fixed[:3] if fixed is not None and len(fixed) >= 3 else np.zeros(3)
    T = np.eye(4)
    T[:3, :3] = M
    T[:3, 3] = t + c - M @ c
    return LPS @ T @ LPS


def _itk_affine_via_sitk(path):
    import SimpleITK as sitk
    tx = sitk.ReadTransform(path)
    o = np.array(tx.TransformPoint((0.0, 0.0, 0.0)))
    T = np.eye(4)
    for i in range(3):
        e = [0.0, 0.0, 0.0]; e[i] = 1.0
        T[:3, i] = np.array(tx.TransformPoint(tuple(e))) - o
    T[:3, 3] = o
    return LPS @ T @ LPS


def fsl_vox2mm(shape, affine, zooms):
    """FSL's scaled-voxel ('fsl-mm') coordinates: x reversed when the sform determinant is positive."""
    F = np.diag([zooms[0], zooms[1], zooms[2], 1.0])
    if np.linalg.det(affine[:3, :3]) > 0:
        F[0, 0] = -zooms[0]
        F[0, 3] = (shape[0] - 1) * zooms[0]
    return F


def read_fsl_affine(path, src_grid, dst_grid):
    """FLIRT .mat -> 4x4 RAS pull (dst RAS -> src RAS)."""
    M = np.loadtxt(str(path))
    ss, As, zs = _grid(src_grid)
    sd, Ad, zd = _grid(dst_grid)
    Fs, Fd = fsl_vox2mm(ss, As, zs), fsl_vox2mm(sd, Ad, zd)
    fwd = Ad @ np.linalg.inv(Fd) @ M @ Fs @ np.linalg.inv(As)
    return np.linalg.inv(fwd)


# --------------------------------------------------------------------------- components

class Affine:
    """x_src = T x (4x4, RAS)."""
    def __init__(self, T, label=""):
        self.T = np.asarray(T, float); self.label = label

    def pull(self, X):
        return X @ self.T[:3, :3].T + self.T[:3, 3]

    def inverse(self):
        return Affine(np.linalg.inv(self.T), f"inv({self.label})")


class Warp:
    """x_src = K x + B d(V x): K, V 4x4, B 3x3, d a displacement field sampled linearly on its grid.
    itk_lps_disp: K=I, B=diag(-1,-1,1); ras_disp: K=I, B=I; fsl_rel/fsl_abs: see _fsl()."""
    def __init__(self, path, convention, src_grid=None, label=""):
        img = nib.load(str(path))
        d = np.asarray(img.dataobj, dtype=np.float32)
        d = d.reshape(d.shape[:3] + (-1,))
        if d.shape[3] != 3:
            raise ValueError(f"{path}: displacement field must have 3 components, got {d.shape}")
        self.d = [np.ascontiguousarray(d[..., k]) for k in range(3)]
        self.A = img.affine.astype(np.float64)
        self.V = np.linalg.inv(self.A)
        self.label = label or f"{convention}:{Path(path).name}"
        self.absolute = False
        if convention == "itk_lps_disp":
            self.K, self.B = np.eye(4), np.diag([-1.0, -1.0, 1.0])
        elif convention == "ras_disp":
            self.K, self.B = np.eye(4), np.eye(3)
        elif convention in ("fsl_rel", "fsl_abs"):
            if src_grid is None:
                raise ValueError(f"{path}: FSL warps need src_grid")
            ss, As, zs = _grid(src_grid)
            Fw = fsl_vox2mm(img.shape[:3], self.A, np.asarray(img.header.get_zooms()[:3], float))
            Fs = fsl_vox2mm(ss, As, zs)
            G = As @ np.linalg.inv(Fs)                     # src fsl-mm -> src RAS
            self.B = G[:3, :3]
            if convention == "fsl_rel":
                self.K = G @ Fw @ self.V                   # x -> fsl-mm on the warp grid -> read as src fsl-mm
            else:
                self.K = np.zeros((4, 4)); self.K[3, 3] = 1.0; self.K[:3, 3] = G[:3, 3]
                self.absolute = True
        else:
            raise ValueError(f"unknown warp convention {convention}")

    def disp(self, X):
        v = X @ self.V[:3, :3].T + self.V[:3, 3]
        c = v.T
        out = np.empty_like(X)
        for k in range(3):
            out[:, k] = map_coordinates(self.d[k], c, order=1, mode="constant", cval=0.0)
        if self.absolute:                                   # outside the FOV an absolute warp is undefined
            inside = np.all((v > -0.5) & (v < np.asarray(self.d[0].shape) - 0.5), axis=1)
            out[~inside] = np.nan
        return out @ self.B.T

    def pull(self, X):
        return X @ self.K[:3, :3].T + self.K[:3, 3] + self.disp(X)

    def inverse(self, iters=30, tol=1e-3):
        return _InverseWarp(self, iters, tol)


class _InverseWarp:
    """Pointwise inverse of a Warp: solve K x + B d(V x) = y by x <- K^-1 (y - B d(V x))."""
    def __init__(self, w, iters, tol):
        if w.absolute:
            raise ValueError("cannot invert an absolute FSL warp pointwise; convert it to relative first")
        self.w, self.iters, self.tol = w, iters, tol
        self.Ki = np.linalg.inv(w.K)
        self.label = f"inv({w.label})"
        self.last_residual = None

    def pull(self, Y):
        x = Y @ self.Ki[:3, :3].T + self.Ki[:3, 3]
        for _ in range(self.iters):
            r = Y - self.w.disp(x)
            xn = r @ self.Ki[:3, :3].T + self.Ki[:3, 3]
            step = np.abs(xn - x).max() if len(x) else 0.0
            x = xn
            if step < self.tol:
                break
        res = np.linalg.norm(self.w.pull(x) - Y, axis=1) if len(x) else np.zeros(0)
        self.last_residual = (float(np.median(res)), float(res.max())) if len(res) else (0.0, 0.0)
        return x

    def inverse(self):
        return self.w


# --------------------------------------------------------------------------- entries and chains

class Entry:
    """One registered transform directory; `.pull(X)` maps dst-world points to src-world points."""
    def __init__(self, d):
        self.dir = Path(d)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.src, self.dst = self.meta["src"], self.meta["dst"]
        self.subject = self.meta.get("subject")
        self.inverted = False
        self._comps = None

    def _resolve(self, f):
        p = Path(f)
        return p if p.is_absolute() else self.dir / p

    def components(self):
        if self._comps is None:
            cs = []
            for c in self.meta["components"]:
                f = self._resolve(c["file"])
                sg = c.get("src_grid", self.meta.get("src_grid"))
                dg = c.get("dst_grid", self.meta.get("dst_grid"))
                conv = c["convention"]
                if c["kind"] == "affine":
                    if conv == "itk":
                        T = read_itk_affine(f)
                    elif conv == "fsl":
                        T = read_fsl_affine(f, self._resolve(sg), self._resolve(dg))
                    elif conv == "ras_pull":
                        T = np.loadtxt(str(f))
                    elif conv == "ras_push":
                        T = np.linalg.inv(np.loadtxt(str(f)))
                    else:
                        raise ValueError(f"{self.dir}: unknown affine convention {conv}")
                    cs.append(Affine(T, f"{conv}:{f.name}"))
                elif c["kind"] == "warp":
                    cs.append(Warp(f, conv, self._resolve(sg) if sg else None))
                else:
                    raise ValueError(f"{self.dir}: unknown component kind {c['kind']}")
            self._comps = cs
        return self._comps

    def pull(self, X):
        for c in self.components():
            X = c.pull(X)
        return X

    def qc_status(self):
        return (self.meta.get("qc") or {}).get("status")

    def __repr__(self):
        s = f"{self.src} -> {self.dst}" + (f" [{self.subject}]" if self.subject else "")
        return s + (" (inverted)" if self.inverted else "") + f"  {self.dir}"


class InverseEntry(Entry):
    def __init__(self, e):
        self.__dict__.update(e.__dict__)
        self.src, self.dst = e.dst, e.src
        self.inverted = True
        self._fwd = e
        self._inv = None

    def pull(self, X):
        if self._inv is None:
            self._inv = [c.inverse() for c in reversed(self._fwd.components())]
        for c in self._inv:
            X = c.pull(X)
        return X

    def residuals(self):
        return [c.last_residual for c in (self._inv or []) if hasattr(c, "last_residual")]


def _roots(subject):
    roots = []
    if subject:
        roots.append(DATA() / f"sub-{subject}" / "xfm")
    roots += GLOBAL_ROOTS()
    return roots


def entries(subject=None):
    """Every registered transform visible for `subject` (its own + template-level)."""
    out, seen = [], set()
    for r in _roots(subject):
        if not r.is_dir():
            continue
        cands = [r] if (r / "meta.json").exists() else []
        cands += [p for p in sorted(r.iterdir()) if p.is_dir() and (p / "meta.json").exists()]
        for d in cands:
            if d.resolve() in seen:
                continue
            seen.add(d.resolve())
            try:
                e = Entry(d)
            except (json.JSONDecodeError, KeyError) as exc:
                warnings.warn(f"skipping {d}: bad meta.json ({exc})")
                continue
            if e.subject and subject and e.subject != subject:
                continue
            out.append(e)
    return out


def find_chain(src, dst, subject=None, allow_inverse=True):
    """Shortest list of entries taking images from `src` to `dst` (BFS; forward links preferred). The cross-modal alignment
    (bf-native <-> mri-template) made the stand-in b0: a chain uses it only when nothing else connects the two spaces, so a brain
    reaches the MRI template through its own MRI or through the blockface template, the paper's two routes."""
    es = entries(subject)
    for use in ([e for e in es if {e.src, e.dst} != {"bf-native", "mri-template"}], es):
        chain = _shortest(src, dst, use, allow_inverse)
        if chain is not None:
            return chain
    raise LookupError(f"no chain {src} -> {dst}" + (f" for {subject}" if subject else "") +
                      f"; registered: {[repr(e) for e in es]}")


def _shortest(src, dst, es, allow_inverse):
    edges = {}
    for e in es:
        edges.setdefault(e.src, []).append(e)
    if allow_inverse:
        for e in es:
            edges.setdefault(e.dst, []).append(InverseEntry(e))
    for k in edges:
        edges[k].sort(key=lambda e: e.inverted)
    prev = {src: None}
    q = deque([src])
    while q:
        s = q.popleft()
        if s == dst:
            break
        for e in edges.get(s, []):
            if e.dst not in prev:
                prev[e.dst] = e
                q.append(e.dst)
    if dst not in prev:
        return None
    chain, s = [], dst
    while prev[s] is not None:
        chain.append(prev[s]); s = prev[s].src
    return chain[::-1]


class Chain:
    """Images move src -> ... -> dst along `links`; points are pulled back dst -> src."""
    def __init__(self, links, strict=False):
        self.links = list(links)
        for e in self.links:
            if e.qc_status() not in ("pass", "pass-visual"):
                msg = f"xfm link without passing QC (§0.12): {e!r} qc={e.qc_status()}"
                if strict:
                    raise RuntimeError(msg)
                warnings.warn(msg)

    @classmethod
    def between(cls, src, dst, subject=None, strict=False, allow_inverse=True):
        return cls(find_chain(src, dst, subject, allow_inverse), strict)

    def pull(self, X):
        for e in reversed(self.links):
            X = e.pull(X)
        return X

    def describe(self):
        return [{"src": e.src, "dst": e.dst, "subject": e.subject, "dir": str(e.dir),
                 "inverted": e.inverted, "engine": e.meta.get("engine"), "qc": e.qc_status()}
                for e in self.links]


# --------------------------------------------------------------------------- apply once / compose

def _slabs(shape):
    per = max(1, CHUNK // (shape[0] * shape[1]))
    for z0 in range(0, shape[2], per):
        yield z0, min(shape[2], z0 + per)


def _grid_points(shape, A, z0, z1):
    i, j, k = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(z0, z1), indexing="ij")
    v = np.stack([i.ravel(), j.ravel(), k.ravel()], 1).astype(np.float64)
    return v @ A[:3, :3].T + A[:3, 3]


def apply(chain, moving, ref, out, order=1, allow_tensor=False, cval=0.0):
    """Resample `moving` (in the chain's source space) onto the `ref` grid (in its destination space)
    with ONE interpolation: order 1 = linear, 0 = nearest (labels, masks). Multi-channel images are
    resampled per channel; 6-channel images are refused (tensors need reorientation, not this)."""
    if order not in (0, 1):
        raise ValueError("apply-once supports order 0 (nearest) or 1 (linear) only")
    mimg, rimg = nib.load(str(moving)), nib.load(str(ref))
    data = np.asanyarray(mimg.dataobj)
    if data.dtype.names:                                     # RGB24 structured dtype
        data = np.stack([data[n] for n in data.dtype.names], -1)
    nch = int(np.prod(data.shape[3:])) if data.ndim > 3 else 1
    if nch == 6 and not allow_tensor:
        raise ValueError(f"{moving} has 6 channels (a tensor?): apply() does not reorient; pass allow_tensor=True "
                         "only for non-tensor 6-channel data")
    chans = data.reshape(data.shape[:3] + (nch,))
    Vm = np.linalg.inv(mimg.affine)
    shape, A = rimg.shape[:3], rimg.affine
    odt = data.dtype if order == 0 else np.float32
    res = np.zeros(shape + (nch,), odt)
    src_f = [np.ascontiguousarray(chans[..., c], dtype=np.float32 if order == 1 else chans.dtype)
             for c in range(nch)]
    for z0, z1 in _slabs(shape):
        X = _grid_points(shape, A, z0, z1)
        Y = chain.pull(X)
        bad = ~np.isfinite(Y).all(1)
        Y[bad] = -1e6
        v = (Y @ Vm[:3, :3].T + Vm[:3, 3]).T
        for c in range(nch):
            s = map_coordinates(src_f[c], v, order=order, mode="constant", cval=cval, prefilter=False)
            s[bad] = cval
            res[:, :, z0:z1, c] = s.reshape(shape[0], shape[1], z1 - z0)
    res = res.reshape(shape + data.shape[3:]) if nch > 1 else res[..., 0]
    hdr = rimg.header.copy()
    hdr.set_data_dtype(odt)
    img = nib.Nifti1Image(res, A, hdr)
    img.set_sform(A, int(rimg.header["sform_code"]) or 1)
    img.set_qform(A, int(rimg.header["qform_code"]) or 1)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(out))
    timing.write_provenance(out, inputs=[moving, ref] + [e.dir / "meta.json" for e in chain.links],
                            extra={"xfm_chain": chain.describe(), "order": order,
                                   "inverse_residual_mm": [e.residuals() for e in chain.links if e.inverted]})
    return out


def compose(chain, ref, out, fmt="itk", src_grid=None):
    """Write the chain as ONE displacement field (pull) on the `ref` grid: fmt 'itk' = ITK/ANTs LPS
    5-D vector NIfTI (usable by antsApplyTransforms and by xfm as itk_lps_disp), 'ras' = RAS 4-D,
    'fsl_rel' = FSL relative warp (`applywarp --rel -w`, `vecreg -w`), which needs `src_grid`: the
    displacement is then src-fsl-mm minus ref-fsl-mm, the convention Warp(..., 'fsl_rel') reads."""
    rimg = nib.load(str(ref))
    shape, A = rimg.shape[:3], rimg.affine
    D = np.zeros(shape + (3,), np.float32)
    if fmt == "fsl_rel":
        if src_grid is None:
            raise ValueError("fmt='fsl_rel' needs src_grid (the image the payload lives on)")
        ss, As, zs = _grid(src_grid)
        Fs = fsl_vox2mm(ss, As, zs)                                  # src voxel -> src fsl-mm
        Fr = fsl_vox2mm(shape, A, np.asarray(rimg.header.get_zooms()[:3], float))
        ras_to_srcfsl = Fs @ np.linalg.inv(As)                       # src RAS -> src fsl-mm
        ras_to_reffsl = Fr @ np.linalg.inv(A)                        # ref RAS -> ref fsl-mm
    for z0, z1 in _slabs(shape):
        X = _grid_points(shape, A, z0, z1)
        Y = chain.pull(X)
        if fmt == "fsl_rel":
            d = (Y @ ras_to_srcfsl[:3, :3].T + ras_to_srcfsl[:3, 3]) - (X @ ras_to_reffsl[:3, :3].T + ras_to_reffsl[:3, 3])
        else:
            d = Y - X
            if fmt == "itk":
                d = d * np.array([-1.0, -1.0, 1.0])
        D[:, :, z0:z1, :] = d.reshape(shape[0], shape[1], z1 - z0, 3)
    if fmt == "itk":
        img = nib.Nifti1Image(D[:, :, :, None, :], A)
        img.header.set_intent(1007)                          # NIFTI_INTENT_VECTOR, as ANTs writes
    elif fmt == "fsl_rel":
        img = nib.Nifti1Image(D, A)
        img.header["intent_code"] = 0                        # as MMORF writes its relative warps: an FSL warp intent
        # (2006 = fnirt displacement field) makes vecreg read the field with fnirt's own convention instead, which
        # silently misorients tensors (measured: 20 deg median in V1). applywarp is unaffected because --rel is explicit.
    else:
        img = nib.Nifti1Image(D, A)
    img.set_sform(A, 1); img.set_qform(A, 1)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(out))
    timing.write_provenance(out, inputs=[ref] + [e.dir / "meta.json" for e in chain.links],
                            extra={"xfm_chain": chain.describe(), "format": fmt})
    return out


def jacobian_folds(field_path, fmt="itk"):
    """Fraction of voxels with det(J) <= 0 for a pull displacement field (either format)."""
    img = nib.load(str(field_path))
    d = np.asarray(img.dataobj, np.float32).reshape(img.shape[:3] + (3,))
    if fmt == "itk":
        d = d * np.array([-1, -1, 1], np.float32)
    Minv = np.linalg.inv(img.affine[:3, :3])                 # world-per-voxel -> gradient in world
    g = [np.gradient(d[..., k]) for k in range(3)]           # g[k][a] = d d_k / d vox_a
    J = np.empty(img.shape[:3] + (3, 3), np.float32)
    for k in range(3):
        for a in range(3):
            J[..., k, a] = sum(g[k][b] * Minv[b, a] for b in range(3))
    J += np.eye(3, dtype=np.float32)
    det = np.linalg.det(J)
    return float((det <= 0).mean()), det


# --------------------------------------------------------------------------- registration of new entries

def register(src, dst, components, *, subject=None, dst_grid, src_grid=None, engine, config=None,
             root=None, copy=True, overwrite=False, extra=None):
    """Create a registry entry. `components` = [(kind, convention, path), ...] in PULL order.
    Files are copied in (copy=True) and md5-stamped. An existing entry is backed up to
    $LINC_WORK/xfm_backup/<date>/ before being replaced (overwrite=True), never silently clobbered."""
    if is_subject_space(src) or is_subject_space(dst):
        if not subject:
            raise ValueError(f"{src}->{dst} involves a subject space: pass subject=")
    d = xfm_dir(src, dst, subject, root)
    if (d / "meta.json").exists():
        if not overwrite:
            raise FileExistsError(f"{d} exists (pass overwrite=True; the old entry is backed up)")
        bak = paths.work_root() / "xfm_backup" / _dt.date.today().isoformat() / (subject or "_global") / \
            f"{d.name}.{_dt.datetime.now():%H%M%S}"
        bak.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(d, bak)
        shutil.rmtree(d)
    d.mkdir(parents=True)
    comps, files = [], {}
    for kind, conv, p in components:
        p = Path(p)
        if copy:
            name = p.name if p.name not in files else f"{len(files)}_{p.name}"
            shutil.copy2(p, d / name)
            f = name
        else:
            f = str(p.resolve())
        comps.append({"kind": kind, "file": f, "convention": conv})
        files[f] = _md5(d / f if copy else p)
    meta = {"src": src, "dst": dst, "subject": subject, "engine": engine, "config": config or {},
            "dst_grid": str(Path(dst_grid).resolve()),
            "src_grid": str(Path(src_grid).resolve()) if src_grid else None,
            "components": comps, "files": files,
            "created": timing._now(), "host": timing._host(), "cmd": " ".join(sys.argv),
            "qc": {"status": None, "note": "set by code/regqc.py (§3.13)"}}
    meta.update(extra or {})
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    ins = [p for _, _, p in components] + [dst_grid] + ([src_grid] if src_grid else [])
    timing.write_provenance(d, inputs=ins, extra={"engine": engine, "config": config or {}})
    return Entry(d)


def set_qc(entry_dir, status, **fields):
    """Record the QC outcome in meta.json (called by code/regqc.py). status: pass | pass-visual | fail | excluded."""
    p = Path(entry_dir) / "meta.json"
    m = json.loads(p.read_text())
    m["qc"] = {"status": status, "time": timing._now(), **fields}
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, indent=1))
    os.replace(tmp, p)


def check(subject=None):
    bad = []
    for e in entries(subject):
        for f, h in (e.meta.get("files") or {}).items():
            p = e._resolve(f)
            if not p.exists():
                bad.append((str(e.dir), f, "missing"))
            elif _md5(p) != h:
                bad.append((str(e.dir), f, "md5 changed"))
    return bad


# --------------------------------------------------------------------------- CLI

def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("spaces")
    s = sub.add_parser("list"); s.add_argument("--subject")
    s = sub.add_parser("check"); s.add_argument("--subject")
    for name in ("chain", "apply", "compose"):
        s = sub.add_parser(name)
        s.add_argument("--from", dest="src", required=True)
        s.add_argument("--to", dest="dst", required=True)
        s.add_argument("--subject")
        s.add_argument("--strict", action="store_true", help="refuse links without passing QC")
        s.add_argument("--no-inverse", action="store_true")
        if name in ("apply", "compose"):
            s.add_argument("--ref", required=True, help="target grid NIfTI (in the destination space)")
        if name == "apply":
            s.add_argument("input"); s.add_argument("output")
            s.add_argument("--nearest", action="store_true")
        if name == "compose":
            s.add_argument("output")
            s.add_argument("--format", default="itk", choices=["itk", "ras"])
    s = sub.add_parser("register")
    s.add_argument("--src", required=True); s.add_argument("--dst", required=True)
    s.add_argument("--subject")
    s.add_argument("--component", action="append", required=True, help="kind:convention:path, PULL order")
    s.add_argument("--dst-grid", required=True); s.add_argument("--src-grid")
    s.add_argument("--engine", required=True); s.add_argument("--config", help="JSON string")
    s.add_argument("--root"); s.add_argument("--no-copy", action="store_true")
    s.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    if a.cmd == "spaces":
        for k, v in SPACES.items():
            print(f"{k:<14} {'per-subject' if v['subject'] else 'global':<11} {v['desc']}")
    elif a.cmd == "list":
        for e in entries(a.subject):
            print(repr(e), f"qc={e.qc_status()}")
    elif a.cmd == "check":
        bad = check(a.subject)
        for b in bad:
            print(*b)
        return 1 if bad else 0
    elif a.cmd == "register":
        comps = [tuple(c.split(":", 2)) for c in a.component]
        e = register(a.src, a.dst, comps, subject=a.subject, dst_grid=a.dst_grid, src_grid=a.src_grid,
                     engine=a.engine, config=json.loads(a.config) if a.config else None, root=a.root,
                     copy=not a.no_copy, overwrite=a.overwrite)
        print(repr(e))
    else:
        ch = Chain.between(a.src, a.dst, a.subject, strict=a.strict, allow_inverse=not a.no_inverse)
        if a.cmd == "chain":
            print(json.dumps(ch.describe(), indent=1))
        elif a.cmd == "apply":
            apply(ch, a.input, a.ref, a.output, order=0 if a.nearest else 1)
            print(a.output)
        elif a.cmd == "compose":
            compose(ch, a.ref, a.output, a.format)
            print(a.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
