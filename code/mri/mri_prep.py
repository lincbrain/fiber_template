"""Per-brain preparation of the MRI construction inputs (paper: Methods, "MRI preprocessing" and "Symmetric groupwise construction").

Each of the four MRI brains (MF1-MF4 = LINC MF283-MF286) has three sessions on three grids: diffusion (b0 and tensor, 0.25 mm: the
reference grid), T2 (multi-echo MSME, 0.5 mm) and SANDI microstructure (0.5 mm). The steps, in order:

  1. staging (`stage`, from the preprocessed files): each image under its brain mask's header (the masks were made after
     re-origining, so the mask affine is the authority; same array, no resampling); b0 and T2 N4-corrected; echo 5 of the MSME
     is the T2 image; SANDI fractions renormalized to sum 1 inside the micro brain mask; the tensor's diffusion files under dwi/raw/.
  2. the D9 tensor (`tensor_d9`): dtifit on the eddy-corrected b <= 4800 volumes (1 b0 + 42 DWIs), b-vectors rotated by each
     volume's eddy_correct matrix (polar part), axis convention from MRtrix dwigradcheck; staged on the dwi grid.
  3. the within-MRI registrations T2 -> b0 and micro -> b0 (one ITK displacement field each, the initial affine inside it) are
     inputs here: $LINC_DATA/sub-<ID>/xfm/{t2,micro}-native__to__b0-native (blockface.registration, anatomix + FireANTs).
  4. the canonical frame (mri.canonical: write_frames_nmt -> canonical.json).
  5. the construction inputs (`prepare`): every payload resampled ONCE onto the canonical grid (antsApplyTransforms with the
     canonical rigid first, then the within-MRI warp), every channel cut to the brain's final hand-edited mask:
       t2_modelmask  T2 head x final mask, 1-99 percentile -> (0, 4095]
       frac1, frac2  the session-mask SANDI fractions (renormalized on the carried micro mask) x final mask; voxels of the final mask
                     outside the micro brain mask (~1 %) have no values and stay 0 (never filled)
       tensor_vr     the D9 tensor through FSL vecreg -t (the canonical rigid as a FLIRT matrix; vecreg reorients), x final mask
       b0            the N4-corrected b0 head x final mask (normalized like T2)
       model_mask    the final mask (lossless uint8 copy; the mask file itself is never modified)
     plus every output's exact mirror (<name>_flip; tensors with xy, xz negated).

    python -m mri.mri_prep stage --subject MF283 --preprocessed <folder: dwi/, msme/, dt2/> [--mri-root ...]
    python -m mri.mri_prep prepare --subject MF283 --final-mask <MF283_mask_EDIT.nii.gz> --tensor <tensor_d9_dwigrid>
        [--mri-root $LINC_DATA/sub-MF283/mri] [--xfm-root $LINC_DATA/sub-MF283/xfm] [--canonical <canonical.json>] [--out <dir>]

Ported from the analysis project's mri/code/rebuild_prep.py (`canon --rule finalhand`, the inputs of record), preprocess_mri.py (the
staging helpers) and tensor_refit_d9.py (`--from-existing`, the tensor of record). Paths come from settings/mri_construction.yaml
`prep.layout` relative to each brain's MRI folder.
"""
import argparse
import hashlib
import itertools
import json
import os
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from . import canonical as C
from template.tools import ants, fsl, fsl_env, mrtrix, recipe, run

DERIVED = [("t2_head.nii.gz", "brain_mask.nii.gz", "t2_dwimask.nii.gz"),
           ("t2_head.nii.gz", "model_mask.nii.gz", "t2_modelmask.nii.gz")]
# (layout key, within-MRI space or None for the dwi grid, output, interpolation): every payload of the canonical grid
PAYLOADS = [("b0_n4_brain", None, "b0.nii.gz", "Linear"),
            ("dwi_mask", None, "brain_mask.nii.gz", "NearestNeighbor"),
            ("t2_brain", "t2", "t2.nii.gz", "Linear"),
            ("t2_mask", "t2", "t2_mask.nii.gz", "NearestNeighbor"),
            ("frac1", "micro", "frac1.nii.gz", "Linear"),
            ("frac2", "micro", "frac2.nii.gz", "Linear"),
            ("micro_mask", "micro", "micro_mask.nii.gz", "NearestNeighbor"),
            ("t2_head", "t2", "t2_head.nii.gz", "Linear")]


class Brain:
    """One brain's MRI files: mri_root/<prep.layout[key]> and its transforms under xfm_root."""

    def __init__(self, subject, mri_root=None, xfm_root=None):
        from shared import paths
        self.subject = subject
        self.mri_root = Path(mri_root) if mri_root else paths.subject_dir(subject) / "mri"
        self.xfm_root = Path(xfm_root) if xfm_root else paths.subject_dir(subject) / "xfm"
        self.layout = recipe()["prep"]["layout"]

    def __call__(self, key):
        return self.mri_root / self.layout[key]

    def within_mri_warp(self, space):
        """The single ITK displacement field of <space>-native -> b0-native (the initial affine is inside it: apply it alone)."""
        d = self.xfm_root / recipe()["prep"]["within_mri_xfm"][space]
        comps = json.loads((d / "meta.json").read_text())["components"]
        assert len(comps) == 1 and comps[0]["kind"] == "warp" and comps[0]["convention"] == "itk_lps_disp", d
        return d / comps[0]["file"]


# --------------------------------------------------------------------------- 1. staging (preprocess_mri)

def adopt_geometry(src, ref, dst):
    """src's voxel array under ref's affine/header (shapes must match; nothing is resampled)."""
    s, r = nib.load(str(src)), nib.load(str(ref))
    if s.shape[:3] != r.shape[:3]:
        raise ValueError(f"shape mismatch: {src} {s.shape[:3]} vs ref {ref} {r.shape[:3]}")
    hdr = s.header.copy()
    hdr.set_sform(r.affine, code=int(r.header["sform_code"]))
    hdr.set_qform(r.affine, code=int(r.header["qform_code"]))
    hdr.set_zooms(r.header.get_zooms()[:3] + s.header.get_zooms()[3:])
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asanyarray(s.dataobj), r.affine, hdr), str(dst))


def _fslroi(src, dst, index):
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    run([fsl("fslroi"), src, dst, index, 1], env=fsl_env())


def extract_echo(multi_echo, dst, index=None):
    """Clip negatives (denoising can undershoot), then take one echo."""
    index = recipe()["prep"]["t2_echo_index"] if index is None else index
    tmp = str(dst).replace(".nii.gz", "_tmpclip.nii.gz")
    n = nib.load(str(multi_echo))
    nib.save(nib.Nifti1Image(np.clip(n.get_fdata(), 0, None), n.affine, n.header), tmp)
    try:
        _fslroi(tmp, dst, index)
    finally:
        os.remove(tmp)


def n4(src, mask, dst, bias=None):
    """N4 bias correction; shrink factor so the fitting grid is ~0.5 mm."""
    p = recipe()["prep"]["n4"]
    vox = float(nib.load(str(src)).header.get_zooms()[0])
    shrink = max(1, int(round(float(p["fit_grid_mm"]) / vox)))
    bias = bias or str(dst).replace(".nii.gz", "_bias.nii.gz")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    run([ants("N4BiasFieldCorrection"), "-d", 3, "-v", 0, "-s", shrink, "-b", p["bspline"], "-c", p["convergence"],
         "-i", src, "-x", mask, "-o", f"[{dst},{bias}]"])


def mask_image(src, mask, dst, normalize=False):
    """Zero outside the mask; optionally rescale the inside to (norm_eps, norm_max] on its 1-99 percentiles."""
    c = recipe()["canonical"]
    n = nib.load(str(src))
    img = n.get_fdata()
    m = nib.load(str(mask)).get_fdata()
    if img.ndim == 4 and m.ndim == 3:
        m = np.broadcast_to(m[..., None], img.shape)
    img = np.where(m == 0, 0, img)
    if normalize and img.ndim == 3:
        inside = img[m > 0]
        vmin, vmax = np.percentile(inside, 1), np.percentile(inside, 99)
        eps = float(c["norm_eps"])
        img = np.clip((img - vmin) / (vmax - vmin), 0, 1) * (float(c["norm_max"]) - eps) + eps
        img = np.where(m == 0, 0, img)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(img, n.affine, n.header), str(dst))


def normalize_fracs_and_weights(b):
    """frac1 + frac2 = 1 inside the micro brain mask; the SDs scaled the same way and turned into precision weights."""
    f1, f2 = nib.load(str(b("frac1_raw"))), nib.load(str(b("frac2_raw")))
    s1, s2 = nib.load(str(b("std1_raw"))), nib.load(str(b("std2_raw")))
    mask = nib.load(str(b("micro_mask"))).get_fdata(dtype=np.float32) > 0
    frac1 = np.clip(f1.get_fdata(dtype=np.float32), 0, 1)
    frac2 = np.clip(f2.get_fdata(dtype=np.float32), 0, 1)
    std1 = np.clip(s1.get_fdata(dtype=np.float32), 0, 1)
    std2 = np.clip(s2.get_fdata(dtype=np.float32), 0, 1)
    eps = float(recipe()["prep"]["frac_eps"])
    denom = frac1 + frac2
    valid = mask & (denom > eps)

    def scaled(a):
        o = np.zeros_like(a)
        o[valid] = a[valid] / denom[valid]
        return o
    frac1n, frac2n = scaled(frac1), scaled(frac2)
    std1n, std2n = scaled(std1), scaled(std2)

    def precision(stdn):
        var = stdn * stdn
        w = np.zeros_like(stdn, dtype=np.float32)
        w[valid] = var[valid].mean() / np.maximum(var[valid], eps)
        return np.clip(w, 0, 2).astype(np.float32)
    hdr = f1.header.copy()
    hdr.set_data_dtype(np.float32)
    base = b.mri_root / "micro"
    for arr, out in ((frac1n, b("frac1")), (frac2n, b("frac2")), (std1n, base / "std1_brain_norm.nii.gz"),
                     (std2n, base / "std2_brain_norm.nii.gz"), (precision(std1n), base / "w1_brain.nii.gz"),
                     (precision(std2n), base / "w2_brain.nii.gz")):
        nib.save(nib.Nifti1Image(arr, f1.affine, hdr), str(out))


def stage_raw(b, dwi_series, msme, micro_series):
    """Raw data onto the brain masks' grids (the three session masks in place under masks/). dwi_series / micro_series: the
    bias-corrected 4-D diffusion series (volume 0 = b0); msme: the denoised 32-echo MSME."""
    for d in ("dwi", "t2", "micro"):
        (b.mri_root / d).mkdir(parents=True, exist_ok=True)
    tmp = b.mri_root / "dwi" / "_b0_raw.nii.gz"
    _fslroi(dwi_series, tmp, 0)
    adopt_geometry(tmp, b("dwi_mask"), b("b0")); tmp.unlink()
    adopt_geometry(msme, b("t2_mask"), b.mri_root / "t2" / "msme.nii.gz")
    tmp = b.mri_root / "micro" / "_b0_raw.nii.gz"
    _fslroi(micro_series, tmp, 0)
    adopt_geometry(tmp, b("micro_mask"), b("micro_b0")); tmp.unlink()


def stage_dwi(b):
    """b0 x dwi brain mask, N4, normalized (the b0 of the canonical frame and of the within-MRI registrations)."""
    mask = b("dwi_mask")
    brain = b.mri_root / "dwi" / "b0_brain.nii.gz"
    mask_image(b("b0"), mask, brain)
    n4(brain, mask, b("b0_n4_brain"), bias=str(b("b0_n4_bias")))
    mask_image(b("b0_n4_brain"), mask, b("b0_n4_brain"), normalize=True)


def stage_t2(b):
    mask = b("t2_mask")
    extract_echo(b.mri_root / "t2" / "msme.nii.gz", b("t2_echo"))
    n4(b("t2_echo"), mask, b("t2_head"))
    mask_image(b("t2_head"), mask, b("t2_brain"), normalize=True)


def stage_micro(b, sandi_fracs_mean, sandi_fracs_std):
    """sandi_fracs_mean / _std: the SANDI fit's est.fracs.mean / est.fracs.std (volumes 0 = soma = frac1, 1 = neurite = frac2)."""
    mask = b("micro_mask")
    brain, brain_n4 = b.mri_root / "micro" / "b0_brain.nii.gz", b.mri_root / "micro" / "b0_brain_n4.nii.gz"
    mask_image(b("micro_b0"), mask, brain)                     # the micro b0 (N4) is the moving image of the micro -> b0 affine init
    n4(brain, mask, brain_n4)
    mask_image(brain_n4, mask, brain_n4, normalize=True)
    for i, key in ((0, "frac1_raw"), (1, "frac2_raw")):
        _fslroi(sandi_fracs_mean, b(key), i)
        mask_image(b(key), mask, b(key))
    for i, key in ((0, "std1_raw"), (1, "std2_raw")):
        _fslroi(sandi_fracs_std, b(key), i)
        mask_image(b(key), mask, b(key))
    normalize_fracs_and_weights(b)


def stage(b, preprocessed):
    """One brain's staged inputs from its three folders of preprocessed files (<preprocessed>/dwi, msme, dt2): stage_raw,
    stage_dwi, stage_t2, stage_micro (SANDIdot's fractions), and the tensor's diffusion files copied under dwi/raw/.
    Refuses when the brain's staged b0 exists (released data are never overwritten)."""
    import shutil
    pre = Path(preprocessed); d, t, m = pre / "dwi", pre / "msme", pre / "dt2" / "tortoise"
    if b("b0").exists():
        raise SystemExit(f"{b('b0')} exists: this brain is staged already")
    missing = [str(b(k)) for k in ("dwi_mask", "t2_mask", "micro_mask") if not b(k).exists()]
    if missing:
        raise SystemExit("the brain masks are inputs: missing " + ", ".join(missing))
    stage_raw(b, d / "dwi_biascorr.nii.gz", t / "msme_denoised_las.nii.gz", m / "dwi_biascorr.nii.gz")
    stage_dwi(b); stage_t2(b)
    sandi = m / "microstructure" / "SANDIdot"
    stage_micro(b, sandi / "est.fracs.mean.nii.gz", sandi / "est.fracs.std.nii.gz")
    for key, src in (("dwi_eddy", d / "dwi_eddycorrect.nii.gz"), ("dwi_ecclog", d / "dwi_eddycorrect.ecclog"),
                     ("dwi_bvecs", d / "dwi_orig.bvecs"), ("dwi_bvals", d / "dwi_orig.bvals"), ("dwi_fit_mask", d / "brain_mask.nii.gz")):
        b(key).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, b(key))


# --------------------------------------------------------------------------- 2. the D9 tensor (tensor_refit_d9 --from-existing)

def tensor_d9(eddy_series, ecclog, bvecs, bvals, fit_mask, dwi_mask, out_dir, threads=8, current_tensor=None):
    """dtifit on the eddy-corrected b <= bmax volumes with rotated, gradient-checked b-vectors, staged onto the dwi grid.

    eddy_series: the denoised, eddy_correct-ed 4-D series (all acquired volumes) and its .ecclog; bvecs / bvals: the acquired
    gradient table; fit_mask: the fit support; dwi_mask: the dwi brain mask (its header = the dwi grid). Writes
    out_dir/tensor_d9_dwigrid.nii.gz (+ run.json with the QC against current_tensor when given)."""
    from scipy.linalg import polar
    od = Path(out_dir); od.mkdir(parents=True, exist_ok=True)
    log = od / "commands.log"
    T0, t = time.time(), {}
    bmax = recipe()["prep"]["tensor_bmax"]
    env = fsl_env()
    bv = np.loadtxt(bvecs); bv = bv.T if bv.shape[1] == 3 and bv.shape[0] != 3 else bv
    bl = np.loadtxt(bvals).ravel()
    assert bv.shape == (3, bl.size), (bv.shape, bl.size)
    np.savetxt(od / "orig.bvecs", bv, fmt="%.6f"); np.savetxt(od / "orig.bvals", bl[None], fmt="%g")
    keep = np.where(bl <= bmax)[0]
    t0 = time.time()
    img = nib.load(str(eddy_series)); assert img.shape[3] == bl.size, img.shape
    nib.save(nib.Nifti1Image(np.asanyarray(img.dataobj)[..., keep].astype(np.float32), img.affine, img.header), str(od / "low_corr.nii.gz"))
    t["load_subset_s"] = round(time.time() - t0, 1)
    L = Path(ecclog).read_text().splitlines()
    mats = [np.array([[float(x) for x in L[i + k].split()] for k in range(1, 5)]) for i, line in enumerate(L) if line.startswith("Final result")]
    assert len(mats) == bl.size, len(mats)
    bvk, blk, rot_deg = bv[:, keep].copy(), bl[keep], []
    for j, i in enumerate(keep):
        R, _ = polar(mats[i][:3, :3]); bvk[:, j] = R @ bvk[:, j]
        rot_deg.append(float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))))
    np.savetxt(od / "low_corr.bvecs", bvk, fmt="%.6f"); np.savetxt(od / "low_corr.bvals", blk[None], fmt="%g")
    t0 = time.time()
    run([mrtrix("dwigradcheck"), od / "low_corr.nii.gz", "-fslgrad", od / "low_corr.bvecs", od / "low_corr.bvals", "-mask", fit_mask,
         "-export_grad_fsl", od / "low_checked.bvecs", od / "low_checked.bvals", "-nthreads", threads, "-force"], env=env, log=log)
    t["gradcheck_s"] = round(time.time() - t0, 1)
    chk = np.loadtxt(od / "low_checked.bvecs"); chk = chk if chk.shape[0] == 3 else chk.T
    nz = np.linalg.norm(bvk, axis=0) > 0.5
    best = min(((P, sg) for P in itertools.permutations(range(3)) for sg in itertools.product((1, -1), repeat=3)),
               key=lambda q: np.abs(np.abs((np.array(q[1])[:, None] * bvk[list(q[0])] * chk).sum(0))[nz] - 1).max()
               + np.abs(np.array(q[1])[:, None] * bvk[list(q[0])] - chk)[:, nz].max())
    flip = dict(permutation=list(best[0]), signs=list(best[1]),
                max_residual=float(np.abs(np.array(best[1])[:, None] * bvk[list(best[0])] - chk)[:, nz].max()))
    t0 = time.time()
    run([fsl("dtifit"), "-k", od / "low_corr.nii.gz", "-o", od / "dti", "-m", fit_mask, "-r", od / "low_checked.bvecs",
         "-b", od / "low_checked.bvals", "--save_tensor"], env=env, log=log)
    t["dtifit_s"] = round(time.time() - t0, 1)
    hm = nib.load(str(dwi_mask))
    Tn = np.asanyarray(nib.load(str(od / "dti_tensor.nii.gz")).dataobj).astype(np.float32)
    assert Tn.shape[:3] == hm.shape[:3]
    nib.save(nib.Nifti1Image(Tn, hm.affine, hm.header), str(od / "tensor_d9_dwigrid.nii.gz"))
    rec = dict(n_acquired=int(bl.size), n_used=int(len(keep)), bmax=bmax, bvalues_used=sorted(set(np.round(blk).astype(int).tolist())),
               ref_volume_index_in_subset=int(np.argmin(blk)), rotation_deg_median=float(np.median(rot_deg)),
               rotation_deg_max=float(np.max(rot_deg)), gradcheck_applied=flip, timings_s=t, threads=threads)
    if current_tensor is not None:
        rec["qc_vs_current_tensor"] = _tensor_qc(np.asanyarray(nib.load(str(current_tensor)).dataobj).astype(np.float64),
                                                 Tn.astype(np.float64), np.asanyarray(hm.dataobj) > 0)
    rec["wall_s"] = round(time.time() - T0, 1)
    (od / "run.json").write_text(json.dumps(rec, indent=1))
    return od / "tensor_d9_dwigrid.nii.gz"


def _tensor_qc(To, Tn, M):
    def eig(T6):
        t6 = T6[M]; A = np.zeros((len(t6), 3, 3))
        for k, (i, j) in enumerate([(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]):
            A[:, i, j] = A[:, j, i] = t6[:, k]
        w, v = np.linalg.eigh(A)
        md = w.mean(1); fa = np.sqrt(1.5 * ((w - md[:, None]) ** 2).sum(1) / np.clip((w ** 2).sum(1), 1e-30, None))
        return fa, v[:, :, -1]
    fa_o, v_o = eig(To); fa_n, v_n = eig(Tn)
    wm = (fa_o > 0.3) & (fa_n > 0.3)
    ang = np.degrees(np.arccos(np.clip(np.abs((v_o * v_n).sum(1)), 0, 1)))[wm]
    return dict(fa_r=float(np.corrcoef(fa_o, fa_n)[0, 1]), fa_median_current=float(np.median(fa_o)), fa_median_d9=float(np.median(fa_n)),
                v1_angle_deg_wm_median=float(np.median(ang)), v1_angle_deg_wm_p90=float(np.percentile(ang, 90)), wm_voxels=int(wm.sum()),
                tensor_zero_in_hand_mask=int((np.abs(Tn[M]).sum(-1) == 0).sum()))


# --------------------------------------------------------------------------- 5. construction inputs (rebuild_prep canon, finalhand)

def _apply(src, ref, transforms, dst, interp):
    cmd = [ants("antsApplyTransforms"), "-d", "3", "-i", str(src), "-r", str(ref), "-o", str(dst), "-n", interp, "--float", "1"]
    for t in transforms:
        cmd += ["-t", str(t)]
    run(cmd)


def check_masked(image, mask, tol=0.001, dist_mm=2.0, rel=0.01, label=""):
    """Fail closed if real signal lies away from the brain mask: voxels > dist_mm from the mask whose |value| exceeds rel x the
    in-brain median, over more than tol of the nonzero voxels (tensors: the component norm)."""
    from scipy import ndimage as ndi
    im, mi = nib.load(str(image)), nib.load(str(mask))
    d = np.asarray(im.dataobj); m = np.asarray(mi.dataobj) > 0.5
    a = np.sqrt((np.asarray(d, np.float64) ** 2).sum(axis=tuple(range(3, d.ndim)))) if d.ndim > 3 else np.abs(np.asarray(d, np.float64))
    nz = a > 0
    if nz.shape != m.shape:
        raise RuntimeError(f"{label}: image grid {nz.shape} != mask grid {m.shape}")
    n = int(nz.sum()); out = float((nz & ~ndi.binary_dilation(m)).sum() / max(n, 1))
    vox = np.sqrt((np.asarray(im.affine)[:3, :3] ** 2).sum(0))
    far = ndi.distance_transform_edt(~m, sampling=vox) > dist_mm
    med = float(np.median(a[m & nz])) if (m & nz).any() else 0.0
    bad = float((far & (a > rel * med)).sum() / max(n, 1))
    if bad > tol:
        raise RuntimeError(f"{label}: {100 * bad:.2f} % of nonzero voxels carry signal more than {dist_mm:g} mm from the brain mask")
    return dict(label=label, image=str(image), nonzero=n, out_dilated=round(out, 5), far_signal=round(bad, 5))


def prepare(b, final_mask, tensor, out, canonical_json=None, check_inputs=True):
    """The construction inputs of one brain (see the module docstring) -> out/. `b` is a Brain; final_mask the brain's final
    hand-edited mask on the dwi b0 grid (values exactly {0, 1}); tensor the D9 tensor on the dwi grid; canonical_json the brain's
    frame (default <mri_root>/landmarks/canonical.json)."""
    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists: refusing to overwrite")
    mask_src = Path(final_mask).resolve()
    b0 = nib.load(str(b("b0"))); mk = nib.load(str(mask_src))
    M = np.asanyarray(mk.dataobj)
    assert mk.shape[:3] == b0.shape[:3] and np.allclose(mk.affine, b0.affine, atol=1e-4), f"{mask_src}: not on the dwi/b0 grid"
    vals = np.unique(M)
    if not (set(vals.tolist()) <= {0.0, 1.0} and np.all(np.isfinite(vals))):
        raise SystemExit(f"{mask_src}: values are not exactly {{0, 1}} ({vals[:8]}) -> stop")
    out.mkdir(parents=True)
    hdr = mk.header.copy(); hdr.set_data_dtype(np.uint8)
    cast = out / "model_mask_source_uint8.nii.gz"
    nib.save(nib.Nifti1Image((M > 0.5).astype(np.uint8), mk.affine, hdr), str(cast))
    assert np.array_equal(np.asanyarray(nib.load(str(cast)).dataobj), (M > 0.5).astype(np.uint8))
    mask_md5 = dict(author_file=str(mask_src), author_md5=hashlib.md5(mask_src.read_bytes()).hexdigest(),
                    author_dtype=str(mk.header.get_data_dtype()), cast_md5=hashlib.md5(cast.read_bytes()).hexdigest())
    cj_path = Path(canonical_json) if canonical_json else b.mri_root / "landmarks" / "canonical.json"
    Tm = np.asarray(json.loads(cj_path.read_text())["T_subject_to_canonical"], float)
    ref = C.canonical_ref(out / "grid.nii.gz")
    T = C.itk_from_ras(Tm, out / "to_canonical.mat")
    chain = {"t2": [str(b.within_mri_warp("t2"))], "micro": [str(b.within_mri_warp("micro"))]}

    # the N4-corrected b0 HEAD on its native grid (the N4 field spans the grid)
    nat = out / "_native"; nat.mkdir()
    bias = np.asanyarray(nib.load(str(b("b0_n4_bias"))).dataobj, dtype=np.float64)
    assert (bias > 0).all(), "N4 bias field does not span the grid"
    nib.save(nib.Nifti1Image((np.asanyarray(b0.dataobj, dtype=np.float64) / bias).astype(np.float32), b0.affine), str(nat / "b0_head_n4.nii.gz"))

    payloads = [(b(k), sp, dst, interp) for k, sp, dst, interp in PAYLOADS] + \
               [(cast, None, "model_mask.nii.gz", "NearestNeighbor"), (nat / "b0_head_n4.nii.gz", None, "b0_head.nii.gz", "Linear")]
    for src, sp, dst, interp in payloads:
        _apply(src, ref, [T] + (chain[sp] if sp else []), out / dst, interp)     # canonical rigid first, then the warp
    for src, mask_name, dst in DERIVED:                                         # extraction + normalization on the canonical grid
        n = nib.load(str(out / src))
        m = np.asanyarray(nib.load(str(out / mask_name)).dataobj) > 0.5
        nib.save(nib.Nifti1Image(C.extract(n, m), n.affine, n.header), str(out / dst))
    C.renormalise_compositional(out)                                            # session-mask fractions on the carried micro mask
    rd = lambda f: np.asanyarray(nib.load(str(out / f)).dataobj)                # noqa: E731
    mm = rd("model_mask.nii.gz") > 0.5
    os.replace(out / "b0.nii.gz", out / "b0_dwimask.nii.gz")
    bh = nib.load(str(out / "b0_head.nii.gz"))
    nib.save(nib.Nifti1Image(C.extract(bh, mm), bh.affine, bh.header), str(out / "b0.nii.gz"))
    for k in (1, 2):                                                            # session-mask fractions x final mask
        os.replace(out / f"frac{k}.nii.gz", out / f"frac{k}_handmask.nii.gz")
        fi = nib.load(str(out / f"frac{k}_handmask.nii.gz")); a = np.asanyarray(fi.dataobj).astype(np.float32)
        a[~mm] = 0
        nib.save(nib.Nifti1Image(a, fi.affine, fi.header), str(out / f"frac{k}.nii.gz"))
    extra = ["b0_dwimask.nii.gz", "frac1_handmask.nii.gz", "frac2_handmask.nii.gz", "b0_head.nii.gz"]

    # tensor: vecreg with the canonical rigid as a FLIRT matrix (vecreg reorients), then cut to the final mask
    tn, g = nib.load(str(tensor)), nib.load(str(out / "grid.nii.gz"))
    np.savetxt(out / "native_to_canonical_fsl.mat", C.flirt_matrix(Tm, tn, g))
    tmp = out / "tensor_vr_unmasked.nii.gz"
    run([fsl("vecreg"), "-i", str(tensor), "-r", str(out / "grid.nii.gz"), "-t", str(out / "native_to_canonical_fsl.mat"), "-o", str(tmp)],
        env=fsl_env())
    tv = nib.load(str(tmp)); tvd = np.asanyarray(tv.dataobj).astype(np.float32)
    tvd[~mm] = 0
    nib.save(nib.Nifti1Image(tvd, tv.affine, tv.header), str(out / "tensor_vr.nii.gz"))
    nib.save(nib.Nifti1Image(mm.astype(np.uint8), tv.affine), str(out / "tensor_mask.nii.gz"))
    tmp.unlink()

    has = dict(frac=(rd("frac1.nii.gz") + rd("frac2.nii.gz")) > 0, tensor=np.abs(tvd).sum(-1) > 0,
               t2=rd("t2_modelmask.nii.gz") > 0, t2_head=rd("t2_head.nii.gz") > 0, b0=rd("b0.nii.gz") > 0,
               frac_handmask_old=(rd("frac1_handmask.nii.gz") + rd("frac2_handmask.nii.gz")) > 0, b0_dwimask_old=rd("b0_dwimask.nii.gz") > 0)
    vx = float(np.prod(nib.load(str(out / "model_mask.nii.gz")).header.get_zooms()[:3]))
    cov = dict(model_mask_voxels=int(mm.sum()), voxel_mm3=vx,
               **{f"{k}_covered": float(v[mm].mean()) for k, v in has.items()},
               **{f"{k}_uncovered_voxels": int((mm & ~v).sum()) for k, v in has.items()},
               **{f"{k}_outside_mask_voxels": int((~mm & v).sum()) for k, v in has.items() if k in ("frac", "tensor", "t2", "b0")})
    for dst in sorted(set([p[2] for p in payloads] + [d[2] for d in DERIVED] + ["tensor_mask.nii.gz"] + extra)):
        C.mirror_volume(out / dst, out / dst.replace(".nii.gz", "_flip.nii.gz"), kind="scalar")
    tmpf = out / "_tensor_vr_flip.nii.gz"
    C.mirror_volume(out / "tensor_vr.nii.gz", tmpf, kind="tensor"); os.replace(tmpf, out / "tensor_vr_flip.nii.gz")
    rec = dict(subject=b.subject, model_mask_source=str(mask_src), model_mask_md5=mask_md5, canonical_frame=str(cj_path),
               within_mri_warps={k: v[0] for k, v in chain.items()}, coverage_in_model_mask=cov,
               tensor_source=f"{tensor} via vecreg, zeroed outside model_mask", rule="finalhand",
               created=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    if check_inputs:
        rec["input_mask_check"] = [check_masked(out / c, out / "model_mask.nii.gz", label=f"{b.subject} {c}")
                                   for c in ("t2_modelmask.nii.gz", "frac1.nii.gz", "frac2.nii.gz", "tensor_vr.nii.gz")]
    from shared.provenance import write_provenance
    write_provenance(out, inputs=[mask_src, cj_path, tensor] + [p[0] for p in payloads[:-2]] + list(chain["t2"]) + list(chain["micro"]),
                     md5=False, extra=rec)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="the construction inputs of one brain")
    p.add_argument("--subject", required=True)
    p.add_argument("--final-mask", required=True, help="the brain's final hand-edited mask (dwi b0 grid, values {0, 1})")
    p.add_argument("--tensor", required=True, help="the D9 tensor on the dwi grid (tensor_d9)")
    p.add_argument("--mri-root"); p.add_argument("--xfm-root"); p.add_argument("--canonical")
    p.add_argument("--out", help="default <mri-root>/construction_inputs")
    g = sub.add_parser("stage", help="one brain's staged inputs from its preprocessed files")
    g.add_argument("--subject", required=True); g.add_argument("--preprocessed", required=True, help="the brain's folder: dwi/, msme/, dt2/")
    g.add_argument("--mri-root")
    t = sub.add_parser("tensor", help="the D9 tensor of one brain")
    for k in ("eddy-series", "ecclog", "bvecs", "bvals", "fit-mask", "dwi-mask", "out"):
        t.add_argument(f"--{k}", required=True)
    t.add_argument("--current-tensor"); t.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    if a.cmd == "stage":
        stage(Brain(a.subject, a.mri_root), a.preprocessed)
    elif a.cmd == "prepare":
        b = Brain(a.subject, a.mri_root, a.xfm_root)
        rec = prepare(b, a.final_mask, a.tensor, a.out or b.mri_root / "construction_inputs", a.canonical)
        print(json.dumps(rec["coverage_in_model_mask"], indent=1))
    else:
        print(tensor_d9(a.eddy_series, a.ecclog, a.bvecs, a.bvals, a.fit_mask, a.dwi_mask, a.out, a.threads, a.current_tensor))


if __name__ == "__main__":
    main()
