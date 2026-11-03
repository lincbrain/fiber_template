"""The MRI template channels from a finished construction run (paper: Methods, "Symmetric groupwise construction", "Template channels").

    python -m linc.construction.channels --run <runs dir>/rb_final --out $LINC_TEMPLATE/mri [-j 8]
        [--prep '$LINC_DATA/sub-{id}/mri/construction_inputs'] [--mri-root '$LINC_DATA/sub-{id}/mri'] [--xfm-root '$LINC_DATA/sub-{id}/xfm']

1. Every input of the run (4 brains + their mirrors) is read ONCE from its own native grid: the whole pull chain
       construction warp (<id>_scalar_mod_01_to_template_warp_resampled_unbiased_full of step 8 iteration 3: affine, warp and
       unbiasing; FSL relative, on the canonical grid)  ->  mirror x -> -x (mirrored inputs)  ->  inv(T_subject_to_canonical)
       ->  the within-MRI warp (T2 and micro payloads)
   is composed into one FSL relative field on the template grid (linc.xfm.compose) and applied with applywarp: scalars spline,
   masks trilinear. The tensor keeps two steps: its canonical input (tensor_vr, reoriented by vecreg -t in mri_prep) through the
   construction warp with vecreg -w --rotwarp (vecreg -w alone applies the field's rotation times the reference voxel size, 25 % at
   0.25 mm; --rotwarp with the field divided by the voxel size applies all of it).
   The T2 is extracted in template space: the warped T2 head inside the warped final mask (>= 0.5), 1-99 percentile -> (0, 4095].
   The fractions are renormalized per input inside the warped micro mask, the final mask and where their sum is > 0.
2. Channels, each the mean over the inputs whose warped final mask is >= 0.5 (`mask_threshold`):
     T2, b0         each input scaled to mean 1000 over the voxels all inputs cover, then averaged; clipped at 0
     frac1, frac2   averaged without scaling (quantitative), clipped to [0, 1], renormalized to sum 1 where the sum is > 0
     tensor         log-Euclidean mean (eigenvalues floored at 1e-6 x the median MD), symmetrized in the log domain with the exact
                    mirror rule, then exp; FA, MD, MO, L1-3, V1-3 from fslmaths -tensor_decomp; DEC = |V1| x FA (uint8)
     nmap           number of inputs covering each voxel
     brainmask      mean warped final mask > 0.5; core_t2 / core_micro: coverage >= 0.999, eroded 6 voxels; wm_fa025: FA > 0.25
   Every scalar channel is symmetrized (A + flip(A)) / 2 and every label channel ANDed with its mirror, so each is exactly symmetric;
   T2, b0 and the fractions are zeroed outside the union of the warped masks and its mirror.
3. Acceptance: T2 r >= 0.99 against the driver's own (symmetrized) T2 template inside core_t2; V1 median angle < 2 deg against the
   driver's tensor template inside wm_fa025; FA in [0, 1]; exact symmetry.

Ported from the analysis project's mri/code/ship_template.py (the final path: --from-native, the rebuild profile, --tensor-stem
tensor_vr, --rotwarp) and the pull chain of mri/code/native_chain.py.
"""
import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion

from .. import xfm
from . import canonical as C
from .mri_prep import Brain
from .tools import fsl, recipe, sh

FLIP = np.diag([-1.0, 1.0, 1.0, 1.0])
FIELD_SRC = {"dwi": "b0", "t2": "t2_head", "micro": "frac1"}       # the native image whose grid each composed field points into


def _k():
    return recipe()["channels"]


# --------------------------------------------------------------------------- the pull chain (native_chain)

class _Step:
    def __init__(self, obj, label):
        self.obj, self.label = obj, label

    def pull(self, X):
        return self.obj.pull(X)


class PointChain:
    """Steps applied in listed order: template point -> ... -> native point (the interface linc.xfm.compose/apply need)."""

    def __init__(self, steps, label=""):
        self.steps, self.label, self.links = list(steps), label, []

    def pull(self, X):
        for s in self.steps:
            X = s.pull(X)
        return X

    def describe(self):
        return [s.label for s in self.steps]


class Inputs:
    """Where one run's inputs live: the run folder, each brain's prep folder, MRI folder and transforms ({id} = the brain)."""

    def __init__(self, run, prep=None, mri_root=None, xfm_root=None, step=None, iteration=None):
        from .. import paths
        k = _k()
        self.run = Path(run)
        self.it = self.run / "out" / f"nln_step_{int(step or k['step']):02d}" / f"iteration_{int(iteration or k['iteration']):02d}"
        self.prep = prep or str(paths.data_root() / "sub-{id}" / "mri" / "construction_inputs")
        self.mri_root, self.xfm_root = mri_root, xfm_root
        self.mod1 = recipe()["scalar_modalities"][0]["name"]
        self.ids = [i for i in (self.run / "subject_ids.csv").read_text().split() if i]

    def brain(self, subject):
        f = lambda p: p.format(id=subject) if p else None             # noqa: E731
        return Brain(subject, f(self.mri_root), f(self.xfm_root))

    def prep_dir(self, subject):
        return Path(self.prep.format(id=subject))

    def native(self, subject, payload):
        """(native file, space) of a payload (the rebuild's inputs: N4 b0 head, the final mask, hand-mask fractions)."""
        b, p = self.brain(subject), self.prep_dir(subject)
        return {"b0": (p / "_native" / "b0_head_n4.nii.gz", "dwi"), "model_mask": (p / "model_mask_source_uint8.nii.gz", "dwi"),
                "t2_head": (b("t2_head"), "t2"), "frac1": (b("frac1"), "micro"), "frac2": (b("frac2"), "micro"),
                "micro_mask": (b("micro_mask"), "micro")}[payload]

    def warp(self, sid):
        return self.it / f"{sid}_scalar_mod_01_to_template_warp_resampled_unbiased_full.nii.gz"

    def chain(self, sid, space):
        subject, mirror = sid.rstrip("f"), sid.endswith("f")
        src = self.run / "data" / self.mod1 / sid / self.mod1 / f"{self.mod1}_unbiased_brain.nii.gz"     # the canonical grid
        steps = [_Step(xfm.Warp(self.warp(sid), "fsl_rel", src_grid=src, label=sid), f"construction warp {sid}")]
        if mirror:
            steps.append(_Step(xfm.Affine(FLIP), "mirror x -> -x (canonical)"))
        b = self.brain(subject)
        T = np.asarray(json.loads((b.mri_root / "landmarks" / "canonical.json").read_text())["T_subject_to_canonical"], float)
        steps.append(_Step(xfm.Affine(np.linalg.inv(T)), f"inv(T_subject_to_canonical) {subject}"))
        if space in ("t2", "micro"):
            steps.append(_Step(xfm.Warp(b.within_mri_warp(space), "itk_lps_disp"), f"{space}_to_dwi warp"))
        return PointChain(steps, f"{sid}:{space}")


# --------------------------------------------------------------------------- 1. each input, once from native

def load(p):
    return np.asanyarray(nib.load(str(p)).dataobj).astype(np.float32)


def save(arr, ref, path, dtype=np.float32, intent=None):
    h = ref.header.copy()
    h.set_data_dtype(dtype)
    img = nib.Nifti1Image(arr.astype(dtype), ref.affine, h)
    if intent is not None:
        img.header.set_intent(intent)
    nib.save(img, str(path))
    os.chmod(path, 0o664)
    return path


def native_field(I, sid, space, ref_p, work):
    f = work / "_fields" / f"{sid}__{space}.nii.gz"
    if f.exists():
        return f
    f.parent.mkdir(parents=True, exist_ok=True)
    src, _ = I.native(sid.rstrip("f"), FIELD_SRC[space])
    tmp = f.with_name(f.name.replace(".nii.gz", ".part.nii.gz"))
    xfm.compose(I.chain(sid, space), ref_p, tmp, fmt="fsl_rel", src_grid=src)
    os.replace(tmp, f)
    return f


def _rotwarp(field, ref, work):
    """' --rotwarp=<field / voxel size>' for vecreg -w (written once into the work folder)."""
    vx = float(nib.load(str(ref)).header.get_zooms()[0])
    rot = work / "_rotwarp" / Path(str(field)).name.replace(".nii.gz", "_rotwarp.nii.gz")
    rot.parent.mkdir(parents=True, exist_ok=True)
    if not rot.exists():
        sh(f"{fsl('fslmaths')} {field} -div {vx!r} {rot}")
    return f" --rotwarp={rot}"


def warp_inputs(I, sid, ref_p, work, stems=("b0", "model_mask", "micro_mask", "frac1", "frac2", "tensor", "t2_modelmask")):
    """Every payload of one construction input onto the template grid -> work/<sid>__<stem>.nii.gz."""
    interp = _k()["interp"]
    base = sid.rstrip("f")
    W = lambda stem: work / f"{sid}__{stem}.nii.gz"                     # noqa: E731

    def resample(stem, payload, kind):
        out = W(stem)
        if out.exists():
            return out
        src, space = I.native(base, payload)
        fld = native_field(I, sid, space, ref_p, work)
        tmp = out.with_name(out.name.replace(".nii.gz", ".part.nii.gz"))
        sh(f"{fsl('applywarp')} -i {src} -r {ref_p} -w {fld} --rel --interp={interp[kind]} -o {tmp}")
        os.replace(tmp, out)
        return out

    need = set(stems)
    if "t2_modelmask" in need:
        need.add("model_mask")
    if need & {"frac1", "frac2"}:
        need |= {"frac1", "frac2", "micro_mask", "model_mask"}
    for stem, kind in (("b0", "scalar"), ("model_mask", "mask"), ("micro_mask", "mask"), ("frac1", "scalar"), ("frac2", "scalar")):
        if stem in need:
            resample(stem, stem, kind)
    if "tensor" in stems and not W("tensor").exists():                  # two steps: canonical tensor_vr, then the construction warp
        src = I.prep_dir(base) / f"tensor_vr{'_flip' if sid.endswith('f') else ''}.nii.gz"
        tmp = W("tensor").with_name(W("tensor").name.replace(".nii.gz", ".part.nii.gz"))
        sh(f"{fsl('vecreg')} -i {src} -r {ref_p} -w {I.warp(sid)}{_rotwarp(I.warp(sid), ref_p, work)} -o {tmp}")
        os.replace(tmp, W("tensor"))
    ref = nib.load(str(ref_p))
    if "t2_modelmask" in stems and not W("t2_modelmask").exists():     # extraction in template space
        head = resample("_t2_head", "t2_head", "scalar")
        m = load(W("model_mask")) >= _k()["mask_threshold"]
        save(C.extract(nib.Nifti1Image(load(head).astype(np.float64), ref.affine), m).astype(np.float64), ref, W("t2_modelmask"))
    if {"frac1", "frac2"} & set(stems):                                 # per-input renormalization
        f1, f2 = load(W("frac1")).astype(np.float64), load(W("frac2")).astype(np.float64)
        tot = f1 + f2
        m = (load(W("micro_mask")) >= 0.5) & (tot > 0) & (load(W("model_mask")) >= 0.5)
        save(np.where(m, f1 / np.where(tot > 0, tot, 1), 0.0), ref, W("frac1"))
        save(np.where(m, f2 / np.where(tot > 0, tot, 1), 0.0), ref, W("frac2"))
    return sid


# --------------------------------------------------------------------------- 2. channels

def sym_scalar(a):
    return 0.5 * (a + np.flip(a, axis=0))


def sym_label(m):
    return np.logical_and(m, np.flip(m, axis=0))


def core(frac):
    k = _k()
    m = frac >= k["core_threshold"]
    return binary_erosion(m, iterations=k["core_erode_vox"]) if k["core_erode_vox"] else m


def tensor_mats(t):
    M = np.empty(t.shape[:-1] + (3, 3), np.float64)
    M[..., 0, 0], M[..., 0, 1], M[..., 0, 2] = t[..., 0], t[..., 1], t[..., 2]
    M[..., 1, 0], M[..., 1, 1], M[..., 1, 2] = t[..., 1], t[..., 3], t[..., 4]
    M[..., 2, 0], M[..., 2, 1], M[..., 2, 2] = t[..., 2], t[..., 4], t[..., 5]
    return M


def mats_tensor(M):
    return np.stack([M[..., 0, 0], M[..., 0, 1], M[..., 0, 2], M[..., 1, 1], M[..., 1, 2], M[..., 2, 2]], -1)


def spd_fun(M, fun, floor):
    w, v = np.linalg.eigh(M)
    w = fun(np.maximum(w, floor)) if fun is np.log else fun(w)
    return np.einsum("...ij,...j,...kj->...ik", v, w, v)


def principal(M):
    _, v = np.linalg.eigh(M)
    return v[..., :, -1]


def coverage(ids, work):
    """Warped final masks (>= mask_threshold), stacked over inputs, and the voxels every input covers."""
    thr = _k()["mask_threshold"]
    masks = np.stack([load(work / f"{s}__model_mask.nii.gz") >= thr for s in ids])
    return masks, masks.all(0)


def wmean(stem, ids, work, masks, overlap, normalise):
    """Mean over the inputs covering each voxel; with normalise, each input first scaled to intensity_mean over `overlap`."""
    num = np.zeros(masks.shape[1:], np.float64); den = np.zeros(masks.shape[1:], np.float64)
    scales = {}
    for k, s in enumerate(ids):
        v = load(work / f"{s}__{stem}.nii.gz").astype(np.float64)
        if normalise:
            sc = float(_k()["intensity_mean"]) / v[overlap].mean(); v *= sc; scales[s] = sc
        num += v * masks[k]; den += masks[k]
    return np.where(den > 0, num / np.maximum(den, 1), 0.0), scales


def anatomical_channel(stem, ids, work, masks, overlap):
    """T2 (stem t2_modelmask) or b0: normalized mean, clipped at 0, symmetrized, zeroed outside the masks' union and its mirror."""
    v, scales = wmean(stem, ids, work, masks, overlap, True)
    v = sym_scalar(np.clip(v, 0, None))
    anym = masks.any(0)
    return np.where(anym | np.flip(anym, 0), v, 0.0), scales


def build(I, out, j=8, prefix=None):
    """All channels of the run -> out/<prefix>_<channel>.nii.gz; returns the acceptance record."""
    k = _k()
    prefix = (prefix or k["prefix"]) + "_"
    out = Path(out); work = out / "_work"; work.mkdir(parents=True, exist_ok=True)
    ids, it = I.ids, I.it
    ref_p = next(it.glob("nln_template_*_scalar_mod_01_brain.nii.gz"))
    ref = nib.load(str(ref_p))
    with ThreadPoolExecutor(max(1, j // 2)) as ex:
        list(ex.map(lambda sid: warp_inputs(I, sid, ref_p, work), ids))
    W = lambda sid, stem: work / f"{sid}__{stem}.nii.gz"               # noqa: E731

    masks, overlap = coverage(ids, work)
    nmap = masks.sum(0).astype(np.float32)
    cover = nmap / len(ids)
    cmask = np.stack([load(W(s, "micro_mask")) >= 0.5 for s in ids])
    ch = {}
    ch["T2"], sc_t2 = wmean("t2_modelmask", ids, work, masks, overlap, True)
    ch["b0"], sc_b0 = wmean("b0", ids, work, masks, overlap, True)
    f1, _ = wmean("frac1", ids, work, masks, overlap, False); f2, _ = wmean("frac2", ids, work, masks, overlap, False)
    f1, f2 = np.clip(f1, 0, 1), np.clip(f2, 0, 1)
    s12 = f1 + f2
    renorm_n = int(((s12 > 0) & (np.abs(s12 - 1) > 1e-3)).sum())
    ch["frac1"] = np.where(s12 > 0, f1 / np.maximum(s12, 1e-12), 0.0)
    ch["frac2"] = np.where(s12 > 0, f2 / np.maximum(s12, 1e-12), 0.0)
    for key in ("T2", "b0"):
        ch[key] = np.clip(ch[key], 0, None)
    for key in list(ch):
        ch[key] = sym_scalar(ch[key])
    ch["nmap"] = sym_scalar(nmap)
    brain = sym_label(masks.mean(0) > 0.5)
    ch["brainmask"] = brain
    ch["core_t2"] = sym_label(core(cover))
    ch["core_micro"] = sym_label(core(cmask.mean(0)) & core(cover))
    for key in ("T2", "b0", "frac1", "frac2"):
        ch[key] = np.where(masks.any(0) | np.flip(masks.any(0), 0), ch[key], 0.0)

    # tensor: log-Euclidean mean, symmetrized in the log domain
    sel = np.where(masks.any(0) | np.flip(masks.any(0), 0))
    md_all = []
    for s in ids[:2]:
        t = load(W(s, "tensor"))[sel]
        md_all.append(np.median((t[:, 0] + t[:, 3] + t[:, 5])[masks.any(0)[sel]] / 3))
    floor = float(k["tensor_log_floor_rel_md"]) * float(np.median(md_all))
    L = np.zeros((len(sel[0]), 3, 3)); cnt = np.zeros(len(sel[0]))
    for kk, s in enumerate(ids):
        m = masks[kk][sel]
        t = load(W(s, "tensor"))[sel][m]
        L[m] += spd_fun(tensor_mats(t.astype(np.float64)), np.log, floor)
        cnt[m] += 1
    L /= np.maximum(cnt, 1)[:, None, None]
    Lvol = np.zeros(ref.shape + (6,), np.float32)
    Lvol[sel] = mats_tensor(L)
    Lvol = 0.5 * (Lvol + C.mirror_array(Lvol, "tensor"))
    covered = np.zeros(ref.shape, bool); covered[sel] = cnt > 0
    covered = covered | np.flip(covered, 0)
    idx = np.where(covered)
    T = np.zeros(ref.shape + (6,), np.float32)
    T[idx] = mats_tensor(spd_fun(tensor_mats(Lvol[idx].astype(np.float64)), np.exp, None))
    tensor_p = save(T, ref, out / f"{prefix}tensor.nii.gz")
    sh(f"{fsl('fslmaths')} {tensor_p} -tensor_decomp {work / 'dec'}")
    for q in ("FA", "MD", "MO", "L1", "L2", "L3", "V1", "V2", "V3"):
        dst = out / f"{prefix}{q}.nii.gz"
        shutil.copyfile(work / f"dec_{q}.nii.gz", dst); os.chmod(dst, 0o664)
    FA = load(out / f"{prefix}FA.nii.gz"); V1 = load(out / f"{prefix}V1.nii.gz")
    ch["wm_fa025"] = sym_label((FA > float(k["wm_fa_threshold"])) & brain)
    dec = np.abs(V1) * np.clip(FA, 0, 1)[..., None]
    save(np.round(255 * np.clip(dec, 0, 1)), ref, out / f"{prefix}DEC.nii.gz", np.uint8)
    for key, v in ch.items():
        save(v, ref, out / f"{prefix}{key}.nii.gz", np.uint8 if v.dtype == bool else np.float32)

    # acceptance
    acc = {}
    raw = load(ref_p).astype(np.float64)
    drv = 0.5 * (raw + np.flip(raw, 0))
    c = ch["core_t2"]
    acc["T2_r_vs_driver_sym_core"] = float(np.corrcoef(ch["T2"][c], drv[c])[0, 1])
    drvT = load(next(it.glob("nln_template_*_tensor_mod_01.nii.gz")))
    drvT = 0.5 * (drvT + C.mirror_array(drvT, "tensor"))
    wm = ch["wm_fa025"]
    v_d = principal(tensor_mats(drvT[wm].astype(np.float64)))
    v_s = principal(tensor_mats(T[wm].astype(np.float64)))
    ang = np.degrees(np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", v_d, v_s)), 0, 1)))
    acc["V1_angle_vs_driver_wm_median_deg"] = float(np.median(ang))
    acc["V1_angle_vs_driver_wm_p95_deg"] = float(np.percentile(ang, 95))
    acc["FA_min"], acc["FA_max"] = float(FA.min()), float(FA.max())
    acc["max_abs_mirror_diff"] = {key: float(np.abs(v.astype(np.float32) - np.flip(v, 0).astype(np.float32)).max()) for key, v in ch.items()}
    tm = float(np.abs(T - C.mirror_array(T, "tensor")).max())
    acc["tensor_max_abs_mirror_diff"] = tm
    acc["frac_renormalised_voxels_off_1e-3"] = renorm_n
    acc["voxels"] = {"brainmask": int(brain.sum()), "core_t2": int(c.sum()), "core_micro": int(ch["core_micro"].sum()), "wm_fa025": int(wm.sum())}
    a = k["acceptance"]
    acc["pass"] = {f"T2_r>={a['t2_r_min']}": acc["T2_r_vs_driver_sym_core"] >= a["t2_r_min"],
                   f"V1_median<{a['v1_median_deg_max']}deg": acc["V1_angle_vs_driver_wm_median_deg"] < a["v1_median_deg_max"],
                   "FA_in_[0,1]": acc["FA_min"] >= 0 and acc["FA_max"] <= 1,
                   "exact_symmetry": max(acc["max_abs_mirror_diff"].values()) == 0 and tm <= a["tensor_symmetry_abs"]}
    acc["intensity_scales"] = {"T2": sc_t2, "b0": sc_b0}
    from ..provenance import write_provenance
    write_provenance(out, inputs=[I.warp(s) for s in ids] + [ref_p], outputs=sorted(out.glob(f"{prefix}*.nii.gz")), md5=False,
                     extra={"run": str(I.run), "step": it.parent.name, "iteration": it.name, "acceptance": acc})
    return acc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="the construction run folder (linc.construction.mmorf_run)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prep", default=None, help="each brain's prep folder, {id} = the brain")
    ap.add_argument("--mri-root", default=None, help="each brain's MRI folder, {id} = the brain (default $LINC_DATA/sub-{id}/mri)")
    ap.add_argument("--xfm-root", default=None, help="each brain's transforms, {id} = the brain (default $LINC_DATA/sub-{id}/xfm)")
    ap.add_argument("--step", type=int, default=None); ap.add_argument("--iteration", type=int, default=None)
    ap.add_argument("-j", type=int, default=8)
    a = ap.parse_args(argv)
    I = Inputs(a.run, a.prep, a.mri_root, a.xfm_root, a.step, a.iteration)
    print(json.dumps(build(I, a.out, a.j), indent=1))


if __name__ == "__main__":
    main()
