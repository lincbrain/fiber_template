"""Assemble a self-contained MMORF template-construction run from settings/mri_construction.yaml (paper: Methods, "Symmetric
groupwise construction").

    python -m template.mmorf_run --upstream <mm-template-construction clone> --out <runs dir>/rb_final \
        [--prep '$LINC_DATA/sub-{id}/mri/construction_inputs'] [--leave-out MF283]

Writes, under the run folder:
  data/<MOD>/<ID>[f]/<MOD>/...   the construction inputs per channel (each brain of `inputs` and, with run_assembly.mirror, its exact
                                 mirror), linked from the prep folders
                                 written by mri.mri_prep (scalar: <MOD>_unbiased_brain, <MOD>_brain_mask; tensor:
                                 <MOD>/dti_tensor, tensor_brain_mask). Brain images only: no head image, so the driver takes its
                                 brain-only path.
  subject_ids.csv                MF283, MF283f, ... (an "f" id is the mirror)
  standardspace.nii.gz           the voxelwise mean of the first-channel (T2) inputs: already in the canonical frame (the driver's
                                 6-dof alignment to it cannot rotate the template out of the frame), exactly symmetric with mirrors
  config/                        data_structure.tree (+ one .tree per channel), mmorf_params_stepNN.ini (one per step),
                                 affine_zero_warp_for_deltaWarp.nii.gz (zero displacement on the canonical grid)
  run_spec.json                  channels, weights, quantitative channels, steps, ids
`driver_command()` gives the patched driver's command line (code/template/patches/README.md).

MMORF .ini layout (MMORF 1.0.1): per-modality options that vary by level (lambda_scalar, fwhm_*, use_mask_*, lambda_bias_reg) are
one line per modality with one value per level; estimate_bias, bias_res_init and use_implicit_mask are one line with one value per
modality. Step s uses s + 1 levels. lambda_scale multiplies every lambda_reg term except the first (near-rigid) one. Quantitative
channels (the SANDI fractions) get estimate_bias = 0 here and --quant_scalar_mods in the driver (no bias division, no intensity
normalization), so their template still sums to 1.

Ported from the analysis project's mri/code/build_template_run.py (the run of record: --mods T2M:1 FRAC1M:0.5 FRAC2M:0.5
--tensor-mods DTIV:1 --lambda-scale 10).
"""
import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np

from template.tools import recipe

DATA_TREE_HEAD = """ext_nii=.nii.gz
ext_mat=.mat
ext_ini=.ini
ext_txt=.txt
"""


def _mods(R):
    """[(name, weight, quantitative)] for the scalar channels and [(name, weight)] for the tensor channels, in driver order."""
    return ([(m["name"], float(m["weight"]), bool(m["quantitative"])) for m in R["scalar_modalities"]],
            [(m["name"], float(m["weight"])) for m in R["tensor_modalities"]])


def step_ini(step, R=None):
    """The MMORF config of one nonlinear step (step + 1 levels)."""
    R = R or recipe()
    if R.get("fwhm_floor_applied"):
        raise NotImplementedError("the run of record did not apply a smoothing floor (fwhm_floor_applied: false)")
    sp = R["run_assembly"]["ini_spelling"]
    mods, tmods = _mods(R)
    n = step + 1
    lam = [sp["lambda_reg_first"]] + [f"{float(x) * float(R['lambda_scale']):.4g}" for x in R["lambda_reg"][1:]]
    L = lambda v: " ".join(str(x) for x in v[:n])                                     # noqa: E731
    fwhm = [str(float(x)) for x in R["fwhm_mm"]]
    head = (f"warp_res_init           = {R['warp_res_init_mm']}\n"
            f"warp_scaling            = {L(R['warp_scaling'])}\n"
            f"lambda_reg              = {L(lam)}\n"
            f"hires                   = {R['hires_mm']}\n"
            f"optimiser_max_it_lowres = {R['optimiser_max_it_lowres']}\n"
            f"optimiser_max_it_hires  = {R['optimiser_max_it_hires']}\n")
    body = ""
    cms, cmt = str(int(R["cost_mask_scalar"])), str(int(R["cost_mask_tensor"]))
    for name, w, quant in mods:
        body += (f"\n; --- scalar modality {name}{' (QUANTITATIVE)' if quant else ''}, lambda_scalar {w} ---\n"
                 f"use_mask_ref_scalar     = {' '.join([cms] * n)}\n"
                 f"use_mask_mov_scalar     = {' '.join([cms] * n)}\n"
                 f"fwhm_ref_scalar         = {L(fwhm)}\n"
                 f"fwhm_mov_scalar         = {L(fwhm)}\n"
                 f"lambda_scalar           = {' '.join([f'{w:g}'] * n)}\n"
                 f"lambda_bias_reg         = {' '.join([sp['lambda_bias_reg']] * n)}\n")
    for name, w in tmods:
        body += (f"\n; --- tensor modality {name}, lambda_tensor {w} ---\n"
                 f"use_mask_ref_tensor     = {' '.join([cmt] * n)}\n"
                 f"use_mask_mov_tensor     = {' '.join([cmt] * n)}\n"
                 f"fwhm_ref_tensor         = {L(fwhm)}\n"
                 f"fwhm_mov_tensor         = {L(fwhm)}\n"
                 f"lambda_tensor           = {' '.join([f'{w:g}'] * n)}\n")
    tail = ("\n; --- per-modality scalars (one value each, in modality order) ---\n"
            f"use_implicit_mask       = {' '.join(['0'] * len(mods))}\n"
            f"estimate_bias           = {' '.join('0' if q else '1' for _, _, q in mods)}\n"
            f"bias_res_init           = {' '.join([str(R['bias_res_init_mm'])] * len(mods))}\n")
    return head + body + tail


def _link(src, dst):
    """Hard link (as the run of record); a symbolic link across file systems."""
    dst = Path(dst)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        os.symlink(Path(src).resolve(), dst)


def default_prep():
    from shared import paths
    return str(paths.data_root() / "sub-{id}" / "mri" / "construction_inputs")


def assemble(out, upstream, prep=None, name=None, leave_out=None, R=None):
    """Assemble the run folder `out`. `prep` is the folder of each brain's construction inputs (mri.mri_prep) with
    "{id}" standing for the brain (default $LINC_DATA/sub-{id}/mri/construction_inputs). upstream = a clone of
    mm-template-construction (its tempconfig/data_structure.tree is extended). Returns run_spec."""
    R = R or recipe()
    out = Path(out)
    name = name or out.name
    subjects = list(R["inputs"])
    if leave_out:
        if leave_out not in subjects:
            raise SystemExit(f"--leave-out {leave_out!r} not in {subjects}")
        subjects = [s for s in subjects if s != leave_out]
    files = R["run_assembly"]["input_files"]
    versions = [("", "")] + ([("f", "_flip")] if R["run_assembly"].get("mirror", True) else [])   # the brain, then its mirror
    mods, tmods = _mods(R)
    pattern = prep or default_prep()
    prep = lambda s: Path(pattern.format(id=s))   # noqa: E731
    cfg = out / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    (out / "out").mkdir(parents=True, exist_ok=True)

    ids, vols = [], []
    for mi, (mod, _w, _q) in enumerate(mods):
        spec = files[mod]
        for s in subjects:
            for sid, suffix in [(s + tag, sfx) for tag, sfx in versions]:
                d = out / "data" / mod / sid / mod
                d.mkdir(parents=True, exist_ok=True)
                brain = prep(s) / f"{spec['brain']}{suffix}.nii.gz"
                mask = prep(s) / f"{spec['mask']}{suffix}.nii.gz"
                if not brain.exists():
                    raise SystemExit(f"{brain} missing: run mri.mri_prep for {s} first")
                _link(brain, d / f"{mod}_unbiased_brain.nii.gz")
                _link(mask, d / f"{mod}_brain_mask.nii.gz")
                if mi == 0:
                    ids.append(sid); vols.append(str(brain))
    for mod, _w in tmods:
        spec = files[mod]
        for s in subjects:
            for sid, suffix in [(s + tag, sfx) for tag, sfx in versions]:
                d = out / "data" / mod / sid / mod / mod
                d.mkdir(parents=True, exist_ok=True)
                for fname, src in [("dti_tensor", prep(s) / f"{spec['tensor']}{suffix}.nii.gz"),
                                   ("tensor_brain_mask", prep(s) / f"{spec['mask']}{suffix}.nii.gz")]:
                    if not src.exists():
                        raise SystemExit(f"{src} missing: run mri.mri_prep for {s} first")
                    _link(src, d / f"{fname}.nii.gz")
    (out / "subject_ids.csv").write_text("\n".join(ids) + "\n")

    ref = standardspace(vols, out / "standardspace.nii.gz")

    tree = DATA_TREE_HEAD
    for i, (mod, _w, _q) in enumerate(mods, start=1):
        tree += f"\n!{{scalar_mod_{i:02d}_dir}}\n    ->scalar_mod_{mod} (scalar_mod_{i:02d})\n"
    for i, (mod, _w) in enumerate(tmods, start=1):
        tree += f"\n!{{tensor_mod_{i:02d}_dir}}\n    ->tensor_mod_{mod} (tensor_mod_{i:02d})\n"
    tree += "\n!{config_dir}\n"
    tree += "".join(f"    mmorf_params_step{i:02d}{{ext_ini}} (mmorf_params_{i:02d})\n" for i in range(1, R["n_steps"] + 1))
    tree += "    affine_zero_warp_for_deltaWarp{ext_nii} (zero_warp_affine)\n\n"
    up = Path(upstream) / R["run_assembly"]["upstream"]["dir"] / "tempconfig" / "data_structure.tree"
    tree += up.read_text().split("!{template_dir}", 1)[1].join(["!{template_dir}", ""])     # upstream's output layout, unchanged
    (cfg / "data_structure.tree").write_text(tree)
    for mod, _w, _q in mods:
        (cfg / f"scalar_mod_{mod}.tree").write_text(
            "{sub_id} (scalar_sub_dir)\n"
            f"    {mod} (scalar_mod_dir)\n"
            f"        {mod}_unbiased_brain{{ext_nii}} (scalar_mod_brain)\n"
            f"        {mod}_brain_mask{{ext_nii}} (scalar_mod_brain_mask)\n")
    for mod, _w in tmods:
        (cfg / f"tensor_mod_{mod}.tree").write_text(
            "{sub_id} (tensor_sub_dir)\n"
            f"    {mod} (tensor_mod_dir)\n"
            f"        {mod} (tensor_mod_subdir)\n"
            "            dti_tensor{ext_nii} (tensor_mod)\n"
            "            tensor_brain_mask{ext_nii} (tensor_mod_brain_mask)\n")
    for i in range(1, R["n_steps"] + 1):
        (cfg / f"mmorf_params_step{i:02d}.ini").write_text(step_ini(i, R))
    nib.save(nib.Nifti1Image(np.zeros(ref.shape + (3,), np.float32), ref.affine), str(cfg / "affine_zero_warp_for_deltaWarp.nii.gz"))

    quant_idx = [i for i, (_m, _w, q) in enumerate(mods, start=1) if q]
    spec = {"name": name, "canonical_dir": Path(pattern).name,
            "modalities": [{"name": m, "lambda_scalar": w, "quantitative": q} for m, w, q in mods],
            "tensor_modalities": [{"name": m, "lambda_tensor": w} for m, w in tmods],
            "quant_scalar_mods": quant_idx, "lambda_scale": float(R["lambda_scale"]), "fwhm_floor": bool(R["fwhm_floor_applied"]),
            "n_steps": R["n_steps"], "subject_ids": ids}
    json.dump(spec, open(out / "run_spec.json", "w"), indent=1)
    return spec


def standardspace(vols, out):
    """The voxelwise mean (float64 accumulation, float32 output) of the inputs, on the first input's grid. Returns that grid's image."""
    acc = None
    for v in vols:
        arr = np.asanyarray(nib.load(v).dataobj).astype(np.float64)
        acc = arr if acc is None else acc + arr
    ref = nib.load(vols[0])
    nib.save(nib.Nifti1Image((acc / len(vols)).astype(np.float32), ref.affine), str(out))
    return ref


def driver_command(run, driver, python="python", batch=None, R=None):
    """The patched driver's command line for an assembled run (as the run of record was launched; batch = concurrent MMORF tasks
    per step, scheduling only)."""
    R = R or recipe()
    run = Path(run)
    spec = json.loads((run / "run_spec.json").read_text())
    d = R["run_assembly"]["driver_args"]
    cmd = [python, str(driver), "--input_scalar"] + [str(run / "data" / m["name"]) + "/" for m in spec["modalities"]]
    if spec.get("quant_scalar_mods"):
        cmd += ["--quant_scalar_mods"] + [str(i) for i in spec["quant_scalar_mods"]]
    if spec.get("tensor_modalities"):
        cmd += ["--input_tensor"] + [str(run / "data" / m["name"]) + "/" for m in spec["tensor_modalities"]]
    cmd += ["--config", str(run / "config") + "/", "--tree", str(run / "config" / "data_structure.tree"),
            "--standardspace", str(run / "standardspace.nii.gz"), "--subids", str(run / "subject_ids.csv"),
            "--dof", str(d["dof"]), "-o", str(run / "out") + "/", "--cpuq", d["cpuq"], "--gpuq", d["gpuq"],
            "-aff", "True", "-nln", "True", "-nres", str(d["n_resolutions"]), "-nit", str(d["n_iterations"])]
    b = batch if batch is not None else R["run_assembly"]["batch_of_record"]
    if b:
        cmd += ["-b"] + [str(x) for x in b]
    return cmd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prep", default=None, help="each brain's prep folder, {id} = the brain "
                    "(default $LINC_DATA/sub-{id}/mri/construction_inputs)")
    ap.add_argument("--upstream", required=True, help="clone of mm-template-construction (see code/template/patches/README.md)")
    ap.add_argument("--out", required=True, help="run folder to create")
    ap.add_argument("--name", default=None)
    ap.add_argument("--leave-out", default=None, help="leave-one-brain-out: drop this brain and its mirror")
    ap.add_argument("--driver", default=None, help="print the driver command for this (patched) driver")
    a = ap.parse_args(argv)
    spec = assemble(a.out, a.upstream, a.prep, a.name, a.leave_out)
    print(json.dumps(spec, indent=1))
    if a.driver:
        print(" ".join(driver_command(a.out, a.driver)))


if __name__ == "__main__":
    main()
