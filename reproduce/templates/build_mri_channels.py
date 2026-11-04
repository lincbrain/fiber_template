#!/usr/bin/env python3
"""Build the MRI channels of the template.

You need, for every brain of configs/mri_construction.yaml `inputs`, under $LINC_DATA/sub-<ID>/ (the config's `prep.layout`)
  either its Bruker sessions under $LINC_RAW (the config's `raw`; the raw-* and stage-* steps make the next two items from them)
  or, from the data release, mri/dwi/raw/: the denoised, eddy-corrected diffusion series with its .ecclog, b-vectors, b-values and
  fit mask, and mri/dwi/, t2/, micro/: the staged b0, the 32-echo MSME and the SANDI fractions (start with --from tensor-<ID>);
  mri/masks/: the hand masks of each session and the final hand-edited brain mask (inputs drawn by hand); from the data release
  or made by the nmt-, t2reg- and microreg-<ID> steps: mri/landmarks/ (the SARM labels and the brain -> NMT v2 affine on the b0
  grid) and xfm/t2-native__to__b0-native, xfm/micro-native__to__b0-native (the within-brain registrations);
  FSL (FSLDIR), ANTs (ANTSPATH), MRtrix (MRTRIX), for the raw steps Julia (JULIA, env/julia) and TORTOISE (TORTOISE, a GPU), and
  for the mmorf step MMORF (LINC_MMORF_EXE), the patched template-construction driver (LINC_TEMPLATE_DRIVER,
  linc/construction/patches/README.md) and one 48 GB GPU.

What it does, one step per brain, then together (each step one command; --plan prints them all)
  raw-dwi-<ID>    the diffusion session from its Bruker scans: conversion, denoising, eddy-current and bias correction
  raw-msme-<ID>   the multi-echo spin-echo scan: conversion, denoising
  raw-dt2-<ID>    the microstructure session: TORTOISE's motion and eddy-current correction (GPU), then SANDI
  stage-<ID>      stages those outputs as the build's inputs in the brain's MRI folder (b0, T2 and b0 N4-corrected, fractions)
  nmt-<ID>        registers NMT v2 to the b0 and pulls its SARM labels onto it (for the AC, the PC and the frame's pitch)
  t2reg-<ID>      registers the T2 echo to the b0; microreg-<ID> the microstructure session's b0 to the b0 (one field each)
  tensor-<ID>     fits the diffusion tensor on the b <= 4800 volumes
  landmarks-<ID>  finds the AC and PC from the SARM labels on the b0
  frames-<ID>     maps the brain rigidly into the canonical frame (its own mid-sagittal plane, NMT's orientation, the AC)
  prepare-<ID>    resamples every channel once onto the canonical grid, cuts it to the final mask, and writes its exact mirror
  assemble        writes the MMORF run folder: the inputs, the standard space and one parameter file per step
  mmorf           runs the groupwise construction (the config's schedule; the record: 8 steps x 3 iterations, 3 d 12 h on one GPU)
  channels        makes the template channels from the final step's warps
  links-<ID>      registers the brain's links up to the template: b0-native -> mri-canonical (the frame), mri-canonical -> mri-template (the warp)

You get, in the brain's $LINC_DATA/sub-<ID>/, never over existing files: mri/ (the staged inputs; landmarks/ with the frame canonical.json) and
xfm/ (the transforms). Under --work (default $LINC_WORK/build_mri_channels):
  raw/<ID>/{dwi,msme,dt2}/ (linc.construction.mri_raw), tensor_d9/<ID>/tensor_d9_dwigrid.nii.gz, prep/<ID>/ (the construction inputs and their
  _flip mirrors, NIfTI on the 0.25 mm canonical grid), runs/<run>/ (the MMORF run), channels/linc_mri_*.nii.gz + provenance.json
  (T2, b0, fractions, tensor and its scalars, masks: the template's mri folder)

    python reproduce/templates/build_mri_channels.py --plan
    python reproduce/templates/build_mri_channels.py --from raw-dwi-MF283 --to prepare-MF283   # one brain from its Bruker sessions
    python reproduce/templates/build_mri_channels.py --from tensor-MF283 --to prepare-MF283    # one brain from the data release
    python reproduce/templates/build_mri_channels.py --step prepare-MF283
    python reproduce/templates/build_mri_channels.py --from assemble --to channels

Environment: env/linc.yml. Steps and parameters: reproduce/templates/mri_channels.md and configs/mri_construction.yaml (paper: Methods,
"MRI preprocessing", "Symmetric groupwise construction", "Template channels"). Validated on MF283 (mri_channels.md, "Validation").
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _driver as D  # noqa: E402

SHIM = str(D.REPO / "linc" / "construction" / "blockface_shim")          # `mmorf` on PATH -> $LINC_MMORF_EXE (its TPL rule never fires here)


def make_steps(W, paths):
    from linc.construction.tools import recipe
    R = recipe(); L = R["prep"]["layout"]; ids = list(R["inputs"])               # the brains, grids and schedule all come from the config
    mirror = R["run_assembly"].get("mirror", True); n_in = len(ids) * (2 if mirror else 1)
    last = Path("out") / f"nln_step_{R['n_steps']:02d}" / f"iteration_{R['iterations_per_step']:02d}"            # the final step's warps
    read = Path("out") / f"nln_step_{R['channels']['step']:02d}" / f"iteration_{R['channels']['iteration']:02d}"   # the warps the channels read
    scal = [m["name"] for m in R["scalar_modalities"]]; quant = [str(i) for i, m in enumerate(R["scalar_modalities"], start=1) if m.get("quantitative")]
    tens = [m["name"] for m in R["tensor_modalities"]]; batch = " ".join(str(b) for b in R["run_assembly"]["batch_of_record"])
    w = lambda *p: W.joinpath(*p)                                                     # noqa: E731
    sub = lambda S: paths.subject_dir(S)                                              # noqa: E731
    m = lambda S, k: sub(S) / "mri" / L[k]                                            # noqa: E731
    run = w("runs", R["rebuild_run"])
    bruker = Path(os.environ.get("LINC_RAW", "$LINC_RAW")) / R["raw"]["root"]       # the Bruker sessions (config `raw`)
    drv_dir = os.environ.get("LINC_TEMPLATE_DRIVER", "$LINC_TEMPLATE_DRIVER")
    driver_py = str(Path(drv_dir) / "run_template_construction.py")
    steps = []
    for S in ids:
        tensor = w("tensor_d9", S, "tensor_d9_dwigrid.nii.gz")
        lm, cj = sub(S) / "mri" / "landmarks" / "landmarks.json", sub(S) / "mri" / "landmarks" / "canonical.json"
        ses, raw = R["raw"]["sessions"][S], lambda *q: w("raw", S, *q)               # noqa: E731
        sandi = raw("dt2", "tortoise", "microstructure", "SANDIdot")
        steps += [
            D.Step(f"raw-dwi-{S}", f"per brain: {S}", "mri",
                   ["-m", "linc.construction.mri_raw", "dwi", "--session", bruker / ses["dwi"][0], "--scans", ses["dwi"][1], "--out", raw("dwi")],
                   outputs=[raw("dwi", "dwi_biascorr.nii.gz")], inputs=[bruker / ses["dwi"][0]],
                   est="4 h (CPU; eddy_correct 3 h)", note="Julia (env/julia), MRtrix3, FSL, ANTs; resumes a stopped run"),
            D.Step(f"raw-msme-{S}", f"per brain: {S}", "mri",
                   ["-m", "linc.construction.mri_raw", "msme", "--session", bruker / ses["msme"][0], "--scan", ses["msme"][1], "--out", raw("msme")],
                   outputs=[raw("msme", "msme_denoised_las.nii.gz")], inputs=[bruker / ses["msme"][0]], est="1 min (CPU)"),
            D.Step(f"raw-dt2-{S}", f"per brain: {S}", "mri",
                   ["-m", "linc.construction.mri_raw", "dt2", "--session", bruker / ses["dt2"][0], "--scans", ses["dt2"][1], "--out", raw("dt2")],
                   outputs=[sandi / "est.fracs.mean.nii.gz"], inputs=[bruker / ses["dt2"][0]], gpu=True,
                   est="30 min (TORTOISE 16 min on a GPU, SANDI 4 min on the CPU)", note="TORTOISE V4.1.0 ($TORTOISE)"),
            D.Step(f"stage-{S}", f"per brain: {S}", "mri",
                   ["-m", "linc.construction.mri_prep", "stage", "--subject", S, "--raw", w("raw", S)],
                   outputs=[m(S, k) for k in ("b0", "b0_n4_brain", "t2_head", "t2_brain", "frac1", "frac2", "dwi_eddy")],
                   inputs=[raw("dwi", "dwi_biascorr.nii.gz"), raw("msme", "msme_denoised_las.nii.gz"), sandi / "est.fracs.mean.nii.gz"]
                          + [m(S, k) for k in ("dwi_mask", "t2_mask", "micro_mask")],
                   est="3 min (CPU)", note="writes the brain's MRI folder; refuses when its staged files exist (the data release's)"),
            D.Step(f"nmt-{S}", f"per brain: {S}", "mri",
                   ["-m", "linc.construction.mri_register", "nmt", "--subject", S, "--out", w("register", S, "nmt")],
                   outputs=[m(S, "nmt_affine"), m(S, "sarm_in_subject")], inputs=[m(S, "b0_n4_brain"), m(S, "dwi_mask")], gpu=True,
                   est="8 min (GPU)", note="ANTs affine, then anatomix + FireANTs ($ANATOMIX_PYTHON); NMT v2 from configs/link.yaml"),
            D.Step(f"t2reg-{S}", f"per brain: {S}", "mri",
                   ["-m", "linc.construction.mri_register", "t2", "--subject", S, "--out", w("register", S, "t2")],
                   outputs=[sub(S) / "xfm" / R["prep"]["within_mri_xfm"]["t2"] / "meta.json"],
                   inputs=[m(S, k) for k in ("b0_n4_brain", "dwi_mask", "t2_brain", "t2_mask")], gpu=True,
                   est="6 min (GPU)", note="ANTs affine, then anatomix + FireANTs ($ANATOMIX_PYTHON)"),
            D.Step(f"microreg-{S}", f"per brain: {S}", "mri",
                   ["-m", "linc.construction.mri_register", "micro", "--subject", S, "--out", w("register", S, "micro")],
                   outputs=[sub(S) / "xfm" / R["prep"]["within_mri_xfm"]["micro"] / "meta.json"],
                   inputs=[m(S, "b0_n4_brain"), m(S, "dwi_mask"), sub(S) / "mri" / "micro" / "b0_brain_n4.nii.gz", m(S, "micro_mask")],
                   gpu=True, est="10 min (GPU)", note="ANTs affine, then the symmetric registration ($ANATOMIX_PYTHON), composed into one field"),
            D.Step(f"tensor-{S}", f"per brain: {S}", "mri",
                   ["-m", "linc.construction.mri_prep", "tensor", "--eddy-series", m(S, "dwi_eddy"), "--ecclog", m(S, "dwi_ecclog"),
                    "--bvecs", m(S, "dwi_bvecs"), "--bvals", m(S, "dwi_bvals"), "--fit-mask", m(S, "dwi_fit_mask"),
                    "--dwi-mask", m(S, "dwi_mask"), "--out", w("tensor_d9", S), "--threads", "8"],
                   outputs=[tensor], inputs=[m(S, k) for k in ("dwi_eddy", "dwi_ecclog", "dwi_bvecs", "dwi_bvals", "dwi_fit_mask", "dwi_mask")],
                   est="8 min (CPU, 8 threads)", note="FSL dtifit on the b <= tensor_bmax volumes; MRtrix3 dwigradcheck ($MRTRIX)"),
            D.Step(f"landmarks-{S}", f"per brain: {S}", "mri",
                   ["-m", "linc.construction.canonical", "landmarks", "--sarm", m(S, "sarm_in_subject"), "--b0", m(S, "b0_n4_brain"),
                    "--mask", m(S, "dwi_mask"), "--out", lm, "--subject", S],
                   outputs=[lm], inputs=[m(S, "sarm_in_subject"), m(S, "b0_n4_brain"), m(S, "dwi_mask")], est="seconds"),
            D.Step(f"frames-{S}", f"per brain: {S}", "mri",
                   ["-m", "linc.construction.canonical", "frames", "--b0", m(S, "b0_n4_brain"), "--mask", m(S, "dwi_mask"),
                    "--landmarks", lm, "--nmt-affine", m(S, "nmt_affine"), "--out", cj, "--subject", S],
                   outputs=[cj], inputs=[lm, m(S, "nmt_affine"), m(S, "b0_n4_brain"), m(S, "dwi_mask")],
                   est="1-2 min (CPU)", note="the mid-sagittal plane is fitted by maximizing the b0's mirror correlation (canonical.plane_*)"),
            D.Step(f"prepare-{S}", f"per brain: {S}", "mri",
                   ["-m", "linc.construction.mri_prep", "prepare", "--subject", S, "--final-mask", m(S, "final_mask"), "--tensor", tensor,
                    "--canonical", cj, "--out", w("prep", S)],
                   outputs=[w("prep", S)],
                   inputs=[m(S, "final_mask"), tensor, cj, m(S, "b0"), m(S, "b0_n4_bias"), m(S, "dwi_mask"), m(S, "t2_head"), m(S, "t2_brain"),
                           m(S, "t2_mask"), m(S, "frac1"), m(S, "frac2"), m(S, "micro_mask"),
                           sub(S) / "xfm" / R["prep"]["within_mri_xfm"]["t2"], sub(S) / "xfm" / R["prep"]["within_mri_xfm"]["micro"]],
                   est="2.5 min (CPU)", note="antsApplyTransforms ($ANTSPATH) and vecreg ($FSLDIR); refuses to overwrite its output folder"),
        ]
    prep_pat = str(w("prep", "{id}"))
    steps += [
        D.Step("assemble", "construction", "mri",
               ["-m", "linc.construction.mmorf_run", "--upstream", str(Path(drv_dir).parent), "--out", run, "--prep", prep_pat,
                "--driver", driver_py],
               outputs=[run / "run_spec.json", run / "standardspace.nii.gz", run / "config"], inputs=[w("prep", S) for S in ids],
               est="17 s (CPU)", note=f"links the {n_in} inputs ({len(ids)} brains{' + mirrors' if mirror else ''}), writes one MMORF .ini per step and prints the driver's command"),
    ]

    def mmorf_argv():
        from linc.construction.mmorf_run import driver_command
        return driver_command(str(run), driver_py)[1:]                      # [python, driver, ...] -> the driver and its arguments

    steps += [
        D.Step("mmorf", "construction", "driver", mmorf_argv,
               outputs=[run / last], inputs=[run / "run_spec.json", run / "standardspace.nii.gz"],
               est="3 d 12 h (record: one RTX 6000 Ada 48 GB; affine 7 min, steps 1-5 1.2-2.3 h, step 6 13.6 h, step 7 21.6 h, step 8 40.7 h)",
               gpu=True, setenv={"PATH": f"{SHIM}:$PATH"},
               plan_argv=[driver_py, "--input_scalar"] + [f"{run}/data/{m}/" for m in scal] + (["--quant_scalar_mods"] + quant if quant else [])
                         + (["--input_tensor"] + [f"{run}/data/{m}/" for m in tens] if tens else []) + ["--config", f"{run}/config/", "...", "-b", batch],
               note="resume after an interruption with -aff False -resume <step>:<iteration> (LINC_SKIP_DONE_MMORF=1 keeps finished tasks)"),
        D.Step("channels", "channels", "mri",
               ["-m", "linc.construction.channels", "--run", run, "--out", w("channels"), "--prep", prep_pat, "-j", "8"],
               outputs=[w("channels", f"{R['channels']['prefix']}_T2.nii.gz"), w("channels", "provenance.json")],
               inputs=[run / read] + [w("prep", S) for S in ids],
               est="10.5 min (CPU, 8 threads)", note="acceptance (T2 r, V1 angle, FA range, symmetry) is written to channels/provenance.json"),
    ]
    for S in ids:
        steps.append(D.Step(f"links-{S}", "template links", "mri", ["-m", "linc.construction.mri_register", "links", "--subject", S, "--run", run],
                            outputs=[sub(S) / "xfm" / "b0-native__to__mri-canonical", sub(S) / "xfm" / "mri-canonical__to__mri-template"],
                            inputs=[run / read, sub(S) / "mri" / "landmarks" / "canonical.json"], est="seconds (CPU)",
                            note="with t2reg and microreg, moves images between the brain's spaces and the template (python -m linc.xfm)"))
    return steps


if __name__ == "__main__":
    os.umask(0o002)                                                   # outputs readable by the group
    D.main("Build the MRI channels of the template (reproduce/templates/mri_channels.md)", make_steps, "build_mri_channels")
