#!/usr/bin/env python3
"""Build the MRI channels of the template from the released per-brain products: the recipe of record (reproduce/templates/
mri_channels.md; paper Methods, "MRI preprocessing", "Symmetric groupwise construction", "Template channels"), one step per command,
with the parameters of configs/mri_construction.yaml (nothing is restated here).

    python reproduce/templates/build_mri_channels.py --plan                     every step, its environment, time, inputs, outputs
    python reproduce/templates/build_mri_channels.py --step prepare-MF283       one step (or several, in recipe order)
    python reproduce/templates/build_mri_channels.py --from assemble --to channels

Stages, per brain (MF1-MF4 = MF283-MF286) then together:
    tensor-<ID>      the D9 tensor: dtifit on the eddy-corrected b <= 4800 volumes, b-vectors rotated and gradient-checked [FSL, MRtrix]
    landmarks-<ID>   AC and PC from the SARM labels carried onto the brain's b0 grid
    frames-<ID>      the brain's rigid map into the canonical (NMT v2) frame: its own mid-sagittal plane, NMT's orientation, the AC
    prepare-<ID>     the construction inputs on the canonical grid, cut to the final hand-edited mask, with their exact mirrors [ANTs, FSL]
    assemble         the MMORF run folder (inputs, standard space, one .ini per step) and the driver's command line
    mmorf            MMORF 1.0.1 through the patched template-construction driver: 8 steps x 3 iterations [GPU, 3 d 12 h of record]
    channels         the template channels from the step-8 warps: T2, b0, fractions, tensor and its scalars, masks [FSL]
Every step runs with $PY_MRI (env/mri.yml: MMORF from the FSL channel, fsl-sub, fslpy, file-tree), with FSLDIR (FSL 6.0.7.4), ANTSPATH
(ANTs 2.5.4, prepare) and MRTRIX (MRtrix3 3.0.2, tensor) set; the mmorf step also needs LINC_TEMPLATE_DRIVER (the patched
template_construction_v1 folder: linc/construction/patches/README.md) and LINC_MMORF_EXE (mmorf_cuda), run on one 48 GB GPU for the
batch of record (`run_assembly.batch_of_record`; scheduling only).
Inputs, per brain under $LINC_DATA/sub-<ID>/ (configs/mri_construction.yaml prep.layout, relative to mri/; the within-MRI warps under
xfm/): the staged session images and hand masks (dwi/, t2/, micro/, masks/), the raw diffusion files of the tensor (dwi/raw/), the
final hand-edited mask (masks/final_brain_mask.nii.gz), the SARM labels carried onto the b0 and the subject -> NMT affine (landmarks/),
and the T2 -> b0 and micro -> b0 registrations (xfm/t2-native__to__b0-native, xfm/micro-native__to__b0-native).
Outputs under --work (default $LINC_WORK/build_mri_channels): tensor_d9/<ID>/, frames/<ID>/{landmarks,canonical}.json, prep/<ID>/
(the construction inputs), runs/rb_final/ (the MMORF run), channels/ (linc_mri_*.nii.gz + provenance.json: the template's mri folder).
Nothing is written into $LINC_DATA. Estimated times are the run of record's (mri_channels.md, "How long it takes").
Validated 3 Nov 2026 on MF283 (landmarks, frames, prepare against the products of record: see mri_channels.md "Validation").
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _driver as D  # noqa: E402

SHIM = str(D.REPO / "linc" / "construction" / "blockface_shim")          # `mmorf` on PATH -> $LINC_MMORF_EXE (its TPL rule never fires here)


def make_steps(W, paths):
    from linc.construction.tools import recipe
    R = recipe(); L = R["prep"]["layout"]; ids = list(R["inputs"])
    w = lambda *p: W.joinpath(*p)                                                     # noqa: E731
    sub = lambda S: paths.subject_dir(S)                                              # noqa: E731
    m = lambda S, k: sub(S) / "mri" / L[k]                                            # noqa: E731
    run = w("runs", R["rebuild_run"])
    drv_dir = os.environ.get("LINC_TEMPLATE_DRIVER", "$LINC_TEMPLATE_DRIVER")
    driver_py = str(Path(drv_dir) / "run_template_construction.py")
    steps = []
    for S in ids:
        tensor = w("tensor_d9", S, "tensor_d9_dwigrid.nii.gz")
        lm, cj = w("frames", S, "landmarks.json"), w("frames", S, "canonical.json")
        steps += [
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
               est="17 s (CPU)", note="links the 8 inputs (4 brains + mirrors), writes one MMORF .ini per step and prints the driver's command"),
    ]

    def mmorf_argv():
        from linc.construction.mmorf_run import driver_command
        return driver_command(str(run), driver_py)[1:]                      # [python, driver, ...] -> the driver and its arguments

    steps += [
        D.Step("mmorf", "construction", "driver", mmorf_argv,
               outputs=[run / "out" / "nln_step_08" / "iteration_03"], inputs=[run / "run_spec.json", run / "standardspace.nii.gz"],
               est="3 d 12 h (record: one RTX 6000 Ada 48 GB; affine 7 min, steps 1-5 1.2-2.3 h, step 6 13.6 h, step 7 21.6 h, step 8 40.7 h)",
               gpu=True, setenv={"PATH": f"{SHIM}:$PATH"}, plan_argv=[driver_py, "--input_scalar", f"{run}/data/T2M/", f"{run}/data/FRAC1M/",
                                                                        f"{run}/data/FRAC2M/", "--quant_scalar_mods", "2", "3", "--input_tensor",
                                                                        f"{run}/data/DTIV/", "--config", f"{run}/config/", "...", "-b", "8 8 8 8 8 4 4 3"],
               note="resume after an interruption with -aff False -resume <step>:<iteration> (LINC_SKIP_DONE_MMORF=1 keeps finished tasks)"),
        D.Step("channels", "channels", "mri",
               ["-m", "linc.construction.channels", "--run", run, "--out", w("channels"), "--prep", prep_pat, "-j", "8"],
               outputs=[w("channels", f"{R['channels']['prefix']}_T2.nii.gz"), w("channels", "provenance.json")],
               inputs=[run / "out" / "nln_step_08" / "iteration_03"] + [w("prep", S) for S in ids],
               est="10.5 min (CPU, 8 threads)", note="acceptance (T2 r, V1 angle, FA range, symmetry) is written to channels/provenance.json"),
    ]
    return steps


if __name__ == "__main__":
    D.main("Build the MRI channels of the template (reproduce/templates/mri_channels.md)", make_steps, "build_mri_channels")
