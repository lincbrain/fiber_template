#!/usr/bin/env python3
"""Build the blockface channel of the template from the released products: the recipe of record (reproduce/templates/blockface.md),
one step per command, with the same parameters (the commands' defaults and the recipe's flags; nothing is restated here).

    python reproduce/templates/build_blockface_template.py --plan                 every step, its environment, time, inputs, outputs
    python reproduce/templates/build_blockface_template.py --step prep-frame      one step (or several, in recipe order)
    python reproduce/templates/build_blockface_template.py --from members --to affine-templates

Stages: members -> plan A (7 MMORF steps at 0.5 mm, GPU) -> step 8 (0.25 mm, GPU) -> finish -> publish. Run in env/linc.yml; the
steps marked [mri] run with $PY_MRI (env/mri.yml), and the two [driver] steps run the patched template-construction driver with
MMORF 1.0.1 on a GPU (48 GB for the batch of record): set PY_MRI, LINC_TEMPLATE_DRIVER (the patched template_construction_v1
folder, blockface.md "Environment"), LINC_MMORF_EXE and FSLDIR. Inputs (configs/products_blockface.yaml, products_completion.yaml, under
$LINC_PRODUCTS): the completed volumes and the completion's cache, the frames and the bridge, the MRI template's b0 and mask, the
affine stage the build copied (plan_a_affine) and the precision rule of the finish. Outputs under --work (default
$LINC_WORK/build_blockface_template): canon_prep, canon, canon_tpl, canon2, canon2_tpl, canon2_s8w, the frames and the box, the run
trees runs/d2b and runs/d2bs8, and the template in final/ and final_run113/ (the published files).
One step the recipe leaves implicit is a step here: bf-view, the step-8 run without TPL that the blockface finish averages (record:
bftemplate/d2b/s8/tc/d2bs8_bfview: the run's config and out linked, run_spec.json without TPL).
Estimated times: the two MMORF stages and the finish from the record's file times (4-5 Oct 2026); the CPU steps are estimates.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _driver as D  # noqa: E402

B = ["-m", "linc.construction.blockface_build"]
SHIM = str(D.REPO / "linc" / "construction" / "blockface_shim")


def make_steps(W, paths):
    P1, P4 = paths.products("products_blockface"), paths.products("products_completion")
    root, cache = P4["completed"], P1["completion_cache"]
    affine, precision = P1["bf_plan_a_affine"], P1["bf_precision_rule"]
    w = lambda *p: W.joinpath(*p)                                                     # noqa: E731
    run_a, run_8, view = w("runs", "d2b"), w("runs", "d2bs8"), w("runs", "d2bs8_bfview")
    it7, it8 = run_a / "out" / "nln_step_07" / "iteration_03", run_8 / "out" / "nln_step_08" / "iteration_03"
    s6 = run_a / "out" / "nln_step_06" / "iteration_03"
    drv = {"LINC_MASKWEIGHTED": "1", "PATH": f"{SHIM}:$PATH"}

    def driver(run, *extra):
        def argv():
            from linc.construction.blockface_build import driver_command
            return driver_command(str(run), list(extra))
        return argv

    def driver_plan(run, *extra):
        return ["$LINC_TEMPLATE_DRIVER/run_template_construction.py", "--input_scalar", f"{run}/data/BF/", f"{run}/data/B0/", "--config",
                f"{run}/config/", "--tree", f"{run}/config/data_structure.tree", "--standardspace", f"{run}/standardspace.nii.gz", "--subids",
                f"{run}/subject_ids.csv", "--dof", "6", "-o", f"{run}/out/", "...", *extra]

    def bf_view():
        """The step-8 run without TPL (what the blockface finish averages): its config and out linked, run_spec.json without TPL."""
        view.mkdir(parents=True, exist_ok=False)
        for d in ("config", "out"):
            os.symlink(run_8 / d, view / d)
        spec = json.load(open(run_8 / "run_spec.json"))
        spec["subject_ids"] = [s for s in spec["subject_ids"] if s != "TPL"]
        spec["view"] = "design 2b step-8 blockface view: TPL left out of the blockface finish"
        json.dump(spec, open(view / "run_spec.json", "w"), indent=1)

    S = D.Step
    return [
        # 1. members
        S("prep-frame", "1. members", "linc", [*B, "prep-frame", "--out", w("frame2prep_0p5.nii.gz")],
          [w("frame2prep_0p5.nii.gz")], [P1["frame1c"]], "1 min", "frame1c grown 8 mm: no member can be clipped"),
        S("members", "1. members", "linc", [*B, "members", "--frame", w("frame2prep_0p5.nii.gz"), "--out", w("canon_prep"), "--root", root,
                                            "--cache", cache], [w("canon_prep")], [w("frame2prep_0p5.nii.gz"), root, cache], "30 min"),
        S("crop", "1. members", "linc", [*B, "crop", "--canon", w("canon_prep"), "--pad", "4", "--mult", "4", "--out", w("canon")],
          [w("canon")], [w("canon_prep")], "5 min", "the build frame: the union + 4 mm, x-symmetric"),
        S("tpl-canon-0p5", "1. members", "linc", [*B, "tpl-canon", "--vox", "0.5", "--source-canon", w("canon"), "--out", w("canon_tpl")],
          [w("canon_tpl")], [w("canon"), P1["tpl_b0"], P1["tpl_mask"], P1["bridge"]], "2 min", "TPL beside symlinks to the 21"),
        S("tpl-costmask-0p5", "1. members", "linc", [*B, "tpl-costmask", "--canons", w("canon_tpl"), "--record", w("tpl_costmask.json")],
          [w("canon_tpl", "TPL", "d2_b0_costmask_ones_v1.nii.gz")], [w("canon_tpl")], "1 min",
          "TPL's b0 cost: 0 below the brainstem plane + 1 mm, cosine over 2 mm (rewrites canon_tpl/TPL/d2_b0_costmask in place)"),
        # 2. plan A
        S("run-plan-a", "2. plan A (0.5 mm)", "linc", [*B, "run-plan-a", "--canon", w("canon_tpl"), "--root", w("runs"), "--name", "d2b"],
          [run_a], [w("canon_tpl")], "5 min", "data tree, configs (7 steps, lambda x10, smoothing floors), standardspace"),
        S("copy-affine", "2. plan A (0.5 mm)", "linc", [*B, "copy-affine", "--plan-a-affine", affine, "--run", run_a],
          [run_a / "out" / "affine_step2"], [affine, run_a], "1 min", "the 42 inputs' affine stage of record + TPL at the identity"),
        S("tpl-affine", "2. plan A (0.5 mm)", "linc", [*B, "tpl-affine", "--run", run_a, "--ref-template",
                                                       Path(affine) / "scalar_mod_02_affine_template_brain.nii.gz"],
          [run_a / "out" / "affine_step2" / "TPL" / "tpl_affine.json"], [run_a], "2 min", "TPL's 12-dof affine, symmetrized"),
        S("affine-templates", "2. plan A (0.5 mm)", "mri", [*B, "affine-templates", "--run", run_a],
          [run_a / "out" / "affine_step2" / "d2b_affavg.json"], [run_a], "5 min",
          "the affine templates over all 43 (the patched driver's averaging)"),
        S("plan-a-mmorf", "2. plan A (0.5 mm)", "driver", driver(run_a, "-aff", "False", "-nln", "True", "-nres", "7", "-nit", "3", "-b", "24"),
          [it7], [run_a], "5 h", "MMORF, 7 steps x 3 iterations (record: one 48 GB RTX 6000 Ada, 4 Oct 18:45-23:39)", drv, gpu=True,
          plan_argv=driver_plan(run_a, "-aff", "False", "-nln", "True", "-nres", "7", "-nit", "3", "-b", "24")),
        # 3. step 8
        S("s8-frame", "3. step 8 (0.25 mm)", "linc", [*B, "s8-frame", "--canon", w("canon"), "--out", w("frame_canon_0p5.nii.gz")],
          [w("frame_canon_0p5.nii.gz")], [w("canon")], "1 min"),
        S("s8-inputs", "3. step 8 (0.25 mm)", "linc", [*B, "s8-inputs", "--canon", w("canon"), "--frame", w("frame_canon_0p5.nii.gz"), "--out",
                                                       w("canon2"), "-j", "6"], [w("canon2")], [w("canon"), root, cache], "45 min",
          "the members on the frame's exact 2x lattice"),
        S("tpl-canon-0p25", "3. step 8 (0.25 mm)", "linc", [*B, "tpl-canon", "--vox", "0.25", "--source-canon", w("canon2"), "--out",
                                                            w("canon2_tpl")], [w("canon2_tpl")], [w("canon2")], "3 min"),
        S("tpl-costmask-0p25", "3. step 8 (0.25 mm)", "linc", [*B, "tpl-costmask", "--canons", w("canon_tpl"), w("canon2_tpl"), "--record",
                                                               w("tpl_costmask.json")],
          [w("canon2_tpl", "TPL", "d2_b0_costmask_ones_v1.nii.gz")], [w("canon_tpl"), w("canon2_tpl")], "1 min"),
        S("step8-weights", "3. step 8 (0.25 mm)", "linc", [*B, "step8-weights", "--canon2", w("canon2_tpl"), "--out", w("canon2_s8w")],
          [w("canon2_s8w")], [w("canon2_tpl")], "5 min", "the author's step-8 rule (blockface.md)"),
        S("s8-box", "3. step 8 (0.25 mm)", "linc", [*B, "s8-box", "--template", s6 / "nln_template_06_03_scalar_mod_01_brain.nii.gz",
                                                    s6 / "nln_template_06_03_scalar_mod_02_brain.nii.gz", "--canon2", w("canon2"), "--out",
                                                    w("box_0p25.nii.gz")], [w("box_0p25.nii.gz")], [s6, w("canon2")], "1 min",
          "plan A's support + 4 mm"),
        S("run-step8", "3. step 8 (0.25 mm)", "linc", [*B, "run-step8", "--canon2", w("canon2_s8w"), "--plan-a", run_a, "--box",
                                                       w("box_0p25.nii.gz"), "--root", w("runs"), "--name", "d2bs8"],
          [run_8], [w("canon2_s8w"), run_a, w("box_0p25.nii.gz")], "5 min"),
        S("tpl-standins", "3. step 8 (0.25 mm)", "linc", [*B, "tpl-standins", "--run", run_8, "--canon2", w("canon2_tpl")],
          [run_8 / "data" / "BF" / "TPL" / "BF" / "BF_mmorf_mask.nii.gz"], [run_8, w("canon2_tpl")], "1 min"),
        S("s8-mats", "3. step 8 (0.25 mm)", "linc", [*B, "s8-mats", "--affine-dir", run_a / "out" / "affine_step2", "--canon", w("canon_tpl"),
                                                     "--canon2", w("canon2_s8w"), "--box", w("box_0p25.nii.gz"), "--run", run_8,
                                                     "--standard05", run_a / "standardspace.nii.gz"],
          [run_8 / "out" / "affine_step2" / "s8_mats.json"], [run_a / "out" / "affine_step2", run_8], "2 min", "the 86 rewritten affines"),
        S("s8-step7", "3. step 8 (0.25 mm)", "linc", [*B, "s8-step7", "--plan-a-it", it7, "--canon", w("canon_tpl"), "--canon2",
                                                      w("canon2_s8w"), "--run", run_8, "-j", "6"],
          [run_8 / "out" / "nln_step_07" / "iteration_03" / "s8_step7_pull.json"], [it7, run_8], "40 min", "the 43 inputs' step-7 pulls at 0.25 mm"),
        S("step7-templates", "3. step 8 (0.25 mm)", "mri", [*B, "step7-templates", "--run", run_8],
          [run_8 / "out" / "nln_step_07" / "iteration_03" / "s8_step7_avg.json"], [run_8], "10 min",
          "the step-7 templates and standardspace (the driver's averaging)"),
        S("step8-mmorf", "3. step 8 (0.25 mm)", "driver", driver(run_8, "-aff", "False", "-nln", "True", "--resume_from", "8:1", "-nres", "8",
                                                                 "-nit", "3", "-b", "12"),
          [it8], [run_8], "8 h", "MMORF at 0.25 mm knots, 3 iterations (record: 4 Oct 23:45 - 5 Oct 07:29)", drv, gpu=True,
          plan_argv=driver_plan(run_8, "-aff", "False", "-nln", "True", "--resume_from", "8:1", "-nres", "8", "-nit", "3", "-b", "12")),
        # 4. finish and publish
        S("bf-view", "4. finish and publish", "linc", [], [view], [run_8], "1 s",
          "the step-8 run without TPL, for the blockface finish (implicit in the recipe)", fn=bf_view),
        S("finish", "4. finish and publish", "linc", [*B, "finish", "--run", view, "--arm", "d2bs8", "--canon", w("canon2"), "--iteration",
                                                      view / "out" / "nln_step_08" / "iteration_03", "--precision-json", precision,
                                                      "--out", w("final"), "-j", "6"],
          [w("final", "d2bs8_grey.nii.gz")], [view, w("canon2"), precision], "15 min",
          "every input pulled once through its step-8 warp, precision-weighted, symmetrized (record: 5 Oct 07:29-07:51 with finish-b0)"),
        S("finish-b0", "4. finish and publish", "linc", [*B, "finish-b0", "--run", run_8, "--arm", "d2bs8", "--canon", w("canon2_tpl"),
                                                         "--iteration", it8, "--weights-canon", w("canon2_s8w"), "--out", w("final"), "-j", "6"],
          [w("final", "d2bs8_b0.nii.gz")], [run_8, w("canon2_tpl"), w("canon2_s8w")], "10 min", "the b0 channel, the same warps"),
        S("embed", "4. finish and publish", "linc", [*B, "embed", "--run", run_8, "--src", w("final"), "--dst", w("final_run113")],
          [w("final_run113")], [w("final")], "2 min", "the box products onto the run-113 grid's 2x refinement: the published files"),
    ]


if __name__ == "__main__":
    D.main("Blockface template build (reproduce/templates/blockface.md)", make_steps, "build_blockface_template")
