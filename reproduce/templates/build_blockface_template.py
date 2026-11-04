#!/usr/bin/env python3
"""Build the blockface template: the brains, grids and MMORF schedule come from configs/blockface_template.yaml (the record's values).

You need
  env/linc.yml for every step ($PY_MRI names the same Python for the [mri] steps); a GPU with MMORF 1.0.1 (FSL's MultiMOdal
  Registration Framework, in env/linc.yml; 48 GB for the batch of record) for the two [driver] steps, and the edited template
  construction driver (LINC_TEMPLATE_DRIVER, linc/construction/patches); LINC_MMORF_EXE and FSLDIR;
  under $LINC_PRODUCTS (configs/products_blockface.yaml, products_completion.yaml): every member brain's volume with its missing
  sections synthesized, and the synthesis cache; the MRI template's frames, b0 and mask; the bridge (the rigid transform between
  the frame of record and the template's stereotaxic frame); the affine stage of record (bf_plan_a_affine) and the final average's
  precision rule.
What it does (one line per stage; --plan lists every step with its command, inputs, outputs and time)
  1. members: every brain and its mirror image on the preparation frame, cropped to the build frame; the MRI template's b0 as one
     more member
  2. the first stage: the run folder, the affine stage, then MMORF's first seven nonlinear steps at 0.5 mm
  3. the eighth step: the members on the 0.25 mm grid, the first stage's state taken over, then MMORF's eighth step
  4. the final average: every input resampled once through its final warp and averaged (weighted by expected error, symmetrized);
     the b0 channel; the products embedded on the MRI template's grid
You get
  under --work (default $LINC_WORK/build_blockface_template): the member folders (canon*), the frames and the eighth step's box, the
  two run folders (runs/), and the template in final/ (on the eighth step's box) and final_run113/ (the published files: grayscale,
  RGB, b0, masks, NIfTI).

    python reproduce/templates/build_blockface_template.py --plan                 every step, its environment, time, inputs, outputs
    python reproduce/templates/build_blockface_template.py --step prep-frame      one step (or several, in order)
    python reproduce/templates/build_blockface_template.py --from members --to affine-templates

Times: the two MMORF stages and the final average from the record's file times (4-5 Oct 2026); the CPU steps are estimates.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _driver as D  # noqa: E402

B = ["-m", "linc.construction.blockface_build"]
SHIM = str(D.REPO / "linc" / "construction" / "blockface_shim")
# the brains, grids and MMORF schedule: configs/blockface_template.yaml (the build reads them from the configuration)
TCFG = __import__("yaml").safe_load((D.REPO / "configs" / "blockface_template.yaml").read_text())


def make_steps(W, paths):
    P1, P4 = paths.products("products_blockface"), paths.products("products_completion")
    root, cache = P4["completed"], P1["completion_cache"]
    affine, precision = P1["bf_plan_a_affine"], P1["bf_precision_rule"]
    w = lambda *p: W.joinpath(*p)                                                     # noqa: E731
    G, MM, MB = TCFG["grids"], TCFG["mmorf"], TCFG["members"]
    va, v8 = G["plan_a_voxel_mm"], G["step8_voxel_mm"]
    ta, t8 = (f"{v:g}".replace(".", "p") for v in (va, v8))                         # 0.5 -> 0p5: grid tags in file and step names
    na, ni, s8, ni8 = MM["plan_a"]["steps"], MM["plan_a"]["iterations"], MM["step8"]["step"], MM["step8"]["iterations"]
    nb = len(MB["with_mri"]) + len(MB["without_mri"])                                 # the brains
    nin = nb * (2 if MB["mirrors"] else 1)                                             # their inputs (each brain and its mirror)
    tpl = MB["template_member"]
    ra, r8 = "d2b", "d2bs8"                                                            # the run folders' names (the record's)
    run_a, run_8, view = w("runs", ra), w("runs", r8), w("runs", f"{r8}_bfview")
    it7, it8 = run_a / "out" / f"nln_step_{na:02d}" / f"iteration_{ni:02d}", run_8 / "out" / f"nln_step_{s8:02d}" / f"iteration_{ni8:02d}"
    s6 = run_a / "out" / f"nln_step_{na - 1:02d}" / f"iteration_{ni:02d}"
    pa_lbl, s8_lbl = f"2. the first stage ({va:g} mm)", f"3. step {s8} ({v8:g} mm)"
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
        spec["subject_ids"] = [s for s in spec["subject_ids"] if s != tpl]
        spec["view"] = "design 2b step-8 blockface view: TPL left out of the blockface finish"
        json.dump(spec, open(view / "run_spec.json", "w"), indent=1)

    mm_a = ["-aff", "False", "-nln", "True", "-nres", str(na), "-nit", str(ni), "-b", str(MM["plan_a"]["batch"])]
    mm_8 = ["-aff", "False", "-nln", "True", "--resume_from", f"{s8}:1", "-nres", str(s8), "-nit", str(ni8), "-b", str(MM["step8"]["batch"])]
    S = D.Step
    return [
        # 1. members
        S("prep-frame", "1. members", "linc", [*B, "prep-frame", "--out", w(f"frame2prep_{ta}.nii.gz")],
          [w(f"frame2prep_{ta}.nii.gz")], [P1["frame1c"]], "1 min", f"frame1c grown {G['prep_frame_grow_mm']:g} mm: no member can be clipped"),
        S("members", "1. members", "linc", [*B, "members", "--frame", w(f"frame2prep_{ta}.nii.gz"), "--out", w("canon_prep"), "--root", root,
                                            "--cache", cache], [w("canon_prep")], [w(f"frame2prep_{ta}.nii.gz"), root, cache], "30 min"),
        S("crop", "1. members", "linc", [*B, "crop", "--canon", w("canon_prep"), "--pad", f"{G['build_frame']['pad_mm']:g}", "--mult",
                                         str(G["build_frame"]["multiple_vox"]), "--out", w("canon")],
          [w("canon")], [w("canon_prep")], "5 min", f"the build frame: the union + {G['build_frame']['pad_mm']:g} mm, x-symmetric"),
        S(f"tpl-canon-{ta}", "1. members", "linc", [*B, "tpl-canon", "--vox", f"{va:g}", "--source-canon", w("canon"), "--out", w("canon_tpl")],
          [w("canon_tpl")], [w("canon"), P1["tpl_b0"], P1["tpl_mask"], P1["bridge"]], "2 min", f"{tpl} beside symlinks to the {nb}"),
        S(f"tpl-costmask-{ta}", "1. members", "linc", [*B, "tpl-costmask", "--canons", w("canon_tpl"), "--record", w("tpl_costmask.json")],
          [w("canon_tpl", tpl, "d2_b0_costmask_ones_v1.nii.gz")], [w("canon_tpl")], "1 min",
          f"{tpl}'s b0 cost: 0 below the brainstem plane + 1 mm, cosine over 2 mm (rewrites canon_tpl/TPL/d2_b0_costmask in place)"),
        # 2. plan A
        S("run-plan-a", pa_lbl, "linc", [*B, "run-plan-a", "--canon", w("canon_tpl"), "--root", w("runs"), "--name", ra],
          [run_a], [w("canon_tpl")], "5 min", f"data tree, configs ({na} steps, lambda x{MM['lambda_scale']:g}, smoothing floors), standardspace"),
        S("copy-affine", pa_lbl, "linc", [*B, "copy-affine", "--plan-a-affine", affine, "--run", run_a],
          [run_a / "out" / "affine_step2"], [affine, run_a], "1 min", f"the {nin} inputs' affine stage of record + {tpl} at the identity"),
        S("tpl-affine", pa_lbl, "linc", [*B, "tpl-affine", "--run", run_a, "--ref-template",
                                         Path(affine) / "scalar_mod_02_affine_template_brain.nii.gz"],
          [run_a / "out" / "affine_step2" / tpl / "tpl_affine.json"], [run_a], "2 min", f"{tpl}'s 12-dof affine, symmetrized"),
        S("affine-templates", pa_lbl, "mri", [*B, "affine-templates", "--run", run_a],
          [run_a / "out" / "affine_step2" / f"{ra}_affavg.json"], [run_a], "5 min",
          f"the affine templates over all {nin + 1} (the patched driver's averaging)"),
        S(f"tpl-standins-{ta}", pa_lbl, "linc", [*B, "tpl-standins", "--run", run_a, "--canon2", w("canon_tpl")],
          [run_a / "data" / "BF" / tpl / "BF" / "BF_mmorf_mask.nii.gz"], [run_a, w("canon_tpl")], "1 min", "TPL's stand-in blockface image and ones mask for its MMORF call"),
        S("plan-a-mmorf", pa_lbl, "driver", driver(run_a, *mm_a),
          [it7], [run_a], "5 h", f"MMORF, {na} steps x {ni} iterations (record: one 48 GB RTX 6000 Ada, 4 Oct 18:45-23:39)", drv, gpu=True,
          plan_argv=driver_plan(run_a, *mm_a)),
        # 3. step 8
        S("s8-frame", s8_lbl, "linc", [*B, "s8-frame", "--canon", w("canon"), "--out", w(f"frame_canon_{ta}.nii.gz")],
          [w(f"frame_canon_{ta}.nii.gz")], [w("canon")], "1 min"),
        S("s8-inputs", s8_lbl, "linc", [*B, "s8-inputs", "--canon", w("canon"), "--frame", w(f"frame_canon_{ta}.nii.gz"), "--out",
                                                       w("canon2"), "-j", "6"], [w("canon2")], [w("canon"), root, cache], "45 min",
          "the members on the frame's exact 2x lattice"),
        S(f"tpl-canon-{t8}", s8_lbl, "linc", [*B, "tpl-canon", "--vox", f"{v8:g}", "--source-canon", w("canon2"), "--out",
                                                            w("canon2_tpl")], [w("canon2_tpl")], [w("canon2")], "3 min"),
        S(f"tpl-costmask-{t8}", s8_lbl, "linc", [*B, "tpl-costmask", "--canons", w("canon_tpl"), w("canon2_tpl"), "--record",
                                                  w("tpl_costmask.json")],
          [w("canon2_tpl", tpl, "d2_b0_costmask_ones_v1.nii.gz")], [w("canon_tpl"), w("canon2_tpl")], "1 min"),
        S("step8-weights", s8_lbl, "linc", [*B, "step8-weights", "--canon2", w("canon2_tpl"), "--out", w("canon2_s8w")],
          [w("canon2_s8w")], [w("canon2_tpl")], "5 min", "the author's step-8 rule (blockface.md)"),
        S("s8-box", s8_lbl, "linc", [*B, "s8-box", "--template", s6 / f"nln_template_{na - 1:02d}_{ni:02d}_scalar_mod_01_brain.nii.gz",
                                      s6 / f"nln_template_{na - 1:02d}_{ni:02d}_scalar_mod_02_brain.nii.gz", "--canon2", w("canon2"), "--out",
                                      w(f"box_{t8}.nii.gz")], [w(f"box_{t8}.nii.gz")], [s6, w("canon2")], "1 min",
          f"the first stage's support + {G['step8_box']['pad_mm']:g} mm"),
        S("run-step8", s8_lbl, "linc", [*B, "run-step8", "--canon2", w("canon2_s8w"), "--plan-a", run_a, "--box",
                                         w(f"box_{t8}.nii.gz"), "--root", w("runs"), "--name", r8],
          [run_8], [w("canon2_s8w"), run_a, w(f"box_{t8}.nii.gz")], "5 min"),
        S(f"tpl-standins-{t8}", s8_lbl, "linc", [*B, "tpl-standins", "--run", run_8, "--canon2", w("canon2_tpl")],
          [run_8 / "data" / "BF" / tpl / "BF" / "BF_mmorf_mask.nii.gz"], [run_8, w("canon2_tpl")], "1 min"),
        S("s8-mats", s8_lbl, "linc", [*B, "s8-mats", "--affine-dir", run_a / "out" / "affine_step2", "--canon", w("canon_tpl"),
                                       "--canon2", w("canon2_s8w"), "--box", w(f"box_{t8}.nii.gz"), "--run", run_8,
                                                     "--standard05", run_a / "standardspace.nii.gz"],
          [run_8 / "out" / "affine_step2" / "s8_mats.json"], [run_a / "out" / "affine_step2", run_8], "2 min",
          f"the {2 * (nin + 1)} rewritten affines"),
        S("s8-step7", s8_lbl, "linc", [*B, "s8-step7", "--plan-a-it", it7, "--canon", w("canon_tpl"), "--canon2",
                                        w("canon2_s8w"), "--run", run_8, "-j", "6"],
          [run_8 / "out" / f"nln_step_{s8 - 1:02d}" / f"iteration_{ni:02d}" / "s8_step7_pull.json"], [it7, run_8], "40 min",
          f"the {nin + 1} inputs' step-{s8 - 1} pulls at {v8:g} mm"),
        S("step7-templates", s8_lbl, "mri", [*B, "step7-templates", "--run", run_8],
          [run_8 / "out" / f"nln_step_{s8 - 1:02d}" / f"iteration_{ni:02d}" / "s8_step7_avg.json"], [run_8], "10 min",
          f"the step-{s8 - 1} templates and standardspace (the driver's averaging)"),
        S("step8-mmorf", s8_lbl, "driver", driver(run_8, *mm_8),
          [it8], [run_8], "8 h", f"MMORF at {v8:g} mm knots, {ni8} iterations (record: 4 Oct 23:45 - 5 Oct 07:29)", drv, gpu=True,
          plan_argv=driver_plan(run_8, *mm_8)),
        # 4. finish and publish
        S("bf-view", "4. finish and publish", "linc", [], [view], [run_8], "1 s",
          "the step-8 run without TPL, for the blockface finish (implicit in the recipe)", fn=bf_view),
        S("finish", "4. finish and publish", "linc", [*B, "finish", "--run", view, "--arm", r8, "--canon", w("canon2"), "--iteration",
                                                      view / "out" / f"nln_step_{s8:02d}" / f"iteration_{ni8:02d}", "--precision-json", precision,
                                                      "--out", w("final"), "-j", "6"],
          [w("final", f"{r8}_grey.nii.gz")], [view, w("canon2"), precision], "15 min",
          "every input pulled once through its step-8 warp, precision-weighted, symmetrized (record: 5 Oct 07:29-07:51 with finish-b0)"),
        S("finish-b0", "4. finish and publish", "linc", [*B, "finish-b0", "--run", run_8, "--arm", r8, "--canon", w("canon2_tpl"),
                                                         "--iteration", it8, "--weights-canon", w("canon2_s8w"), "--out", w("final"), "-j", "6"],
          [w("final", f"{r8}_b0.nii.gz")], [run_8, w("canon2_tpl"), w("canon2_s8w")], "10 min", "the b0 channel, the same warps"),
        S("embed", "4. finish and publish", "linc", [*B, "embed", "--run", run_8, "--src", w("final"), "--dst", w("final_run113")],
          [w("final_run113")], [w("final")], "2 min", "the box products onto the run-113 grid's 2x refinement: the published files"),
    ]


if __name__ == "__main__":
    D.main("Blockface template build (reproduce/templates/blockface.md)", make_steps, "build_blockface_template", doc=__doc__)
