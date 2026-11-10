#!/usr/bin/env python3
"""The MRI template construction recipe (paper: Methods, "Symmetric groupwise construction").
The recipe is settings/mri_construction.yaml: number of steps, iterations per step, the knot spacing reached at each step (coarsest
16 mm, 4 mm after step 4, 0.25 mm = one voxel after step 8), the smoothing ladder, the regularization (lambda_scale times the base
lambda_reg series, first level unscaled), the channel weights and the cost masks; plus, for comparison, the smoothing floor that each
native resolution (0.25 / 0.5 / 0.7 mm) would support. Every recipe value is checked against the configs the template was actually
built with (products: mri_runs/<run>/run_spec.json and config/mmorf_params_stepNN.ini), and the iterations are counted in the run's
output folders; a mismatch stops the script.
-> $LINC_NUMBERS/mri_recipe.json
    python code/figures/paper_numbers/mri_recipe.py [--run mmsym_model_dti_v]"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
from shared import paths, provenance  # noqa: E402

RECIPE = Path(__file__).resolve().parents[3] / "settings" / "mri_construction.yaml"


def ini_values(ini, key):
    """All occurrences of `key` in an MMORF ini (one line per modality for per-modality options), each split into tokens."""
    return [line.split("=", 1)[1].split(";")[0].split() for line in Path(ini).read_text().splitlines()
            if line.split("=")[0].strip() == key]


def lambda_reg(R, n):
    """The lambda_reg series written for an n-level step: first term as is, the rest times lambda_scale at 4 significant digits."""
    lam = R["lambda_reg"]
    return [float(lam[0])] + [float(f"{float(x) * R['lambda_scale']:.4g}") for x in lam[1:]][:n - 1]


def check_run(R, run):
    """Stop unless the run folder was built with exactly the recipe in the YAML."""
    spec = json.loads((run / "run_spec.json").read_text())
    expect = {"n_steps": R["n_steps"], "lambda_scale": R["lambda_scale"],
              "modalities": [m["name"] for m in R["scalar_modalities"]], "tensor_modalities": [m["name"] for m in R["tensor_modalities"]],
              "fwhm_floor": R["fwhm_floor_applied"], "subject_ids": [s + f for s in R["inputs"] for f in ("", "f")]}
    got = {"n_steps": spec["n_steps"], "lambda_scale": spec["lambda_scale"], "modalities": [m["name"] for m in spec["modalities"]],
           "tensor_modalities": [m["name"] for m in spec["tensor_modalities"]], "fwhm_floor": spec["fwhm_floor"],
           "subject_ids": spec["subject_ids"]}
    assert got == expect, f"run_spec.json differs from {RECIPE.name}: {got} vs {expect}"
    assert ([m["lambda_scalar"] for m in spec["modalities"]] == [m["weight"] for m in R["scalar_modalities"]]
            and [m["lambda_tensor"] for m in spec["tensor_modalities"]] == [m["weight"] for m in R["tensor_modalities"]]), "weights differ"
    ns, nt = len(R["scalar_modalities"]), len(R["tensor_modalities"])
    for s in range(1, R["n_steps"] + 1):
        ini, n = run / "config" / f"mmorf_params_step{s:02d}.ini", s + 1
        f = lambda key: [[float(x) for x in row] for row in ini_values(ini, key)]  # noqa: E731
        expect = {"warp_res_init": [[R["warp_res_init_mm"]]], "warp_scaling": [R["warp_scaling"][:n]], "lambda_reg": [lambda_reg(R, n)],
                  "hires": [[R["hires_mm"]]], "optimiser_max_it_lowres": [[R["optimiser_max_it_lowres"]]],
                  "optimiser_max_it_hires": [[R["optimiser_max_it_hires"]]],
                  "fwhm_ref_scalar": [R["fwhm_mm"][:n]] * ns, "fwhm_mov_scalar": [R["fwhm_mm"][:n]] * ns,
                  "fwhm_ref_tensor": [R["fwhm_mm"][:n]] * nt, "fwhm_mov_tensor": [R["fwhm_mm"][:n]] * nt,
                  "lambda_scalar": [[m["weight"]] * n for m in R["scalar_modalities"]],
                  "lambda_tensor": [[m["weight"]] * n for m in R["tensor_modalities"]],
                  "use_mask_ref_scalar": [[R["cost_mask_scalar"]] * n] * ns, "use_mask_mov_scalar": [[R["cost_mask_scalar"]] * n] * ns,
                  "use_mask_ref_tensor": [[R["cost_mask_tensor"]] * n] * nt, "use_mask_mov_tensor": [[R["cost_mask_tensor"]] * n] * nt,
                  "lambda_bias_reg": [[R["lambda_bias_reg"]] * n] * ns, "bias_res_init": [[R["bias_res_init_mm"]] * ns],
                  "estimate_bias": [[0 if m["quantitative"] else 1 for m in R["scalar_modalities"]]]}
        for key, v in expect.items():
            assert f(key) == v, f"{ini.name} {key}: {f(key)} vs {RECIPE.name} {v}"
    return spec


def main():
    R = yaml.safe_load(RECIPE.read_text())
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default=R["run"]); a = ap.parse_args()
    run = paths.products("products_mri")["mri_runs"] / a.run
    check_run(R, run)
    N = R["n_steps"]
    # iterations: the highest iteration index on disk per step (earlier iteration folders may have been thinned)
    its = {s: max(int(p.name.split("_")[1]) for p in (run / "out" / f"nln_step_{s:02d}").glob("iteration_*")) for s in range(1, N + 1)}
    assert set(its.values()) == {R["iterations_per_step"]}, f"iterations on disk {its} vs {RECIPE.name} {R['iterations_per_step']}"
    res0, sc, fw = float(R["warp_res_init_mm"]), [float(x) for x in R["warp_scaling"]], [float(x) for x in R["fwhm_mm"]]
    steps = {s: {"levels": s + 1, "final_knot_mm": res0 / float(np.prod(sc[:s + 1])), "final_fwhm_mm": fw[s]} for s in range(1, N + 1)}
    lam = lambda_reg(R, N + 1)
    floor = lambda mm: R["fwhm_floor_mtf_constant"] / R["fwhm_floor_f_eff_cyc_per_mm"][round(mm, 2)]  # noqa: E731
    out = {"run": a.run, "n_steps": N, "iterations_per_step": sorted(set(its.values())),
           "warp_res_init_mm": res0, "warp_scaling": " ".join(str(x) for x in R["warp_scaling"][:N + 1]),
           "steps": steps, "knot_step4_mm": steps[4]["final_knot_mm"], "knot_final_mm": steps[N]["final_knot_mm"],
           "fwhm_first_mm": fw[0], "fwhm_last_mm": fw[N],
           "lambda_scale": R["lambda_scale"], "lambda_reg_first": lam[0], "lambda_reg_rest": lam[1:],
           "weights": {m["name"]: m["weight"] for m in R["scalar_modalities"] + R["tensor_modalities"]},
           "cost_mask_scalar_values": [R["cost_mask_scalar"]], "cost_mask_tensor_values": [R["cost_mask_tensor"]],
           "fwhm_floor_mm": {"native_0p25": floor(0.25), "native_0p5": floor(0.5), "native_0p7": floor(0.7)}}
    ins = [RECIPE, run / "run_spec.json"] + [run / "config" / f"mmorf_params_step{s:02d}.ini" for s in range(1, N + 1)]
    provenance.write_numbers("mri_recipe", out, inputs=ins, md5=False, cmd=" ".join(["python code/figures/paper_numbers/mri_recipe.py"] + sys.argv[1:]))


if __name__ == "__main__":
    main()
