#!/usr/bin/env python3
"""The parameters of the deformable blockface-to-b0 registration, as one record (paper: Methods, "Blockface to MRI registration and
synthesis of missing sections").

    python reproduce/numbers/registration_params_r073.py

You need
  configs/registration.yaml (to_own_mri and direct.direct_placement, each with its featurereg block), the defaults of
  linc/registration/featurereg.py register(), and one registration of record in the products (products_blockface `data`:
  sub-MR252/mri/nonlinear_r1v3mr/MR252_metrics.json).
What it does
  It copies the settings of the registration of a brain's blockface to its own b0 and of the cross-modal alignment (the same
  deformable settings, asserted): the working grid and padding, the features, the similarity, the two stages' levels, iterations
  and correlation windows, the smoothing of the update and of the deformation (voxels and mm), and the linear starts; and, from
  MM2's run of record, the channels, stages, grid and time.
You get
  $LINC_NUMBERS/registration_params_r073.json.

Ported from the analysis project's paper/r073_l1/registration_params_r073.py, which read the same files.
"""
import inspect
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from linc import paths, provenance  # noqa: E402
from linc.registration import featurereg  # noqa: E402


def main():
    cfg_p = REPO / "configs" / "registration.yaml"; cfg = yaml.safe_load(cfg_p.read_text())
    sig = {k: v.default for k, v in inspect.signature(featurereg.register).parameters.items() if v.default is not inspect.Parameter.empty}
    met_p = paths.products("products_blockface")["data"] / "sub-MR252/mri/nonlinear_r1v3mr/MR252_metrics.json"
    met = json.loads(met_p.read_text())
    ints = lambda s: [int(x) for x in str(s).split(",")]  # noqa: E731
    own = cfg["to_own_mri"]; dp = cfg["direct"]["direct_placement"]
    assert own["featurereg"]["smooth_warp_sigma"] == dp["featurereg"]["smooth_warp_sigma"] == met["config"]["smooth_warp_sigma"]
    out = dict(
        grid_mm=own["resolution_mm"], pad_rule=own["pad_rule"], pad_mm_min=float(own["pad_rule"].split(":")[0]),
        backbone=sig["backbone"], anatomix_channels=32, mindssc_channels=12, mindssc_radius=1, mindssc_dilation=2,
        loss=own["featurereg"]["loss"], lr=own["featurereg"]["lr"], fixed_clip_percentile=sig["fixed_clip_percentile"],
        stage1=dict(shrink=ints(sig["scales"]), iterations=ints(sig["iterations"]), cc_window_vox=ints(sig["cc"])),
        stage2=dict(shrink=ints(sig["stage2_scales"]), iterations=ints(sig["stage2_iterations"]), cc_window_vox=ints(sig["stage2_cc"])),
        smooth_grad_sigma_vox=own["featurereg"]["smooth_grad_sigma"], smooth_warp_sigma_vox=own["featurereg"]["smooth_warp_sigma"],
        smooth_warp_sigma_mm=own["featurereg"]["smooth_warp_sigma"] * own["resolution_mm"],
        smooth_grad_sigma_mm=own["featurereg"]["smooth_grad_sigma"] * own["resolution_mm"],
        gate_features_by_mask=dict(to_own_mri=own["featurereg"]["gate_features_by_mask"], cross_modal=dp["featurereg"]["gate_features_by_mask"]),
        linear=dict(to_own_mri="FLIRT 7 then 9 dof, correlation ratio, weighted to the photographed slab (slab_inset_mm %s)" % own["slab_inset_mm"],
                    cross_modal="FLIRT 6 dof on the brain masks (full-sphere search), 7 then 9 dof (scales kept in %s)" % dp["dof9_scale_range"]),
        run_of_record_MM2=dict(channels=met["channels"], stages=met["stages"], grid_voxel_mm=met["grid"]["voxel_mm"],
                               grid_mvox=met["grid_vox_M"], wall_s=met["register_wall_seconds"], engine=met["engine_version"]),
        note="sigma in voxels of the 0.25 mm working grid; smoothing applied to the update gradient and to the deformation after each step")
    print(json.dumps(out, indent=1))
    provenance.write_numbers("registration_params_r073", out, inputs=[cfg_p, REPO / "linc/registration/featurereg.py", met_p], md5=False,
                             cmd="python reproduce/numbers/registration_params_r073.py")


if __name__ == "__main__":
    main()
