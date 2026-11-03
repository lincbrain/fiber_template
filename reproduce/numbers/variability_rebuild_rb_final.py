#!/usr/bin/env python3
"""Map how much the template's inputs differ from the template, voxel by voxel (the variability figure and its text).

You need (configs/products_mri.yaml)
  crossmodal_cache: the four-channel variant's warped inputs; template_mri: the deliverable T2 core, its white matter, the T2 grid.

What it does
  1. measures each input's deviation from the variant's mean template: T2 |dz|, soma and neurite fraction |d|, principal-eigenvector angle;
  2. averages the deviations over the inputs;
  3. reports each map's mean in the core and in its white matter, with the two voxel counts.

You get
  $LINC_NUMBERS/variability_rebuild_rb_final.json, and the maps $LINC_WORK/variability/rb_final__<metric>.nii.gz for the figure

    python reproduce/numbers/variability_rebuild_rb_final.py [--plan]
"""
import sys
from pathlib import Path

import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from linc import paths, provenance  # noqa: E402
from linc.evaluation import variability as VB  # noqa: E402
from linc.evaluation.variants import IDS, cached, load  # noqa: E402

NAME = "variability_rebuild_rb_final"
ARMS = ["rb_final"]                     # the four-channel template


def main():
    P = paths.products("products_mri")
    cache, tpl = P["crossmodal_cache"], P["template_mri"]
    out_dir = paths.work_root() / "variability"
    if "--plan" in sys.argv[1:]:                                             # print what this script reads and writes; run nothing
        from linc.evaluation import plan
        return plan.show(NAME, env="env/mri.yml (PY_MRI)", est="a few minutes (CPU)",
                         inputs={"crossmodal_cache": cache, "template_mri": [tpl / "linc_mri_T2.nii.gz", tpl / "linc_mri_core_t2.nii.gz", tpl / "linc_mri_wm_fa025.nii.gz"],
                                 "warped inputs": [cached(cache, arm, m, s) for arm in ARMS for m, _ in VB.METRICS for s in IDS]},
                         outputs=[paths.numbers_root() / f"{NAME}.json"] + [out_dir / f"{arm}__{m}.nii.gz" for arm in ARMS for m, _ in VB.METRICS],
                         notes=["the maps under $LINC_WORK/variability are what the variability figure reads"])
    ref = nib.load(str(tpl / "linc_mri_T2.nii.gz"))
    core = load(tpl / "linc_mri_core_t2.nii.gz") > 0
    wm = (load(tpl / "linc_mri_wm_fa025.nii.gz") > 0) & core
    summ = {}
    for arm in ARMS:
        maps = VB.arm_maps(cache, arm, core, ref, out_dir=out_dir)
        summ[arm] = VB.summary(maps, core, wm)
        print(arm, {k: round(v["core_mean"], 3) for k, v in summ[arm].items()}, flush=True)
    ins = [cached(cache, arm, m, s) for arm in ARMS for m, _ in VB.METRICS for s in IDS]
    ins += [tpl / "linc_mri_T2.nii.gz", tpl / "linc_mri_core_t2.nii.gz", tpl / "linc_mri_wm_fa025.nii.gz"]
    provenance.write_numbers(NAME, {"support_core_voxels": int(core.sum()), "wm_voxels": int(wm.sum()), "arms": summ},
                             inputs=ins, md5=False, cmd=f"python reproduce/numbers/{NAME}.py")


if __name__ == "__main__":
    main()
