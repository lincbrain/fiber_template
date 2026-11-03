#!/usr/bin/env python3
"""Between-input variability of the four-channel MRI template (paper: Results "MRI template", the variability figure and its text).

For the four-channel variant, the voxelwise deviation of each of the eight inputs from the variant's own mean template, averaged over
the inputs (linc/evaluation/variability.py): T2 |Δz| (z-units), soma and neurite fraction |Δ| (percentage points), and the angle
between principal eigenvectors (degrees). Reported: the mean of each map in the deliverable T2 core (template linc_mri_core_t2) and
in its white matter (linc_mri_wm_fa025 inside the core), and the two voxel counts.
The maps are written to $LINC_WORK/variability/rb_final__<metric>.nii.gz for the figure script.
Inputs (products_mri): crossmodal_cache (the warped inputs), template_mri (core, white matter, the T2 grid).
-> $LINC_NUMBERS/variability_rebuild_rb_final.json
    python reproduce/numbers/variability_rebuild_rb_final.py"""
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
