#!/usr/bin/env python3
"""Measure how much the four brains differ from their mean in each template variant (the variability figure and its text).

You need (settings/internal/products_mri.yaml)
  crossmodal_cache: each variant's warped inputs; template_mri: the T2 core and the white matter (FA > 0.25).

What it does, for each of the seven variants, on the four brains alone (no mirrors), because a mirror is not a separate brain
  1. measures each brain's deviation from the four brains' mean: T2 |dz|, soma and neurite fraction |d|, angle between principal eigenvectors;
  2. averages the deviations over the brains;
  3. reports each map's mean in the core and in the white matter, with the two voxel counts.

You get
  $LINC_NUMBERS/variability_four_brains_r061.json, and the maps $LINC_WORK/variability/<variant>_four__<metric>.nii.gz

    python code/figures/paper_numbers/variability_four_brains_r061.py [--plan]
"""
import sys
from pathlib import Path

import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
from shared import paths, provenance  # noqa: E402
from figures.evaluation import variability as VB  # noqa: E402
from figures.evaluation.variants import IDS, cached, load  # noqa: E402

NAME = "variability_four_brains_r061"
ARMS = ["rb_t2", "rb_micro", "rb_dti", "rb_t2micro", "rb_t2dti", "rb_dtimicro", "rb_final"]     # the seven variants (Table 2 order)
BRAINS = [s for s in IDS if not s.endswith("f")]
# the record's own descriptions, fixed (they name the analysis project's script and folder)
INPUTS_TEXT = ("the four brains MF283-MF286 only (no mirrors), per template from its own transformed inputs; reference = their mean; "
               "definitions of paper/fig_variability.py; WM = today's linc_mri_wm_fa025")
MAPS_TEXT = "mri/tc/_variability/<arm>_four__<metric>.nii.gz"


def main():
    P = paths.products("products_mri")
    cache, tpl = P["crossmodal_cache"], P["template_mri"]
    out_dir = paths.work_root() / "variability"
    ins = [cached(cache, arm, m, s) for arm in ARMS for m, _ in VB.METRICS for s in BRAINS]
    if "--plan" in sys.argv[1:]:                                             # print what this script reads and writes; run nothing
        from figures.evaluation import plan
        return plan.show(NAME, env="settings/env/linc.yml (PY_MRI)", est="about 10 minutes (CPU)",
                         inputs={"crossmodal_cache": cache, "template_mri": [tpl / "linc_mri_core_t2.nii.gz", tpl / "linc_mri_wm_fa025.nii.gz"], "warped inputs": ins},
                         outputs=[paths.numbers_root() / f"{NAME}.json"] + [out_dir / f"{arm}_four__{m}.nii.gz" for arm in ARMS for m, _ in VB.METRICS])
    ref = nib.load(str(tpl / "linc_mri_wm_fa025.nii.gz"))
    core = load(tpl / "linc_mri_core_t2.nii.gz") > 0.5
    wm = load(tpl / "linc_mri_wm_fa025.nii.gz") > 0.5
    out_dir.mkdir(parents=True, exist_ok=True)
    summ = {}
    for arm in ARMS:
        maps = VB.arm_maps(cache, arm, core, ref, ids=BRAINS)
        for k, v in maps.items():
            nib.save(nib.Nifti1Image(v.astype("float32"), ref.affine), str(out_dir / f"{arm}_four__{k}.nii.gz"))
        summ[f"{arm}_four"] = VB.summary(maps, core, wm)
        print(arm, {k: round(v["core_mean"], 4) for k, v in summ[f"{arm}_four"].items()}, flush=True)
    provenance.write_numbers(NAME, {"support_core_voxels": int(core.sum()), "wm_voxels": int(wm.sum()), "arms": summ, "inputs": INPUTS_TEXT, "maps": MAPS_TEXT},
                             inputs=ins, md5=False, cmd=f"python code/figures/paper_numbers/{NAME}.py")


if __name__ == "__main__":
    main()
