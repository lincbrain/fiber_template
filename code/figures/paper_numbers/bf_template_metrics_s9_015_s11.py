#!/usr/bin/env python3
"""The published blockface template's numbers (Results, the blockface template): step 9 at 0.15 mm on its 11 brains, read from its
saved products only (no template recomputed).

    python code/figures/paper_numbers/bf_template_metrics_s9_015_s11.py

You need
  the template's products (settings/internal/products_blockface.yaml: `bftemplate_step9`, the template on its own grid, and
  `bftemplate_step9_in_mri`, the same in the MRI template's frame; `bftemplate_step8`'s metrics.json for step 8 beside it).
What it does
  The members, the grid, the brain, photographed and predicted volumes and the predicted fraction (predicted = brain with a
  photographed fraction < 0.5), the brain where no photograph contributes at all, and the symmetry (max |V - mirror(V)| of the
  grayscale, color and both masks); the brain and predicted volumes in the MRI template's frame; step 8's beside them.
You get
  $LINC_NUMBERS/bf_template_metrics_s9_015_s11.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
from shared import paths, provenance  # noqa: E402

# the record's `definition`, verbatim (it is part of the record's data)
DEFINITION = ('s9_015_numbers_s11.py = s9_015_numbers.py for S11 (option C; TESTS.md addendum 16) -> paper/numbers/bf_template_metrics_s9_015_s11.json.\n'
              'The bf_template_metrics numbers for the step-9 blockface template on the 0.15 mm grid (TESTS.md addenda 13 and 16; option C, S11),\n'
              'from its saved products only, beside the 2b record (bf_template_metrics_d2b.json, d2bs8 at 0.25 mm, unchanged):\n'
              '  members: S11 = 11 brains (S12 without MR263), 22 blockface inputs (each brain and its mirror), no b0-only member;\n'
              "  brain = the template's brainmask (P >= 0.5); photographed = brain & photo_share >= 0.5 (the real mask);\n"
              '  synthesized (the paper\'s "predicted") = brain & photo_share < 0.5, the same rule as d2bs8\'s synth_mask; also brain & photo_share = 0;\n'
              '  symmetry: max |V - mirror(V)| of grey, rgb, brainmask, real mask (exact 0 expected);\n'
              "  the same in the MRI template's frame (out_015_s11/in_mri_template, through d2bs8_s5sym, 0.25 mm), for information.\n"
              '    python step9/s9_015_numbers_s11.py  -> paper/numbers/bf_template_metrics_s9_015_s11.json')
LABEL, ARM = "step 9 at 0.15 mm, S11 (option C, the author's go 15 Nov)", "s9_015_s11"   # the record's strings


def main():
    import yaml
    cfg = yaml.safe_load((Path(__file__).resolve().parents[3] / "settings" / "internal" / "products_blockface.yaml").read_text())
    prod = paths.products("products_blockface")
    c9 = yaml.safe_load((Path(__file__).resolve().parents[3] / "settings" / "blockface_template.yaml").read_text())["step9"]
    members, tag = c9["members"], c9["tag"]
    T, MRI = prod["bftemplate_step9"] / f"T3{tag}", prod["bftemplate_step9_in_mri"] / f"T3{tag}_015_in_mri"
    members_text = f"{tag.upper()}: " + " ".join(members)                    # "S11: MF278 ..." (settings/blockface_template.yaml step9)
    L = lambda k: nib.load(f"{T}_{k}.nii.gz")                                   # noqa: E731
    bi = L("brainmask"); B = np.asanyarray(bi.dataobj) > 0; A = bi.affine
    PS = np.asanyarray(L("photo_share").dataobj).astype(np.float32)
    vox = float(A[0, 0]); v = vox ** 3 / 1000
    x = A[0, 3] + vox * np.arange(B.shape[0]); assert np.allclose(x, -x[::-1], atol=1e-6), "grid not centred on x = 0"
    syn = B & (PS < 0.5); zero = B & (PS <= 0)
    sym = {}
    for k in ("grey", "rgb", "brainmask", "real_mask"):
        a = np.asanyarray(L(k).dataobj).astype(np.float32); sym[k] = float(np.abs(a - np.flip(a, 0)).max())
    mb = np.asanyarray(nib.load(f"{MRI}_brainmask.nii.gz").dataobj) > 0; ms = np.asanyarray(nib.load(f"{MRI}_synthesized_mask.nii.gz").dataobj) > 0
    m8 = json.load(open(prod["bftemplate_step8"] / "metrics.json")); m8 = m8.get("data", m8)
    row = dict(label=LABEL, arm=ARM, folder=cfg["bftemplate_step9"],
               n_brains=len(members), n_blockface_inputs=2 * len(members), members=members_text,
               grid_mm=vox, grid_shape=list(B.shape), rule="addendum 10 (tissue probability, mask-normalized values, precision weights)",
               brain_cm3=float(B.sum() * v), photographed_cm3=float((B & (PS >= 0.5)).sum() * v),
               predicted_cm3=float(syn.sum() * v), predicted_share_of_brain=float(syn.sum() / B.sum()),
               photo_share_zero_cm3=float(zero.sum() * v), symmetry=dict(max_abs_mirror_diff=sym),
               in_mri_template=dict(grid_mm=0.25, link="d2bs8_s5sym", brain_cm3=float(mb.sum() * 0.25 ** 3 / 1000),
                                    predicted_cm3=float(ms.sum() * 0.25 ** 3 / 1000), predicted_share_of_brain=float(ms.sum() / mb.sum())),
               d2bs8_for_reference=dict(brain_cm3=m8["brain_cm3"], predicted_cm3=m8["synth_cm3"],
                                        predicted_share_of_brain=m8["synth_frac_of_brain"]))
    print(f"{ARM} brain {row['brain_cm3']:.2f} cm3 predicted {row['predicted_cm3']:.2f} cm3 ({100 * row['predicted_share_of_brain']:.2f} %) "
          f"zero {row['photo_share_zero_cm3']:.2f} cm3 | MRI frame {row['in_mri_template']['predicted_cm3']:.2f} cm3 "
          f"({100 * row['in_mri_template']['predicted_share_of_brain']:.2f} %) | sym {sym}")
    provenance.write_numbers("bf_template_metrics_s9_015_s11", dict(templates=[row], definition=DEFINITION),
                             inputs=[f"{T}_{k}.nii.gz" for k in ("brainmask", "photo_share", "grey", "rgb", "real_mask")]
                             + [f"{MRI}_brainmask.nii.gz", f"{MRI}_synthesized_mask.nii.gz", prod["bftemplate_step8"] / "metrics.json"],
                             cmd="python code/figures/paper_numbers/bf_template_metrics_s9_015_s11.py", md5=False)


if __name__ == "__main__":
    main()
