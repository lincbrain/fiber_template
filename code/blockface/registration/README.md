# Registration of a blockface volume

`linc-register` registers a brain's blockface volume to its own b0 or to the blockface template, and a b0 to the template.
It runs in `$PY` (`settings/env/linc.yml`) with FSL and the anatomix environment, with the settings in `settings/registration.yaml`.
The tutorials cover it in [01, Step B2](../../../docs/tutorials/01_mri.md#step-b2-register-the-b0-to-the-template), [02, Steps 3 and 4](../../../docs/tutorials/02_blockface.md#step-3-register-the-volume-to-the-brains-own-b0-a-brain-with-mri) and [02, Checks](../../../docs/tutorials/02_blockface.md#4-checks).

## Commands

| Command | What it registers |
|---|---|
| `linc-register mri` | the blockface volume to the brain's own b0 |
| `linc-register mri-template` | the brain's b0 to the template |
| `linc-register blockface-template` | the blockface volume to the blockface template: a brain without MRI, and every brain with histology, whose labels take this route |
| `linc-register direct-placement` | a brain without MRI: its blockface volume straight to the template's b0 (the cross-modal alignment), which gives its stand-in b0 |
| `linc-register qc` | checks one registration; presets `bf2b0`, `direct`, `b02template`, `bf2bftemplate`, `template2template`; limits in `settings/registration.yaml` |

A brain with MRI uses the first two, and a brain without MRI the next two.
A brain with histology also runs `blockface-template`, MRI or not, and then the refinement at 0.15 mm of [02, Step 4](../../../docs/tutorials/02_blockface.md#step-4-register-the-volume-to-the-blockface-template-a-brain-without-mri-or-with-histology), because its labels reach the template that way; the route through its own b0 is the other one the paper reports.

## Inputs and outputs

[01, Step B2](../../../docs/tutorials/01_mri.md#step-b2-register-the-b0-to-the-template) and [02, Steps 3 and 4](../../../docs/tutorials/02_blockface.md#step-3-register-the-volume-to-the-brains-own-b0-a-brain-with-mri) list what each command reads and writes; working files and checks also go to `$LINC_WORK/qc/`.

## Files

| | |
|---|---|
| `linc-register` | the command |
| `to_own_mri.py`, `direct.py` | the registrations |
| `featurereg.py` | the deformable engine (anatomix features and FireANTs) |
| `qc.py` | the checks |
