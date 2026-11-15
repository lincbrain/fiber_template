# Registration of a blockface volume

Registers a brain's blockface volume to its own b0 or to the blockface template, and a b0 to the template.
Environment: `settings/env/linc.yml` (`$PY`), with FSL and the anatomix environment. Settings: `settings/registration.yaml`.
Step by step: [01, Step B2](../../../docs/tutorials/01_mri.md#step-b2-register-the-b0-to-the-template), [02, Steps 3 and 4](../../../docs/tutorials/02_blockface.md#step-3-register-the-volume-to-the-brains-own-b0-a-brain-with-mri) and [02, Checks](../../../docs/tutorials/02_blockface.md#4-checks).

## Commands

| Command | What it registers |
|---|---|
| `linc-register mri` | the blockface volume to the brain's own b0 |
| `linc-register mri-template` | the brain's b0 to the template |
| `linc-register blockface-template` | a brain without MRI: its blockface volume to the blockface template |
| `linc-register direct-placement` | a brain without MRI: its blockface volume straight to the template's b0, the cross-modal alignment, which gives its stand-in b0 |
| `linc-register qc` | checks one registration; presets `bf2b0`, `direct`, `b02template`, `bf2bftemplate`, `template2template`; limits in `settings/registration.yaml` |

A brain with MRI uses the first two, a brain without MRI the next two.

## Inputs

- the blockface volume and its mask, in `$LINC_DATA/sub-<ID>/blockface/` ([code/blockface](../README.md));
- the brain's b0 and its mask, prepared by `linc-mri prep` ([code/mri](../../mri/README.md));
- the template's b0 and brain mask (`$LINC_TEMPLATE/mri/`), or the blockface template.

## Outputs

- each transform, `$LINC_DATA/sub-<ID>/xfm/<from>__to__<to>/`: its warp, its linear parts and `meta.json` with the check's verdict;
- working files and checks in `$LINC_WORK/register/<ID>/`, `$LINC_WORK/b0_to_template/<ID>/` and `$LINC_WORK/qc/`;
- the stand-in b0 (`direct-placement`), in `$LINC_DATA/sub-<ID>/mri/stand_in/`.

## Files

| | |
|---|---|
| `linc-register` | the command |
| `to_own_mri.py`, `direct.py` | the registrations |
| `featurereg.py` | the deformable engine (anatomix features and FireANTs) |
| `qc.py` | the checks |
