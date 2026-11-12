# Registration of a blockface volume

Registers a brain's blockface volume to its own MRI or to the template.
Environment: `settings/env/linc.yml`, with FSL and the anatomix environment. Settings: `settings/registration.yaml`.

| Command | What it registers |
|---|---|
| `linc-register mri` | the blockface volume to the brain's own b0 |
| `linc-register mri-template` | the brain's b0 to the MRI template |
| `linc-register blockface-template` | a brain without MRI, to the blockface template |
| `linc-register direct-placement` | a brain without MRI, to the MRI template's b0 (gives its stand-in b0, which `code/blockface/synthesis/` reads) |
| `linc-register qc` | checks one registration; presets `bf2b0`, `direct`, `b02template`, `bf2bftemplate`, `template2template`; limits in `settings/registration.yaml` |

A brain with MRI uses the first two; a brain without MRI uses the next two.
How to run each, its GPU memory and time, what it writes and how to check it: [02, Steps 3 and 4](../../../docs/tutorials/02_blockface.md#step-3-register-the-volume-to-the-brains-own-b0-a-brain-with-mri) and [02, Checks](../../../docs/tutorials/02_blockface.md#4-checks).

| | |
|---|---|
| `linc-register` | the command |
| `to_own_mri.py`, `direct.py` | the registrations |
| `featurereg.py` | the deformable engine (anatomix features and FireANTs) |
| `qc.py` | the checks |
