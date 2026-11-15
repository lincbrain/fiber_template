# Building the templates

There are three builds, the MRI template, the blockface template and the template link, each with one driver that reads its brains, grids and parameters from its settings file.
You do not need them to align a new brain: a rebuild makes a new template ([00](../../docs/tutorials/00_start_here.md#adding-a-new-brain-two-different-things)).
[01, part A](../../docs/tutorials/01_mri.md#a-the-mri-template-on-the-papers-four-brains) and [02, Step 7](../../docs/tutorials/02_blockface.md#step-7-the-blockface-template-and-its-link-to-the-mri-template) go through them step by step, with times and memory.

## Commands

| Driver | Builds | Settings |
|---|---|---|
| `build_mri_channels.py` | the MRI template | `settings/mri_construction.yaml` |
| `build_blockface_template.py` | the blockface template | `settings/blockface_template.yaml` |
| `build_link.py` | the template link, and the registrations to the published templates | `settings/link.yaml` |

```text
$PY code/template/build_mri_channels.py --plan               every step; runs nothing
$PY code/template/build_mri_channels.py --step <name>        one step
$PY code/template/build_mri_channels.py --from <a> --to <b>  a range
```

A step whose outputs exist does not run again without `--force`, and each step writes a provenance file.
The groupwise registrations run MMORF through the template-construction driver of Arthofer et al. with our edits ([patches/](patches/README.md)).

## Inputs and outputs

The members are listed in the settings files, and [01, part A](../../docs/tutorials/01_mri.md#a-the-mri-template-on-the-papers-four-brains) and [02, Step 7](../../docs/tutorials/02_blockface.md#step-7-the-blockface-template-and-its-link-to-the-mri-template) list what each build writes under `$LINC_WORK/`.

## Files

| | |
|---|---|
| `mmorf_run.py`, `channels.py`, `tools.py` | the MRI template |
| `blockface_build.py`, `blockface_mmorf.py`, `blockface_finish.py` | the blockface template |
| [blockface_shim/](blockface_shim/README.md) | the MMORF wrapper of the blockface build |
| [link/](link/README.md) | the template link, and a b0's registration to the template |
| [patches/](patches/README.md) | our edits to the template-construction driver |
