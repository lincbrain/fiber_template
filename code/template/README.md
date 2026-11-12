# Building the templates

Three builds, one driver each. Each reads its brains, grids and parameters from its settings file.

| Driver | Builds |
|---|---|
| `build_mri_channels.py` | the MRI template (`settings/mri_construction.yaml`) |
| `build_blockface_template.py` | the blockface template (`settings/blockface_template.yaml`) |
| `build_link.py` | the two templates registered (`settings/link.yaml`) |

```text
$PY code/template/build_mri_channels.py --plan               every step; runs nothing
$PY code/template/build_mri_channels.py --step <name>        one step
$PY code/template/build_mri_channels.py --from <a> --to <b>  a range
```

A step whose outputs exist is not run again without `--force`. Each step writes a provenance file.
The builds step by step, with their times and memory: [01, part A](../../docs/tutorials/01_mri.md#a-the-mri-template-on-the-papers-four-brains) and [02, Step 7](../../docs/tutorials/02_blockface.md#step-7-the-blockface-template-and-its-link-to-the-mri-template).
A rebuild makes a new template: [00](../../docs/tutorials/00_start_here.md#adding-a-new-brain-two-different-things).

Both builds run through the template-construction driver of Arthofer et al. with our edits: [patches/](patches/README.md).

| | |
|---|---|
| `mmorf_run.py`, `channels.py`, `tools.py` | the MRI template |
| `blockface_build.py`, `blockface_mmorf.py`, `blockface_finish.py` | the blockface template |
| [link/](link/README.md) | the registration between the two |
