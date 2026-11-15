# Documentation

Every page and command of the repository, and what each answers.

## Pages

| Page | Kind | Answers |
|---|---|---|
| [overview.md](overview.md) | explanation | What are the parts, spaces, templates and networks, and how do they connect? Where do files and results go? What can it not do? What does a term mean? |
| [tutorials/00_start_here.md](tutorials/00_start_here.md) | tutorial | How do I set up and run a step? Aligning a brain to the template versus adding it into the template |
| [tutorials/01_mri.md](tutorials/01_mri.md) | tutorial | How was the MRI template built? How do I register a b0 to it and move images into it, or to NMT v2, INIA19 and CIVM-R? |
| [tutorials/02_blockface.md](tutorials/02_blockface.md) | tutorial | How do photographs become a volume in the template, with or without MRI? How is the missing tissue synthesized, and the blockface template built? |
| [tutorials/03_histology.md](tutorials/03_histology.md) | tutorial | How are histology sections matched and registered, and their labels carried into the template? |
| [how-to/new_brain.md](how-to/new_brain.md) | how-to | What do I do, in order, for a new brain? |
| [troubleshooting.md](troubleshooting.md) | how-to | Something went wrong: why, and what to do |
| [faq.md](faq.md) | questions | Short answers, each with its link |
| [brain_identifiers.md](brain_identifiers.md) | reference | What are MF283, MF1 and M1, and which brain was used for what? |
| [media/README.md](media/README.md) | reference | How are the README's animations made? |
| each folder's `README.md`, [steps.yaml](../steps.yaml) | reference | A folder's commands, inputs, outputs and files; every step of the paper with its command, check and needs |
| [tests/README.md](../tests/README.md) | explanation | What decides whether a rerun matches the paper? |

## Commands

Every command prints what it needs, what it does and what it writes with `--help`.
Run them from the top of the repository, after `source settings/roots.sh`.

| Command | What it does | Explained in |
|---|---|---|
| `bash install.sh` | builds the Python environments | [settings/env](../settings/env/README.md) |
| `bash tests/selftest.sh` | checks the installation, without data or a GPU | [tests](../tests/README.md) |
| `$PY run.py --plan`, `--run`, `--status` | runs the paper's steps with their checks | [00](tutorials/00_start_here.md#two-ways-to-run-a-step) |
| `$PY code/mri/linc-mri prep` | prepares a b0: one volume, its mask, its bias correction | [01, Step B1](tutorials/01_mri.md#step-b1-prepare-the-b0) |
| `$PY_TIRL code/blockface/linc-reconstruct` | photographs to a blockface volume | [02, Step 1](tutorials/02_blockface.md#step-1-reconstruct-the-volume) |
| `$PY code/blockface/registration/linc-register` | `mri`, `mri-template`, `blockface-template`, `direct-placement`: the registrations; `qc`: their check | [01, Step B2](tutorials/01_mri.md#step-b2-register-the-b0-to-the-template), [02, Step 3](tutorials/02_blockface.md#step-3-register-the-volume-to-the-brains-own-b0-a-brain-with-mri) |
| `$PY_DEEP code/blockface/synthesis/linc-complete` | synthesizes the missing tissue; `train` retrains the network | [02, Step 5](tutorials/02_blockface.md#step-5-synthesize-the-tissue-that-was-never-photographed) |
| `$PY_TIRL code/histology/linc-histology` | `propose`: the match table; `all`: the registration | [03, Step B](tutorials/03_histology.md#step-b-the-match-table) |
| `code/labels/linc-labels` | `push`, `interpolate`, `carry`, `connect`: labels into the template | [03, Step D](tutorials/03_histology.md#step-d-tracer-labels) |
| `$PY -m shared.xfm` | `list`, `chain`, `apply`: a brain's transforms; images moved between spaces | [overview](overview.md#spaces-and-transforms) |
| `$PY_DEEP code/shared/linc-segment` | `predict`, `train`: the segmentation networks | [code/shared/segment](../code/shared/segment/README.md) |
| `$PY code/shared/linc-info` | what the installation runs with, for a bug report | [code/shared](../code/shared/README.md) |
| `$PY code/template/build_mri_channels.py`, `build_blockface_template.py`, `build_link.py` | rebuild the templates and the template link | [01, part A](tutorials/01_mri.md#a-the-mri-template-on-the-papers-four-brains), [02, Step 7](tutorials/02_blockface.md#step-7-the-blockface-template-and-its-link-to-the-mri-template) |
| `make -C code/figures numbers`, `figures`, `check` | remake and check the paper's numbers and figures | [code/figures](../code/figures/README.md) |
| `$PY tests/check_numbers.py`, `tests/verify_weights.py`, `tests/compare/*.py` | compare a numbers record, the weights or a result with the paper's | [tests](../tests/README.md) |
| `$PY_TIRL code/docs/tutorial_figures.py` | draws the tutorials' figures | [code/docs](../code/docs/README.md) |

## Repository layout

| | |
|---|---|
| [install.sh](../install.sh) | builds the Python environments |
| [run.py](../run.py) | runs the paper's steps and checks each against the paper's files |
| [steps.yaml](../steps.yaml) | every step of the paper |
| [code/](../code/README.md) | the pipeline, one folder per stage |
| [settings/](../settings/README.md) | every parameter: one file per brain and one per stage |
| [models/](../settings/models/README.md) | the trained networks' weights |
| [tests/](../tests/README.md) | checks of the installation and of the results |
| [docs/](README.md) | this documentation |
