# Documentation

If you are new, start with [tutorials/00_start_here.md](tutorials/00_start_here.md) and read the tutorials in order.

## Pages

- [overview.md](overview.md) explains how everything fits together and what the package cannot do, and ends with a glossary.
- [tutorials/00_start_here.md](tutorials/00_start_here.md) sets you up and introduces the three tutorials: [01_mri.md](tutorials/01_mri.md), [02_blockface.md](tutorials/02_blockface.md) and [03_histology.md](tutorials/03_histology.md).
- [how-to/new_brain.md](how-to/new_brain.md) lists what to do, in order, for a new brain.
- [troubleshooting.md](troubleshooting.md) covers problems we have met and how to fix them.
- [faq.md](faq.md) gives short answers, each with a link.
- [brain_identifiers.md](brain_identifiers.md) explains names such as MF283, MF1 and M1, and which brain was used for what.
- [media/README.md](media/README.md) says how the README's animations are made.

Each folder's `README.md` describes its commands and files, [steps.yaml](../steps.yaml) lists every step of the paper with its command, check and needs, and [tests/README.md](../tests/README.md) explains what decides whether a rerun matches the paper.

## Commands

Run every command from the top of the repository, after `source settings/roots.sh`.
Each one prints what it needs, what it does and what it writes with `--help`.

| Command | What it does | Explained in |
|---|---|---|
| `bash install.sh` | builds the Python environments | [settings/env](../settings/env/README.md) |
| `bash tests/selftest.sh` | checks the installation, without data or a GPU | [tests](../tests/README.md) |
| `$PY run.py --plan`, `--run`, `--status` | runs the paper's steps with their checks | [00](tutorials/00_start_here.md#two-ways-to-run-a-step) |
| `$PY code/mri/linc-mri prep` | prepares a b0: one volume, its mask, its bias correction | [01, Step B1](tutorials/01_mri.md#step-b1-prepare-the-b0) |
| `$PY_TIRL code/blockface/linc-reconstruct` | turns photographs into a blockface volume | [02, Step 1](tutorials/02_blockface.md#step-1-reconstruct-the-volume) |
| `$PY code/blockface/registration/linc-register` | the registrations (`mri`, `mri-template`, `blockface-template`, `direct-placement`) and their check (`qc`) | [01, Step B2](tutorials/01_mri.md#step-b2-register-the-b0-to-the-template), [02, Step 3](tutorials/02_blockface.md#step-3-register-the-volume-to-the-brains-own-b0-a-brain-with-mri) |
| `$PY_DEEP code/blockface/synthesis/linc-complete` | synthesizes the missing tissue; `train` retrains the network | [02, Step 5](tutorials/02_blockface.md#step-5-synthesize-the-tissue-that-was-never-photographed) |
| `$PY_TIRL code/histology/linc-histology` | `propose` makes the match table, `all` runs the registration | [03, Step B](tutorials/03_histology.md#step-b-the-match-table) |
| `code/labels/linc-labels` | `push`, `interpolate`, `carry`, `connect`: labels into the template | [03, Step D](tutorials/03_histology.md#step-d-tracer-labels) |
| `$PY -m shared.xfm` | `list`, `chain` and `apply` a brain's transforms, moving images between spaces | [overview](overview.md#spaces-and-transforms) |
| `$PY_DEEP code/shared/linc-segment` | `predict` and `train` the segmentation networks | [code/shared/segment](../code/shared/segment/README.md) |
| `$PY code/shared/linc-info` | reports what the installation runs with, for a bug report | [code/shared](../code/shared/README.md) |
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
