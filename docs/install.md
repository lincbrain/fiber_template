# Install

The pipeline's stages need different environments and programs; set up only those of the stages you run. Each stage's page in `docs/` names its environment.

**1. Python environments.** One script builds them all:

```bash
export LINC_TIRL_SRC=<folder with the TIRL sources>
bash env/install.sh
source envs/linc_envs.sh          # in every new shell: points the pipeline at these environments
```

`env/install.sh` creates each environment from its file in `env/` into `envs/` in your clone (`--prefix` gives another folder),
runs the source installs the files' headers describe, installs this package into each, and ends with the installation check
(`docs/reproduce.md`). It takes about 20–25 minutes, most of it downloading. It is safe to run again: an environment it already built
is skipped, and nothing is deleted. After `git pull` changes an `env/*.yml` file, remove that environment's folder under
`envs/` and run `env/install.sh` again: it rebuilds only what is missing. `--only linc,deep` builds some of them.

TIRL 3.6.3 and its three script packages (tirlscripts-oxford-scriptutils 1.1.1, tirlscripts-stanford-core 1.2.2,
tirlscripts-mgh-linc 1.0.1) are installed from a local folder that holds their four source trees, named by
`LINC_TIRL_SRC`. TIRL is not yet publicly distributed: ask its authors for the sources. Without them, the TIRL environment is created
without TIRL, and the reconstruction and histology steps refuse to run.

The environments, because their packages conflict (TIRL needs numpy 1.x):

| Environment | Used by |
|---|---|
| `env/linc.yml` | `linc-register` (its linear stages and products) and `linc-register mri-template`, `linc-mri prep`, the MRI channels' construction (MMORF 1.0.1), `linc-labels carry` and `linc-labels interpolate` (SimpleITK demons), evaluation, the numbers and the figures (`docs/paper_to_code.md` lists each figure's) |
| `env/deep.yml` | the four segmentation models (`linc-segment`) and the synthesis of the missing caudal tissue (`linc-complete`), the synthesis numbers and the LGN/STN budget, and their three supplementary figures |
| `env/anatomix.yml` | the deformable registration (anatomix features + FireANTs), called by `linc-register` |
| `env/tirl.yml` | `linc-reconstruct` (its Python steps and the `zarr2nii` / `nii2zarr` exports), `linc-histology`, `linc-labels push` (TIRL 3.6.3, installed from source) |
| `env/viz3d.yml` | optional (`--with-figures-3d`): the 3D tracer scene (pyvista 0.44.2 on a headless VTK) |

Until 3 Nov 2026, the MRI construction and the label interpolation had environments of their own; they were merged into
`env/linc.yml`, in which the registration and the interpolation are byte-identical.

**By hand**, for one environment: create it from its file and install this package into it with `--no-deps` (each file already
pins every dependency at the version of record; without `--no-deps`, pip would upgrade some of them, e.g. numpy). Each file's
header gives its exact versions and notes.

```bash
export PYTHONNOUSERSITE=1      # keep packages in ~/.local out of the environments, when building and when running
conda env create -f env/linc.yml && conda activate linc && pip install -e . --no-deps
```

Set `PYTHONNOUSERSITE=1` for both building and running. Python puts a user's own `~/.local` packages ahead of the environment's,
and pip counts them as already installed. They can then replace the pinned versions, or stand in for packages that pip skips.

Two engines are not on PyPI at the paper's versions and are installed from their sources. The steps are in the file headers:
- anatomix (pinned commit) and its FireANTs fork: `env/anatomix.yml`;
- TIRL 3.6.3 and its three script packages: `env/tirl.yml`.

**2. External programs**, at the paper's versions. Set the variables below so the pipeline can find them, or put the programs on
`PATH`.

| Program | Version | Variable |
|---|---|---|
| FSL | 6.0.7.4 | `FSLDIR` |
| MMORF | 1.0.1, installed with the environments (`fsl-mmorf`, from the FSL conda channel); `envs/linc_envs.sh` sets its path | `MMORF` (the `mmorf` executable) |
| ANTs | 2.5.4 | `ANTSPATH` |
| MRtrix3 | 3.0.2, for the MRI from the scanner's files and the tensor step of the MRI channels' build (`docs/build_everything.md`) | `MRTRIX` |
| TORTOISE | V4.1.0, for the second diffusion session only (`mri_raw dt2`): a free download from the NIH after registration, run as a container on an NVIDIA GPU | `TORTOISE` (the folder that holds `TORTOISEProcess`) |
| Julia | 1.11.4, for the MRI from the scanner's files only, with the packages of `env/julia` installed in `JULIA_DEPOT_PATH` | `JULIA` |
| the anatomix environment's Python | see `env/anatomix.yml` | `ANATOMIX_PYTHON` |
| the deep-learning environment's Python, for the segmentation models | see `env/deep.yml` | `LINC_SEGMENT_PYTHON` |
| TIRL | 3.6.3 (`env/tirl.yml`), found beside the running Python | `TIRL_BIN` only to point at another installation |
| zarr2nii | nifti-zarr 1.0.0rc6, installed in `env/tirl.yml` | found on `PATH` (`ZARR2NII_BIN` only to point at another installation) |

The wrappers check each version before running and stop on a mismatch. `LINC_SKIP_VERSION_CHECK=1` lets them continue, but the
results are then not the paper's.

**3. Fonts** (figures only). Liberation Sans, from the `fonts-liberation` package.

**4. Network weights.** Download them with the data release. Put them under `$LINC_MODELS`, as `models/*.md` describes, and check
them:

```bash
$PY models/verify_weights.py models/segmentation.sha256 --root $LINC_MODELS/segment
$PY models/verify_weights.py models/completion_public.sha256 --root $LINC_MODELS/completion
```

**5. Where your data live.** Set the roots once per shell (`docs/data_layout.md` explains each one):

```bash
export LINC_RAW=/data/linc/photographs     # blockface photographs as taken
export LINC_DATA=/data/linc/brains         # per-brain data: $LINC_DATA/sub-<ID>/...
export LINC_TEMPLATE=/data/linc/template   # the released template
export LINC_MODELS=/data/linc/models       # the trained networks
export LINC_PRODUCTS=/data/linc/products   # the paper's intermediate products (to reproduce the paper)
export LINC_NUMBERS=$PWD/numbers           # numbers records
export LINC_WORK=$PWD/work                 # scratch
```

Paths recorded inside the products (input lists, transform metadata) name the machine that made them. Before running anything
that maps them (the `reproduce/` scripts, `linc-labels carry`), copy `configs/path_placeholders.yaml` to
`configs/path_placeholders.local.yaml` (not in git) and set each left-hand side to the prefix your products record; the example file
explains how.

Brains are named in files by their Specimen ID (`sub-MR252`). The paper's names (MF1–MF10, MM1–MM15) map to them in
`docs/specimen_labels.json`.

**6. Check the install.** `cli/linc-info` prints what the installation runs with: the repository's commit, this Python and its
torch, CUDA and GPUs, the external programs beside the versions the paper used, the interpreters the pipeline is pointed at
(`LINC_SEGMENT_PYTHON`, `ANATOMIX_PYTHON`, `PY_TIRL`, `PY_DEEP`) with their own versions, and the data roots. It needs only
Python's standard library, so it runs outside the environments too, and a missing program or variable prints "not found" or
"(not set)" instead of an error. It takes a few seconds, up to about a minute when an interpreter imports torch for the first time;
`--json` prints one JSON object. `make test` then runs the tests and toy runs of each command line (`docs/reproduce.md`).

```bash
python3 cli/linc-info
```

An excerpt of its output, with the paths shortened:

```
external_programs:
  ants:
    version: 2.5.4
    path: <ANTs>/bin/antsRegistration
    paper: 2.5.4
interpreters:
  LINC_SEGMENT_PYTHON:
    path: <envs>/linc-deep/bin/python
    python: 3.11.13
    torch: 2.6.0+cu124
    cuda: 12.4
```
