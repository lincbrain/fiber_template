# fiber_template: code for a multimodal macaque template

Code for *A multimodal macaque template for aligning ex vivo MRI and microscopy* (Gonzales et al.). It runs the method from the
blockface photographs and the preprocessed MRI (`docs/inputs.md`):
- builds the MRI template from four brains and the blockface template from 21;
- registers the blockface template into the MRI template;
- aligns each brain's histology and its labels to the template;
- evaluates every registration step and writes the paper's numbers and figures;
- brings new brains into the template.

`docs/build_everything.md` runs it from start to end, and `docs/inputs.md` lists every input it assumes.

> **Status: release in preparation.** This repository holds the analysis code of the paper, ported and checked against the
> paper's products; it will be public on GitHub. The template, the paper's products and the network weights are a separate data
> release on DANDI, in the LINC archive (see [Data](#data)). The links will follow. The public repository starts from the code
> as released; the development history is not part of it.

## What the template is

The template is a symmetric population template of the macaque brain. It lives in the NMT v2 symmetric frame on a 0.25 mm grid and
has two kinds of channels.

- **MRI channels.** T2-weighted, diffusion tensor, and the SANDI soma and neurite fractions. They were built from the ex vivo MRI of
  four *Macaca fascicularis* brains and their mirror images; all four contrasts drove one alignment.
- **Blockface channel.** A population average of the photographs taken while 21 *M. mulatta* and *M. fascicularis* brains were
  sectioned for histology (8 of them never scanned). It was built as the blockface template in its own space, then registered to the
  MRI template.

Histology is aligned to the template through the blockface photograph of its own section, with or without MRI of its brain: by the
MRI-based alignment (through the brain's own MRI) or the blockface-based alignment (through the blockface template). The paper
measures the accuracy of each registration step against references that the registration had not used.

## What is in this repository

```
linc/          the pipeline, as a Python package: blockface reconstruction, segmentation networks, synthesis of missing tissue,
               registration, template construction, the blockface template's registration into the MRI template, histology and
               labels, evaluation
cli/           entry points for new data: linc-mri, linc-reconstruct, linc-segment, linc-register, linc-complete, linc-histology,
               linc-labels;
               linc-info, which prints what an installation runs with
configs/       one YAML per pipeline, holding the paper's parameters; each comment names the Methods subsection it comes from
reproduce/     the scripts behind the paper: numbers/ (one per numbers record), figures/ (one per figure), templates/ (how the
               templates were built), check_numbers.py
models/        model cards, checksums and a checker for the trained networks (the weights are downloaded, not in git)
env/           the Python environments
tests/         tests of the package
docs/          one page per stage (install, mri, reconstruct, register, complete, histology, labels, reproduce,
               build_everything) and for the segmentation models, a walkthrough of four brains, the paper → code map, the
               data layout, FAQ
CONVENTIONS.md brains and names, data layout, spaces and transforms, provenance
```

Start with `docs/paper_to_code.md` to find where a part of the paper lives in the code.

## Install

1. **Python environments.** One script builds them all into `envs/` in your clone (about 20–25 minutes):

   ```bash
   export LINC_TIRL_SRC=<folder with the TIRL sources>
   bash env/install.sh
   source envs/linc_envs.sh      # in every new shell
   ```

   TIRL and its script packages are installed from a local copy of their sources, which their authors provide (`docs/install.md`);
   `docs/install.md` also says how to build one environment by hand.
2. **External programs.** FSL 6.0.7.4 and ANTs 2.5.4, found through `FSLDIR` and `ANTSPATH` or on `PATH`; MMORF 1.0.1 comes with
   the environments. The wrappers check each version before running.
3. **Weights and data.** Download the network weights with the data release, check them with `models/verify_weights.py`, and set the
   data roots (`LINC_DATA`, `LINC_TEMPLATE`, `LINC_MODELS` and the others in `docs/install.md`).

`python3 cli/linc-info` then prints what the installation runs with, and `make test` checks it (below). Every detail, including the variables of the other programs
and the fonts of the figures, is in `docs/install.md`.

## Use the template with a new brain

The commands below run in the order a new brain needs them, from the repository root. Each starts with the Python of its
environment, in brackets (`$PY` for `env/linc.yml`, `$PY_DEEP`, `$PY_TIRL`), which `source envs/linc_envs.sh` sets. Each
command's page in `docs/` gives its inputs and options, and `--help` lists every option. A brain's own settings live in
`configs/subjects/<ID>.yaml`; start from the commented `configs/subjects/TEMPLATE.yaml`.

1. **Prepare the MRI**, for a brain with MRI: a b0 or T2-weighted image, its brain mask by the MRI brain-mask network, and the
   bias-corrected brain [`env/linc.yml`] (`docs/mri.md`):

   ```bash
   $PY cli/linc-mri prep --subject <ID> --image <b0.nii.gz>
   ```

   In: a b0 or T2-weighted image. Out: `sub-<ID>/mri/masks/dwi_brain_mask.nii.gz` and `sub-<ID>/mri/dwi/b0_brain_n4.nii.gz`.

2. **Reconstruct the blockface volume** from the photographs [`env/tirl.yml`] (`docs/reconstruct.md`):

   ```bash
   $PY_TIRL cli/linc-reconstruct all --subject <ID>
   ```

   In: the photographs and `configs/subjects/<ID>.yaml`. Out: `sub-<ID>/blockface/<ID>_blockface.ome.zarr`, with its brain mask and
   marker tubes.

3. **Register the brain into the template's frame** (`docs/register.md`). For a brain with MRI, first to its own b0 [`env/linc.yml`],
   with the files of step 1, then its b0 to the MRI template [`env/linc.yml`]:

   ```bash
   M=$LINC_DATA/sub-<ID>/mri
   $PY cli/linc-register mri --subject <ID> --place-b0 $M/dwi/b0_brain_n4.nii.gz $M/masks/dwi_brain_mask.nii.gz --new-brain
   $PY cli/linc-register mri-template --subject <ID> --template-b0 $LINC_TEMPLATE/mri/linc_mri_b0.nii.gz \
       --template-mask $LINC_TEMPLATE/mri/linc_mri_brainmask.nii.gz
   ```

   In: the blockface volume and the prepared b0. Out: `sub-<ID>/xfm/bf-native__to__b0-native/` and
   `sub-<ID>/xfm/b0-native__to__mri-template/`.

   A brain without MRI is registered once, onto the MRI template's b0, with `cli/linc-register direct-placement`. Out:
   `sub-<ID>/xfm/bf-native__to__mri-template/`.

4. **Synthesize the missing caudal tissue** [`env/deep.yml`] (`docs/complete.md`):

   ```bash
   $PY_DEEP cli/linc-complete prepare --subject <ID> --out $LINC_WORK/prepared
   $PY_DEEP cli/linc-complete --brain $LINC_WORK/prepared/<ID> --out $LINC_WORK/complete
   ```

   In: the blockface volume and its registration. Out: `$LINC_WORK/complete/<ID>/<ID>_bf_completed.nii.gz` and its color version.

5. **Register the histology** to its blockface photographs [`env/tirl.yml`] (`docs/histology.md`):

   ```bash
   $PY_TIRL cli/linc-histology all --subject <ID> --store $LINC_DATA/sub-<ID>/blockface/<ID>_blockface.ome.zarr \
       --histology <sections dir> --table <match table> --regions <ID>_regions.nii.gz \
       --out $LINC_DATA/sub-<ID>/histology/registration --threads 12
   ```

   In: the section images, the match table and the regions volume (`docs/inputs.md`). Out: each section's registration, under
   `sub-<ID>/histology/registration/`.

6. **Transform the labels into the template** (`docs/labels.md`):

   ```bash
   $PY_TIRL cli/linc-labels push        --subject <ID>   # [env/tirl.yml] the annotations onto the blockface
   $PY      cli/linc-labels interpolate --subject <ID>   # [env/linc.yml] between the sections
   $PY      cli/linc-labels carry       --subject <ID>   # [env/linc.yml] transformed into the template
   ```

   In: the annotations and the section map, with the histology registration. Out: `sub-<ID>/labels/blockface/` and
   `sub-<ID>/labels/template/`.

The segmentation networks can also be run on their own (`cli/linc-segment`, [`env/deep.yml`]).

## Reproduce the paper

`make test` checks the installation without any data. The other three commands need the paper's intermediate products
(`$LINC_PRODUCTS`, from the data release):

```bash
make test           # the tests and toy runs of the command lines; a few minutes, no data, no GPU
make numbers        # every numbers record the paper reads, into $LINC_NUMBERS
make figures        # every figure, into outputs/figures
make check          # compare the rewritten records with the paper's
```

`docs/reproduce.md` gives the Python of each target, the records that need a GPU or raw data, and what decides whether a rerun
matches the paper bit for bit. `docs/build_everything.md` builds the whole method, the templates included, step by step. `docs/paper_to_code.md`
lists which records and scripts belong to each part of the paper.

## Data

The template, its transforms, the per-brain products and the network weights will be released on DANDI, in the LINC archive,
under the Creative Commons Attribution 4.0 license (CC BY 4.0) *(link to follow)*.

ONPRC18 and CIVM-R image intensities are not redistributed, because their licenses do not allow it. Only our transforms to them
are released. NMT v2 and CIVM-R are also inputs that you download yourself, for the comparison of registration targets and the tracer
topography (which reads the CHARM key from your NMT v2 copy): `configs/targets.yaml` gives both sources and the folder layout
(`configs/link.yaml` `inputs.atlas_reference`).

## Citation

Please cite the paper (`CITATION.cff`):

> Gonzales RA, Habte E, Wu J, Gong T, Maffei C, Celestine MR, Dey N, Huszar IN, París G, Shao J, Bell E, Lehman JF, Haber SN,
> Balbastre Y, Yendiki A. A multimodal macaque template for aligning ex vivo MRI and microscopy. *(in preparation)*

## License

The code is licensed under the Apache License 2.0 (`LICENSE`); the copyright and the terms of the third-party patches it carries
are in `NOTICE`. The data release is licensed under CC BY 4.0.

## Contact

Ricardo A. Gonzales, Athinoula A. Martinos Center for Biomedical Imaging, Massachusetts General Hospital
(rgonzalesvera@mgh.harvard.edu).
