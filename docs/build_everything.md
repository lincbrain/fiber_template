# Build everything

The whole method, from the raw data to the paper's numbers and figures.

**You need** each brain's raw data and hand-made inputs (`docs/inputs.md`), the environments (`docs/install.md`) and a GPU.

**What it does**, in this order:
1. prepares the MRI of each brain with MRI;
2. builds the MRI template from four of those brains;
3. reconstructs each brain's blockface volume from its photographs;
4. aligns each blockface volume to the MRI template, through the brain's own MRI or, without MRI, directly;
5. synthesizes each volume's missing caudal tissue;
6. builds the blockface template from the 21 volumes with their synthesized tissue;
7. registers the blockface template into the MRI template (the link) and maps the template to NMT v2, INIA19 and CIVM-R;
8. registers each brain's histology to its photographs and transforms its labels into the template;
9. writes the numbers and figures of the paper, which hold the evaluation.

**You get** the products below. Per-brain products go to `$LINC_DATA/sub-<ID>/`, the templates and other results to `$LINC_WORK`,
and numbers records to `$LINC_NUMBERS`.

| Product | Command | Config | Environment | Time |
|---|---|---|---|---|
| a brain's prepared MRI (brain mask, bias-corrected b0) | `cli/linc-mri prep` | `mri_construction.yaml` | linc, GPU | under a minute |
| the MRI template | `reproduce/templates/build_mri_channels.py` | `mri_construction.yaml` | linc; MMORF on a 48 GB GPU | 3.5 days on the GPU, about 1 hour on a CPU |
| a brain's blockface volume | `cli/linc-reconstruct all` | `blockface.yaml`, `subjects/<ID>.yaml` | tirl, GPU | about 2.5 hours for 1,400 photographs |
| the blockface volume in its own b0 | `cli/linc-register mri` | `registration.yaml` | linc, GPU | 13 minutes |
| the b0 in the MRI template | `cli/linc-register mri-template` | `registration.yaml` | linc, GPU | 4 minutes |
| a brain without MRI in the MRI template | `cli/linc-register direct-placement` | `registration.yaml` | linc, GPU | 5 minutes |
| the blockface volume with its synthesized tissue | `cli/linc-complete prepare`, then `cli/linc-complete` | `completion.yaml` | deep, GPU | 3 minutes |
| the blockface template | `reproduce/templates/build_blockface_template.py` | `blockface_template.yaml` | linc; MMORF on a 48 GB GPU | 13 hours on the GPU, about 3 hours on a CPU |
| the link, and the mappings to published templates | `reproduce/templates/build_link.py` | `link.yaml` | linc, GPU | about 1 hour |
| a brain's histology, registered to its photographs | `cli/linc-histology all` | `histology.yaml` | tirl | about 2 hours |
| a brain's labels in the template | `cli/linc-labels push`, `interpolate`, `carry` | `labels.yaml` | tirl, then linc | about 40 minutes |
| a segmentation network, trained on the hand-drawn and hand-corrected masks | `cli/linc-segment train --task configs/segment/<model>.json --fold all` | `segment/<model>.json` | deep, GPU | about 2 hours per network |
| the numbers records | `make numbers` | `reproduce/numbers/` | each record's own | `docs/reproduce.md` |
| the figures | `make figures` | `figures.yaml` | linc; viz3d for Fig. 10 | `docs/reproduce.md` |

The weights of the segmentation networks come with the data release; training them again is optional, and `--plan` lists a task's
cases and brains first. Configs are in `configs/`. The times are from one node with an NVIDIA RTX 6000 Ada (48 GB). The per-brain commands are explained one
by one in `docs/walkthrough.md`.

## The three template builds

Each build has a driver in `reproduce/templates/`, with its recipe beside it (`mri_channels.md`, `blockface.md`, `link.md`).

```bash
$PY reproduce/templates/build_mri_channels.py --plan
$PY reproduce/templates/build_mri_channels.py --step prepare-MF283
$PY reproduce/templates/build_mri_channels.py --from assemble --to channels
```

`--plan` runs nothing. It prints every step with its environment, its time, its command, its inputs and its outputs. `--step` runs
one step, and `--from` and `--to` run a range. Each step writes a provenance JSON beside its output and is not rerun without `--force`.
Outputs go to `--work`, by default under `$LINC_WORK`.

The MMORF steps run through the template-construction driver (`LINC_TEMPLATE_DRIVER`, `linc/construction/patches/README.md`) and
`LINC_MMORF_EXE`. The MRI build also needs FSL, ANTs and MRtrix3 (`FSLDIR`, `ANTSPATH`, `MRTRIX`). The link needs `ANATOMIX_PYTHON`.

The CPU steps of each build were rerun for this release, and they reproduce the paper's files. The GPU steps were not rerun.

## Rebuilding a template

Each build reads its brains, grids and parameters from its config:
- `configs/mri_construction.yaml`: `inputs` lists the MRI template's brains, and `canonical` gives its grid;
- `configs/blockface_template.yaml`: `members` lists the blockface template's brains, with and without MRI, and `grids` gives its grids;
- `configs/link.yaml`: the link's inputs and its grid.

Edit the list or the grid, then run the driver with `--plan` to see the steps and the inputs it will read before anything runs.
