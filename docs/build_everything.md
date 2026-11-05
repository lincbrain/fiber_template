# Build everything

The whole method, from the blockface photographs and the preprocessed MRI to the paper's numbers and figures.

**You need** each brain's data and hand-made inputs (`docs/inputs.md`), the environments (`docs/install.md`) and a GPU.

**What it does**, in this order:
1. builds the MRI template from the preprocessed MRI of four brains;
2. prepares the MRI of each sectioned brain with MRI for its registration;
3. reconstructs each brain's blockface volume from its photographs;
4. aligns each blockface volume to the MRI template, through the brain's own MRI or, without MRI, through the blockface template, once it
   is built (the cross-modal alignment gives such a brain its stand-in b0);
5. synthesizes each volume's missing caudal tissue;
6. builds the blockface template from the 21 volumes with their synthesized tissue;
7. registers the blockface template to the MRI template and registers the template to NMT v2, INIA19 and CIVM-R;
8. registers each brain's histology to its photographs and transforms its labels into the template;
9. writes the numbers and figures of the paper, which hold the evaluation.

**You get** the products below. Per-brain products go to `$LINC_DATA/sub-<ID>/`, the templates and other results to `$LINC_WORK`,
and numbers records to `$LINC_NUMBERS`.

| Product | Command | Config | Environment | Time |
|---|---|---|---|---|
| the MRI template, from the four brains' preprocessed files | `reproduce/templates/build_mri_channels.py` | `mri_construction.yaml` | linc; MRtrix3, FSL, ANTs; a GPU; MMORF on a 48 GB GPU | about 40 minutes per brain, then 3.5 days on the GPU |
| a brain's prepared MRI (brain mask, bias-corrected b0) | `cli/linc-mri prep` | `mri_construction.yaml` | linc, GPU | under a minute |
| a brain's blockface volume | `cli/linc-reconstruct all` | `blockface.yaml`, `subjects/<ID>.yaml` | tirl, GPU | about 10 hours per brain |
| the blockface volume in its own b0 | `cli/linc-register mri` | `registration.yaml` | linc, GPU | 13 minutes |
| the b0 in the MRI template | `cli/linc-register mri-template` | `registration.yaml` | linc, GPU | 4 minutes |
| a brain without MRI: its stand-in b0 | `cli/linc-register direct-placement` | `registration.yaml` | linc, GPU | 5 minutes |
| the blockface volume with its synthesized tissue | `cli/linc-complete prepare`, then `cli/linc-complete` | `completion.yaml` | deep, GPU | 3 minutes |
| the blockface template | `reproduce/templates/build_blockface_template.py` | `blockface_template.yaml` | linc; MMORF on a 48 GB GPU | about 25 hours on one card (estimated; the paper's run took 11 h 35 min on three), about 3 hours on a CPU |
| the blockface template registered to the MRI template, and its registrations to published templates | `reproduce/templates/build_link.py` | `link.yaml` | linc, GPU | about 1 hour |
| a new brain without MRI in the blockface template | `cli/linc-register blockface-template` | `registration.yaml` | linc, GPU | about 15 minutes |
| a brain's histology, registered to its photographs | `cli/linc-histology all` | `histology.yaml` | tirl | about 2 hours |
| a brain's labels in the template | `cli/linc-labels push`, `interpolate`, `carry`, `connect` | `labels.yaml` | tirl, then linc | about 40 minutes |
| a segmentation network, trained on the masks drawn or corrected by hand | `cli/linc-segment stage` (the blockface networks' training cases), then `cli/linc-segment train --task configs/segment/<model>.json --fold all` | `segment/<model>.json` | deep, GPU with about 10 GB free | about 2 hours per network |
| the synthesis network, trained on each brain's b0 and photographs | `cli/linc-complete train --root <prepared folders>` | `completion.yaml` | deep, GPU with about 8 GB free | about 1 hour per network |
| the numbers records | `make numbers`, `make numbers-gpu`, `make numbers-figures` | `reproduce/numbers/` | each record's own | `docs/reproduce.md` |
| the figures | `make figures` | `figures.yaml` | linc; viz3d for Fig. 10 | `docs/reproduce.md` |

From the preprocessed files, the MRI build reproduces the paper's files, with the differences `docs/mri.md` gives.
The networks' weights come with the data release; training them again is optional, and `--plan` lists the brains and any missing
files first. Configs are in `configs/`. The times are from one node with an NVIDIA RTX 6000 Ada (48 GB). The per-brain commands are explained one
by one in `docs/walkthrough.md`.

## The three template builds

Each build has a driver in `reproduce/templates/`, with its recipe beside it (`mri_channels.md`, `blockface.md`, `link.md`).

```bash
$PY reproduce/templates/build_mri_channels.py --plan
$PY reproduce/templates/build_mri_channels.py --step prepare-MF283
$PY reproduce/templates/build_mri_channels.py --from assemble --to channels
```

The MRI build starts each brain from its preprocessed files under `$LINC_MRI_PREPROCESSED/<ID>/` (step `stage-<ID>`; `docs/mri.md` lists them).

`--plan` runs nothing. It prints every step with its environment, its time, its command, its inputs and its outputs. `--step` runs
one step, and `--from` and `--to` run a range. Each step writes a provenance JSON beside its output and is not rerun without `--force`.
Outputs go to `--work`, by default under `$LINC_WORK`.

The MMORF steps run through the template-construction driver (`LINC_TEMPLATE_DRIVER` for the MRI build, `LINC_BLOCKFACE_DRIVER` for the
blockface build; `linc/construction/patches/README.md`) and
`LINC_MMORF_EXE`. The MRI build also needs FSL, ANTs and MRtrix3 (`FSLDIR`, `ANTSPATH`, `MRTRIX`). The registration between the two templates needs `ANATOMIX_PYTHON`.


## What was rerun for this release

Rerun with this code, and identical to the paper's files or differing as the pages named here state:
- the reconstruction of MM2, MF8 and MM14 from their photographs (`docs/walkthrough.md`);
- the registrations of MM2 and MF8, including MF8 through the blockface template, their synthesis, their histology, and MM2's labels
  (`docs/walkthrough.md`);
- `linc-histology propose` on the series of the analysis's 12 rule-written tables and on the five brains with hand-made tables
  (`linc/histology/README.md`);
- one full training of two networks: `mri_brainmask`, whose weights came out byte-identical to the shipped ones on the same GPU model; and
  the synthesis network, the same recipe but not the same bits (`docs/complete.md`);
- the MRI template's build from the preprocessed files, step by step: `stage` and `tensor` on all four brains, voxel-identical to the
  paper's; `nmt`, `t2reg`, `landmarks`, `frames` and `prepare` on all four brains and `microreg` on MF1, each step on its own, which differ
  as `docs/mri.md` states;
- the construction of the MRI template on the paper's four inputs: `assemble`; the `mmorf` step's affine stage and step 1, through the
  release's driver, byte-identical to the paper's run (202 of 202 comparable files); and `channels` on the paper's run (20 of 20 files identical);
- the blockface template's build from its 21 brains through the seven steps of its first stage on one card: steps 1 to 5 identical to the
  paper's files, steps 6 and 7 slightly different, starting at the first 1 mm level (`reproduce/templates/blockface.md`);
- the registration between the two templates and to NMT v2, whose GPU step gives transforms within 1 mm of the paper's, with medians under
  0.1 mm (`reproduce/templates/link.md`).

Not rerun: `microreg` on MF2 to MF4; steps 2 to 8 of the MRI template's construction; step 8 of the blockface template; the registrations to INIA19
and CIVM-R; and the full training of the other networks. Their code and parameters are the paper's.

## Rebuilding a template

Each build reads its brains, grids and parameters from its config:
- `configs/mri_construction.yaml`: `inputs` lists the MRI template's brains, and `canonical` gives its grid;
- `configs/blockface_template.yaml`: `members` lists the blockface template's brains, with and without MRI, and `grids` gives its grids;
- `configs/link.yaml`: the inputs and the grid of the registration between the two templates.

Edit the list or the grid, then run the driver with `--plan` to see the steps and the inputs it will read before anything runs.
