# Inputs

Everything the code assumes it is given, in three kinds: data, made by an instrument; hand-made inputs, made once per brain by a
person; and downloads. Everything else is made by the code. Paths are relative to a brain's folder, `$LINC_DATA/sub-<ID>/`, unless
they start with another root (`docs/data_layout.md`).

## Data

| Input | Format and location | Read by |
|---|---|---|
| blockface photographs | JPEG, one per section, RGB, numbered in cutting order; one folder per brain under `$LINC_RAW`, named by `raw` in `configs/subjects/<ID>.yaml`, the numbers read by its `raw_regex` | `linc-reconstruct` |
| preprocessed MRI of the MRI template's brains | NIfTI, under `$LINC_MRI_PREPROCESSED/<ID>/` (the data release's `mri_preprocessed/`): `dwi/`, `msme/` and `dt2/tortoise/` (`docs/mri.md` lists the files) | the MRI template build |
| a b0 or T2-weighted image | NIfTI, for a brain that is only registered to the template | `linc-mri prep` |
| histology sections | JPEG 2000, one per section (`sourcedata/sub-<ID>/micr/<section>.jp2`), and their OME-Zarr stacks (`rawdata/sub-<ID>/micr/`), from LINC dataset 000003 | `linc-histology`, `linc-labels` |

We assume the MRI inputs are preprocessed: for each brain of the MRI template, the 0.25 mm diffusion series, the multi-echo series and
the 0.5 mm multi-shell series made for Gong et al. 2026, with the SANDIdot fit. This repository adds the N4 correction, the diffusion
tensor, the registrations within each brain and to NMT v2, the canonical frame and the template (`docs/mri.md`). A new brain that is only
registered to the template needs one b0 or T2-weighted image (`linc-mri prep`).

## Hand-made inputs

| Input | Format and location | How a new brain gets it | Read by |
|---|---|---|---|
| pixel size and section step | `vx`, `vz` (µm) in `configs/subjects/<ID>.yaml` | a ruler in a photograph; the microtome's step | `linc-reconstruct` |
| camera jumps | `jumps` in the same file: [last section before, first after], 0-based | by eye, on the stacked photographs of `linc-reconstruct stack` | `linc-reconstruct` |
| missing or damaged sections, leading frames, caudal cutoff | `missing` in the same file | by eye, the same way | `linc-reconstruct` |
| corrected photograph masks (optional) | PNG, `blockface/01_brainmask/01_final/` | `linc-reconstruct edit-photo-masks export`, edit, `import` | `linc-reconstruct` |
| corrected 3D brain mask and marker tubes (optional) | NIfTI, `blockface/curated/<ID>_mask_edited.nii.gz`, `<ID>_holes_edited.nii.gz` | edit the network's `<ID>_mask.nii.gz`, `<ID>_holes.nii.gz` | `linc-reconstruct` |
| MRI brain masks | NIfTI, `mri/masks/`: one per session (`dwi_brain_mask`, `t2_brain_mask`, `micro_brain_mask`), on that session's grid, a network's prediction edited by hand, and the final edited b0 mask of each MRI template brain (`final_brain_mask`) | optional: `linc-mri prep` makes one with the network | `linc-mri`, the MRI template build |
| match table | text, in the brain's `histology/` folder: section number → photograph number, per component (`linc/histology/match.py`). Every step uses the first of: the path given by flag, `match_table_edited.txt`, `match_table.txt`, `match_table_proposed.txt` | proposed by `linc-histology propose`, then reviewed row by row by a person, who corrects it in a copy named `match_table_edited.txt`; or made by hand. Once per brain and series | `linc-histology`, `linc-labels` |
| regions volume | NIfTI, `histology/regions.nii.gz` on the blockface: 1 left, 2 right, 3 cerebellum and brainstem, 4 tubes | transformed from the template (`linc/histology/README.md`) | `linc-histology` |
| MRI-side trim (the paper's brains) | NIfTI labels on the b0 frame, named by `register` in `configs/completion.yaml` `prepare.inputs`; label 1 is tissue the MRI shows and the photographs lack; made by the detector below and hand trims | none: `linc-complete prepare` detects that tissue for a new brain with MRI (`prepare.trim` in the same file); a brain without MRI has none | `linc-complete prepare` |
| section map | text, `histology/section_map.tsv`: stack index → section number | matched by content, then checked by hand | `linc-labels` |
| tracer annotations | OME-Zarr, one stack per class aligned with the section stack, `histology/annotations/<class>.ome.zarr` (the dataset's `rawdata/sub-<ID>/micr/masks/`) | drawn on the sections by an anatomist | `linc-labels` |

## Downloads

| Input | Location | Read by |
|---|---|---|
| the template | `$LINC_TEMPLATE` (the data release) | `linc-register`, `linc-labels` |
| network weights | `$LINC_MODELS/segment/<model>/` and `$LINC_MODELS/completion/<name>/` (the data release; `docs/models.md`) | `linc-mri`, `linc-reconstruct`, `linc-segment`, `linc-complete` |
| anatomix's feature network | `env/install.sh` fetches it once from Hugging Face, at a pinned revision, and checks it | `linc-register`, the registration between the two templates |
| NMT v2 (symmetric), with its SARM and CHARM labels | your own copy, laid out as `configs/targets.yaml` says, under `configs/link.yaml` `inputs.atlas_reference` | the MRI template's frame and landmarks, the registrations to published templates, the comparison of targets, the label tables |
| CIVM-R, INIA19 | your own copies, the same way (CIVM-R's license does not allow redistribution) | the registrations to published templates, the comparison of targets |
| the blockface template in its own space | `$LINC_PRODUCTS/bftemplate/d2b/final_s8/` (the data release), with the frame and bridge of `$LINC_TEMPLATE` | `linc-register blockface-template` |
| the paper's products | `$LINC_PRODUCTS` (the data release) | `make numbers`, `make figures`, the template builds |

A new brain's settings go in `configs/subjects/<ID>.yaml`, copied from `configs/subjects/TEMPLATE.yaml`. `tools/subjects_overview.py` prints all the brains' settings as a table, one row per brain, and with `--data` which hand-made files sit in each brain's folder; nothing reads that table, so changes go in the yaml files.
