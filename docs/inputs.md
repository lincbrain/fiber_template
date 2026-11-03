# Inputs

Everything the code assumes it is given, in three kinds: data, made by an instrument (the MRI given preprocessed: denoised,
eddy-corrected and, for SANDI, fitted); hand-made inputs, made once per brain by a person; and downloads. Everything else is made by the code. Paths are relative to a brain's folder, `$LINC_DATA/sub-<ID>/`, unless
they start with another root (`docs/data_layout.md`).

## Data

| Input | Format and location | Read by |
|---|---|---|
| blockface photographs | JPEG, one per section, RGB, numbered in cutting order; one folder per brain under `$LINC_RAW`, named by `raw` in `configs/subjects/<ID>.yaml`, the numbers read by its `raw_regex` | `linc-reconstruct` |
| diffusion MRI, preprocessed | NIfTI: the b0, `mri/dwi/b0.nii.gz` (0.25 mm), and the denoised, eddy-corrected series with its log, gradient table and fit mask, `mri/dwi/raw/` | `linc-mri`, the MRI template build |
| T2-weighted MRI, preprocessed | NIfTI, `mri/t2/`: the denoised 32-echo multi-echo series (0.5 mm), whose fifth echo is the T2-weighted image | the MRI template build |
| SANDI maps | NIfTI, `mri/micro/frac1_brain.nii.gz` and `frac2_brain.nii.gz` (soma and neurite fractions), from a second diffusion session (0.5 mm) | the MRI template build |
| NMT v2 landmarks | NIfTI and ITK affine, `mri/landmarks/`: NMT v2's SARM labels on the b0 and the b0 → NMT v2 affine, from a registration of NMT v2 to the b0 | the MRI template build |
| histology sections | JPEG 2000, one per section (`sourcedata/sub-<ID>/micr/<section>.jp2`), and their OME-Zarr stacks (`rawdata/sub-<ID>/micr/`), from LINC dataset 000003 | `linc-histology`, `linc-labels` |

The MRI was acquired ex vivo on a 4.7 T Bruker scanner and processed by the lab's pipelines: conversion, MP-PCA denoising (MRtrix3),
FSL `eddy_correct`, TORTOISE 4.1.0 for the second diffusion session, and the SANDI fit (Microstructure.jl). A new brain that is only
registered to the template needs one b0 or T2-weighted image (`linc-mri prep`).

## Hand-made inputs

| Input | Format and location | How a new brain gets it | Read by |
|---|---|---|---|
| pixel size and section step | `vx`, `vz` (µm) in `configs/subjects/<ID>.yaml` | a ruler in a photograph; the microtome's step | `linc-reconstruct` |
| camera jumps | `jumps` in the same file: [last section before, first after], 0-based | by eye, on the stacked photographs of `linc-reconstruct stack` | `linc-reconstruct` |
| missing or damaged sections, leading frames, caudal cutoff | `missing` in the same file | by eye, the same way | `linc-reconstruct` |
| corrected photograph masks (optional) | PNG, `blockface/01_brainmask/01_final/` | `linc-reconstruct edit-photo-masks export`, edit, `import` | `linc-reconstruct` |
| corrected 3D brain mask and marker tubes (optional) | NIfTI, `blockface/curated/<ID>_mask_edited.nii.gz`, `<ID>_holes_edited.nii.gz` | edit the network's `<ID>_mask.nii.gz`, `<ID>_holes.nii.gz` | `linc-reconstruct` |
| MRI brain masks | NIfTI, `mri/masks/`: one per session (`dwi_brain_mask`, `t2_brain_mask`, `micro_brain_mask`), drawn on its grid, and the final edited b0 mask of each MRI template brain (`final_brain_mask`) | optional: `linc-mri prep` makes one with the network | `linc-mri`, the MRI template build |
| match table | text, `histology/match_table.txt`: section number → photograph number, per component (`linc/histology/match.py`) | by hand, once per brain and series | `linc-histology` |
| regions volume | NIfTI, `histology/regions.nii.gz` on the blockface: 1 left, 2 right, 3 cerebellum and brainstem, 4 tubes | transformed from the template (`linc/histology/README.md`) | `linc-histology` |
| MRI-side trim (optional) | NIfTI labels on the b0 frame, given with `--register`; label 1 is tissue the MRI shows and the photographs lack; the paper's 21 brains have one each | none is needed, or one drawn by hand | `linc-complete prepare` |
| section map | text, `histology/section_map.tsv`: stack index → section number | matched by content, then checked by hand | `linc-labels` |
| tracer annotations | OME-Zarr, one stack per class aligned with the section stack, `histology/annotations/<class>.ome.zarr` (the dataset's `rawdata/sub-<ID>/micr/masks/`) | drawn on the sections by an anatomist | `linc-labels` |

## Downloads

| Input | Location | Read by |
|---|---|---|
| the template | `$LINC_TEMPLATE` (the data release) | `linc-register`, `linc-labels` |
| network weights | `$LINC_MODELS/segment/<model>/` and `$LINC_MODELS/completion/<name>/` (the data release; `docs/models.md`) | `linc-mri`, `linc-reconstruct`, `linc-segment`, `linc-complete` |
| anatomix's feature network | fetched from Hugging Face on the first registration, then cached | `linc-register`, the link |
| NMT v2 (symmetric), with its SARM and CHARM labels | your own copy, laid out as `configs/targets.yaml` says, under `configs/link.yaml` `inputs.atlas_reference` | the MRI template's frame and landmarks, the mappings, the comparison of targets, the label tables |
| CIVM-R, INIA19 | your own copies, the same way (CIVM-R's license does not allow redistribution) | the mappings, the comparison of targets |
| the paper's products | `$LINC_PRODUCTS` (the data release) | `make numbers`, `make figures`, the template builds |

A new brain's settings go in `configs/subjects/<ID>.yaml`, copied from `configs/subjects/TEMPLATE.yaml`.
