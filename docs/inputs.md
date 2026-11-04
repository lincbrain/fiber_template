# Inputs

Everything the code assumes it is given, in three kinds: data, made by an instrument; hand-made inputs, made once per brain by a
person; and downloads. Everything else is made by the code. Paths are relative to a brain's folder, `$LINC_DATA/sub-<ID>/`, unless
they start with another root (`docs/data_layout.md`).

## Data

| Input | Format and location | Read by |
|---|---|---|
| blockface photographs | JPEG, one per section, RGB, numbered in cutting order; one folder per brain under `$LINC_RAW`, named by `raw` in `configs/subjects/<ID>.yaml`, the numbers read by its `raw_regex` | `linc-reconstruct` |
| MRI sessions | Bruker ParaVision session folders (numbered scans, each with its `method`, `acqp` and `pdata/<n>/2dseq`): the diffusion scans, the 32-echo multi-echo scan and a second diffusion session for SANDI, under `$LINC_RAW`, as `configs/mri_construction.yaml` `raw` names them | the MRI template build |
| a b0 or T2-weighted image | NIfTI, for a brain that is only registered to the template | `linc-mri prep` |
| histology sections | JPEG 2000, one per section (`sourcedata/sub-<ID>/micr/<section>.jp2`), and their OME-Zarr stacks (`rawdata/sub-<ID>/micr/`), from LINC dataset 000003 | `linc-histology`, `linc-labels` |

The MRI was acquired ex vivo on a 4.7 T Bruker scanner. The MRI template build makes everything else from the sessions: the
preprocessed diffusion, T2-weighted and SANDI images, the registration of NMT v2 to each brain, and the registrations between each
brain's sessions. A new brain that is only registered
to the template needs one b0 or T2-weighted image (`linc-mri prep`).

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
| anatomix's feature network | `env/install.sh` fetches it once from Hugging Face, at a pinned revision, and checks it | `linc-register`, the registration between the two templates |
| NMT v2 (symmetric), with its SARM and CHARM labels | your own copy, laid out as `configs/targets.yaml` says, under `configs/link.yaml` `inputs.atlas_reference` | the MRI template's frame and landmarks, the registrations to published templates, the comparison of targets, the label tables |
| CIVM-R, INIA19 | your own copies, the same way (CIVM-R's license does not allow redistribution) | the registrations to published templates, the comparison of targets |
| the paper's products | `$LINC_PRODUCTS` (the data release) | `make numbers`, `make figures`, the template builds |

A new brain's settings go in `configs/subjects/<ID>.yaml`, copied from `configs/subjects/TEMPLATE.yaml`.
