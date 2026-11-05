# Rebuilding the MRI channels of the template

The MRI channels (`$LINC_TEMPLATE/mri/linc_mri_*.nii.gz`: T2, b0, frac1, frac2, tensor and its FA/MD/MO/L1-3/V1-3/DEC, brain and
core masks) come from a symmetric groupwise construction of four brains (MF1–MF4 = LINC MF283–MF286) and their exact mirrors with
MMORF 1.0.1 (paper: Methods, "MRI preprocessing", "Symmetric groupwise construction"). Every parameter is in
`configs/mri_construction.yaml`; the run of record is `rebuild_run: rb_final`.

Five steps: **inputs → per-brain preparation → run assembly → MMORF through the patched driver → channels.** Steps 2, 3 and 5 run on
CPU; step 4 needs one GPU (48 GB for the batch of record). The commands below are also one driver, `reproduce/templates/
build_mri_channels.py` (`--plan` to list every step, `--step` / `--from --to` to run them; last section).

## Environment

- Python environment `env/linc.yml` (`PY_MRI` in `reproduce/Makefile` names its Python): the environment the MRI channels, their numbers
  and figures were made in (MMORF 1.0.1 from the FSL conda channel), with this repository installed (`pip install -e .`).
- External programs, found from the environment and version-checked by `linc/construction/tools.py`
  (`tools:` in the config): `FSLDIR` = FSL 6.0.7.4; `ANTSPATH` = ANTs 2.5.4; `MRTRIX` = MRtrix3 3.0.2 (the tensor only);
  MMORF 1.0.1 for step 4.
- `LINC_DATA` (per-brain data), `LINC_TEMPLATE` (output), `LINC_WORK` (runs), as in `linc/paths.py`.

## 1. Inputs

Per brain, under `$LINC_DATA/sub-<ID>/` (file names: `prep.layout` in the config, relative to `mri/`):

| Input | Where it comes from |
|---|---|
| diffusion series (bias-corrected; volume 0 = b0), the eddy-corrected series + `.ecclog`, b-table, fit mask | the preprocessed files, `$LINC_MRI_PREPROCESSED/<ID>/dwi/` (`docs/mri.md`) |
| 32-echo MSME (denoised) | the preprocessed files, `msme/` |
| SANDI fit `est.fracs.mean` / `est.fracs.std` + the micro session's diffusion series | the preprocessed files, `dt2/tortoise/` |
| brain masks: dwi, T2, micro (one per session; from the masks' network loop, edited by hand at the end) | released with the data |
| the **final** hand-edited brain mask on the b0 grid (values exactly {0, 1}) | released with the data (`<ID>_mask_EDIT`) |
| `xfm/t2-native__to__b0-native/`, `xfm/micro-native__to__b0-native/` (one ITK displacement field each; the initial affine is inside it) | `linc.construction.mri_register t2` / `micro` (ANTs affine, then anatomix + FireANTs) |
| the brain → NMT_v2.0_sym ITK affine and the SARM level-6 labels transformed to the b0 grid | `linc.construction.mri_register nmt` (NMT registered to the b0) |

## 2. Per-brain preparation (`linc/construction/mri_prep.py`, `linc/construction/canonical.py`)

For each of MF283, MF284, MF285, MF286 (`$ID`):

```bash
# staging, from the preprocessed files (the brain masks in place under masks/): images under their brain mask's header; N4 b0
# and T2; echo 5 of the MSME; SANDI fractions renormalized; the tensor's files under dwi/raw/
python -m linc.construction.mri_prep stage --subject $ID --preprocessed $LINC_MRI_PREPROCESSED/$ID
# the tensor (config prep.tensor_bmax): dtifit on the eddy-corrected b <= 4800 volumes, rotated + gradient-checked b-vectors
python -m linc.construction.mri_prep tensor --eddy-series <dwi_eddycorrect.nii.gz> --ecclog <dwi_eddycorrect.ecclog> \
    --bvecs <dwi_orig.bvecs> --bvals <dwi_orig.bvals> --fit-mask <fit brain_mask> \
    --dwi-mask $LINC_DATA/sub-$ID/mri/masks/dwi_brain_mask.nii.gz --out $LINC_WORK/tensor_d9/$ID --threads 8
# canonical frame (config `canonical`): AC/PC from the transformed SARM labels, then the rigid map into the NMT v2 frame
M=$LINC_DATA/sub-$ID/mri
python -m linc.construction.canonical landmarks --sarm $M/landmarks/SARM6_in_subject.nii.gz \
    --b0 $M/dwi/b0_brain_n4.nii.gz --mask $M/masks/dwi_brain_mask.nii.gz --out $M/landmarks/landmarks.json --subject $ID
python -m linc.construction.canonical frames --b0 $M/dwi/b0_brain_n4.nii.gz --mask $M/masks/dwi_brain_mask.nii.gz \
    --landmarks $M/landmarks/landmarks.json --nmt-affine $M/landmarks/nmt_affine_0GenericAffine.mat \
    --out $M/landmarks/canonical.json --subject $ID
# construction inputs on the canonical grid, every channel cut to the final mask, + exact mirrors
python -m linc.construction.mri_prep prepare --subject $ID --final-mask <$ID final mask> \
    --tensor $LINC_WORK/tensor_d9/$ID/tensor_d9_dwigrid.nii.gz          # -> $M/construction_inputs/
```

## 3. Run assembly (`linc/construction/mmorf_run.py`)

```bash
git clone https://git.fmrib.ox.ac.uk/cart/mm-template-construction.git $LINC_WORK/mm-template-construction
git -C $LINC_WORK/mm-template-construction checkout f4907f4
python $REPO/linc/construction/patches/apply_edits.py \
    $REPO/linc/construction/patches/mm-template-construction_v1_linc.edits.json $LINC_WORK/mm-template-construction
python -m linc.construction.mmorf_run --upstream $LINC_WORK/mm-template-construction --out $LINC_WORK/runs/rb_final \
    --driver $LINC_WORK/mm-template-construction/template_construction_v1/run_template_construction.py
```

This writes the run folder (inputs, `subject_ids.csv`, `standardspace.nii.gz` = mean of the 8 T2 inputs, `config/` with one MMORF
`.ini` per step, `run_spec.json`) from the config's recipe: `n_steps`, `iterations_per_step`, `warp_res_init_mm`, `warp_scaling`,
`lambda_reg` × `lambda_scale`, `fwhm_mm`, `hires_mm`, `optimiser_max_it_*`, `bias_res_init_mm`, `lambda_bias_reg`,
`scalar_modalities` / `tensor_modalities` (weights, quantitative channels), `cost_mask_*`, `run_assembly`. It prints the driver
command.

## 4. MMORF through the patched driver

Run the printed command in the driver's environment (`linc/construction/patches/README.md`: Python 3.11, `fsl-sub` 2.11 shell
plugin, `fslpy`, `file-tree`; MMORF 1.0.1 first on `PATH`, then FSL 6.0.7.4):

```bash
python run_template_construction.py --input_scalar <run>/data/T2M/ <run>/data/FRAC1M/ <run>/data/FRAC2M/ \
    --quant_scalar_mods 2 3 --input_tensor <run>/data/DTIV/ --config <run>/config/ --tree <run>/config/data_structure.tree \
    --standardspace <run>/standardspace.nii.gz --subids <run>/subject_ids.csv --dof 6 -o <run>/out/ --cpuq cpu.q --gpuq gpu.q \
    -aff True -nln True -nres 8 -nit 3 -b 8 8 8 8 8 4 4 3
```

`-b` (concurrent MMORF tasks per step, `run_assembly.batch_of_record`) only bounds GPU memory; it does not change results. After an
interruption, add `-aff False -resume <step>:<iteration>` (and `LINC_SKIP_DONE_MMORF=1` to keep that iteration's finished tasks).
The channels use the construction warps of step 8, iteration 3 (`channels.step`, `channels.iteration`).

## 5. Channels (`linc/construction/channels.py`)

```bash
python -m linc.construction.channels --run $LINC_WORK/runs/rb_final --out $LINC_TEMPLATE/mri -j 8
```

Every input is read once from its native grid through one composed field (construction warp, mirror, canonical rigid, within-MRI
warp); T2 and b0 normalized and averaged, fractions averaged and renormalized, the tensor a log-Euclidean mean, everything exactly
symmetric (config `channels`). Acceptance (T2 r against the driver's template, V1 angle, FA range, symmetry) is written to
`$LINC_TEMPLATE/mri/provenance.json`.

## How long it takes

From the logs of the run of record (`rb_final`):

| Stage | Wall time | Hardware |
|---|---|---|
| the tensor, per brain | 8 min (481 s for MF283, 8 threads) | CPU |
| construction inputs (`prepare`), per brain | 2.5 min (145 s for MF283, re-measured 21 Oct 2026) | CPU |
| run assembly | 17 s | CPU |
| MMORF construction, 30 Sep 14:39:51 → 4 Oct 02:59:43 | **3 d 12 h 20 min** | one RTX 6000 Ada (48 GB) |
| channels | 10.5 min (8 threads) | CPU |

MMORF time per stage: affine 7 min; nonlinear steps 1–5 1.2, 1.5, 1.7, 1.8, 2.3 h; step 6 13.6 h; step 7 21.6 h; step 8 40.7 h
(three iterations each; batch 8 for steps 1–5, 4 for steps 6–7, 4 then 3 for step 8; the batch was changed by resuming at 7:1 and
8:2). The card was not shared with another build.

## Validation of the release code (21 Oct 2026; CPU, no MMORF rerun)

| Check | Result |
|---|---|
| run assembly vs `rb_final/config/*`, `subject_ids.csv`, `run_spec.json`, the 64 linked inputs | identical (every `.ini` and `.tree` byte-identical; zero warp identical uncompressed) |
| `standardspace.nii.gz` recomputed from the inputs | exact (max abs diff 0, same header) |
| canonical frame of MF283 recomputed (symmetry-plane fit included) vs `canonical.json` | exact (max abs diff 0 in T) |
| construction inputs of MF283 recomputed (`prepare`) vs the inputs of record | exact (all 37 images, max abs diff 0, same headers) |
| T2 channel recomputed from `rb_final`'s step-8 iteration-3 warps vs `linc_mri_T2.nii.gz` (16 composed fields, 24 warped inputs, then the channel) | exact (max abs diff 0, 0 voxels differ, same header; intensity scales identical) |

## The build driver (3 Nov 2026)

`reproduce/templates/build_mri_channels.py` runs the recipe above as one step per command (`--plan` lists every step with its
environment, inputs, outputs and estimated time; `--step NAME ...` or `--from A --to B` run steps; a step whose outputs exist is not
rerun without `--force`; each step that ran leaves `<work>/provenance/<NN>_<step>.json`). Steps: `stage-<ID>`, `nmt-<ID>`, `t2reg-<ID>`,
`microreg-<ID>`, `tensor-<ID>`, `landmarks-<ID>`, `frames-<ID>`, `prepare-<ID>` for each of the four brains, then `assemble`, `mmorf` (the patched driver on a GPU), `channels` and
`links-<ID>`, which registers the brain's links up to the template (`b0-native` -> `mri-canonical`, `mri-canonical` -> `mri-template`). The
per-brain inputs are the `prep.layout` names of `configs/mri_construction.yaml` under `$LINC_DATA/sub-<ID>/mri/` (the final mask
`masks/final_brain_mask.nii.gz`, the tensor's raw diffusion files under `dwi/raw/`, the SARM labels and NMT affine under `landmarks/`)
and the within-MRI registrations under `xfm/`; everything the driver makes goes under `--work` (default `$LINC_WORK/build_mri_channels`).

Validation of the driver (3 and 4 Nov 2026; MF283 unless said; the same products of record as the table above; "clean clone" = a fresh clone and install, an empty HOME, the preprocessed files as its only MRI inputs besides the brain masks):

| Step | Result |
|---|---|
| `stage-MF283` (4 Nov, clean clone) | from the preprocessed files: all 17 staged images (diffusion b0, T2, microstructure b0, fractions and their SDs) voxel-identical to the paper's, N4 included; 200 s, 7.2 GB of RAM, 6 GB of disk |
| `nmt-MF283`, `t2reg-MF283` (4 Nov, clean clone) | on a GPU, so no identical result to expect, and the paper's NMT registration ran on an earlier version of the engine. SARM labels agree on 93.3 % of the brain's voxels; the T2 field differs from the paper's by 0.019 mm (median), 0.19 mm (99th percentile), 0.52 mm (largest); 460 s + 352 s; 15.3 and 15.6 GB of GPU memory |
| `microreg-MF283` (4 Nov, a copy of the clean clone, on a second card of the same model) | the field differs from the paper's by 0.033 mm (median), 0.42 mm (99th percentile), 1.9 mm (largest); 379 s; 20.1 GiB of GPU memory, 22.1 GB in `nvidia-smi` |
| `tensor-MF283` (4 Nov, clean clone) | from `stage-MF283`'s files: voxel-identical to the tensor of record, same header, the same axis order and signs from `dwigradcheck`; 286 s (481 s of record) |
| `landmarks-MF283`, `frames-MF283` | AC/PC equal to the record's `landmarks.json`; `T_subject_to_canonical` identical (max abs diff 0; the symmetry-plane fit converges to the same plane, mirror r 0.5687); 6 s + 59 s. On the clean clone's `nmt-MF283`: the frame differs from the paper's by 0.04 deg and 0.16 mm; 6 s + 37 s |
| `prepare-MF283` (with the frame and the tensor of record) | all 36 construction inputs voxel-identical to the inputs of record, same headers; 507 s under full machine load (145 s of record). On the clean clone's frame, tensor, `t2reg-MF283` and `microreg-MF283`: every image is resampled through that slightly different frame, so all differ; inside both masks T2 r 0.968, b0 r 0.957, fractions r 0.923 and 0.909, tensor r 0.995, mask Dice 0.989; 62 s |
| `nmt`, `t2reg`, `landmarks`, `frames`, `prepare` on MF284, MF285, MF286 (4 Nov, clean clone; `prepare` with the paper's `microreg`, which was not rerun on these three) | SARM labels agree on 94.3, 94.3 and 94.1 % of the brain's voxels; the T2 field differs from the paper's by 0.018, 0.037 and 0.009 mm (median), 0.49, 0.84 and 0.42 mm (largest); 205 to 413 s per GPU step, 11.4 to 14.5 GiB of GPU memory. The frame differs from the paper's by 0.20, 0.15 and 0.81 deg, and 0.29, 0.17 and 0.38 mm, so all of `prepare`'s images differ: inside both masks T2 r 0.937, 0.975 and 0.885, b0 r 0.894, 0.975 and 0.879, fractions r 0.916, 0.961 and 0.871, and 0.905, 0.945 and 0.845, tensor r 0.989, 0.996 and 0.981, mask Dice 0.986, 0.993 and 0.979; 72 to 76 s |
| The frame's effect on these comparisons (MF283 to MF286) | with the difference between the frames undone, T2 r is 0.991 to 0.993, b0 r 0.987 to 0.993, the fractions' r 0.942 to 0.974. Rerunning `nmt` and `frames` can land on a different mid-sagittal plane. The symmetry-plane fit returns the same plane from the same start, and the NMT affine moves its start by 0.2 to 0.3 deg on all four brains. On MF286 that start reaches a different optimum, 0.8 deg from the paper's plane (a mean move of 0.31 mm), and the paper's plane scores higher, mirror r 0.635 against 0.628; on MF283 the two planes are 0.04 deg apart. The effect on a finished template was not tested |
| `assemble` (the four brains' inputs of record) | the 14 config files byte-identical to `rb_final/config`, `subject_ids.csv` identical, `standardspace.nii.gz` identical, the 64 input links in place; `run_spec.json` differs only in the label `canonical_dir` (the prep folder pattern) |
| `channels` (on the run of record `rb_final`, the four brains' inputs of record) | all 20 channel files voxel-identical to `$LINC_TEMPLATE/mri` (same headers and dtypes); acceptance T2 r 0.9946, V1 median 0.46 deg, exact symmetry; 840 s under full machine load (10.5 min of record) |
| `mmorf` (4 Nov) | the affine stage and step 1, on the paper's four brains' inputs: byte-identical to the run of record (202 of 202 files that can be compared); steps 2 to 8 not rerun (3 d 12 h of GPU) |
| `links` (4 Nov; the paper's run; `mri/xfm_rb_final/` of the records) | `t2-native` -> `mri-template` on MF283's `echo_n4`: r 0.987 with the driver's resampling of its T2 input, r 0.945 with the T2 channel, inside the brain; 129 s under load. The inverse, `mri-template` -> `t2-native`: r 0.927 with its T2 brain; 81 s |
