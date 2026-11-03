# Rebuilding the MRI channels of the template

The MRI channels (`$LINC_TEMPLATE/mri/linc_mri_*.nii.gz`: T2, b0, frac1, frac2, tensor and its FA/MD/MO/L1-3/V1-3/DEC, brain and
core masks) come from a symmetric groupwise construction of four brains (MF1–MF4 = LINC MF283–MF286) and their exact mirrors with
MMORF 1.0.1 (paper: Methods, "MRI preprocessing", "Symmetric groupwise construction"). Every parameter is in
`configs/mri_construction.yaml`; the run of record is `rebuild_run: rb_final`.

Five steps: **inputs → per-brain preparation → run assembly → MMORF through the patched driver → channels.** Steps 2, 3 and 5 run on
CPU; step 4 needs one GPU (48 GB for the batch of record). The commands below are also one driver, `reproduce/templates/
build_mri_channels.py` (`--plan` to list every step, `--step` / `--from --to` to run them; last section).

## Environment

- Python environment `env/mri.yml` (`PY_MRI` in `reproduce/Makefile`): the environment the MRI channels, their numbers and figures were
  made in (MMORF 1.0.1 from the FSL conda channel), with this repository installed (`pip install -e .`).
- External programs, found from the environment and version-checked by `linc/construction/tools.py`
  (`tools:` in the config): `FSLDIR` = FSL 6.0.7.4; `ANTSPATH` = ANTs 2.5.4; `MRTRIX` = MRtrix3 3.0.2 (D9 tensor only);
  MMORF 1.0.1 for step 4.
- `LINC_DATA` (per-brain data), `LINC_TEMPLATE` (output), `LINC_WORK` (runs), as in `linc/paths.py`.

## 1. Inputs

Per brain, under `$LINC_DATA/sub-<ID>/` (file names: `prep.layout` in the config, relative to `mri/`):

| Input | Where it comes from |
|---|---|
| diffusion series (bias-corrected; volume 0 = b0), its CS reconstruction's dtifit tensor, the eddy-corrected series + `.ecclog`, b-table, fit mask | the acquisition and its upstream processing |
| 32-echo MSME (denoised) | the T2 session |
| SANDI fit `est.fracs.mean` / `est.fracs.std` + the micro session's diffusion series | the microstructure session |
| hand-drawn masks: dwi, T2, micro (one per session) | released with the data |
| the **final** hand-edited brain mask on the b0 grid (values exactly {0, 1}) | released with the data (`<ID>_mask_EDIT`) |
| `xfm/t2-native__to__b0-native/`, `xfm/micro-native__to__b0-native/` (one ITK displacement field each; the initial affine is inside it) | `linc.registration` (anatomix + FireANTs) |
| the subject → NMT_v2.0_sym ITK affine and the SARM level-6 labels carried onto the b0 grid | `linc.registration` (NMT registered to the b0) |

## 2. Per-brain preparation (`linc/construction/mri_prep.py`, `linc/construction/canonical.py`)

For each of MF283, MF284, MF285, MF286 (`$ID`):

```bash
# staging: raw images under their hand mask's header; N4 b0 and T2; echo 5 of the MSME; SANDI fractions renormalized
python - <<EOF
from linc.construction import mri_prep as P
b = P.Brain("$ID")                       # $LINC_DATA/sub-$ID/mri and .../xfm
P.stage_raw(b, dwi_series, dwi_tensor, msme, micro_series, dwi_mask, t2_mask, micro_mask)
P.stage_dwi(b); P.stage_t2(b); P.stage_micro(b, sandi_fracs_mean, sandi_fracs_std)
EOF
# D9 tensor (config prep.tensor_bmax): dtifit on the eddy-corrected b <= 4800 volumes, rotated + gradient-checked b-vectors
python -m linc.construction.mri_prep tensor --eddy-series <dwi_eddycorrect.nii.gz> --ecclog <dwi_eddycorrect.ecclog> \
    --bvecs <dwi_orig.bvecs> --bvals <dwi_orig.bvals> --fit-mask <fit brain_mask> \
    --dwi-mask $LINC_DATA/sub-$ID/mri/masks/dwi_brain_mask.nii.gz --out $LINC_WORK/tensor_d9/$ID --threads 8
# canonical frame (config `canonical`): AC/PC from the carried SARM labels, then the rigid map into the NMT v2 frame
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
| D9 tensor, per brain | 8 min (481 s for MF283, 8 threads) | CPU |
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
rerun without `--force`; each step that ran leaves `<work>/provenance/<NN>_<step>.json`). Steps: `tensor-<ID>`, `landmarks-<ID>`,
`frames-<ID>`, `prepare-<ID>` for each of the four brains, then `assemble`, `mmorf` (the patched driver on a GPU) and `channels`. The
per-brain inputs are the `prep.layout` names of `configs/mri_construction.yaml` under `$LINC_DATA/sub-<ID>/mri/` (the final mask
`masks/final_brain_mask.nii.gz`, the tensor's raw diffusion files under `dwi/raw/`, the SARM labels and NMT affine under `landmarks/`)
and the within-MRI registrations under `xfm/`; everything the driver makes goes under `--work` (default `$LINC_WORK/build_mri_channels`).

Validation of the driver (3 Nov 2026; CPU; MF283 unless said; the same products of record as the table above):

| Step | Result |
|---|---|
| `landmarks-MF283`, `frames-MF283` | AC/PC equal to the record's `landmarks.json`; `T_subject_to_canonical` identical (max abs diff 0; the symmetry-plane fit converges to the same plane, mirror r 0.5687); 6 s + 59 s |
| `prepare-MF283` (with the frame and the tensor of record) | all 36 construction inputs voxel-identical to the inputs of record, same headers; 507 s under full machine load (145 s of record) |
| `assemble` (the four brains' inputs of record) | the 14 config files byte-identical to `rb_final/config`, `subject_ids.csv` identical, `standardspace.nii.gz` identical, the 64 input links in place; `run_spec.json` differs only in the label `canonical_dir` (the prep folder pattern) |
| `channels` (on the run of record `rb_final`, the four brains' inputs of record) | all 20 channel files voxel-identical to `$LINC_TEMPLATE/mri` (same headers and dtypes); acceptance T2 r 0.9946, V1 median 0.46 deg, exact symmetry; 840 s under full machine load (10.5 min of record) |
| `mmorf` | `--plan` only (3 d 12 h of GPU; not rerun) |
