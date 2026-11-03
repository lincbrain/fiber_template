# Synthesis networks: model cards

One 3D U-Net architecture predicts the blockface tissue that a brain's photographs lack, from its b0 image (paper: Methods,
"Blockface to MRI registration and synthesis of missing sections"; the network: Supplementary Section S1). The paper's Data and Code
Availability promises their weights.

**The weights are not in git.**
- Put each network's folder (`last.pt`, `config.json`, `train_log.jsonl`) under `$LINC_MODELS/completion/<name>/`.
- Check them with `$PY models/verify_weights.py models/completion_public.sha256 --root $LINC_MODELS/completion`.
- Download location: with the data release.
- **The released files are copies without machine paths or host names.** The weights are
  those of record, tensor for tensor; only the configuration's folder paths (re-rooted to `$LINC_PRODUCTS`), its quoted design
  notes (`author_*`, dropped) and the training log's host field changed. `models/completion_checksum_map.tsv` maps each file of record (`models/completion.sha256`) to its copy.
  Checked: `cli/linc-complete` from the copies is bitwise identical to the volumes of record (MF278, own b0; MR248,
  stand-in b0).

## The architecture (all four networks)

- **Input.** One channel: the b0 image at 0.5 mm, normalized to [0, 1] inside the brain mask and 0 outside it.
- **Output.** Blockface grey at 0.25 mm, on [0, 1], on the input box's center-aligned 2x refinement.
- **Model** (`linc/completion/network.py`).
  - MONAI BasicUNet at 0.5 mm (features 32/32/64/128/256/32, LeakyReLU 0.1, instance norm), 32 output features.
  - A x2 transposed convolution to 0.25 mm, two 3x3x3 convolution blocks, a 1x1x1 convolution and a sigmoid.
  - 5,810,945 parameters.
- **Training recipe** (`linc/completion/train.py`, `configs/completion.yaml`).
  - Random paired patches: 64³ input voxels (32 mm) → 128³ output voxels; 500 patches per epoch, batch 2, 80 epochs.
  - One random physical-space transform per patch, applied to input and target alike; intensity augmentation of the input.
  - Loss: l1 + 0.2 gradient-l1 + 0.6 (1 − multi-scale SSIM), on photographed tissue only.
  - Adam 2e-4 (5-epoch warm-up, cosine decay); the released weights are the exponential moving average (0.999).
  - 13 brains with MRI: MF5, MF6, MF7 and MM1–MM10. MF6 is not one of the 12 evaluation brains.
- **Use.** `cli/linc-complete` (prediction, intensity match, 2 mm cross-fade at the clean cut face).
- **File.** `last.pt` = {"G": the weights, "cfg": the training configuration, "epoch": 79}. Load it with
  `linc.completion.network.load`. `cfg["root"]` records the training folder (in the released copies, under `$LINC_PRODUCTS`);
  nothing reads it.

## 1. Synthesis from the brain's own b0: `d3_full_A_s0`

- **Made.** The synthesized tissue of the 13 brains with MRI.
- **Input.** The brain's own b0, normalized by its 0.5th and 99.5th percentiles inside the brain.
- **Training.** Protocol `full`: all photographed tissue rostral of each brain's clean cut face. Seed 0. 55.5 min on one GPU.

## 2. Synthesis from a stand-in b0: `d3p_full_A_s0`

- **Made.** The synthesized tissue of the 8 brains without MRI (MF8–MF10, MM11–MM15).
- **Input.** The stand-in b0: the MRI template's b0 transformed into the brain by its cross-modal alignment.
  - Normalized as x = clip(b0 / p99.5, 0, 1) inside the brain.
  - Trained on the stand-in b0s of the same 13 brains, whose 0.5th percentile was 0, so this is the same scaling.
- **Training.** As network 1, with the stand-in b0 as input. 55.7 min on one GPU.

## 3. Leave-slab-out, own b0: `d3_lso_A_s0`

- **Purpose.** The withheld-slab evaluation (the paper's accuracy of the synthesis, measured on withheld tissue).
- **Training.** As network 1, but in each of the 12 evaluation brains a slab of photographed tissue was withheld from training.
  - The slab lies just rostral of the clean cut face.
  - Its photographed volume equals the brain's missing volume: 6.0 mm median, range 5.0–9.0 mm.
- **Accuracy.**
  - On the withheld slabs, joined as in the volumes of record: mean SSIM 0.684 over 12 brains (`d3_domain_shift`,
    `summary.ssim.mean_own`).
  - From a stand-in b0 instead of the own b0, the same network scores 0.559 (difference −0.125, 95 % CI −0.165 to −0.083).
- **Reproduce.** `reproduce/numbers/d3_domain_shift.py`.

## 4. Leave-slab-out, stand-in b0: `d3p_lso_A_s0`

- **Purpose.** The stand-in counterpart of network 3. No number in the paper comes from it.
- **Included because** `d3_methods` records its configuration.

## Provenance

- The numbers above are from the records `d3_methods`, `d3_domain_shift` and `d3_predicted_share` (`reproduce/reference/`).
  - Their scripts (`reproduce/numbers/d3_*.py`) regenerate them identically from these files and the products of record.
- Training ran on 3–4 Oct 2026 on an NVIDIA RTX 6000 Ada GPU in the environment `env/completion.yml`, now `env/deep.yml` with the same pins (torch 2.6.0, CUDA 12.4,
  MONAI 1.4.0).
- Checksums: `models/completion.sha256` (sha256 of the files as used for the paper, 21 Oct 2026).
  - The md5 of each `last.pt` is also in `d3_methods` (`models.<group>.checkpoint_md5`).
