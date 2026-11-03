# Blockface segmentation networks: model cards

Three nnU-Net v2 networks (nnU-Net 2.5.1) segment the blockface photographs and the stacked volumes in the reconstruction
(paper: Methods, "Blockface reconstruction", steps 1, 4 and 6). The paper's Data and Code Availability promises their weights.

**The weights are not in git.**
- Put the three nnU-Net results folders under `$LINC_MODELS/nnUNet_results/`.
- Check them with `$PY models/verify_weights.py models/blockface_segmentation.sha256`.
- Download location: with the data release.

In the pipeline, every automatic mask is only a starting point: a hand-corrected mask, when present, is used instead
(`cli/linc-reconstruct`).

## 1. Brain mask in each photograph (2D): `Dataset500_SLICE`

- **Task.** Brain vs background in each blockface photograph.
- **Input.** The photograph as a PNG, three channels (R, G, B). Labels: background 0, brain 1.
- **Model.**
  - nnU-Net 2D, trainer `nnUNetTrainer`, plans `nnUNetPlans`, patch 1024 × 1536 pixels.
  - Five folds were trained; the paper's masks used **fold 0 alone** (`nnUNetv2_predict -c 2d -f 0`, configs/blockface.yaml
    `photo_segmentation.folds`). On MM13 its masks agree with the ensemble of folds 0–3 to Dice 0.998, at about a quarter of the time.
  - Then nnU-Net's post-processing runs (`crossval_results_folds_0_1_2_3_4/postprocessing.pkl`).
- **Training data.** 149 photographs. It was trained on 26 Dec 2024, before this work; the brains they came from are not recorded.
  It was used unchanged (paper: Supplementary S1.1).
- **Accuracy.**
  - Dice against the hand-curated masks of 14 brains: mean of the per-brain medians 0.974 (SD 0.015), lowest brain 0.923.
  - Record: `s1_segmentation_r057.json` `mask2d` (`reproduce/numbers/s1_segmentation_r057.py`, which holds the model-card record
    `bf_models_card.json` `mask2d` of the analysis project).
  - These scores are **not cross-validated**: the curated masks were corrected from this network's own output.
- **Files.** `fold_{0..4}/checkpoint_final.pth` (453 MB each; the pipeline loads fold 0 only), `plans.json`, `dataset.json`,
  `dataset_fingerprint.json`, and `crossval_results_folds_0_1_2_3_4/{postprocessing.pkl, plans.json, dataset.json}` (the
  post-processing chosen by the five-fold cross-validation).

## 2. Brain mask of the aligned stack (3D): `Dataset506_BFmask5`

- **Task.** Brain vs background in the aligned color stack with its background.
- **Input.** The RGB stack on the uncropped L4 grid, 0.32 mm in-plane and 0.40 mm between sections, as three channels.
  Labels: background, brain.
- **Model.**
  - nnU-Net `3d_fullres`, plans `nnUNetResEncUNetMPlans` (residual-encoder U-Net, size M), trainer `nnUNetTrainer_250epochs`.
  - Fold 0, `checkpoint_final.pth`, patch 128³.
- **Post-processing.** The largest 3D connected component, holes filled (`linc/blockface/segment3d.py`).
- **Training data.** 13 volumes: 12 brains and one held-out slab of one brain.
- **Human in the loop, 5 rounds.** Each round's network segmented every brain; the predictions for the brains added next were
  corrected by hand and joined the training set. The training set grew from 1 to 2, 4, 8 and 12 brains. Rounds 1-4 are documented, not
  recomputable: their training folders were deleted after round 5 (sources in the record).
- **Accuracy.** Dice 0.991 on its single validation volume. One volume only, so this is not a cross-validated estimate.
- **Files.** `fold_0/checkpoint_final.pth` (782 MB), `plans.json`, `dataset.json`, `dataset_fingerprint.json`.

## 3. Marker tubes (3D): `Dataset508_holes3`

- **Task.** The drilled marker tubes in the masked color brain.
- **Input.** The masked RGB brain at 0.32 × 0.32 × 0.40 mm, three channels. Labels: background, hole.
- **Model.** The same configuration as Dataset506 (ResEnc-M `3d_fullres`, `nnUNetTrainer_250epochs`, fold 0).
- **Post-processing.** Tubes are kept only in the left hemisphere, where they were drilled; small specks are dropped.
- **Training data.** 16 volumes (16 brains with hand-painted tubes).
- **Human in the loop, 3 rounds,** as for the brain mask: 1, 4 and 16 brains (round 1 documented, not recomputable).
- **Accuracy.** Mean Dice 0.71 over the 4 volumes of its validation fold (0.67–0.76). One fold only.
- **Files.** `fold_0/checkpoint_final.pth` (782 MB), `plans.json`, `dataset.json`, `dataset_fingerprint.json`.

## Provenance

- The numbers above are the paper's record `s1_segmentation_r057.json` (Supplementary S1.1; `reproduce/numbers/s1_segmentation_r057.py`
  recomputes it: training sets and rounds from the networks' files, accuracy taken from the analysis project's model-card record
  `bf_models_card.json`, written by `code/bf_models_card.py`) and each network's own `dataset.json` and `plans.json`.
- Checksums: `models/blockface_segmentation.sha256` (sha256 of the files as used for the paper, 21 Oct 2026).
