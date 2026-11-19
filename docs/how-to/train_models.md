# Train the networks again

The paper uses six networks trained in this project: four that segment images and two that synthesize the blockface tissue a brain's photographs lack.
Two more synthesis networks were trained only for the paper's evaluation of the synthesis.
You do not need to train anything to use the repository: the weights are in `models/`, and `settings/roots.sh` puts them together in `$LINC_MODELS` ([settings/models](../../settings/models/README.md)).
Train again to check the recipe, or to train on other data.

anatomix, the network whose image features the registrations compare, is used as its authors released it, and MIND-SSC is a fixed image descriptor; neither is trained here.

## The networks

| Network | What it does | Where it is used | Trained on | Weights |
|---|---|---|---|---|
| `mri_brainmask` | brain mask of an ex vivo b0 or T2-weighted image | the MRI images ([01](../tutorials/01_mri.md)) | 38 b0 and T2-weighted images of 23 brains, plus bias-corrected copies of 9 of the T2-weighted images, and their masks of record | `models/segment/mri_brainmask/` |
| `bf_mask2d` | brain mask of each blockface photograph | the reconstruction ([02, Step 1](../tutorials/02_blockface.md#step-1-reconstruct-the-volume)) | 16,078 photographs of 19 brains, at a quarter of their size, labeled from the brains' 3D masks of record | `models/segment/bf_mask2d/` |
| `bf_mask3d` | brain mask of a blockface volume | the reconstruction | the 21 photographed brains' volumes and their manually edited masks | `models/segment/bf_mask3d/` |
| `bf_holes` | the fiducial markers (the round holes in the block face) in a blockface volume | the reconstruction, which fills them, and the check of the histology registration ([03](../tutorials/03_histology.md)) | the 21 volumes and their manually corrected markers | `models/segment/bf_holes/` |
| `d3_full_A_s0` | the missing tissue, from the brain's own b0 | the synthesis ([02, Step 5](../tutorials/02_blockface.md#step-5-synthesize-the-tissue-that-was-never-photographed)), 13 brains with MRI | the 13 brains with MRI | `models/completion/d3_full_A_s0/` |
| `d3p_full_A_s0` | the same, from a stand-in b0 | the synthesis, 8 brains without MRI | the same 13 brains, each with a stand-in b0 as input | `models/completion/d3p_full_A_s0/` |
| `d3_lso_A_s0`, `d3p_lso_prod_A_s0` | the same two, each trained without a withheld slab of each brain | the paper's evaluation of the synthesis only, on withheld slabs (`d3p_lso_prod_A_s0`: its test of the stand-in b0) | the same 13 brains; `d3p_lso_prod_A_s0` with the same stand-in b0s as `d3p_full_A_s0` | `models/completion/` |

Their training data are in the data folder, or are made from it with the commands below.
Each segmentation network's card, `card.txt`, in the data folder's `models/`, gives its cases, folds, held-out scores and known limits.

## The segmentation networks

The training code is in [code/shared/segment](../../code/shared/segment/README.md) (`train.py`, `net.py`, `losses.py`, `augment.py`, `data.py`), and each network's cases, folds and settings are in its task file, `settings/segment/<network>.json`.
The MRI task reads its images and masks where they are in the data folder.
The three blockface tasks are first rebuilt from each brain's reconstruction and its manually edited masks:

```bash
$PY code/shared/linc-segment stage --task settings/segment/bf_mask3d.json --out $LINC_WORK/segment/cases/bf_mask3d   # bf_mask2d and bf_holes the same way
$PY_DEEP code/shared/linc-segment train --task $LINC_WORK/segment/cases/bf_mask3d/bf_mask3d.json --plan              # the cases, folds and missing files
$PY_DEEP code/shared/linc-segment train --task $LINC_WORK/segment/cases/bf_mask3d/bf_mask3d.json --fold all          # the shipped network
$PY_DEEP code/shared/linc-segment train --task settings/segment/mri_brainmask.json --fold all                         # the MRI task needs no staging
```

`stage` runs in `$PY`, because it reads the color volumes from their OME-Zarr stores; `train` runs in `$PY_DEEP`.
`--fold all` trains on every brain and gives the network the repository ships; `--fold 0` to `4` each hold out a fifth of the brains and score them at the end, which is how the held-out results were measured.
A run writes `model.pt`, `spec.json` and `record.json` (the recipe, every epoch's loss, the held-out scores) to `$LINC_WORK/segment/runs/<network>/fold_<k>/`.

Each run of the paper took about 2 hours on one RTX 6000 Ada card (1.7 to 2.1 hours), using about 10 GB of it: 200 epochs of 250 iterations, the last epoch's weights kept.
There were 24 runs, the four networks times five folds and `all`.

To check a retrained network against the paper's:

- `--fold all`: compare `sha256sum $LINC_WORK/segment/runs/<network>/fold_all/model.pt` with the network's line in `settings/models/segmentation.sha256`.
  On the same GPU model, a rerun of `mri_brainmask` gave the shipped file bit for bit; the other three have not been retrained this way, and other GPU models have not been tried.
- `--fold 0` to `4`: compare the held-out scores in each `record.json` with the network's card and the paper's record, `tests/reference/segmodels_<network>_2026-11.json`.

### The earlier networks behind the paper's masks

The brain masks and markers of the paper's 21 photographed brains were made before these four networks existed, by earlier networks and manual correction, as the paper's Supplementary Material describes.
For a new brain, the four networks above take their place.
The earlier ones were trained with nnU-Net and its own commands (version 2.5.1 for the two 3D networks), not with the code here, and only on fold 0 of nnU-Net's split:

- the photograph network: nnU-Net's 2D configuration with its default trainer and plans, on 119 of 149 photographs with manually drawn brain masks, the other 30 for validation, in December 2024, before this work;
- the blockface-volume and marker networks: nnU-Net's 3D residual-encoder configuration (`nnUNetv2_plan_and_preprocess -d <dataset> -pl nnUNetPlannerResEncM`, then `nnUNetv2_train <dataset> 3d_fullres 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainer_250epochs`: 250 epochs, the final checkpoint), in rounds, the predictions of the brains added next corrected manually after each round.
  The blockface-volume network had 5 rounds, from 1 to 12 brains; the marker network had 3, and in the last one 12 of 16 brains with manually painted markers trained it and 4 checked it.

Their weights and training sets stay with the laboratory.

## The synthesis networks

The training code is in [code/blockface/synthesis](../../code/blockface/synthesis/README.md) (`train.py`, `network.py`, `losses.py`, `augment.py`, `data.py`), and the settings are under `training` in `settings/completion.yaml`, which lists the 13 brains.
Each brain's prepared folder is in the data folder, `brains/sub-<ID>/intermediate/synthesis/prepared/`, and `linc-complete prepare` makes it again.
The stand-in network's folders are the same with the stand-in b0 as input, which `--targets-from` builds from the own-b0 folders.
The paper's two stand-in networks, `d3p_full_A_s0` and `d3p_lso_prod_A_s0`, were trained on the final stand-in b0s, the MRI template's b0 carried into each of the 13 brains as for a brain without MRI; the data folder keeps them as `{mri}/nonlinear_s2val_prod/<ID>_mri_full.nii.gz`, and `--stand-in-b0` gives them:

```bash
$PY_DEEP code/blockface/synthesis/linc-complete prepare --subject <ID> --out $LINC_WORK/prepared                       # each of the 13 brains
$PY_DEEP code/blockface/synthesis/linc-complete prepare --subject <ID> --out $LINC_WORK/prepared_stand_in \
    --targets-from $LINC_WORK/prepared --stand-in-b0 '{mri}/nonlinear_s2val_prod/{ID}_mri_full.nii.gz'                # the stand-in network's folders
$PY_DEEP code/blockface/synthesis/linc-complete train --root $LINC_WORK/prepared --plan                               # the brains and missing files
$PY_DEEP code/blockface/synthesis/linc-complete train --root $LINC_WORK/prepared                                      # d3_full_A_s0
$PY_DEEP code/blockface/synthesis/linc-complete train --root $LINC_WORK/prepared_stand_in                             # d3p_full_A_s0
$PY_DEEP code/blockface/synthesis/linc-complete train --root $LINC_WORK/prepared --protocol lso                       # d3_lso_A_s0; prepared_stand_in: d3p_lso_prod_A_s0
```

Without `--stand-in-b0`, `--targets-from` takes each brain's stand-in b0 from what `linc-register direct-placement` wrote ([02, Step 4](../tutorials/02_blockface.md#step-4-register-the-volume-to-the-blockface-template-a-brain-without-mri-or-with-histology)), which computes them the same way.
The data folder also keeps an earlier, development set, `{mri}/nonlinear_s2val/`, made before the final cross-modal alignment; the paper's networks did not use it.
A run writes `last.pt`, `config.json` and `train_log.jsonl` to `$LINC_WORK/completion/runs/<prepared folder>_<protocol>_s<seed>/`, and `linc-complete --model <that folder>` uses it.

Each network of the paper took about an hour on one RTX 6000 Ada card (0.90 to 0.93 hours; `d3_lso_A_s0` 1.5 hours on an RTX 3090), using about 8 GB of it: 80 epochs of 500 pairs of patches.

A retrained synthesis network follows the same recipe but does not reproduce the shipped weights bit for bit.
To compare it with the paper's, synthesize one brain with each (the shipped network as in [02, Step 5](../tutorials/02_blockface.md#step-5-synthesize-the-tissue-that-was-never-photographed)) and compare the two volumes:

```bash
$PY_DEEP code/blockface/synthesis/linc-complete --brain $LINC_WORK/prepared/MF278 --out $LINC_WORK/complete_retrained --model <your run>
$PY tests/compare/compare_registration.py image $LINC_WORK/complete_retrained/MF278/MF278_bf_completed.nii.gz \
    $LINC_WORK/complete/MF278/MF278_bf_completed.nii.gz --min-r 0.9
```

A retraining of `d3_full_A_s0` on the same GPU model ended within 0.06 % of the paper's training loss, and its synthesized tissue on MF278 correlated with the shipped network's at r = 0.95.
