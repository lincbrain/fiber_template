# Segmentation models

Four networks segment new data: the brain in an ex vivo MRI image, the brain in a blockface photograph, the brain in a blockface
volume, and the marker tubes in a blockface volume. They share one recipe and one code path, `linc/segment`, run by
`cli/linc-segment`. Each was trained once, on every mask of the project that was drawn or corrected by hand,
and evaluated by a five-fold cross-validation grouped by brain. The paper's Supplementary Section S5 and Table S8 give the held-out
results; each model's card gives them per brain, with the model's known limitations.

## Files

Each model is a folder `$LINC_MODELS/segment/<task>/`:
- `model.pt`, the weights;
- `spec.json`, the input, normalization, prediction and post-processing;
- `card.md`, the training data, the held-out results and the known limitations;
- `SHA256SUMS`, the checksums of `model.pt` and `spec.json`. `linc-register` checks them before it uses the MRI model.

They come with the data release (`derivatives/models/segment/<model>/`). Check a downloaded copy with
`$PY models/verify_weights.py models/segmentation.sha256 --root $LINC_MODELS/segment`; `models/segmentation.md` gives each
model's input, output, post-processing and parameters.

## Environment

`env/deep.yml`, with the pinned versions of the environment the four models were trained and evaluated in. A GPU is the default device (`--device cuda`);
`--device` chooses another.

## Running a model

One case, one file per channel:

```bash
$PY_DEEP cli/linc-segment predict --model mri_brainmask --out <ID>_mask.nii.gz <ID>_b0.nii.gz
```

`--model` takes a task name, which reads `$LINC_MODELS/segment/<task>/`, or a model folder. With `--in DIR` instead of the image
files, every case in the folder is segmented; the cases are named `<case>_0000<ext>` (, one file per channel; an RGB
photograph is one PNG), and each mask is written to `<out>/<case><ext>`. Every mask has the values 0 and 1, on its input's grid.

For a folder of native photographs (named `<case>_0000.png`, as the reconstruction's photographs are):

```bash
$PY_DEEP cli/linc-segment predict --model bf_mask2d --native --in <photographs> --out <masks>
```

Ten native photographs of 3872 × 2592 pixels took 23 s on a GPU, about 1 s each after a few seconds of start-up. Each mask comes
back at the photograph's native size, as `<case>.png`, with its holes filled.

Prediction averages the network's output over every combination of flips; `--mirror off` makes one pass, 4 times
faster for a photograph and 8 times for a volume.

| Model | Input | Output and post-processing |
|---|---|---|
| `mri_brainmask` | one ex vivo b0 or T2-weighted volume (NIfTI), resampled to 0.5 mm and the mask back to the input's grid | the brain mask, with no post-processing |
| `bf_mask2d` | one RGB blockface photograph; with `--native`, a photograph at its native size, reduced 4 × 4 by block means first | the brain mask at the input's size, its interior holes filled, as the pipeline fills its photograph masks |
| `bf_mask3d` | the background-intact RGB volume that `linc-reconstruct mask` reads | the brain mask: the largest connected component, its holes filled |
| `bf_holes` | the masked color volume that `linc-reconstruct tubes` reads | the tubes, kept only in the left hemisphere, where they were drilled, and inside the brain mask, without pieces under 3 voxels; it needs the brain mask on the same grid (`--masks DIR`, or `--mask` for one case), or `--postprocess none` |

Each model applies the post-processing its `spec.json` names, as in the table. `--postprocess` replaces it with `none`,
`largest_component`, `largest_cc_fill` (the largest connected component, its holes filled) or `fill_holes` (enclosed holes filled
on the output grid, so at native size for a native photograph).

On a GPU, `predict` needs about 9 GB of free memory for `bf_mask3d` and `bf_holes`, 5 GB for `mri_brainmask` and 1 GB for
`bf_mask2d`.

## Known limitations

Each card lists them in full. In short:
- **All four:** they were trained on this project's brains, camera setups and MRI acquisitions; a new setup, stain or contrast needs
  checking against a few hand-made masks.
- **`mri_brainmask`:** it was trained on b0 and T2-weighted images only. Its masks are used as predicted, like the hand-made masks,
  which leave the spaces inside the brain unfilled; a few held-out masks had small detached parts, which `--postprocess
  largest_component` removes.
- **`bf_mask2d`:** its labels are the volumes' masks, one label section for several photographs, so they lag the photograph where
  the outline changes quickly, in the first photographs of a stack (the frontal pole); most of its low scores are there. Its mask
  fills holes, which the labels leave out (the ventricles and fissures).
- **`bf_mask3d`:** at surfaces facing the front of the brain, where tissue below the cut surface shows through the ice, its mask
  reaches slightly beyond the masks of record, which follow the cut face.
- **`bf_holes`:** the tubes are thin, so its Dice is low even when it detects the tubes; its median centerline Dice (clDice) over
  the held-out brains is 0.940. Its rule keeps tubes in the left hemisphere
  only, where they were drilled; a brain with tubes on the right is outside its scope.

## In the blockface reconstruction

The reconstruction runs these models for its three segmentation tasks (`photos`, `mask`, `tubes`). Their output gets the
pipeline's post-processing once (`linc/blockface/mask_rules.py`), and a hand-corrected mask still replaces a network's. On MM9, for all three tasks, the reconstruction's masks are identical to those of `linc-segment` run on its own.

In the registration, `linc-register mri --place-b0 <b0>` runs `mri_brainmask` on a new brain's b0 when no mask is given
(`docs/register.md`).
