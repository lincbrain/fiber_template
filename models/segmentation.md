# Segmentation models

Four networks, one recipe and one code path (`linc/segment`, `cli/linc-segment`). They replace the earlier networks of the paper:

| Model | Input | Output | Post-processing (in its spec) | Parameters |
|---|---|---|---|---|
| `mri_brainmask` | one scalar ex vivo MRI (b0 or T2), resampled to 0.5 mm and back | brain mask | none | 31,195,594 |
| `bf_mask2d` | one blockface photograph (RGB); native photographs with `--native` (4 × 4 block means first) | brain mask | enclosed holes filled, at native size | 20,619,658 |
| `bf_mask3d` | the background-intact RGB blockface volume (three channel files), resampled to the median spacing and back | brain mask | largest 26-connected component, holes filled | 31,197,322 |
| `bf_holes` | the masked color blockface volume, same spacing | marker tubes | left hemisphere only, inside the brain mask, parts under 3 voxels dropped (needs the brain mask) | 31,197,322 |

**The weights are not in git.**
- Put each model's folder (`model.pt`, `spec.json`, `card.md`, `SHA256SUMS`) under `$LINC_MODELS/segment/<model>/`.
- Check them with `$PY models/verify_weights.py models/segmentation.sha256 --root $LINC_MODELS/segment`.
- Download location: with the data release (`derivatives/models/segment/<model>/`).
- `cli/linc-segment predict --model <model> ...` then finds them by name; `linc-reconstruct` and `linc-register` call it through
  `LINC_SEGMENT_PYTHON` (`env/deep.yml`).

**Each model's card** (`card.md`, beside its weights) gives its data and labels, the recipe, the inference, the held-out
performance by brain, the comparison with the model it replaces, and its known limitations. The numbers of the cards and of the
paper's Supplement come from the records `segmodels_<model>_2026-11` (`reproduce/numbers`).

**Recipe** (`linc/segment/train.py`; the same for all four):
- a U-Net with instance norm, leaky ReLU and deep supervision;
- Dice + cross-entropy; SGD (Nesterov 0.99), learning rate 1e-2 with polynomial decay; 200 epochs × 250 iterations;
- augmentation on the GPU (rotation, scale, mirroring, noise, blur, brightness, contrast, low resolution, gamma);
- five folds grouped by brain for the held-out performance, and one model on all brains, the one shipped;
- the final epoch's weights, with no selection on held-out data.

**Inference:**
- sliding window with the training patch (step 1/2, Gaussian weighting);
- test-time mirroring (the mean over every combination of flips);
- class probabilities resampled back linearly to the input's grid, then the argmax and the model's post-processing.

`--postprocess` overrides the post-processing: `spec`, `none`, `largest_component`, `largest_cc_fill` or `fill_holes`.

Trained and checked in `env/deep.yml` on NVIDIA RTX 6000 Ada GPUs, November 2026.
