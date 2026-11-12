# Synthesis of the missing tissue

No brain was photographed to its caudal end. A 3D network predicts the missing blockface tissue from
the brain's b0 and joins it to the photographs at one plane per brain. A brain without MRI uses a
stand-in b0: the MRI template's b0 moved into the brain. The predicted voxels are marked.
Environment: `settings/env/deep.yml`. Settings: `settings/completion.yaml`.

```bash
$PY_DEEP code/blockface/synthesis/linc-complete prepare --subject <ID> --out <prepared>
$PY_DEEP code/blockface/synthesis/linc-complete --brain <prepared>/<ID> --out <out> [--stand-in]
```

`prepare` needs the brain's registration and takes about a minute on a CPU. The second command takes
about 20 seconds on a GPU with 2 GB free. It fits one brightness map from the prediction to the
photographs near the cut and blends the two over 2 mm.

You get, in `<out>/<ID>/`, on the brain's 0.25 mm grid: `<ID>_bf_completed.nii.gz` (and `_rgb` in color),
`<ID>_bf_synth` (the prediction), `<ID>_bf_real` (the photographs used), `<ID>_photo_fraction`, and
`<ID>_synth_mask` (the predicted voxels).

## Train the network again

```bash
$PY_DEEP code/blockface/synthesis/linc-complete train --root <prepared folders> [--plan]
```

About 1 hour on a GPU with 8 GB free, on the 13 brains with MRI. A retrained network has the same
recipe but not the same bits. The trained networks are in the data, under `models/`.

- `prepare.py`          makes a brain's prepared folder
- `network.py`          the network
- `train.py`, `losses.py`, `augment.py`, `data.py`    training
- `export.py`           writes the results
- `linc-complete`       the command
