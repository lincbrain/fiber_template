# Synthesis of the missing tissue

No brain was photographed to its caudal end.
A 3D network predicts the missing blockface tissue from the brain's b0 (for a brain without MRI, from a stand-in b0: the MRI template's b0 moved into the brain), joins it to the photographs at one plane per brain, and marks the predicted voxels.
Environment: `settings/env/deep.yml`. Settings: `settings/completion.yaml`.

```bash
$PY_DEEP code/blockface/synthesis/linc-complete prepare --subject <ID> --out <prepared>
$PY_DEEP code/blockface/synthesis/linc-complete --brain <prepared>/<ID> --out <out> [--stand-in]
```

What they do, how long they take and what they write: [02, Step 5](../../../docs/tutorials/02_blockface.md#step-5-predict-the-tissue-that-was-never-photographed).

## Train the network again

```bash
$PY_DEEP code/blockface/synthesis/linc-complete train --root <prepared folders> [--plan]
```

About 1 hour on a GPU with 8 GB free, on the 13 brains with MRI. A retrained network has the same recipe but not the same bits.
The trained networks are in the data, under `models/`.

| | |
|---|---|
| `prepare.py` | makes a brain's prepared folder |
| `network.py` | the network |
| `train.py`, `losses.py`, `augment.py`, `data.py` | training |
| `export.py` | writes the results |
| `linc-complete` | the command |
