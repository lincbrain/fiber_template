# Synthesis of the missing tissue

No brain was photographed to its caudal end.
`linc-complete` predicts the missing blockface tissue from the brain's b0 (for a brain without MRI, from its stand-in b0), joins it to the photographs at the junction plane, and marks the synthesized voxels.
It runs in `$PY_DEEP` (`settings/env/deep.yml`) with the settings in `settings/completion.yaml`, and [02, Step 5](../../../docs/tutorials/02_blockface.md#step-5-synthesize-the-tissue-that-was-never-photographed) walks through it.

## Commands

```bash
$PY_DEEP code/blockface/synthesis/linc-complete prepare --subject <ID> --out <prepared>
$PY_DEEP code/blockface/synthesis/linc-complete --brain <prepared>/<ID> --out <out> [--stand-in]
$PY_DEEP code/blockface/synthesis/linc-complete train --root <prepared folders> [--plan]
```

## Inputs and outputs

[02, Step 5](../../../docs/tutorials/02_blockface.md#step-5-synthesize-the-tissue-that-was-never-photographed) lists what it reads and writes; the networks are joined into `$LINC_MODELS/completion/`.

## Training

`train` takes about 1 hour on a GPU with 8 GB free, on the 13 brains with MRI; a retrained network has the same recipe but not the same bits.
`models/completion/` also holds `d3_lso_A_s0` and `d3p_lso_A_s0`, the leave-slab-out networks of the paper's evaluation.

## Files

| | |
|---|---|
| `linc-complete` | the command |
| `prepare.py` | makes a brain's prepared folder |
| `network.py` | the network |
| `train.py`, `losses.py`, `augment.py`, `data.py` | training |
| `export.py` | writes the results |
