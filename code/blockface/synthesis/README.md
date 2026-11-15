# Synthesis of the missing tissue

No brain was photographed to its caudal end.
The synthesis network predicts the missing blockface tissue from the brain's b0 (for a brain without MRI, from its stand-in b0), joins it to the photographs at the junction plane, and marks the synthesized voxels.
Environment: `settings/env/deep.yml` (`$PY_DEEP`). Settings: `settings/completion.yaml`.
Step by step: [02, Step 5](../../../docs/tutorials/02_blockface.md#step-5-synthesize-the-tissue-that-was-never-photographed).

## Commands

```bash
$PY_DEEP code/blockface/synthesis/linc-complete prepare --subject <ID> --out <prepared>
$PY_DEEP code/blockface/synthesis/linc-complete --brain <prepared>/<ID> --out <out> [--stand-in]
$PY_DEEP code/blockface/synthesis/linc-complete train --root <prepared folders> [--plan]
```

## Inputs

- `prepare`: the brain's blockface volume and its registration to its own b0 ([code/blockface/registration](../registration/README.md)), or, for a brain without MRI, its stand-in b0, from `$LINC_DATA/sub-<ID>/`;
- the network: `models/completion/d3_full_A_s0` (own b0) or `d3p_full_A_s0` (stand-in b0), joined into `$LINC_MODELS/completion/`.

## Outputs

In `<out>/<ID>/`, on the brain's 0.25 mm grid: `<ID>_bf_completed` (the whole brain; `_rgb` in color), `<ID>_bf_synth`, `<ID>_bf_real`, `<ID>_photo_fraction`, `<ID>_synth_mask`, `<ID>_seam_ramp`, `<ID>_seam_depth` and `<ID>_b0_input_0p5`, with a provenance file.

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
