# Segmentation networks

Four networks, one recipe, one code path: brain masks and marker tubes.
Environment: `settings/env/deep.yml` (`$PY_DEEP`).
The weights are in this repository under `models/segment/` (a file over 100 MB in two parts, which `settings/roots.sh` joins); each network's card, `card.txt`, with its training data, held-out results and known limitations (Supplementary Section S5 of the paper), is in the data folder's `models/`.

| Network | Input | Output | GPU free |
|---|---|---|---|
| `mri_brainmask` | one ex vivo b0 or T2-weighted image | brain mask | 5 GB |
| `bf_mask2d` | one blockface photograph, or a folder of them | brain mask | 1 GB |
| `bf_mask3d` | the blockface volume with its background | brain mask | 9 GB |
| `bf_holes` | the masked color volume and its brain mask | marker tubes | 9 GB |

## Commands

```bash
$PY_DEEP code/shared/linc-segment predict --model mri_brainmask --out mask.nii.gz b0.nii.gz
$PY_DEEP code/shared/linc-segment predict --model bf_mask2d --native --in <photos> --out <masks>
$PY_DEEP code/shared/linc-segment train --help      # about 2 hours on a GPU with 10 GB free
```

## Inputs and outputs

`bf_mask2d` reads a folder of blockface photographs named `<case>_0000.png`.
Every mask has the values 0 and 1, on its input's grid; `--postprocess` changes the clean-up step.
The training tasks are in [settings/segment](../../../settings/segment/README.md).

The networks were trained on this project's brains, cameras and MRI acquisitions: a new setup, stain or contrast needs checking against a few manual masks.
`bf_holes` keeps tubes in the left hemisphere only, where they were drilled.
