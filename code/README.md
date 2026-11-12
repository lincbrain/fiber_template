# The pipeline

One folder per stage. Each stage's commands and scripts are in its folder.

| | |
|---|---|
| [mri/](mri/README.md) | prepares a brain's MRI |
| [blockface/](blockface/README.md) | makes a 3D volume from the blockface photographs |
| [blockface/registration/](blockface/registration/README.md) | registers a blockface volume to its MRI or to the template |
| [blockface/synthesis/](blockface/synthesis/README.md) | predicts the tissue a brain's photographs lack |
| [histology/](histology/README.md) | registers histology sections to their photographs |
| [labels/](labels/README.md) | carries labels drawn on the sections into the template |
| [template/](template/README.md) | builds the MRI and blockface templates |
| [template/link/](template/link/README.md) | registers the two templates and places brains in them |
| [figures/](figures/README.md) | the paper's figures, numbers and evaluation |
| [shared/](shared/README.md) | paths, transforms, provenance and the segmentation networks |
| [docs/](docs/README.md) | the tutorials' figures |

Every command prints what it needs, what it does and what it writes with `--help`.
Every setting of the paper is in [settings/](../settings/README.md), with a comment naming the Methods section it comes from.
