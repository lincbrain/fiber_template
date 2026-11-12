# The pipeline

One folder per stage. Each stage's commands and scripts are in its folder.

|  |  |
|---|---|
| `mri/`                      | prepares a brain's MRI |
| `blockface/`                | makes a 3D volume from the blockface photographs |
| `blockface/registration/`   | registers a blockface volume to its MRI or to the template |
| `blockface/synthesis/`      | predicts the tissue a brain's photographs lack |
| `histology/`                | registers histology sections to their photographs |
| `labels/`                   | carries labels drawn on the sections into the template |
| `template/`                 | builds the MRI and blockface templates |
| `template/link/`            | registers the two templates and places brains in them |
| `figures/`                  | the paper's figures, numbers and evaluation |
| `shared/`                   | paths, transforms, provenance and the segmentation networks |

Every command prints what it needs, what it does and what it writes with `--help`.
Every setting of the paper is in `settings/`, with a comment naming the Methods section it comes from.
