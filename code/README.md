# The pipeline

One folder per stage, each with its commands and a README: what it does, its commands, inputs, outputs and files.
Every command prints what it needs, what it does and what it writes with `--help`, and every setting is in [settings/](../settings/README.md).

| | |
|---|---|
| [mri/](mri/README.md) | prepares a brain's MRI: one volume, its brain mask, its bias correction |
| [blockface/](blockface/README.md) | makes a 3D volume from a brain's blockface photographs |
| [blockface/registration/](blockface/registration/README.md) | registers a blockface volume to its own b0 or to the blockface template, and a b0 to the template |
| [blockface/synthesis/](blockface/synthesis/README.md) | synthesizes the tissue a brain's photographs do not cover |
| [histology/](histology/README.md) | registers histology sections to their blockface photographs |
| [labels/](labels/README.md) | carries labels drawn on the sections into the template |
| [template/](template/README.md) | builds the MRI template, the blockface template and the template link |
| [figures/](figures/README.md) | the paper's figures, numbers and evaluation |
| [shared/](shared/README.md) | paths, transforms, provenance and the segmentation networks |
| [docs/](docs/README.md) | draws the tutorials' figures |
