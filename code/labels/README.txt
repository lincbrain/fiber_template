LABELS

Carries labels drawn on histology sections into the template. Settings: settings/labels.yaml.

    $PY_TIRL code/labels/linc-labels push        --subject <ID>   (settings/env/tirl.yml)
    $PY      code/labels/linc-labels interpolate --subject <ID>   (settings/env/linc.yml)
    $PY      code/labels/linc-labels carry       --subject <ID>   (settings/env/linc.yml)
    $PY      code/labels/linc-labels connect     --subject <ID>   (settings/env/linc.yml)

    push          moves the labels through each section's registration onto the blockface
    interpolate   fills the sections between (demons), bundles and terminal fields as one group each
    carry         resamples the labels once into the MRI template
    connect       joins the bundles into one pathway and marks each join as inferred

You need: the labels (one OME-Zarr per class), the section map, the match table, the regions volume
and the histology registration, all in the brain's histology folder. About 40 minutes, no GPU.

You get: the labels on the blockface (labels/blockface/) and in the template (labels/template/). In a
file name, density is the fraction of each voxel, solid is where the density exceeds 0.5, and
connected has 1 for charted and 2 for inferred voxels.

    labels.py          push
    interpolate.py     interpolate
    carry.py           carry
    connect.py         connect
    linc-labels        the command
