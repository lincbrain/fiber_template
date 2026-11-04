# Transform labels into the template

**You need** the brain's annotations and section map (`docs/inputs.md`), its histology registration (`docs/histology.md`) and the
template.

**What it does**
1. `push`: transforms the annotations through each section's registration onto the blockface;
2. `interpolate`: fills the labels between sections;
3. `carry`: transforms them into the template, in one resampling. In its files, `solid` is the label where the density exceeds 0.5;
4. `connect`: joins the carried bundle label into one connected pathway, and marks each join as inferred.

**You get** the labels on the blockface, `$LINC_DATA/sub-<ID>/labels/blockface/`, and in the template, `labels/template/`.

The label steps read and write under `$LINC_DATA/sub-<ID>/`, as `configs/labels.yaml` lists:
- the annotations (one OME-Zarr per class), the section map, the match table and the atlas regions on the blockface frame
  (`regions.nii.gz`: 1 left, 2 right, 3 cerebellum; the volume `linc-histology --regions` takes) under `histology/`;
- the histology registration (`docs/histology.md`). For the paper's brains, the label steps read each brain's registration of
  record; for MM1, MM3 and MF8 it includes the re-registration of a few components that corrected two registration errors (the
  paper's Methods; `configs/products_blockface.yaml`, `labels_histology_arm`);
- the template from `$LINC_TEMPLATE`.

The four stages, each with the Python of its environment:

```bash
$PY_TIRL cli/linc-labels push        --subject MR252   # env/tirl.yml: the annotations through each section's registration onto the blockface
$PY      cli/linc-labels interpolate --subject MR252   # env/linc.yml: interpolate the labels between sections (demons)
$PY      cli/linc-labels carry       --subject MR252   # env/linc.yml: carry the labels into the MRI template, in one resampling
$PY      cli/linc-labels connect     --subject MR252   # env/linc.yml: join the bundle label into one connected pathway
```

The interpolation interpolates the bundles and the terminal fields as one group each, keeping each density class as a part of its group,
pairing sections within each hemisphere; a group present on only one side of a gap shrinks to a point at its middle. It writes
labels only, under `labels/blockface/tracer/`. `--method v1`, the earlier method, interpolates each class on its own and also writes
the RGB volume.

On MM2 (`sub-MR252`), the push and the interpolation reproduce the paper's labels byte for byte; the interpolation took
34 minutes on a shared node.

`connect` reads `carry`'s densities, `labels/template/tracer/<ID>_<group>_mri-template_density.nii.gz`, and the template's brain mask
and white matter (FA > 0.25). It keeps the largest component of the bundles and joins every other component to it along the cheapest
path inside the brain, which is cheap where the density is high and cheaper in white matter. It drops only specks smaller than 8 voxels.
It takes 10 to 15 seconds on a CPU for MM1 to MM3 and 77 seconds for MF8, with at most 1.2 GB of RAM. It writes in the same folder:
- `<ID>_<group>_mri-template_solid2.nii.gz`, one per group;
- `<ID>_Bundles_mri-template_connected.nii.gz`: 1 = charted, 2 = inferred;
- `<ID>_Bundles_mri-template_solid_v2c.nii.gz`: the pathway;
- `<ID>_Bundles_mri-template_bridges.nii.gz`: each join's component number;
- `<ID>_connect_v2c.json`: every join with its length, the components dropped and the inferred fraction.

From the paper's carried densities, `connect` reproduces the paper's files of all four cases voxel for voxel. `linc-labels all` ends with it.
