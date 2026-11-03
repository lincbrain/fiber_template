# Register histology to its blockface photographs

Environment: `env/tirl.yml`.


```bash
$PY_TIRL cli/linc-histology all --subject MR252 --store $LINC_DATA/sub-MR252/blockface/MR252_blockface.ome.zarr \
    --histology <sections dir> --table <match table> --regions <ID>_regions.nii.gz \
    --out $LINC_DATA/sub-MR252/histology/registration --threads 12
```

The steps run in order: match each section to its photograph, cut the pieces, mask them, then register them with TIRL. TIRL's
result depends on the thread count; with 12 threads per piece (the default), the paper's registrations are reproduced exactly.

Two per-brain inputs are prepared before this step:
- **The match table** maps each histology section to the blockface photograph of each of its pieces. It is made by hand, once
  per brain and series; its format is in `linc/histology/match.py`.
- **The regions volume** (`<ID>_regions.nii.gz`: 1 left, 2 right, 3 cerebellum + brainstem, 4 marker tubes) gives the pieces the
  sections were cut into. It is the atlas regions carried from the template into the brain's blockface volume.
  [`linc/histology/README.md`](../linc/histology/README.md) explains how the paper's volumes were made and how to make one for a new
  brain (then pass `--regions-frame current`).

Write the output to `$LINC_DATA/sub-<ID>/histology/registration`, as above: the label steps (`docs/labels.md`) read it from there.

A regions volume made at another spacing than the brain's current one is given with that spacing, `--regions-spacing-um VX VZ`
(in-plane and between sections, in µm). For MF8 (`sub-MF191`), the paper's registration used an earlier regions volume, made before
its section step was corrected; it ships as the product `hidden_arm_regions` (`configs/products_blockface.yaml`), and
`--regions <that file> --regions-spacing-um 23.75 600` reproduces all 58 of its pieces byte for byte.
