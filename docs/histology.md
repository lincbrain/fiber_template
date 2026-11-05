# Register histology to its blockface photographs

**You need** the brain's blockface volume, its histology sections, its match table and its regions volume (`docs/inputs.md`).

**What it does**
1. matches each section to its photograph;
2. divides each section into components (the hemispheres, the cerebellum and brainstem) by the regions volume;
3. masks the components;
4. registers each component to its photograph (TIRL).

**You get**, under the `--out` folder: each component's registration, with the registered images.

Environment: `env/tirl.yml`.


```bash
$PY_TIRL cli/linc-histology all --subject MR252 --store $LINC_DATA/sub-MR252/blockface/MR252_blockface.ome.zarr \
    --histology <sections dir> --table <match table> --regions <ID>_regions.nii.gz \
    --out $LINC_DATA/sub-MR252/histology/registration --threads 12
```

The steps run in order: match each section to its photograph, divide it into components, mask them, then register them with TIRL. TIRL's
result depends on the thread count; with 12 threads per component (the default), the paper's registrations are reproduced exactly.

Two per-brain inputs are prepared before this step:
- **The match table** maps each histology section to the blockface photograph of each of its components, once per brain and series.
  `linc-histology propose` proposes it for a person to review row by row ([`linc/histology/README.md`](../linc/histology/README.md));
  its format and the order in which steps find it are in `linc/histology/match.py`.
- **The regions volume** (`<ID>_regions.nii.gz`: 1 left, 2 right, 3 cerebellum + brainstem, 4 marker tubes) gives the components the
  sections were cut into. It is the atlas regions transformed from the template into the brain's blockface volume.
  [`linc/histology/README.md`](../linc/histology/README.md) explains how the paper's volumes were made and how to make one for a new
  brain.

Write the output to `$LINC_DATA/sub-<ID>/histology/registration`, as above: the label steps (`docs/labels.md`) read it from there.

A regions volume made at another spacing than the brain's current one is given with that spacing, `--regions-spacing-um VX VZ`
(in-plane and between sections, in µm). For MF8 (`sub-MF191`), the paper's registration used its own regions volume, made at a section step of 600 µm, and
`--regions <that file> --regions-spacing-um 23.75 600` reproduces all 58 of its components byte for byte.
