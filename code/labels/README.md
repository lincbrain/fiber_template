# Labels

Carries labels drawn on histology sections onto the blockface volume and into the template.
Settings: `settings/labels.yaml`. No GPU.
Step by step: [03, Steps D and E](../../docs/tutorials/03_histology.md#step-d-tracer-labels).

## Commands

```text
$PY_TIRL code/labels/linc-labels push        --subject <ID> [--kind structures]   (settings/env/tirl.yml)
$PY      code/labels/linc-labels interpolate --subject <ID>                       (settings/env/linc.yml)
$PY      code/labels/linc-labels carry       --subject <ID>                       (settings/env/linc.yml)
$PY      code/labels/linc-labels connect     --subject <ID>                       (settings/env/linc.yml)
```

| Subcommand | What it does |
|---|---|
| `push` | moves the labels through each component's registration onto the blockface (`--kind structures`: the LGN and STN) |
| `interpolate` | fills the slices between charted sections (demons), the bundles and the terminal fields as one group each |
| `carry` | resamples the labels into the template in one resampling |
| `connect` | joins the bundles into one pathway and marks each join as inferred |

## Inputs

In `$LINC_DATA/sub-<ID>/histology/`: the labels (`annotations/<class>.ome.zarr`), the section map (`section_map.tsv`), the match table, the regions volume and the histology registration ([code/histology](../histology/README.md)); for the LGN and STN, `structures/`.

## Outputs

`$LINC_DATA/sub-<ID>/labels/blockface/tracer/` and `structures/` (on the blockface), and `$LINC_DATA/sub-<ID>/labels/template/tracer/` (in the template).

## Files

| | |
|---|---|
| `linc-labels` | the command |
| `labels.py` | push |
| `interpolate.py` | interpolate |
| `carry.py` | carry |
| `connect.py` | connect |
