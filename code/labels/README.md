# Labels

`linc-labels` carries labels drawn on histology sections onto the blockface volume and into the template.
Its settings are in `settings/labels.yaml`, it needs no GPU, and [03, Steps D and E](../../docs/tutorials/03_histology.md#step-d-tracer-labels) walk through it.

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

## Inputs and outputs

It reads the labels, the section map, the match table, the regions volume and the histology registration ([code/histology](../histology/README.md)) from `$LINC_DATA/sub-<ID>/histology/`, and writes to `$LINC_DATA/sub-<ID>/labels/`.
[03, Steps D and E](../../docs/tutorials/03_histology.md#step-d-tracer-labels) list the files.

## Files

| | |
|---|---|
| `linc-labels` | the command |
| `labels.py` | push |
| `interpolate.py` | interpolate |
| `carry.py` | carry |
| `connect.py` | connect |
