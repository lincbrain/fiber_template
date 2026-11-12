# Labels

Carries labels drawn on histology sections into the template. Settings: `settings/labels.yaml`.

```text
$PY_TIRL code/labels/linc-labels push        --subject <ID>   (settings/env/tirl.yml)
$PY      code/labels/linc-labels interpolate --subject <ID>   (settings/env/linc.yml)
$PY      code/labels/linc-labels carry       --subject <ID>   (settings/env/linc.yml)
$PY      code/labels/linc-labels connect     --subject <ID>   (settings/env/linc.yml)
```

| Subcommand | What it does |
|---|---|
| `push` | moves the labels through each section's registration onto the blockface |
| `interpolate` | fills the sections between (demons), bundles and terminal fields as one group each |
| `carry` | resamples the labels once into the MRI template |
| `connect` | joins the bundles into one pathway and marks each join as inferred |

What they read and write, the file names and how long they take: [03, Steps D and E](../../docs/tutorials/03_histology.md#step-d-tracer-labels).

| | |
|---|---|
| `labels.py` | push |
| `interpolate.py` | interpolate |
| `carry.py` | carry |
| `connect.py` | connect |
| `linc-labels` | the command |
