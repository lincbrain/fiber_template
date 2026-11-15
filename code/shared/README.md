# Shared tools

The tools every stage uses: paths, transforms, provenance and the segmentation networks.

## Commands

```text
$PY -m shared.xfm list  --subject <ID>                                       a brain's transforms, with their verdicts
$PY -m shared.xfm chain --from bf-native --to mri-template --subject <ID>   the links a chain uses
$PY -m shared.xfm apply --from <a> --to <b> [--subject <ID>] --ref <grid> [--nearest] <in> <out>
$PY_DEEP code/shared/linc-segment predict|train ...                          the segmentation networks (segment/)
$PY code/shared/linc-info                                                    what this installation runs with; attach it to a bug report
```

A transform `<from>__to__<to>` moves images from the first space to the second; its folder holds the transform files and `meta.json`.
World coordinates are RAS, in millimeters, and a chain is applied with one interpolation.
An inverted link is solved at every voxel of the target grid, so use 0.3 mm or coarser.
The spaces and their names: [overview, Spaces and transforms](../../docs/overview.md#spaces-and-transforms).

## Files

| | |
|---|---|
| `paths.py` | finds every input and output from the variables `settings/roots.sh` sets |
| `xfm.py` | the transforms: finds them, chains them and applies them once |
| `provenance.py` | records each product's inputs, command and tool versions |
| `linc-segment`, [segment/](segment/README.md) | runs and trains the segmentation networks |
| `linc-info` | prints what this installation runs with |
| `link_data.py` | gives the code its names for the files of the data folder (`settings/files.tsv`) |
| `make_local_data.sh` | builds the working folders from the data folder |
| `join_weights.sh` | joins the networks' weights kept in two parts, and checks them |
| `subjects_overview.py` | prints all the brains' settings as one table |
| `run_until.py` | runs a command until a file appears, then stops it |
