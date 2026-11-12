# Shared tools

|  |  |
|---|---|
| `paths.py`              | finds every input and output from the variables `settings/roots.sh` sets |
| `xfm.py`                | the transforms: finds them, chains them and applies them once |
| `provenance.py`         | records each product's inputs, command and tool versions |
| `linc-segment`          | runs and trains the segmentation networks (`segment/`) |
| `linc-info`             | prints what this installation runs with; attach it to a bug report |
| `link_data.py`          | gives the code its working names for the files of the data tree |
| `make_local_data.sh`    | builds the working folders from the data tree |
| `subjects_overview.py`  | prints all the brains' settings as one table |
| `run_until.py`          | runs a command until a file appears, then stops it |

## Spaces and transforms

The spaces and their names: [00, Spaces and transforms](../../docs/tutorials/00_start_here.md#spaces-and-transforms).

A transform `<from>__to__<to>` moves images from the first space to the second. Its folder holds the
transform files and `meta.json`. World coordinates are RAS, in millimeters. A chain is applied with one
interpolation. An inverted link is solved at every voxel of the target grid, so use 0.3 mm or coarser.

```text
$PY -m shared.xfm chain --from bf-native --to mri-template --subject <ID>   lists the links
$PY -m shared.xfm apply --from <a> --to <b> [--subject <ID>] --ref <grid> [--nearest] <in> <out>
```
