# Blockface reconstruction

Makes a brain's 3D color volume from its blockface photographs (Methods, "Blockface reconstruction").
Environment: `settings/env/tirl.yml` (`$PY_TIRL`). Settings: `settings/blockface.yaml` and `settings/brains/<ID>.yaml`.
Step by step: [02, Steps 1, 2 and 6](../../docs/tutorials/02_blockface.md#step-1-reconstruct-the-volume).

## Commands

```bash
$PY_TIRL code/blockface/linc-reconstruct all --subject <ID>
$PY_TIRL code/blockface/linc-reconstruct <step> --subject <ID> [--keep-work]
$PY_TIRL code/blockface/linc-reconstruct edit-photo-masks export|import --subject <ID>
```

`all` runs the steps `photos`, `stack`, `prepare`, `mask`, `volume`, `tubes` and `spacing`, in that order.

## Inputs

- the photographs, in `$LINC_RAW/<raw>`, named as `raw_glob` and `raw_regex` say;
- `settings/brains/<ID>.yaml`: pixel size, section step, camera jumps, missing sections;
- optionally, manually corrected masks in `$LINC_DATA/sub-<ID>/blockface/curated/`.

## Outputs

In `$LINC_DATA/sub-<ID>/blockface/`: `<ID>_blockface.ome.zarr` (the volume, in color, at four resolutions), `<ID>_blockface_L2.nii.gz` (a lighter copy), `<ID>_mask.nii.gz`, `<ID>_holes.nii.gz` (the marker tubes) and `<ID>.info`.

## Files

| | |
|---|---|
| `linc-reconstruct` | the command |
| `reconstruct.py`, `prepare.py`, `photos.py`, `jumps.py`, `spacing.py`, `curate.py` | the steps |
| `subject.py` | reads a brain's settings |
| `series.py` | the recorded alignment to an external series that MF191 and MF196 replay |
| [registration/](registration/README.md) | registers the volume to its own b0 or to the blockface template |
| [synthesis/](synthesis/README.md) | synthesizes the tissue the photographs do not cover |
