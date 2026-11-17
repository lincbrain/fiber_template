# Blockface reconstruction

`linc-reconstruct` turns a brain's blockface photographs into a 3D color volume (Methods, "Blockface reconstruction").
It runs in `$PY_TIRL` (`settings/env/tirl.yml`) with the settings in `settings/blockface.yaml` and `settings/brains/<ID>.yaml`, and [02, Steps 1, 2 and 6](../../docs/tutorials/02_blockface.md#step-1-reconstruct-the-volume) walk through it.

## Commands

```bash
$PY_TIRL code/blockface/linc-reconstruct all --subject <ID> [--keep-work]
$PY_TIRL code/blockface/linc-reconstruct <step> --subject <ID>             # volume and tubes also take --keep-work
$PY_TIRL code/blockface/linc-reconstruct edit-photo-masks export|import --subject <ID>
```

`all` runs the steps `photos`, `stack`, `prepare`, `mask`, `volume`, `tubes` and `spacing`, in that order.

## Inputs and outputs

It reads the photographs from `$LINC_RAW/<raw>`, the brain's settings file and any corrected masks in `$LINC_DATA/sub-<ID>/blockface/curated/`, and writes the volume, its brain mask and its fiducial markers (the round holes in the block face) to `$LINC_DATA/sub-<ID>/blockface/`.
[02, Step 1](../../docs/tutorials/02_blockface.md#step-1-reconstruct-the-volume) lists the files.

## Files

| | |
|---|---|
| `linc-reconstruct` | the command |
| `reconstruct.py`, `prepare.py`, `photos.py`, `jumps.py`, `spacing.py`, `curate.py` | the steps |
| `subject.py` | reads a brain's settings |
| `series.py` | the recorded alignment to an external series that MF191 and MF196 replay |
| [registration/](registration/README.md) | registers the volume to its own b0 or to the blockface template |
| [synthesis/](synthesis/README.md) | synthesizes the tissue the photographs do not cover |
