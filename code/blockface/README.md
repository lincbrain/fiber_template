# Blockface reconstruction

Makes a brain's 3D color volume from its blockface photographs (Methods, "Blockface reconstruction").
Environment: `settings/env/tirl.yml`. Settings: `settings/blockface.yaml` and `settings/brains/<ID>.yaml`.

```bash
$PY_TIRL code/blockface/linc-reconstruct all --subject <ID>
$PY_TIRL code/blockface/linc-reconstruct <step> --subject <ID>
$PY_TIRL code/blockface/linc-reconstruct edit-photo-masks export|import --subject <ID>
```

`all` runs the steps `photos`, `stack`, `prepare`, `mask`, `volume`, `tubes` and `spacing`, in that order.
What each one does, needs and writes, the manual corrections, and a new brain's settings file: [02, Steps 1, 2 and 6](../../docs/tutorials/02_blockface.md#step-1-reconstruct-the-volume).

| | |
|---|---|
| `linc-reconstruct` | the command |
| `reconstruct.py`, `prepare.py`, `photos.py`, `jumps.py`, `spacing.py`, `curate.py` | the steps |
| `subject.py` | reads a brain's settings |
| `series.py` | the recorded alignment to an external series that MF191 and MF196 replay |
| [registration/](registration/README.md) | registers the volume to its MRI or to the template |
| [synthesis/](synthesis/README.md) | predicts the tissue the photographs lack |
