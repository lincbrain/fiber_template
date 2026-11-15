# Histology registration

`linc-histology` registers each histology section to the blockface photograph of its own section (Methods, "Histology registration and label propagation").
It runs in `$PY_TIRL` (`settings/env/tirl.yml`) with the settings in `settings/histology.yaml`, needs no GPU, and is walked through in [03, Steps A to C](../../docs/tutorials/03_histology.md#step-a-the-regions-volume).

## Commands

```bash
$PY_TIRL code/histology/linc-histology propose --subject <ID> --store <ID>_blockface.ome.zarr \
    --histology <sections folder> --regions <regions.nii.gz> --out <histology>/registration
$PY_TIRL code/histology/linc-histology all --subject <ID> --store <ID>_blockface.ome.zarr \
    --histology <sections folder> --regions <regions.nii.gz> --out <histology>/registration [--threads 12 --jobs 2]
```

`propose` proposes the match table.
`all` runs the steps `match`, `pieces`, `masks`, `prepare`, `register` (TIRL) and `rgb`, in that order, starting with `propose` when there is no match table yet.

## Inputs and outputs

It reads the sections (JPEG 2000, in cutting order), the brain's blockface volume `<ID>_blockface.ome.zarr` ([code/blockface](../blockface/README.md)), its regions volume and, if there is one, a match table.
[03, Steps A to C](../../docs/tutorials/03_histology.md#step-a-the-regions-volume) describe each input, the order in which a match table is looked for, and every output.

## Files

| | |
|---|---|
| `linc-histology` | the command |
| `match.py`, `propose.py` | the match table |
| `pieces.py`, `sections.py` | divide sections into components |
| `register.py` | the registration |
| [patches/](patches/README.md) | our one change to TIRL |
