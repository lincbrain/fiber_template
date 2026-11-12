# Histology registration

Registers each histology section to the blockface photograph of its own section (Methods, "Histology registration and label propagation").
Environment: `settings/env/tirl.yml`. Settings: `settings/histology.yaml`. No GPU.

```bash
$PY_TIRL code/histology/linc-histology propose --subject <ID> --store <ID>_blockface.ome.zarr \
    --histology <sections folder> --regions <regions.nii.gz> --out <histology>/registration
$PY_TIRL code/histology/linc-histology all --subject <ID> --store <ID>_blockface.ome.zarr \
    --histology <sections folder> --regions <regions.nii.gz> --out <histology>/registration
```

`all` runs the steps `match`, `pieces`, `masks`, `prepare`, `register` (TIRL) and `rgb`, in that order, and `propose` first when there is no match table yet.
How to run them, the two inputs made per brain (the regions volume and the match table), the stiffness rule and the checks: [03, Steps A to C](../../docs/tutorials/03_histology.md#step-a-the-regions-volume).

| | |
|---|---|
| `match.py`, `propose.py` | the match table |
| `pieces.py`, `sections.py` | divides sections into components |
| `register.py` | the registration |
| `linc-histology` | the command |
| [patches/](patches/README.md) | our one change to TIRL |
