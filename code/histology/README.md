# Histology registration

Registers each histology section to the blockface photograph of its own section (Methods, "Histology registration and label propagation").
Environment: `settings/env/tirl.yml` (`$PY_TIRL`). Settings: `settings/histology.yaml`. No GPU.
Step by step: [03, Steps A to C](../../docs/tutorials/03_histology.md#step-a-the-regions-volume).

## Commands

```bash
$PY_TIRL code/histology/linc-histology propose --subject <ID> --store <ID>_blockface.ome.zarr \
    --histology <sections folder> --regions <regions.nii.gz> --out <histology>/registration
$PY_TIRL code/histology/linc-histology all --subject <ID> --store <ID>_blockface.ome.zarr \
    --histology <sections folder> --regions <regions.nii.gz> --out <histology>/registration [--threads 12 --jobs 2]
```

`propose` proposes the match table; `all` runs the steps `match`, `pieces`, `masks`, `prepare`, `register` (TIRL) and `rgb`, in that order, and `propose` first when there is no match table yet.

## Inputs

- the sections, as JPEG 2000, in cutting order;
- the brain's blockface volume, `<ID>_blockface.ome.zarr` ([code/blockface](../blockface/README.md));
- its regions volume ([03, Step A](../../docs/tutorials/03_histology.md#step-a-the-regions-volume));
- optionally, a match table: `--table`, or `match_table_edited.txt`, `match_table.txt`, `match_table_proposed.txt`, the first that exists.

## Outputs

- `propose`: `match_table_proposed.txt`, `match_review.csv` and `match_review/` beside the `--out` folder;
- `all`: in the `--out` folder, `matched/`, `pieces/`, `tissue_masks/`, `moving/`, `fixed/` and `tirl/<component>/` (TIRL's transforms and the registered image), each step with a `provenance.json`.

## Files

| | |
|---|---|
| `linc-histology` | the command |
| `match.py`, `propose.py` | the match table |
| `pieces.py`, `sections.py` | divides sections into components |
| `register.py` | the registration |
| [patches/](patches/README.md) | our one change to TIRL |
