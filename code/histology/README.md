# Histology registration

Registers each histology section to the blockface photograph of its own section (Methods, "Histology
registration and label propagation"). Environment: `settings/env/tirl.yml`. Settings:
`settings/histology.yaml`. No GPU.

```bash
$PY_TIRL code/histology/linc-histology all --subject <ID> --store <ID>_blockface.ome.zarr \
    --histology <sections folder> --regions <regions.nii.gz> --out <histology>/registration
```

The steps are propose, match, pieces (each section divided into components: hemispheres, cerebellum
with brainstem), masks, prepare, register (TIRL) and rgb. About 2 hours per brain with `--jobs` 2 (24
cores). A run that stops resumes where it stopped. Use 12 threads per component (the default): TIRL's
result depends on the thread count, and 12 threads reproduce the paper's registrations exactly.

You get, in `--out`: each component's registration in `tirl/<component>/`, and the registered images.

## The two inputs made per brain

The match table says which photograph goes with each section. `linc-histology` propose proposes it.
A person then reviews it row by row, starting with the flagged rows, in a copy named
`match_table_edited.txt`, which nothing overwrites. A step uses the first of: the path given by flag,
`match_table_edited.txt`, `match_table.txt`, `match_table_proposed.txt`. It prints which one it used.

The regions volume is a label image on the blockface grid: 1 left, 2 right, 3 cerebellum and
brainstem, 4 marker tubes. For a new brain, carry the template's regions into it through its own
registration (the command is in the top `README.txt`), then write its left-hemisphere tubes as label 4.

|  |  |
|---|---|
| `match.py`, `propose.py`           | the match table |
| `pieces.py`, `sections.py`         | divides sections into components |
| `register.py`                    | the registration |
| `linc-histology`                 | the command |
