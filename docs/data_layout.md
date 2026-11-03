# Data layout

Nothing in this repository holds a path to a machine. Every input and output is located through seven folders ("roots"), each set
by an environment variable. `linc/paths.py` resolves them; `CONVENTIONS.md` gives the rules for spaces, transforms and provenance.

| Variable | Holds | Needed for |
|---|---|---|
| `LINC_RAW` | the blockface photographs as taken, one folder per brain (`configs/subjects/<ID>.yaml` names it); for a few records, the raw MRI (`macaque_mri`, `mri_mm`), the microstructure fits' source series (`microstructure`, read by `r057_mri_facts_b` only) and the LINC dataset (`linc_000003`: histology, participants) | reconstruction; those records |
| `LINC_DATA` | per-brain data: `$LINC_DATA/sub-<ID>/{blockface,mri,histology,labels,xfm}/` | using the template on your own brains |
| `LINC_TEMPLATE` | the released template: `mri/` (MRI channels), `blockface/` (blockface channel and template), `xfm/`, `frames/` | registering into the template |
| `LINC_MODELS` | the trained networks: `segment/<model>/` (the segmentation networks) and `completion/<name>/` (`docs/models.md`) | reconstruction, synthesis of missing tissue |
| `LINC_PRODUCTS` | intermediate products of the paper (reconstructions, registrations, template builds) | reproducing figures and numbers |
| `LINC_NUMBERS` | the numbers records (`<record>.json`); default `./numbers` | reproducing the paper |
| `LINC_WORK` | scratch for intermediate files; default `./work` | every pipeline |

Example:

```bash
export LINC_RAW=/data/linc/photographs
export LINC_DATA=/data/linc/brains
export LINC_MODELS=/data/linc/models
export LINC_TEMPLATE=/data/linc/template
export LINC_PRODUCTS=/data/linc/products
export LINC_NUMBERS=$PWD/numbers
export LINC_WORK=$PWD/work
```

## Per-brain data (`LINC_DATA`)

One folder per brain, named by its Specimen ID (`sub-MR252`). `docs/specimen_labels.json` maps the identifiers to the paper's
names (MM2 = MR252). The layout inside each folder is in `CONVENTIONS.md` (Data layout); transforms live in `xfm/<src>__to__<dst>/`
with a `meta.json`.

## Products (`LINC_PRODUCTS`) and the product maps

Scripts that read the paper's intermediate products never name a path themselves. They read a product map,
`configs/products_<owner>.yaml`, which gives each product a short name and a path relative to `$LINC_PRODUCTS`. For example,
`configs/products_figures.yaml` lists the inputs of the pipeline and tracer figures (`reproduce/figures/pipeline.py`, `tracer.py`).

The data release fixes `LINC_PRODUCTS` and, where its folder names differ from ours, edits only these YAML files.

## Numbers records (`LINC_NUMBERS`)

Every number in the paper's text and tables comes from a record, `<record>.json` = `{"data": ..., "_provenance": ...}`:
- `reproduce/numbers/<record>.py` writes it;
- `reproduce/check_numbers.py` compares it with the paper's copy (counts exact, floats to 1e-9).

Each record holds its provenance: inputs with size, modification time and md5, the command and the host. Figures carry the same
in `<figure>.provenance.json`.

## Outputs

- **Figures:** `reproduce/figures/*.py --out <file.png>` writes the PNG, a PDF beside it and its provenance. The default is
  `figures/` under the current folder.
- **Pipelines:** results go under `$LINC_DATA/sub-<ID>/`, following `CONVENTIONS.md`; scratch goes under `$LINC_WORK`.

## Preparing the data release (notes for the release, not for users)

- **The atlas-region volumes** by which the histology sections were divided into components (`labels/regions_t2_final/<ID>_regions.nii.gz` in the
  analysis project) are still in the blockface spacing frame used before 2 Oct 2026. `linc-histology --regions` needs them in the
  store's world frame. Before release, rescale each brain's header along the section axis (axis 1), about the world origin, without
  resampling: new affine = diag(1, f, 1, 1) · old affine.
  - The factor f is the brain's own: the ratio of its section step as used to its nominal step (numbers record `bf_spacing_used`,
    `per_brain.<ID>.vz_um / nominal_vz_um`).
  - For the six histology brains: MM2 and MM3 0.9, MM1 0.915, MF5 and MF6 0.94, MF8 1.0.
- **Thread count.** TIRL's result depends on the number of threads. The paper's histology registrations ran with 12 threads per
  piece (`configs/histology.yaml`, `tirl.threads`). At that setting, `linc-histology` reproduces MM2's 47 registered sections bit for
  bit.

## Not in this repository

- The image data.
- The released template.
- The network weights: they come with the data release; `models/` has the model cards, the checksum lists and a script that
  checks a downloaded copy against them.
- ONPRC18 and CIVM-R image intensities: their licenses do not allow redistribution. Only our transforms to them are released.
  NMT v2 and CIVM-R are also inputs that you download yourself, for the comparison of registration targets and the tracer
  topography (which reads the CHARM key from your NMT v2 copy): `configs/targets.yaml` gives both sources and the folder layout
  (`configs/link.yaml` `inputs.atlas_reference`).
