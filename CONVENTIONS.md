# Conventions

This file is the contract between the modules of `linc`. Everything a module reads or writes follows it.

## Brains and names

- **Paper names:** MF1–MF10 (*Macaca fascicularis*) and MM1–MM15 (*Macaca mulatta*).
- **LINC identifiers:** files and folders use the LINC identifier (for example `sub-MR252`). The mapping is
  `docs/specimen_labels.json`.
- **MF1–MF4** are M1–M4 of Gong et al. (2026), the four MRI brains of the template's MRI channels.

## Roots (environment variables; `linc/paths.py`)

| Variable | Holds |
|---|---|
| `LINC_RAW` | raw acquisitions as delivered (photographs, MRI, histology); read only |
| `LINC_DATA` | per-brain data and products, `sub-<ID>/` |
| `LINC_TEMPLATE` | the released template |
| `LINC_MODELS` | network weights (`models/` has cards and checksums) |
| `LINC_PRODUCTS` | intermediate products read by `reproduce/` (mapped in `configs/products_*.yaml`) |
| `LINC_NUMBERS` | where regenerated number records go (default `./numbers`) |
| `LINC_WORK` | scratch (default `./work`) |

Paths recorded inside the products (input lists, transform metadata) name the machine that made them. Before running anything
that maps them (the `reproduce/` scripts, `linc-labels carry`), copy `configs/path_placeholders.yaml` to
`configs/path_placeholders.local.yaml` (not in git) and set each left-hand side to the prefix your products record; the example file
explains how.

## Data layout

Everything per brain lives under `$LINC_DATA/sub-<ID>/`:

```
sub-<ID>/
  blockface/   photographs (input) and the reconstruction: <ID>_blockface.ome.zarr, masks, holes
  mri/         b0 (and T2 where acquired), brain mask
  histology/   sections (OME-Zarr) and their registrations to the blockface
  labels/      annotations pushed to the blockface, interpolated, and carried to the template
  xfm/         one folder per transform, <src>__to__<dst>/ with meta.json (see below)
```

The released template lives under `$LINC_TEMPLATE/`:
- `mri/`: the MRI channels;
- `blockface/`: the blockface channel and template;
- `xfm/`: template-level transforms;
- `frames/`: reference grids.

## Spaces and transforms (`linc/xfm.py`)

**Named spaces:**

| Space | Meaning |
|---|---|
| `bf-native` | a blockface reconstruction's own grid |
| `b0-native`, `t2-native`, `micro-native` | an MRI as acquired |
| `mri-template` | the MRI template, 0.25 mm |
| `bf-template` | the blockface template in its own space |
| `histo-piece:<piece>` | one histology tissue piece |
| `atlas:<name>` | a published template |

**Folders.** A transform `<src>__to__<dst>` moves IMAGES from src to dst. Its folder holds the transform files and a `meta.json`
listing its components in PULL order: a dst-space world point goes through them first to last.

**Component conventions** (all are pulls, dst world → src world):

| Kind | Convention | Meaning |
|---|---|---|
| affine | `itk` | ITK/ANTs, LPS |
| affine | `fsl` | FLIRT .mat; needs both grids |
| affine | `ras_pull`, `ras_push` | 4 × 4 text, RAS |
| warp | `itk_lps_disp` | ITK/FireANTs displacement field |
| warp | `ras_disp` | the same with RAS components |
| warp | `fsl_rel`, `fsl_abs` | FSL warps; need the source grid |

**Chains.** Chains are composed and applied with one interpolation. Inverses are computed on the fly: analytically for affines, by
fixed-point iteration for warps.

**Validation.** Validated against antsApplyTransforms, applywarp and flirt (`linc/xfm.py` docstring). The release port reproduces the
analysis project's transforms exactly (pull and inverse, maximum difference 0.0 on a real blockface → b0 chain, 21 Oct 2026).

## World coordinates and display

- **World coordinates:** RAS, millimeters.
- **Figures:** neurological convention (subject's left on the left). Sagittal views have anterior to the right; axial views have
  anterior up.

## Parameters

Every parameter of the paper lives in `configs/*.yaml`, with a comment naming the Methods section it comes from. Code reads the
config; it does not repeat values.

## Provenance (`linc/provenance.py`)

- **Products:** every product gets `provenance.json`, recording its inputs (size, mtime, md5), the script, the command and the
  host.
- **Numbers:** every number of the paper is written by `write_numbers(name, data, inputs)` to `$LINC_NUMBERS/<name>.json`.
  `reproduce/check_numbers.py` compares a regenerated record with the paper's.

## External tools

The pipeline calls these as external programs, at the versions used in the paper:
- FSL 6.0.7.4
- MMORF 1.0.1
- ANTs
- TIRL 3.6.2
- nnU-Net 2.5.1

FireANTs 1.5.1 and anatomix are Python dependencies. Wrappers check versions before running.

## Not in this repository

- The image data, the released template and the network weights. Use the download script in `models/` and the data release.
- ONPRC18 and CIVM-R image intensities. Their licenses do not allow redistribution, so only our transforms to them are released.
  NMT v2 and CIVM-R are also inputs that you download yourself, for the comparison of registration targets and the tracer
  topography (which reads the CHARM key from your NMT v2 copy): `configs/targets.yaml` gives both sources and the folder layout
  (`configs/link.yaml` `inputs.atlas_reference`).
