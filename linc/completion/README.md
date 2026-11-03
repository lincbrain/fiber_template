# Caudal completion

No brain was photographed to its caudal end. A 3D U-Net predicts the missing blockface tissue from the brain's b0 image, its own or a
stand-in b0, and the prediction is joined to the photographs at one plane per brain, the clean cut face (paper: Methods,
"Registration to the brain's own MRI and caudal completion"; the network: Gonzales et al. 2026, ISBI).

## What runs from this package

```
linc-complete prepare --subject <ID> --out <dir> [...]               # one brain's products -> its prepared folder (prepare.py)
linc-complete --brain <prepared folder> --out <dir> [--stand-in]      # one brain -> its completed volume (cli/linc-complete)
python -m linc.completion.train --root <prepared folders> --protocol full --seed 0 --out <dir>   # the training recipe
python reproduce/numbers/d3_methods.py (d3_predicted_share, d3_domain_shift, d3_sigma2_pred)    # the paper's numbers
python reproduce/numbers/r057_s1_synthesis.py (r057_missing_extent, d3p_vs_d3_pseudo)            # Supplementary S1's numbers
```

Parameters: `configs/completion.yaml`. Weights: `models/completion.md`. Environment: `env/deep.yml` (the environment of record; `env/completion.yml` until 3 Nov 2026, same pins).

## linc-complete starts from a prepared folder

The network reads one **prepared folder** per brain (layout in `__init__.py`):
- the b0 on a 0.5 mm box;
- the photographs, their masks and their photographed fraction on the box's 0.25 mm refinement;
- `seam.json`, the cut plane.

The release provides the prepared folders of the paper's 21 brains as products of record. `linc-complete prepare`
(`prepare.py`) builds the folder of any brain from its products; the steps are below.

Three roots of the paper's prepared folders link into a sibling root: `canon_d3p_prod` and `canon_d3p_dev` (the stand-in network's
training inputs) into `canon_d3`, and `canon_d3n_prod_lo0` (the 8 brains without MRI, as deployed) into `canon_d3n_prod`. On the
analysis machine those links are absolute. A copy that works elsewhere:

```
python -m linc.completion.relocate <root> <new root> [--mode copy|relative]   # copy: each link replaced by what it points to
```

In copy mode the JSON records (`seam.json`, `canon_root.json`) name their sources as `$LINC_PRODUCTS/...` placeholders instead of
analysis-machine paths; `linc.paths.rerooted` resolves them. The tool checks every other file of the copy against the file it stands
for (md5), and every JSON record against the record with its paths scrubbed. `linc-complete` on a relocated folder gives the same
volumes.

## linc-complete prepare: a brain's prepared folder from its products

```
cli/linc-complete prepare --subject MF278 --out $LINC_WORK/prepared            # a brain with MRI (the defaults: MF278's inputs)
cli/linc-complete prepare --subject MR248 --out $LINC_WORK/prepared_stand_in --lo0 \
    --nonlin '{mri}/nonlinear_s2lin_prod' --matrix '{mri}/s2lin_prod/{ID}_bf2mri_dof9.mat' \
    --brain-mask '{mri}/{ID}_pseudo_brain_n1_0p5_prod.nii.gz' --stand-in-b0 '{nonlin}/{ID}_mri_full.nii.gz'   # no MRI
```

One process on the CPU (numpy, scipy, nibabel; about 30-50 s and 2.3 GB per brain). It writes `<out>/<ID>/` (`seam.json`,
`coarse/`, `fine/`) and never writes over an existing folder; `--keep DIR` also writes the intermediate grids. The flags are path
templates: `{ID}`, `{mri}` = `$LINC_DATA/sub-<ID>/mri`, `{bf}` = `$LINC_DATA/sub-<ID>/blockface`, `{nonlin}` = the `--nonlin` folder.

| Flag | Default (MF278's folder of record) | What it names |
|---|---|---|
| `--nonlin` | `{mri}/nonlinear_r1v3mr` | the registration's products on its 0.25 mm grid: `{ID}_mri` (the grid), `{ID}_bf_nonlin`, `{ID}_bf_nonlin_rgb`, `{ID}_warp`, `{ID}_mri_full` (the b0) |
| `--matrix` | `{mri}/r1v3mr/{ID}_bf2mri_dof9.mat` | the registration's linear start (FLIRT, blockface iso grey -> frame) |
| `--brain-mask` | `{mri}/{ID}_mri_mask_completion_reg_2026-10-03.nii.gz` | the completion brain on the 0.5 mm frame |
| `--register` | `{mri}/{ID}_register_2026-10-03.nii.gz` when it exists (`none`: no trim) | the MRI-side register; label 1 is trimmed before the cut face |
| `--stand-in-b0` | none | a brain without MRI: the stand-in b0, which is also its frame; no warp |
| `--lo0` | off | with `--stand-in-b0`: the deployed input `clip(b0 / p99.5, 0, 1)` (`stand_in_lo`) |

The other inputs have fixed names. They are the frame `{mri}/{ID}_mri.nii.gz` (brains with MRI) and, in `{bf}`,
`{ID}_blockface.ome.zarr` (only the level-1 shape and affine, read from its JSON metadata), `{ID}.info`, `{ID}_holes.nii.gz`,
`{ID}_blockface_iso_grey.nii.gz` and `{ID}_mask_iso.nii.gz`. The brain mask, the register and the stand-in b0 come from steps
outside this package.

`seam.json` keeps the fields of the folders of record and adds `provenance`: every input as `$LINC_DATA/...` with its md5, the
parameters and the clean-face numbers. Its paths are written as `$LINC_DATA/...` (`linc.paths.rerooted` resolves them). Not
written: `batch`, `sources.unrotated`, `sources.seam_root` and `sources.norm_md5`, which named the analysis project's
intermediate caches.

**Validation (3 Nov 2026).** MF278 against `canon_d3/MF278` and MR248 against `canon_d3n_prod/MR248` and
`canon_d3n_prod_lo0/MR248`:
- every array of `coarse/` and `fine/` is identical in voxels, dtype, shape and affine;
- every numeric field of `seam.json` is equal (z_f, d*, y_dep, the flat grid and R, the boxes, the normalization, the counts);
- `deploy.regions` gives the same sets for both protocols.

The intermediate grids are identical to the batch caches of record, step by step.

## How the paper's prepared folders were made

`prepare.py` runs steps 2 and 4 to 7 and the trim of step 3. Its inputs come from step 1 (placement) and from the register of
step 3.

1. **Placement.**
   - A brain with MRI: the blockface volume is registered to its own b0 (`linc-register --route mri`).
   - A brain without MRI: direct placement into the MRI template's b0 (`--route direct`). The template's b0 carried into the brain
     is its stand-in b0.
2. **Resampling onto the b0's frame.** The registered blockface volume goes onto the 0.25 mm lattice that refines the b0's 0.5 mm
   frame grid by 2: a 0.25 mm voxel 2k sits on 0.5 mm voxel k.
   - The grid is the b0's frame cropped to the brain with a 3 mm margin, at 0.25 mm.
   - It is then zero-padded so that the brain keeps at least 4 mm on every side.
   - Grey: the photographs, mapped per brain to [0, 1] by the 1st and 99th percentiles of the photographed tissue.
   - Color: the photographs' RGB on the same grid.
   - Brain mask: the b0's brain mask, mapped by nearest neighbor.
   - Photographed mask: where photographs exist.
3. **The author's correspondence rule.** Tissue that the MRI shows but the photographs lack (for example a longer brainstem) is
   deleted from the MRI side.
4. **Photographed fraction.** Each voxel's share that comes from photographed sections. Interpolated sections and filled marker tubes
   count as not photographed.
5. **Section normal.** The normal of the photographed sections, carried through the linear part of the registration. A grid
   rotated so that this normal is one of its axes turns every plane parallel to the sections into a grid plane.
6. **Clean cut face** (`configs/completion.yaml`, `clean_face`), on the planes of that rotated grid.
   - A plane is complete when the photographs cover at least 95 % of the brain mask eroded by 1 mm.
   - z_c is the most caudal plane from which every plane over the next 40 mm is complete.
   - The cut is z_f = z_c + 1.5 mm. Everything caudal of it is predicted, including incomplete last sections.
   - For the withheld-slab evaluation, the slab d* holds the complete photographed planes just rostral of z_f whose photographed
     volume equals the brain volume missing caudal of z_f.
7. **The prepared folder.**
   - The b0 is normalized (`input_percentiles`; a stand-in b0 with `stand_in_lo`), masked by the brain and cut to the 0.5 mm box
     around the brain.
   - The 0.25 mm arrays are cut to the box's 2x refinement.
   - `y_dep(k) = y0 + g . k` maps every 0.25 mm voxel to its plane on the rotated grid.
   - The training targets (`valid`) are the photographed tissue minus each anterior-posterior column's caudal-most 0.25 mm.

## Products (per brain)

See `export.py`. They are the completed volume (grey and color), the prediction alone, the photographs where they are kept, the
cross-fade weight, the signed distance from the cut, the photographed fraction, the predicted-voxel mask and the network's input.
