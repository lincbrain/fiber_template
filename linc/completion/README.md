# Synthesis of the missing tissue

No brain was photographed to its caudal end. A 3D U-Net (an encoder-decoder convolutional network with skip connections) predicts the missing
blockface tissue from the brain's b0 image, and the prediction is joined to the photographs at one plane per brain, the junction plane. A brain
without MRI uses a stand-in b0: the MRI template's b0 transformed into the brain. The paper describes this in Methods, "Blockface to MRI
registration and synthesis of missing sections", and in Supplementary Section S1.

## What runs from this package

```
cli/linc-complete prepare --subject <ID> --out <dir>     # one brain's products -> its prepared folder (prepare.py; CPU)
cli/linc-complete --brain <prepared folder> --out <dir> [--stand-in]   # one brain -> its volume with synthesized tissue (GPU)
cli/linc-complete train --root <prepared folders> [--plan]             # the network, trained again (train.py; GPU)
python reproduce/numbers/d3_methods.py (d3_predicted_share, d3_domain_shift, d3_sigma2_pred)    # the paper's numbers
python reproduce/numbers/r057_s1_synthesis.py (r057_missing_extent, d3p_vs_d3_pseudo --final)    # Supplementary S1's numbers
```

Parameters: `configs/completion.yaml`. Weights: `models/completion.md`. Environment: `env/deep.yml`.

## The prepared folder

The network reads one **prepared folder** per brain, named by its Specimen ID (layout in `__init__.py`):
- the b0 on a 0.5 mm box;
- the photographs, their masks and each voxel's photographed fraction on the box's 0.25 mm refinement;
- `seam.json`, the junction plane.

The data release provides the prepared folders of the paper's 21 brains. Three of their roots link into a sibling root: `canon_d3p_prod` and
`canon_d3p_dev` (the stand-in network's training inputs) into `canon_d3`, and `canon_d3n_prod_lo0` (the 8 brains without MRI, as used) into
`canon_d3n_prod`. A copy that works on another machine:

```
python -m linc.completion.relocate <root> <new root> [--mode copy|relative]   # copy: each link replaced by what it points to
```

In copy mode the JSON records (`seam.json`, `canon_root.json`) name their sources as `$LINC_PRODUCTS/...` placeholders; `linc.paths.rerooted`
resolves them. The tool checks every other file of the copy against the file it stands for (md5), and every JSON record against the record with
its paths scrubbed. `linc-complete` on a relocated folder gives the same volumes.

## linc-complete prepare

`cli/linc-complete prepare --subject <ID> --out <dir>` reads the brain's inputs from `configs/completion.yaml`, `prepare.inputs`. Each of the paper's
21 brains has its row there. Any other brain needs no paths: it uses what `linc-register mri` wrote (`new_brain`) when that exists, else what
`linc-register direct-placement` wrote in `mri/stand_in/` (`new_brain_stand_in`, a brain without MRI). To use other folders, edit the table. One process on the CPU, about a minute and 2.3 GB per brain; an existing folder is never written over.

Each row names:

| Key | What it names |
|---|---|
| `nonlin` | the registration's products on its 0.25 mm grid: `{ID}_mri` (the grid), `{ID}_bf_nonlin`, `{ID}_bf_nonlin_rgb`, `{ID}_warp`, `{ID}_mri_full` (the b0) |
| `matrix` | the registration's linear start (a FLIRT matrix, blockface grey -> the b0 frame) |
| `brain_mask` | the brain on the 0.5 mm frame |
| `register` (the paper's brains) | labels on the frame; label 1 is tissue the MRI shows and the photographs lack, removed before the junction plane is detected |
| `detect_trim` (a new brain) | detect that tissue instead (`trim` in the same file), and write it as `<ID>_mri_only_tissue.nii.gz` beside the folder |
| `stand_in_b0` | a brain without MRI: the stand-in b0, which is also its frame; there is no deformation |

The other inputs have fixed names: the frame `{mri}/{ID}_mri.nii.gz` (brains with MRI) and, in `$LINC_DATA/sub-<ID>/blockface`, `{ID}_blockface.ome.zarr`
(only the level-1 shape and affine, read from its JSON metadata), `{ID}.info`, `{ID}_holes.nii.gz`, `{ID}_blockface_iso_grey.nii.gz` and
`{ID}_mask_iso.nii.gz`.

`seam.json` records the junction plane, the normalization and every input as `$LINC_DATA/...` with its md5.

**To train the stand-in network**, a brain with MRI needs a folder with its stand-in b0 as the input and its own photographs as the targets:

```
cli/linc-complete prepare --subject MF278 --out <dir> --targets-from <own-b0 prepared folders>
```

The detector's rule is in `configs/completion.yaml`, `prepare.trim`: unphotographed brain voxels within the photographed extent, opened by a 1 mm ball to drop the thin surface shell and fissures; the components of at least 0.1 ml that begin within 20 mm of the photographed end and lie in the lower half of the brain, because only the brainstem should be removed, not cortex at the cut; and their caudal continuation. On the paper's 12 evaluation brains it detects tissue in the same 6 as the labels of record and in none of the other 6. It matches the label of record exactly on the one brain whose detection of record used the same brain mask (Dice coefficient 1.000), and with a Dice coefficient of 0.92 to 0.96 on the others, which used an older outline. The paper's brains also carry hand trims, which a new brain does not get.

The stand-in b0 is `{mri}/stand_in/{ID}_mri_full.nii.gz`, which `linc-register direct-placement` writes for a brain with MRI too; `--stand-in-b0` names another. Run `prepare` in `env/deep.yml`: in `env/linc.yml` (other numpy and scipy versions) `photo_fraction` differs from the record by up to 1.1e-5.

**Validation.** All 21 prepared folders of record rebuild identically: every array of `coarse/` and `fine/` in voxels, dtype, shape and affine, and every
numeric field of `seam.json` (z_f, d*, y_dep, the boxes, the normalization). The stand-in network's training folders do too.

## How a prepared folder is made

`prepare.py` runs steps 2 to 7. Step 1 comes before it.

1. **Registration.** A brain with MRI: the blockface volume is registered to its own b0 (`linc-register mri`). A brain without MRI: it is registered to
   the MRI template's b0 (`linc-register direct-placement`), and the template's b0, transformed into the brain, is its stand-in b0.
2. **Resampling onto the b0's frame.** The network's output grid must refine its input grid, so the registered blockface volume goes onto the 0.25 mm
   lattice that refines the b0's 0.5 mm frame by 2: a 0.25 mm voxel 2k sits on 0.5 mm voxel k.
   - The grid is the frame cropped to the brain with a 3 mm margin, zero-padded so that the brain keeps at least 4 mm on every side.
   - Grey: the photographs, mapped per brain to [0, 1] by the 1st and 99th percentiles of the photographed tissue.
   - Color: the photographs' RGB on the same grid.
   - Brain mask: the b0's brain mask, resampled by nearest neighbor.
3. **Tissue only the MRI shows.** A longer brainstem in the MRI, for example, has no counterpart in the photographs, so it is removed on the MRI side.
4. **Photographed fraction.** The fraction of each voxel that comes from photographed sections. Interpolated sections and filled marker tubes count as
   not photographed.
5. **Section normal.** The normal of the photographed sections, transformed by the linear part of the registration. A grid rotated so that this normal is
   one of its axes turns every plane parallel to the sections into a grid plane, because the junction plane must be one such plane.
6. **Junction plane** (`configs/completion.yaml`, `clean_face`), on the planes of that rotated grid.
   - A plane is complete when the photographs cover at least 95 % of the brain mask eroded by 1 mm.
   - z_c is the most caudal plane from which every plane over the next 40 mm is complete.
   - The junction plane is z_f = z_c + 1.5 mm. Everything caudal of it is synthesized, including incomplete last sections.
   - For the withheld-slab evaluation, the slab d* holds the complete photographed planes just rostral of z_f whose photographed volume equals the brain
     volume missing caudal of z_f.
7. **The folder.**
   - The b0 is normalized (`input_percentiles`; a stand-in b0 with `stand_in_lo`), masked by the brain and cut to the 0.5 mm box around the brain.
   - The 0.25 mm arrays are cut to the box's 2x refinement.
   - `y_dep(k) = y0 + g . k` maps every 0.25 mm voxel to its plane on the rotated grid.
   - The training targets (`valid`) are the photographed tissue minus each anterior-posterior column's caudal-most 0.25 mm.

## Products (per brain)

`cli/linc-complete -h` lists them (`export.py`): the volume with its synthesized tissue (grey and color), the mapped prediction alone, the photographs where
they are kept, the cross-fade weight, the signed distance from the junction plane, the photographed fraction, the synthesized-voxel mask and the network's
input.
