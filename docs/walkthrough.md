# Walkthrough: four brains of the paper

This page runs the pipeline on brains of the paper, one for each way through the pipeline, with the commands exactly as they ran from a fresh
clone. Each step's results can be compared with the paper's own. The times are from one cluster node with an NVIDIA RTX 6000 Ada
GPU, shared with other jobs.

| Brain | Files | What it shows |
|---|---|---|
| MF1 | `sub-MF283` | the preparation of a brain's MRI |
| MM2 | `sub-MR252` | the MRI-based alignment: reconstruction, registration to its own MRI and to the MRI template, synthesis, histology, labels |
| MF8 | `sub-MF191` | a brain without MRI: the cross-modal alignment, synthesis from a stand-in b0, histology |
| MM14 | `sub-MR263` | the reconstruction of a brain without MRI |

## Before you start

- The setup in `docs/install.md` (on the lab cluster, `source env/lab.sh`).
- A GPU, for the networks and the deformable registrations.

Every command below writes its results under `LINC_DATA` or `LINC_WORK`. The reconstruction reads `LINC_REBUILD_DATA`, a data folder
that holds only the brains' hand-made inputs, so that it starts from the photographs.

## 1. Prepare a brain's MRI (MF1)

```bash
$PY cli/linc-mri prep --subject MF283 --image $LINC_PRODUCTS/mri/data/sub-MF283/dwi/b0.nii.gz
```

23 s: the network's brain mask in 2 s on the GPU, then the bias-field correction (N4, about 4 CPU-minutes). It writes, under
`$LINC_DATA/sub-MF283/mri/`:
- `dwi/b0.nii.gz`, the b0 as given;
- `masks/dwi_brain_mask.nii.gz`, the network's brain mask, with a `.json` that names the model and its checksums;
- `dwi/b0_brain.nii.gz` and `dwi/b0_brain_n4.nii.gz` (with its `_bias`), the masked and the bias-corrected brain;
- `dwi/b0_brain_n4.provenance.json`, every step with its tool and version.

It ends by printing the next command, the first registration (section 3). With the brain's hand-made mask instead
(`--mask <mask>`), it reproduces the paper's preprocessed files of MF1 voxel for voxel. `docs/mri.md` gives the options.

## 2. Reconstruct the blockface volumes (MM2, MM14)

Each volume is rebuilt from the brain's photographs and its hand-made inputs, in the data folder `$LINC_REBUILD_DATA` (Before you
start):

```bash
LINC_DATA=$LINC_REBUILD_DATA $PY_TIRL cli/linc-reconstruct all --subject MR252
LINC_DATA=$LINC_REBUILD_DATA $PY_TIRL cli/linc-reconstruct all --subject MR263
```

It runs its steps in order and prints each as it starts: `=== <ID>: photos`, then `stack`, `prepare`, `mask`, `volume`, `tubes` and
`spacing`. In `photos`, the photographs are read and the network makes their brain masks, one line per photograph; `volume` ends with
`=== DONE <ID> in <s>s`. On one node, reading the photographs took about 1.5 s each, their masks 0.8 to 4 s each on a shared GPU, and
`prepare` about 30 minutes for about 1,400 photographs. The whole reconstruction of MM2 (1,440 photographs) or MM14 (1,401) takes
about 2.5 hours.

Compare a volume with the paper's (here MM2's):

```bash
$PY tools/compare_reconstruction.py MR252 $LINC_REBUILD_DATA/sub-MR252/blockface $LINC_PRODUCTS/data/sub-MR252/blockface
```

It compares every level of the OME-Zarr stores (arrays, scale and translation), the NIfTI products (arrays and affines) and
`MR252.info`, one line each, and exits with status 0 when all are identical. The results of this comparison for MM2 and MM14 are
not yet on this page.

MF8's volume of record was rebuilt on 24–26 Sep 2026 with an earlier padding of its canvas. The release reproduces its aligned
photographs, brain mask and tubes exactly, and its volume up to a shift of the canvas by less than a voxel (its origin moves by 0.5
and 2 µm; 26 % of the voxels differ, by a mean of 0.5 on a scale of 255).

## 3. Register the brains into the template's frame

### MM2, through its own MRI

First the blockface volume to its own b0 (on the GPU, through `ANATOMIX_PYTHON`):

```bash
$PY cli/linc-register mri --subject MR252
```

About 13 minutes on a shared GPU, with 23 GB of GPU memory. It ends with
`done: $LINC_WORK/register/MR252 | entry $LINC_DATA/sub-MR252/xfm/bf-native__to__b0-native`. It writes the registration's entry,
`$LINC_DATA/sub-MR252/xfm/bf-native__to__b0-native/` (the warp `MR252_warp.nii.gz`, its two pull matrices and `meta.json`), and the
products in `$LINC_WORK/register/MR252/`, among them the blockface volume in the b0 (`MR252_bf_nonlin.nii.gz`) and the
registration's measures (`MR252_metrics.json`). Compare the entry with the paper's:

```bash
for f in MR252_warp.nii.gz b0native_to_placed_pull.txt placed_to_bf_pull.txt; do
  cmp $LINC_DATA/sub-MR252/xfm/bf-native__to__b0-native/$f $LINC_DATA/sub-MR252/xfm_record/bf-native__to__b0-native/$f && echo "$f identical"
done
```

On the GPU model of the paper's runs, all three were identical; in another GPU session or on another model, the warp can differ by
floating point (`docs/reproduce.md`, Exact reproduction).


Then its b0 to the MRI template (on the GPU, through `ANATOMIX_PYTHON`):

```bash
$PY cli/linc-register mri-template --subject MR252 \
    --template-b0 $LINC_TEMPLATE/mri/linc_mri_b0.nii.gz --template-mask $LINC_TEMPLATE/mri/linc_mri_brainmask.nii.gz
```

About 4 minutes (4 min 16 s). It writes the registration's entry, `$LINC_DATA/sub-MR252/xfm/b0-native__to__mri-template/`. The
paper's entry sits beside it, in `xfm_record/`; the two differ by floating point only (`docs/reproduce.md`, Exact reproduction).
The chain from the blockface volume to the template then reads:

```bash
$PY -m linc.xfm chain --from bf-native --to mri-template --subject MR252
```

### MF8, without MRI

One registration, the blockface volume directly to the MRI template's b0 (on the GPU):

```bash
$PY cli/linc-register direct-placement --subject MF191 \
    --template-b0 $LINC_TEMPLATE/mri/linc_mri_b0.nii.gz --template-mask $LINC_TEMPLATE/mri/linc_mri_brainmask.nii.gz
```

About 5 minutes, with 16 GB of GPU memory. It ends with
`done: $LINC_WORK/direct_placement/MF191 | linear dof9 | entry $LINC_DATA/sub-MF191/xfm/bf-native__to__mri-template`. Compare with
the paper's entry:

```bash
for f in MF191_warp.nii.gz MF191_linear_pull_ras.txt; do
  cmp $LINC_DATA/sub-MF191/xfm/bf-native__to__mri-template/$f $LINC_DATA/sub-MF191/xfm_record/bf-native__to__mri-template/$f && echo "$f identical"
done
```

Both were identical. Run the registrations one at a time on one GPU: two at once need about 40 GB.

## 4. Synthesize the missing caudal tissue (MM2, MF8)

Each brain in two steps: `prepare` makes its folder from its products (on a CPU), then the networks synthesize (on the GPU).

```bash
$PY_DEEP cli/linc-complete prepare --subject MR252 --out $LINC_WORK/prepared
$PY_DEEP cli/linc-complete --brain $LINC_WORK/prepared/MR252 --out $LINC_WORK/complete
```

About 2 minutes, then 32 s. The first prints `MR252: coarse [129, 165, 107] fine [258, 330, 214] z_f 67 d* 20 valid 4777773`, the
second `MR252: predicted share 0.099, cut z_f 67`.

MF8 has no MRI, so its folder is prepared from the stand-in b0, the MRI template's b0 transformed into the brain:

```bash
$PY_DEEP cli/linc-complete prepare --subject MF191 --out $LINC_WORK/prepared --nonlin '{mri}/nonlinear_s2lin_prod' \
    --matrix '{mri}/s2lin_prod/{ID}_bf2mri_dof9.mat' --brain-mask '{mri}/{ID}_pseudo_brain_n1_0p5_prod.nii.gz' \
    --stand-in-b0 '{nonlin}/{ID}_mri_full.nii.gz' --lo0
$PY_DEEP cli/linc-complete --brain $LINC_WORK/prepared/MF191 --stand-in --out $LINC_WORK/complete
```

About 40 s, then 27 s. Keep the single quotes: `{mri}` stands for `$LINC_DATA/sub-<ID>/mri`, `{nonlin}` for the `--nonlin` folder and
`{ID}` for the brain, and the command fills them in.

Each brain's results are in `$LINC_WORK/complete/<ID>/`: the volume with its synthesized tissue (`<ID>_bf_completed.nii.gz`, and `_rgb` in color), the
synthesized and the photographed parts (`_bf_synth`, `_bf_real`), the junction (`_seam_ramp`, `_seam_depth`), the photographed
fraction of each voxel (`_photo_fraction`), the synthesized mask, the b0 used, the color model (`<ID>_chroma.json`) and the
provenance. Every output volume of both brains is identical, voxel for voxel, to the volumes the paper used.

## 5. Register the histology (MM2, MF8)

Each section of histology is matched to its blockface photograph, divided into its components (the hemispheres, the cerebellum
with the brainstem), masked, and registered to the photograph with TIRL:

```bash
$PY_TIRL cli/linc-histology all --subject MR252 --store $LINC_DATA/sub-MR252/blockface/MR252_blockface.ome.zarr \
    --histology $LINC_RAW/linc_000003/sourcedata/sub-MR252/micr --table $LINC_PRODUCTS/labels/match_tables/num_MR252_plus.txt \
    --regions $LINC_PRODUCTS/labels/regions_t2_final/MR252_regions_t2.nii.gz --out $LINC_DATA/sub-MR252/histology/registration \
    --threads 12 --jobs 2
```

For MF8, the same command with `MF191` in place of `MR252` and the match table `num_MF191.txt`.

It prints each step as it starts and ends (`match`, `pieces`, `masks`, `prepare`, `register`, `rgb`), one line per component while it
registers, and at the end, for MM2, `TIRL histology -> blockface: 47 ok, 0 failed`. On a shared 48-core node, MM2's 47 components took
about 40 minutes for the masks, 6 minutes to prepare, 80 minutes to register two at a time and 5 minutes for the color images:
about 2 hours 10 minutes in all. MF8's 58 components took about 45, 5, 75 and 7 minutes. Each component uses 12 threads, so `--jobs 2` needs 24 cores, and the masks about 12 GB of memory per job. The
thread count is what makes the result reproducible: run with 12 threads, as the paper did.

A run that stops resumes from its failed step, keeping the earlier steps' outputs: `$PY_TIRL cli/linc-histology register` with the
same arguments, then `rgb`. The pipeline always uses the TIRL of its environment, even where FSL, which brings an older TIRL of its own, is set up; to call
TIRL yourself, use `$(dirname $PY_TIRL)/tirl`.

The results are in `$LINC_DATA/sub-<ID>/histology/registration/`: the matched sections, the components and their tissue masks,
and per component in `tirl/<piece>/` the registered section (`moving4_nonlinear.png`, in color `moving4_nonlinear_rgb.png`) and TIRL's final
transform (`fixed4_nonlinear.timg`).

Compare MM2's registered sections with the paper's:

```bash
for p in $(ls $LINC_DATA/sub-MR252/histology/registration/tirl); do
  cmp -s $LINC_DATA/sub-MR252/histology/registration/tirl/$p/moving4_nonlinear.png \
         $LINC_PRODUCTS/data_arm_final/sub-MR252/blockface/06_histo2bf/tirl/$p/moving4_nonlinear.png && echo "$p identical" || echo "$p differs"
done
```

All 47 of MM2's components were identical, byte for byte, at every stage.

For MF8, the paper's registration divided its sections with an earlier regions volume, made before MF8's section step was corrected.
That volume ships as the product `hidden_arm_regions` (`configs/products_blockface.yaml`). Run with it, at the spacing it was made at,
all 58 of MF8's components are identical to the paper's, byte for byte:

```bash
$PY_TIRL cli/linc-histology all --subject MF191 <the other arguments, as for MM2> \
    --regions <the hidden_arm_regions file> --regions-spacing-um 23.75 600
```

This took 18 minutes for the masks, 5 minutes to prepare and 27 minutes to register the 31 components it divides, two at a time.

## 6. Transform the labels into the template (MM2)

Three steps, each with the Python of its environment: the labels go through each section's registration onto the blockface, are
interpolated between the sections, and are transformed into the template.

```bash
$PY_TIRL cli/linc-labels push        --subject MR252
$PY      cli/linc-labels interpolate --subject MR252
$PY      cli/linc-labels carry       --subject MR252 \
    --entry $LINC_DATA/sub-MR252/xfm_record/bf-native__to__b0-native --entry $LINC_DATA/sub-MR252/xfm_record/b0-native__to__mri-template
```

About 3 minutes, half an hour and 3 minutes. `carry`, as written, goes through the paper's registrations of MM2 (`xfm_record/`);
without `--entry` it goes through your own, from section 3 (`xfm/`). The interpolation writes the interpolated labels under
`$LINC_DATA/sub-MR252/labels/blockface/tracer/`. Each step reproduces the paper's labels: the pushed sections and the interpolated
labels byte for byte, and the densities in the template identical.
