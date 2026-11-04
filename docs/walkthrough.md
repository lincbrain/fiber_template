# Walkthrough: four brains of the paper

This page runs the pipeline on brains of the paper, from the Haber laboratory's collection, one for each way through the pipeline, with the commands exactly as they ran from a fresh
clone. Each step's results can be compared with the paper's own. The times are from one cluster node with an NVIDIA RTX 6000 Ada
GPU, shared with other jobs.

| Brain | Files | What it shows |
|---|---|---|
| MF1 | `sub-MF283` | the preparation of a brain's MRI |
| MM2 | `sub-MR252` | the MRI-based alignment: reconstruction, registration to its own MRI and to the MRI template, synthesis, histology, labels |
| MF8 | `sub-MF191` | a brain without MRI: the cross-modal alignment, synthesis from a stand-in b0, the blockface-based alignment, histology, labels |
| MM14 | `sub-MR263` | the reconstruction of a brain without MRI |

## Before you start

- The setup in `docs/install.md` (on the lab cluster, `source env/lab.sh`).
- A GPU, for the networks and the deformable registrations. Each GPU step below gives the memory it needs; check the card's free
  memory with `nvidia-smi`, and if the card is busy, choose another one with `CUDA_VISIBLE_DEVICES`.

Every command below writes its results under `LINC_DATA` or `LINC_WORK`. The reconstruction reads `LINC_REBUILD_DATA`, a data folder
that holds only the brains' hand-made inputs, so that it starts from the photographs.

## 1. Prepare a brain's MRI (MF1)

```bash
$PY cli/linc-mri prep --subject MF283 --image $LINC_PRODUCTS/mri/data/sub-MF283/dwi/b0.nii.gz
```

The command takes 23 s: 2 s for the network's brain mask on the GPU and about 4 CPU-minutes for the N4 bias-field correction. It writes, under
`$LINC_DATA/sub-MF283/mri/`:
- `dwi/b0.nii.gz`, the b0 as given;
- `masks/dwi_brain_mask.nii.gz`, the network's brain mask, with a `.json` that names the model and its checksums;
- `dwi/b0_brain.nii.gz` and `dwi/b0_brain_n4.nii.gz` (with its `_bias`), the masked and the bias-corrected brain;
- `dwi/b0_brain_n4.provenance.json`, every step with its tool and version.

It ends by printing the next command for the brain it prepared, its first registration; section 3 shows it for MM2. `docs/mri.md` gives the options.

With the brain's hand-edited mask, `masks/dwi_brain_mask.nii.gz`, it reproduces the paper's preprocessed files of MF1 voxel for voxel
(32 s, on a CPU):

```bash
$PY cli/linc-mri prep --subject MF283 --image $LINC_PRODUCTS/mri/data/sub-MF283/dwi/b0.nii.gz \
    --mask $LINC_PRODUCTS/mri/data/sub-MF283/masks/dwi_brain_mask.nii.gz --mri-root $LINC_WORK/MF283_hand_mask
$PY tools/compare_nifti.py $LINC_WORK/MF283_hand_mask/dwi $LINC_PRODUCTS/mri/data/sub-MF283/dwi \
    b0_brain.nii.gz b0_brain_n4.nii.gz b0_brain_n4_bias.nii.gz
```

The three files were identical to the paper's.

## 2. Reconstruct the blockface volumes (MM2, MM14)

To obtain the 3D volume that every later registration uses, each brain's photographs are reconstructed with its hand-made inputs
(its camera jumps, discontinuities in the stack caused by movement of the camera or specimen holder; the sections to skip; and the
masks corrected by hand), in the data folder `$LINC_REBUILD_DATA` (Before you
start):

```bash
LINC_DATA=$LINC_REBUILD_DATA $PY_TIRL cli/linc-reconstruct all --subject MR252
LINC_DATA=$LINC_REBUILD_DATA $PY_TIRL cli/linc-reconstruct all --subject MR263
```

It runs its steps in order and prints each as it starts: `=== <ID>: photos`, then `stack`, `prepare`, `mask`, `volume`, `tubes` and
`spacing`. In `photos`, the photographs are read and the network makes their brain masks, one line per photograph; `volume` ends with
`=== DONE <ID> in <s>s`. A reconstruction takes about 10 hours per brain. For MM2 (1,440 photographs), the paper's run
(`MR252_timing.tsv` in its blockface folder) took 1.6 hours for `photos`, 1.4 hours for `stack`, 4.9 hours for `prepare` and 2.7
hours for `mask`, `volume` and `tubes`; MM14's (1,401 photographs) took 2.2, 0.8, 4.6 and 0.8 hours. The longest step is
`prepare`, whose camera-jump registrations run one after another. Only the segmentation steps use the GPU: about 10 GB of free memory for `mask` and `tubes`, 1.5 GB for
`photos`. A reconstruction also needs memory (RAM) in proportion to its stack: 88 GB
for MM2 and 85 GB for MM14, 9 GB for MF8's 77 photographs.

Compare a volume with the paper's (here MM2's):

```bash
$PY tools/compare_reconstruction.py MR252 $LINC_REBUILD_DATA/sub-MR252/blockface $LINC_PRODUCTS/data/sub-MR252/blockface
```

It compares every level of the OME-Zarr stores (arrays, scale and translation), the NIfTI products (arrays and affines) and
`MR252.info`, one line each, and exits with status 0 when all are identical. It needs about 7 GB of memory and 12 minutes for
MM2. MM2's volume, rebuilt from its photographs in 9 hours 10 minutes on a shared
node, was identical to the paper's in every item. So was MM14's, whose configuration asks for one extra illumination
correction (`shading: poly2+zref` in `configs/subjects/MR263.yaml`), as the paper's reconstruction of that brain had.

## 3. Register the brains to the template

### MM2, through its own MRI

First the blockface volume to its own b0 (on the GPU, through `ANATOMIX_PYTHON`):

```bash
$PY cli/linc-register mri --subject MR252
```

It takes about 13 minutes on a shared GPU, with 23 GB of GPU memory (15 to 25 GB, depending on the brain). It ends with
`done: $LINC_WORK/register/MR252 | entry $LINC_DATA/sub-MR252/xfm/bf-native__to__b0-native`. It writes the registration's entry, the folder that holds its transform and metadata:
`$LINC_DATA/sub-MR252/xfm/bf-native__to__b0-native/` (the warp `MR252_warp.nii.gz`, its two matrices and `meta.json`), and the
products in `$LINC_WORK/register/MR252/`, among them the blockface volume in the b0 (`MR252_bf_nonlin.nii.gz`) and the
registration's measures (`MR252_metrics.json`). Compare the entry with the paper's:

```bash
for f in MR252_warp.nii.gz b0native_to_placed_pull.txt placed_to_bf_pull.txt; do
  cmp $LINC_DATA/sub-MR252/xfm/bf-native__to__b0-native/$f $LINC_DATA/sub-MR252/xfm_record/bf-native__to__b0-native/$f && echo "$f identical"
done
```

On the GPU model of the paper's runs, all three were identical; in another GPU session or on another model, the warp can differ by
floating point (`docs/reproduce.md`, Exact reproduction).

To catch a failed registration before later steps use it, check it. `qc` checks its linear part (scales and rotations), the
normalized cross-correlation of the two images, the voxels of its warp with a negative Jacobian determinant and its displacement, and the outline of the moved brain against the target's, and records PASS or
FAIL in the registration's entry, which clears the warning "link without passing QC". This check is a safeguard of the software for new brains; the paper
measures accuracy against references the registration had not used, and does not report it. For one of the paper's brains, the outline
target is the b0 mask the registration used (`registration_to_own_mri` in `configs/subjects/<ID>.yaml`); for a new brain, the mask
`--place-b0` wrote (`<ID>_mri_mask.nii.gz`):

```bash
R=$LINC_WORK/register/MR252; D=$LINC_DATA/sub-MR252
$PY cli/linc-register qc --preset bf2b0 --fixed $R/MR252_mri.nii.gz --moved $R/MR252_bf_nonlin.nii.gz --warp $R/MR252_warp.nii.gz \
    --mat $R/linear/MR252_bf2mri_dof9.mat --mat-kind fsl --mat-src $D/blockface/MR252_blockface_iso_grey.nii.gz \
    --mat-ref $R/linear/MR252_b0_masked.nii.gz --outline-target $D/mri/MR252_mri_mask_r1pair_reg_2026-10-03.nii.gz \
    --slab-iso-mask $D/blockface/MR252_mask_iso.nii.gz --entry $D/xfm/bf-native__to__b0-native --name MR252
```

It takes 30 s on a CPU and ends with `outline PASS`.


Then its b0 to the MRI template (on the GPU, through `ANATOMIX_PYTHON`):

```bash
$PY cli/linc-register mri-template --subject MR252 \
    --template-b0 $LINC_TEMPLATE/mri/linc_mri_b0.nii.gz --template-mask $LINC_TEMPLATE/mri/linc_mri_brainmask.nii.gz
```

It takes about 4 minutes (4 min 16 s), with 13 GB of GPU memory (12 to 14 GB, depending on the brain). It writes the registration's entry, `$LINC_DATA/sub-MR252/xfm/b0-native__to__mri-template/`, with its own outline check, so
no `qc` step follows it. The
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

It takes about 5 minutes, with 16 GB of GPU memory. It ends with
`done: $LINC_WORK/direct_placement/MF191 | linear dof9 | entry $LINC_DATA/sub-MF191/xfm/bf-native__to__mri-template`. Compare with
the paper's entry:

```bash
for f in MF191_warp.nii.gz MF191_linear_pull_ras.txt; do
  cmp $LINC_DATA/sub-MF191/xfm/bf-native__to__mri-template/$f $LINC_DATA/sub-MF191/xfm_record/bf-native__to__mri-template/$f && echo "$f identical"
done
```

Both were identical. Run the registrations one at a time on one GPU: two at once need about 40 GB. The command's last stage, on a
CPU, writes `$LINC_DATA/sub-MF191/mri/stand_in/`: the brain aligned to the MRI template's frame and its stand-in b0, which `prepare`
reads in section 4.

Check it the same way. For the cross-modal alignment, the check runs on the part of the template the brain covers, with the
brainstem cut flat, which `direct-placement` writes into its output folder (`_fixed_0.25.nii.gz` and its mask); against the whole
template, the outline would fail where the photographed stack ends:

```bash
P=$LINC_WORK/direct_placement/MF191
$PY cli/linc-register qc --preset direct --fixed $P/_fixed_0.25.nii.gz --moved $P/MF191_bf_in_template.nii.gz --warp $P/MF191_warp.nii.gz \
    --mat $P/MF191_linear_world.txt --mat-kind world --fixed-mask $P/_fixed_mask_0.25.nii.gz \
    --moved-mask $P/MF191_mask_in_template.nii.gz --outline-target $P/_fixed_mask_0.25.nii.gz \
    --entry $LINC_DATA/sub-MF191/xfm/bf-native__to__mri-template --name MF191
```

It takes 19 s on a CPU and ends with `outline PASS`; the anisotropy of the scales is reported for a cross-modal alignment, not checked.

### A brain without MRI, through the blockface template

The paper aligns a brain without MRI to the template through the blockface template (the blockface-based alignment); the cross-modal
alignment above serves its stand-in b0. MF8 is a member of the blockface template, so its route in the paper is its own warp from the
template's construction (section 6). A new brain runs `blockface-template`, then its check, and the chain to the template:

```bash
$PY cli/linc-register blockface-template --subject <ID>
P=$LINC_WORK/register_direct/<ID>
$PY cli/linc-register qc --preset bf2bftemplate --fixed $P/_fixed_0.25.nii.gz --moved $P/<ID>_bf_in_template.nii.gz --warp $P/<ID>_warp.nii.gz \
    --mat $P/<ID>_dof7_world.txt --mat-kind world --fixed-mask $P/_fixed_mask_0.25.nii.gz \
    --moved-mask $P/<ID>_mask_in_template_cut.nii.gz --outline-target $P/_fixed_mask_0.25.nii.gz \
    --entry $LINC_DATA/sub-<ID>/xfm/bf-native__to__bf-template --name <ID>
$PY -m linc.xfm chain --from bf-native --to mri-template --subject <ID>
```

It writes `$LINC_DATA/sub-<ID>/xfm/bf-native__to__bf-template/`. On MF8, as a demonstration, it took 15 minutes with the machine and the
card busy, with 19 GB of GPU memory and 2.4 GB of RAM, and the check passed. There is no record to compare it with, because MF8 is a
member. The two routes are not interchangeable: for MF8, the point each gives in the blockface volume differs by a median of 0.52 mm
(99th percentile 4.0 mm, largest 5.2 mm).

## 4. Synthesize the missing caudal tissue (MM2, MF8)

Each brain in two steps: `prepare` makes its folder from its products (on a CPU), then the networks synthesize (on the GPU).

```bash
$PY_DEEP cli/linc-complete prepare --subject MR252 --out $LINC_WORK/prepared
$PY_DEEP cli/linc-complete --brain $LINC_WORK/prepared/MR252 --out $LINC_WORK/complete
```

The two commands take about 2 minutes on a CPU and 32 s on a GPU, with about 2 GB of its memory. The first prints `MR252: coarse [129, 165, 107] fine [258, 330, 214] z_f 67 d* 20 valid 4777773`, the
second `MR252: synthesized fraction 0.099, junction plane z_f 67`.

MF8 has no MRI, so its folder is prepared from the stand-in b0, the MRI template's b0 transformed into the brain:

```bash
$PY_DEEP cli/linc-complete prepare --subject MF191 --out $LINC_WORK/prepared
$PY_DEEP cli/linc-complete --brain $LINC_WORK/prepared/MF191 --stand-in --out $LINC_WORK/complete
```

They take about 40 s and 27 s.

Each brain's results are in `$LINC_WORK/complete/<ID>/`: the volume with its synthesized tissue (`<ID>_bf_completed.nii.gz`, and `_rgb` in color), the
synthesized and the photographed parts (`_bf_synth`, `_bf_real`), the junction (`_seam_ramp`, `_seam_depth`), the photographed
fraction of each voxel (`_photo_fraction`), the synthesized mask, the b0 used, the color model (`<ID>_chroma.json`) and the
provenance. Compare them with the paper's volumes (here MM2's; for MF8, `MF191` in place of `MR252`):

```bash
$PY tools/compare_nifti.py $LINC_WORK/complete/MR252 $LINC_PRODUCTS/synthesis/out_d3_full/MR252 \
    MR252_bf_completed.nii.gz MR252_bf_completed_rgb.nii.gz MR252_bf_synth.nii.gz MR252_synth_mask.nii.gz
```

Every output volume of both brains was identical, voxel for voxel, to the paper's.

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

It prints each step as it starts and ends (`match`; `pieces`, which divides each section into components; `masks`, `prepare`, `register`, `rgb`), one line per component while it
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
R=$LINC_PRODUCTS/data_arm_final/sub-MR252/blockface/06_histo2bf/tirl
for d in $LINC_DATA/sub-MR252/histology/registration/tirl/*/; do p=$(basename $d)
  cmp -s $d/moving4_nonlinear.png $R/$p/moving4_nonlinear.png && echo "$p identical" || echo "$p differs"
done
```

All 47 of MM2's components were identical, byte for byte, at every stage.

For MF8, the paper's registration divided its sections with its own regions volume, made at a section step of 600 µm. Run with it, at
that spacing,
and into a folder of its own:

```bash
$PY_TIRL cli/linc-histology all --subject MF191 \
    --store $LINC_DATA/sub-MF191/blockface/MF191_blockface.ome.zarr \
    --histology $LINC_RAW/linc_000003/sourcedata/sub-MF191/micr \
    --table $LINC_PRODUCTS/labels/match_tables/num_MF191.txt \
    --regions $LINC_PRODUCTS/labels/regions_t2_mf191_made_at_vz600/MF191_regions_t2.nii.gz --regions-spacing-um 23.75 600 \
    --out $LINC_WORK/mf191_hidden --threads 12 --jobs 2
```

The `--out` folder must be new: a folder that already holds a registration is resumed, and every component with results is
skipped. About 18 minutes for the masks, 5 minutes to prepare and 75 minutes to register the 58 components, two at a time. Compare
them with the paper's:

```bash
R=$LINC_PRODUCTS/data_arm_final/sub-MF191/blockface/06_histo2bf/tirl
for d in $LINC_WORK/mf191_hidden/tirl/*/; do p=$(basename $d)
  cmp -s $d/moving4_nonlinear.png $R/$p/moving4_nonlinear.png && echo "$p identical" || echo "$p differs"
done
```

All 58 of MF8's components, with their masks, were identical to the paper's, byte for byte.

## 6. Transform the labels into the template (MM2)

Four steps, each with the Python of its environment: the labels go through each section's registration onto the blockface, are
interpolated between the sections and are transformed into the template, and the bundles are joined into one connected pathway.

```bash
R=$LINC_PRODUCTS/data_arm_preinp/sub-MR252/blockface/06_histo2bf/tirl
$PY_TIRL cli/linc-labels push        --subject MR252 --registrations $R
$PY      cli/linc-labels interpolate --subject MR252 --registrations $R
$PY      cli/linc-labels carry       --subject MR252 \
    --entry $LINC_DATA/sub-MR252/xfm_record/bf-native__to__b0-native --entry $LINC_DATA/sub-MR252/xfm_record/b0-native__to__mri-template
$PY      cli/linc-labels connect     --subject MR252
```

The paper's labels went through the histology registration made on the photographs before the marker tubes' holes were filled,
so `push` and `interpolate` are given that registration
(`--registrations`); without it, they use your own from section 5. The four steps take about 3 minutes, 30 minutes, 3 minutes and 15 seconds. `carry`, as written, goes through the paper's registrations of MM2 (`xfm_record/`);
without `--entry` it goes through your own, from section 3 (`xfm/`). The interpolation writes the interpolated labels under
`$LINC_DATA/sub-MR252/labels/blockface/tracer/`. Compare each step's labels with the paper's (the gzip bytes differ, so the arrays
are compared):

```bash
D=$LINC_DATA/sub-MR252/labels
P=$LINC_PRODUCTS/labels
# push: the 6 labels drawn on the sections (the bundle classes, Fibers, the injection site), on the blockface grid
$PY tools/compare_nifti.py $D/blockface/tracer $P/push_tracer_r058d/MR252 $(cd $P/push_tracer_r058d/MR252 && ls MR252_*_bfnative_L2.nii.gz)
# interpolate: the 7 interpolated labels (those 6 and the bundle group)
$PY tools/compare_nifti.py $D/blockface/tracer $P/tracer_v2/MR252 $(cd $P/tracer_v2/MR252 && ls MR252_*_bfnative_L2_demons.nii.gz)
# carry: the 21 files in the template (7 labels, each as a mask, a density and a footprint)
$PY tools/compare_nifti.py $D/template/tracer $P/tracer_template_v2/MR252 \
    $(cd $P/tracer_template_v2/MR252 && ls MR252_*_mri-template.nii.gz MR252_*_mri-template_density.nii.gz MR252_*_mri-template_footprint.nii.gz)
# connect: the connected pathway, its solid label and its joins
$PY tools/compare_nifti.py $D/template/tracer $P/tracer_template_v2c/MR252 \
    MR252_Bundles_mri-template_connected.nii.gz MR252_Bundles_mri-template_solid_v2c.nii.gz MR252_Bundles_mri-template_bridges.nii.gz
```

All 6 labels on the sections, 7 interpolated labels and 21 template files were identical to the paper's. `connect` was run from the
paper's carried densities, and its 3 files were identical.

MF8's labels go through its warp from the construction of the blockface template, then the registration of that template to the MRI
template. The transform is valid in photographed tissue only:

```bash
E=$LINC_DATA/sub-MF191/xfm_record/bf-native__to__bf-template
$PY cli/linc-labels carry --subject MF191 --out $LINC_WORK/labels_mf8 \
    --labels $LINC_PRODUCTS/labels/tracer_v2/MF191/MF191_*_bfnative_L2_demons.nii.gz \
    --map $LINC_TEMPLATE/xfm/bf-template__to__mri-template:full \
    --warp $E/MF191_d2bs8_warp.nii.gz:fsl_rel:$E/src_grid.nii.gz --ras-pull $E/MF191_rigid_pull_ras.txt
P=$LINC_PRODUCTS/labels/tracer_template_v2/MF191
$PY tools/compare_nifti.py $LINC_WORK/labels_mf8 $P \
    $(cd $P && ls MF191_*_mri-template.nii.gz MF191_*_mri-template_density.nii.gz MF191_*_mri-template_footprint.nii.gz)
```

The first command takes about 1 minute on a CPU with 1.8 GB of RAM, and about 5 minutes on a loaded node. All 30 files were identical to
the paper's.
