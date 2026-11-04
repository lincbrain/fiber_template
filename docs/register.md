# Register a brain into the template's frame

**You need** the brain's blockface volume (`docs/reconstruct.md`), its prepared b0 if it has MRI (`docs/mri.md`), and the template.

**What it does**
1. with MRI: registers the blockface volume to the brain's own b0, then the b0 to the MRI template;
2. without MRI: registers the blockface volume directly onto the MRI template's b0.

**You get** each registration as one entry, `$LINC_DATA/sub-<ID>/xfm/<from>__to__<to>/` (the warp, its matrices and `meta.json`),
and its products and measures under `$LINC_WORK/`.

A brain with MRI and a brain without MRI are registered differently:

```bash
# the MRI-based alignment, for a brain with MRI: two registrations
$PY cli/linc-register mri --subject MR252          # blockface -> the brain's own b0 (env/linc.yml, FSLDIR, ANATOMIX_PYTHON)
$PY cli/linc-register mri-template --subject MR252 --template-b0 $LINC_TEMPLATE/mri/linc_mri_b0.nii.gz \
    --template-mask $LINC_TEMPLATE/mri/linc_mri_brainmask.nii.gz                               # its b0 -> the MRI template (env/linc.yml)

# the cross-modal alignment, for a brain without MRI: one registration onto the MRI template's b0 (env/linc.yml)
$PY cli/linc-register direct-placement --subject MF191 --template-b0 $LINC_TEMPLATE/mri/linc_mri_b0.nii.gz \
    --template-mask $LINC_TEMPLATE/mri/linc_mri_brainmask.nii.gz
```

Each runs on a GPU, and its memory grows with the brain's working grid. `mri` needs 15 to 25 GB, depending on the brain: MM2,
MM5 and MM8 need more than a 24 GB card has free. `mri-template` needs 12 to 14 GB, and `direct-placement` 15 to 17 GB. Each prints its need and
the card's free memory before its GPU stage, and stops if the card is short; then choose another card with `CUDA_VISIBLE_DEVICES`,
or check the cards first with `nvidia-smi`. Run one registration at a time on a card.

The MRI template's b0 channel and brain mask come with the released template, under `$LINC_TEMPLATE/mri/`. For a brain that is not one of the
paper's (`cli/linc-mri prep`, `docs/mri.md`, makes its b0 and brain mask):
- align its b0 to the template's frame first, with `--place-b0 <b0>` on the first `linc-register mri` run. No hand mask is needed:
  the MRI brain-mask network (`linc-segment`'s `mri_brainmask`, in `$LINC_MODELS/segment/mri_brainmask`) makes the b0's mask,
  run with `$LINC_SEGMENT_PYTHON`. To use your own mask instead, give `--place-b0 <b0> <brain mask>`;
- add `--new-brain`, which estimates its starting pose.

For the paper's brains, the b0 on the template's frame comes with the data release. The b0 → template step also reads the frame bridge named in
`configs/link.yaml` (`inputs.bridge`, under `$LINC_PRODUCTS`).

Each registration records its transform in the brain's registry, `$LINC_DATA/sub-<ID>/xfm/`; `--registry` changes the folder. An
existing entry stops the run before the GPU stage unless `--overwrite` is given, which replaces it and backs up the old one. The
products go to `--out`. Later steps find the transforms there:

```bash
$PY -m linc.xfm chain --from bf-native --to mri-template --subject MR252   # the links from the blockface volume to the template
```

The registry warns "link without passing QC" for entries written by `linc-register`, and `--strict` refuses them: the
registrations of a new brain are unchecked until they are checked. The released entries of the paper's 21 brains hold the result
of the paper's quality checks (all passed).

## Check a registration

`cli/linc-register qc` runs the paper's quality checks on one registration: the scales and rotations of its linear part, the
image correlation, the folds and the drift of its warp, the overlap of the two brain masks, and the distance between the moved
brain's outline and its target's. It writes a three-plane PNG and a JSON row (`--out`, by default `$LINC_WORK/qc`), and with
`--entry` it records pass or fail in the registration's registry entry, which then stops the warning. A preset sets the paper's
settings for each kind of registration: `bf2b0` (blockface to its own b0), `direct` (blockface to the MRI template), `b02template`
(b0 to the MRI template) and `template2template` (the blockface template to the MRI template). `--help` lists the inputs; the thresholds are in
`configs/registration.yaml` (`qc`). On three of the paper's registrations, its measures equal the paper's to 1e-6, with the same
outcome.

For a brain registered to its own b0 by `linc-register mri --subject <ID>`, with its products in `$LINC_WORK/register/<ID>/`:

```bash
R=$LINC_WORK/register/<ID>
$PY cli/linc-register qc --preset bf2b0 \
    --fixed $R/<ID>_mri.nii.gz --moved $R/<ID>_bf_nonlin.nii.gz --warp $R/<ID>_warp.nii.gz \
    --mat $R/linear/<ID>_bf2mri_dof9.mat --mat-kind fsl \
    --mat-src $LINC_DATA/sub-<ID>/blockface/<ID>_blockface_iso_grey.nii.gz --mat-ref $R/linear/<ID>_b0_masked.nii.gz \
    --outline-target $LINC_DATA/sub-<ID>/mri/<ID>_mri_mask.nii.gz \
    --slab-iso-mask $LINC_DATA/sub-<ID>/blockface/<ID>_mask_iso.nii.gz \
    --entry $LINC_DATA/sub-<ID>/xfm/bf-native__to__b0-native --name <ID>
```

The outline target is the b0's brain mask: the one on the template's frame, as here, or, for one of the paper's brains, its mask of record. This
command ran on the paper's registration of MM9 (`sub-MR257`, from the folders of the paper's products), in 25 s on a CPU, and
printed, in part (with the run's name shortened):

```
PASS: MR257 -> $LINC_WORK/qc/MR257.json, $LINC_WORK/qc/MR257.png
  NCC -0.608  NMI 1.054  envDice 0.968
  rot xyz +3.1 +1.2 -1.3 deg (absorbed none)  scales 1.006 1.014 1.024
  det J min 0.242 max 4.06 std 0.185  folds 0  drift p95 1.2 mm
  outline PASS
```

A failed check names its reason, for example `FAIL: rotation y/z -0.0/45.0 deg >= 30.0` for a linear part rotated 45° about one
axis, and records "fail" in the entry when `--entry` is given.
