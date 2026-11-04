# linc/histology

Histology sections -> their blockface photographs -> the template, and annotations along the same path. Entry points:
`cli/linc-histology` (match, pieces, masks, prepare, register, rgb) and `cli/linc-labels` (push, interpolate, carry). Parameters:
`configs/histology.yaml`, `configs/labels.yaml`. Environments: `env/tirl.yml` (TIRL 3.6.2; 12 threads per piece reproduce the
registrations of record bit for bit), `env/labels-interp.yml` (the demons interpolation).

## The atlas regions volume (input of `linc-histology pieces`)

**What it is.** One label volume on the brain's blockface grid (the reconstruction's world frame), uint8:
1 = left hemisphere, 2 = right hemisphere, 3 = cerebellum + brainstem, 4 = marker tubes (drilled in the left hemisphere; counted as
left). `linc-histology pieces` samples it on every matched photograph to cut the photograph into the pieces the sections were cut
into (whole section where the hemispheres are joined, left / right hemisphere, cerebellum + brainstem), restricted to the photograph's
tissue; unlabeled tissue takes the nearest label, and two physically separate side-by-side lobes are split geometrically.

**How the paper's volumes were made** (the analysis project's `labels/regions_via_t2.py` and `labels/regions3d.py`; one volume per
histology brain):
1. Atlas regions in NMT v2 (symmetric) space, from three files of the NMT v2 distribution:
   - 1 left / 2 right: `supplemental_masks/NMT_v2.0_sym_LR_brainmask.nii.gz` (its two values; left = the value whose voxels lie on the
     -x, i.e. anatomical left, side);
   - 3 cerebellum + brainstem, inside that brain mask: `supplemental_masks/NMT_v2.0_sym_cerebellum_mask.nii.gz` (> 0) united with the
     SARM level-1 labels 162 (mesencephalon), 204 (metencephalon) and 279 (myelencephalon) of
     `supplemental_SARM/SARM_1_in_NMT_v2.0_sym.nii.gz` (values from `SARM_key_1.txt`).
2. Template level, once: NMT v2 was registered to the template's T2 channel (ANTs affine, then the feature-based registration of
   `linc/registration`) and the regions were resampled with nearest neighbor: **`regions_in_t2.nii.gz`** (uint8, 0-3, the 0.25 mm
   template grid; released with the regions of record). It was made on the template of 27 Sep 2026, before the final rebuild of the
   MRI channels (same grid; the anatomy differs slightly).
3. Brain level: the template's T2 (moving) was registered to the brain's masked grayscale blockface volume (fixed): FLIRT 12 degrees of
   freedom for the start, the template cropped to the blockface's own anterior-posterior extent (the photographed stacks end short of
   the caudal pole), then the feature-based registration. The same transform carried `regions_in_t2.nii.gz` into the blockface
   (nearest neighbor); the brain's tube mask (`<ID>_holes`) was written as label 4.
4. These volumes are in each brain's spacing frame of 27 Sep (before the 2 Oct spacing calibration): `linc-histology pieces` scales
   them by the brain's own factors (`configs/histology.yaml` `regions.made_at_spacing_um`; validated: the pieces of record are
   reproduced exactly for MM2 and MF5).

**For a new brain.** Carry the template-space regions into the brain's blockface grid through the brain's own registration to the
template (from `cli/linc-register`, which writes its transforms to the registry `$LINC_DATA/sub-<ID>/xfm/`), with nearest-neighbor
resampling, in one line (the registry finds the chain, mri-template -> bf-native, and inverts the links it needs):

    python -m linc.xfm apply --from mri-template --to bf-native --subject <ID> --nearest \
        --ref $LINC_DATA/sub-<ID>/blockface/<ID>_blockface_L2.nii.gz \
        $LINC_TEMPLATE/labels/regions_in_t2.nii.gz $LINC_DATA/sub-<ID>/histology/<ID>_regions.nii.gz

(`python -m linc.xfm chain --from mri-template --to bf-native --subject <ID>` prints the links it will use.)

Then write the brain's tube mask as label 4 where it falls in the left hemisphere, and pass the result to `linc-histology pieces
--regions ...`. A new brain needs no flag: its volume is made on its current frame. A volume made now for one of the paper's
brains passes that frame's spacing with `--regions-spacing-um VX VZ` (the brain's vx and vz). This carries the regions through the brain's
registration to the template, whereas the paper registered the T2 template to each blockface directly (step 3): check the pieces on a
few photographs before registering.

## The test of the registration settings (Supplementary S4, Table S7)

Eight single changes to the TIRL settings were tested against the settings of record (C0) on the same 284 tube-bearing pieces of
the six histology brains, scored by the offset of the hidden marker tubes and by tissue Dice, under a decision rule fixed before
the runs. None was adopted. `configs/histology_sweep.yaml` holds each setting as its change to `configs/histology.yaml`:

| setting | change (record) | key under `tirl.config` |
|---|---|---|
| C0 | none: the settings of record | |
| C1 | regularization weight 0.15 (0.30) | `regparams.nonlinear.regweight` |
| C2 | regularization weight 0.60 (0.30) | `regparams.nonlinear.regweight` |
| C3 | affine scaling bounds 0.90-1.10 (0.95-1.05) | `regparams.init.affine.lb`, `.ub` |
| C4 | rotation search step 2 degrees (5) | `regparams.rotsearch.coarse` |
| C5 | tissue masks in the similarity: automask threshold 0.02 on both images (none) | `histology.mask.automask`, `block.mask.automask` |
| C6 | MIND neighborhood truncated at 3.0 (1.5) | `regparams.nonlinear.truncate` |
| C8 | regularization weight 0.40 (0.30), chosen after the C1-C6 result | `regparams.nonlinear.regweight` |
| C9 | regularization weight 0.45 (0.30), chosen after the C1-C6 result | `regparams.nonlinear.regweight` |

To register one piece with one setting:

    python -m linc.histology.sweep C2 > histology_C2.yaml
    cli/linc-histology register --config histology_C2.yaml --out <folder> --pieces <piece>

With 12 threads per piece this reproduces the test's transforms exactly (checked for C2 on one MF5 piece, 1 Nov 2026). The test's
moving images were made from tissue masks with the corrected masking step on every piece. This release's `masks` step makes the
original masks (the correction is not in this snapshot yet), so pass corrected masks to `prepare` with `--masks`. The fixed images
are those of record (the blockface with the marker tubes inpainted).

The results are the reference records `tirl_sweep_r058` (C0-C6) and `tirl_sweep_r058b` (C0-C6, C8, C9) in `reproduce/reference`.
This release does not regenerate them: their scoring reads the test's 2,556 TIRL outputs (284 pieces x 9 settings), which are not
released (`configs/numbers_exceptions.yaml`).

## Known issue (TIRL 3.6.2)

The affine stage can stop with nlopt's `ValueError: bounds 0 fail -1 <= -1 <= 1`. When a diagonal entry of the affine matrix ends a
coarse level exactly on its bound (0.95 or 1.05, `configs/histology.yaml`), its normalized start for the next level can fall a
round-off error below -1, and BOBYQA refuses that start. No registration of record failed this way. At least one piece of record
(MF8) sits on the bound at every affine level and escaped only by round-off, and with wider bounds (C3 of the test above) the same
piece failed. The fix is to clip the normalized start into [-1, 1] in TIRL's `OptNL.__call__` (`tirl/optimisation/optnl.py`), just
before `self.opt.optimize(x0)`. `env/tirl.yml` does not apply it, so the registrations of record are reproduced unchanged.
