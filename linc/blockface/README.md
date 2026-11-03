# Blockface reconstruction (`linc/blockface`, `cli/linc-reconstruct`)

Turns a brain's blockface photographs into the 3D color volume the template is built from (Methods, "Blockface
reconstruction"; Fig. 3). The code is the paper's pipeline, ported unchanged; the parameters are in `configs/blockface.yaml`,
each brain's inputs in `configs/subjects/<ID>.yaml`.

## Usage

```
linc-reconstruct all --subject <ID>
```

or step by step:

| Step | Command | What it does (paper step) |
|---|---|---|
| photos | `linc-reconstruct photos --subject <ID>` | renames the photographs into a section series and segments the brain in each with the 2D nnU-Net (1) |
| stack | `linc-reconstruct stack --subject <ID>` | stacks the photographs and their masks |
| prepare | `linc-reconstruct prepare --subject <ID>` | corrects the camera jumps with TIRL (2), marks and pre-fills missing sections, drops small 3D components, crops, corrects illumination and brightness, exports the background-intact volume |
| mask | `linc-reconstruct mask --subject <ID>` | segments the brain in 3D with an nnU-Net (4) |
| volume | `linc-reconstruct volume --subject <ID>` | the reconstruction: interpolates missing sections by demons registration (3), corrects illumination, brightness steps and color (5), fills the marker tubes (6) |
| tubes | `linc-reconstruct tubes --subject <ID>` | segments the marker tubes in 3D with an nnU-Net (6) and re-fills them if they changed |
| spacing | `linc-reconstruct spacing --subject <ID>` | writes the calibrated spacing into the headers |

Outputs go to `$LINC_DATA/sub-<ID>/blockface/`: `<ID>_blockface.ome.zarr` (in-plane levels L1 to L4, native section step),
`<ID>_mask.nii.gz`, `<ID>_holes.nii.gz` (L4), `<ID>.info`, and QC images. `cli/linc-register` reads them.

## Environments

- The Python steps: `env/tirl.yml` (TIRL 3.6.2 is also imported to read the camera-jump registrations back).
- nnU-Net 2.5.1: `env/nnunet.yml`; set `NNUNET_BIN` to its bin folder and `LINC_MODELS` to the folder holding
  `nnUNet_results/` (`models/blockface_segmentation.md`, `models/verify_weights.py`).
- `tirl` (with tirlscripts-mgh-linc 1.0.1): `TIRL_BIN`; zarr2nii (nifti-zarr 1.0.0rc6): installed in `env/tirl.yml` (found on PATH; `ZARR2NII_BIN` points at another installation).
- `LINC_RAW`: the root the photograph folders in `configs/subjects/<ID>.yaml` (`raw`) are relative to.

Each external tool's version is checked before it runs (`linc/blockface/external.py`).

## A new brain

Copy `configs/subjects/TEMPLATE.yaml` to `configs/subjects/<ID>.yaml`; every field is explained there (required or optional,
units, defaults). Fill in the photographs (`raw`, `raw_glob`, `raw_regex`, `idx_range`, `rotate180`), the pixel size and section
step (`vx`, `vz`, um), and, from the stacked photographs, the camera jumps (`jumps`) and the missing or damaged sections and
caudal cutoff (`missing`), all listed by hand. Then run the steps in order.

## Hand corrections

All masks of the paper were corrected by hand where needed; `configs/subjects/<ID>.yaml` `masks_of_record` lists which.

- **3D brain mask:** after `mask`, correct `curated/<ID>_mask.nii.gz` on `curated/<ID>_bg.nii.gz` and save it as
  `curated/<ID>_mask_edited.nii.gz`. It wins over the model's output in every later step.
- **Marker tubes:** after `volume`, save a corrected `curated/<ID>_holes_edited.nii.gz` the same way (`curate.fold_edits` folds
  edits made on the final grid back onto the curated grid; `curate.reinpaint` re-fills the tubes without a full rebuild).
- **2D photo masks:** optional, before `prepare`: `linc-reconstruct edit-photo-masks export`, edit `nii/01_brainmask_L0.nii.gz`,
  then `linc-reconstruct edit-photo-masks import`.

## The spacing

The paper's volumes were reconstructed at each brain's nominal pixel size and section step, and then given the spacing
calibrated against the brain's own MRI by rescaling the headers about the world origin; the pixels are unchanged. Because some
steps depend on the spacing (for example, the smoothing width of the brightness correction, in sections), the release does the
same: it reconstructs at `reconstruction_spacing_um` and then runs `spacing`. For a new brain, both spacings are the same and
`spacing` does nothing.

## MF8 and MF9

The camera of these two older brains moved on nearly every frame. Instead of listing camera jumps, each photograph was
registered to its counterpart in an external, already aligned series of the same sections. That fit was a rescue step: it
needed the external series and a separate 2D mask network for it, so it is not part of the general pipeline and is not
released. What is released is its record (`sub-<ID>/blockface/wvig/align.json`) and `linc/blockface/series.py`, which
`photos` uses to replay it, so the two brains reconstruct like the others.

## Validation (21-22 Oct 2026)

- **MM1, end to end** (from its photographs and 2D masks of record, with its hand-edited 3D mask and tubes of record as
  distributed, through `stack`, `prepare`, `mask`, `volume`, `tubes` and `spacing`): every array and header of the products of
  record reproduced bit for bit: the aligned stack and its mask (every level), the reconstruction (`<ID>_blockface.ome.zarr`,
  L1 to L4), the pre-inpaint store, the L4 mask and tubes, the view NIfTI, the curated background-intact volume, the tube
  model's output and `<ID>.info`. The one difference is in the 3D brain-mask model's own output (2 of 1,727,616 voxels), which
  the reconstruction does not use (the hand-edited mask wins); the analysis project's own code, rerun the same day, gives the
  same 2-voxel difference, so it comes from rerunning the network, not from the port.
- **MM1's photographs** (`photos`, from the raw JPGs): all 164 renamed photographs identical. The 2D network's masks are not
  bit-reproducible run to run on a GPU: two runs on the same day and GPU (the release's and the analysis project's own code)
  differ by 1,432 of 1.65 billion pixels, and each differs from the masks of record by about as much (1,411 and 1,583),
  a few boundary pixels per photograph.
- **MF8's alignment** replayed from its record: all 77 photographs and masks identical to the aligned ones of record.
