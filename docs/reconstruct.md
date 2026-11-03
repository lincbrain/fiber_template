# Reconstruct a blockface volume from the photographs

**You need** the brain's photographs and its settings, `configs/subjects/<ID>.yaml`, with its hand-made inputs (`docs/inputs.md`).

**What it does**
1. `photos`: copies the photographs in cutting order and makes their brain masks with the network;
2. `stack`: stacks the photographs and their masks;
3. `prepare`: corrects the camera jumps (TIRL), marks and fills the missing sections, and evens the illumination;
4. `mask`: makes the 3D brain mask;
5. `volume`: writes the volume;
6. `tubes`: detects the marker tubes;
7. `spacing`: writes the pixel size and section step into the headers.

**You get**, under `$LINC_DATA/sub-<ID>/blockface/`: the volume, `<ID>_blockface.ome.zarr`, with its brain mask (`<ID>_mask.nii.gz`)
and marker tubes (`<ID>_holes.nii.gz`).

Run each entry point from the repository root, in the environment its page names. `--help` lists every option.

A brain's own settings live in `configs/subjects/<ID>.yaml`: its photographs, pixel size and section step, camera jumps, missing
sections and masks of record. A new brain needs one; start from the commented `configs/subjects/TEMPLATE.yaml`.

Environment: `env/tirl.yml`, which brings TIRL and zarr2nii; the segmentation networks run through `LINC_SEGMENT_PYTHON`
(`env/deep.yml`), which `envs/linc_envs.sh` sets (`docs/models.md`).

```bash
$PY_TIRL cli/linc-reconstruct all --subject MR252
```

This writes `$LINC_DATA/sub-<ID>/blockface/<ID>_blockface.ome.zarr`. The steps can also run one at a time: `photos`, `stack`,
`prepare`, `mask`, `volume`, `tubes`, `spacing`, each with `--subject <ID>`. To hand-correct the 2D brain masks before `prepare`,
use `cli/linc-reconstruct edit-photo-masks export|import --subject <ID>`. The steps, hand corrections and spacing are explained in
[`linc/blockface/README.md`](../linc/blockface/README.md).

For MF8 (`sub-MF191`), the volume of record was rebuilt on 24–26 Sep 2026 with an earlier padding of its canvas. The release reproduces its aligned
photographs, brain mask and tubes exactly, and its volume up to a shift of the canvas by less than a voxel (its origin moves by 0.5
and 2 µm; 26 % of the voxels differ, by a mean of 0.5 on a scale of 255).
