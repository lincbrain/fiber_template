# The MRI-based alignment: a brain's own b0 → the MRI template

This registration aligns each brain that has MRI to the MRI template through its own b0, the MRI contrast every such brain has.
Paper: Methods (sec:m-link), the MRI-based alignment. Code:
`linc/link/b0_to_template.py`. Parameters: `configs/link.yaml` `b0_to_template`. Products of record: the 13 registry entries
`data/sub-<ID>/xfm/b0-native__to__mri-template/`.

## What it does

For each brain with MRI, its native b0 is registered to the MRI template's b0 channel:

1. **Initial alignment.** The b0 is aligned rigidly (FLIRT, 6 dof) to the 0.5 mm frame of the first MRI template run
   (`$LINC_TEMPLATE/frames/mri_template_0p5.nii.gz`), then transformed rigidly from that frame to the template's grid (the frame bridge).
2. **Input check.** The b0 is re-debiased with ANTs N4 when its residual field exceeds 1.5. Of the brains of record, only MR253 needed it.
3. **Linear start.** A 7-dof similarity is fitted, then a 12-dof affine. The affine is rejected, leaving the similarity only, when its
   anisotropy exceeds 1.5.
4. **Working grid.** The b0 is resampled once onto a centered 0.35 mm grid with 8 mm of zeros on every side.
5. **Brainstem cut.** The moving input is cut below the template brain's lowest axial slice (z = −2.0 mm).
6. **Deformable registration.** The feature-based engine runs at 0.35 mm, with warp smoothing σ = 1.25 voxels.

The registry entry holds two components, in pull order: `<ID>_warp.nii.gz` (`itk_lps_disp`, on the working grid), then
`<ID>_start_pull.txt` (`ras_pull`). Beside it the script writes:
- the moved b0 and the uncropped moved mask on the template grid;
- the outline check of the moved mask, which is also recorded as the entry's QC.

## Inputs

These follow the data layout (`$LINC_DATA/sub-<ID>/mri/`):
- `<ID>_mri_b0_native.nii.gz`;
- the brain's mask of record, `<ID>_mri_mask_native.nii.gz`;
- the initial rigid alignment, `<ID>_mri2template.mat` with its reference `<ID>_mri.nii.gz`. `--place-b0 <the 0.5 mm frame>` remakes it
  with FLIRT.

You also need the template's `mri/linc_mri_b0.nii.gz` and `mri/linc_mri_brainmask.nii.gz`, and the frame bridge, the rigid transform
from the 0.5 mm frame to the template's grid (`configs/link.yaml` `inputs.bridge`, or `--bridge`).

## Command

```bash
export ANATOMIX_PYTHON=/path/to/anatomix-env/bin/python   # env/anatomix.yml (the engine)
export LINC_DATA=... LINC_PRODUCTS=... LINC_WORK=$PWD/work
python -m linc.link.b0_to_template --subject MR252 \
    --template-b0 $LINC_TEMPLATE/mri/linc_mri_b0.nii.gz --template-mask $LINC_TEMPLATE/mri/linc_mri_brainmask.nii.gz \
    [--b0 B0 --mask M --placement-mat MAT --placed-b0 IMG] [--registry DIR] [--out DIR] [--device cuda:0] [--overwrite]
```

The entry goes to `$LINC_DATA/sub-<ID>/xfm/b0-native__to__mri-template`, or under `--registry`. The working files go to
`$LINC_WORK/b0_to_template/<ID>`. A run takes about 3.5 min: about 2 min of linear fits on CPU and about 70 s on the GPU (11 GB peak).
ANTs 2.5.4 is needed only when N4 applies.

## Validation (22 Oct 2026, MR252, NVIDIA RTX 6000 Ada)

The run used the record's Python stack for orchestration (`mri/env/mmorf`: numpy 2.4.6, scipy 1.17.1, nibabel 5.4.2) and the
anatomix engine environment (`env/anatomix-upstream`, torch 2.13.0+cu126). It was compared with the entry of record:

| Check | Result |
|---|---|
| similarity, affine fits (NCC, evaluation count, parameters) | identical to the run log of record (0.2868 / 947, 0.4025 / 2120) |
| guard anisotropy, fallback | 1.2484, no fallback: identical to meta.json |
| `<ID>_start_pull.txt` | byte-identical |
| working grid (208 × 256 × 160 at 0.35 mm, 8 mm margin), engine input clip (1315.8587646484375) | identical |
| warp | The record ran on an RTX 3090, so only floating-point agreement is expected. Displacement difference at the 3.3 M template-brain voxels, through the two registry chains: median 0.0011 mm, p99 0.031 mm, max 0.37 mm |
| native b0 pulled through the two chains | r = 0.99997 inside the template brain |
| moved mask | Dice 0.99986 with the record's; outline check PASS on both (anterior overshoot p99 1.03 mm, max 1.82 mm) |
| QC numbers of record (NCC at the b0's 0.7 mm FWHM, envelope Dice) | 0.731 and 0.978, as recorded |

Not ported: the analysis harness's regqc store, montages and working-grid edge diagnostics; the other linear modes of the test
script; the declared-box outline scoring (no box is declared for this alignment).

### Second brain, through the release CLI (22 Oct 2026, MF6 = MF275)

`linc-register mri-template` end to end on MF6, in environments built fresh from the then `env/mri.yml` (since merged into
`env/linc.yml`) and `env/anatomix.yml` (PYTHONNOUSERSITE=1),
on an NVIDIA RTX 6000 Ada: entry written, QC pass (outline PASS); `<ID>_start_pull.txt` md5-identical to the record's; warp
displacement difference at the 3.3 M template-brain voxels median 0.0052 mm, p99 0.090 mm, max 0.448 mm (the record ran on an
RTX 3090). Control: the same command in the environments of record on the same GPU gave a warp md5-identical to the fresh-env run,
so the residual comes from the GPU model, not the environment. `python -m linc.xfm chain --from bf-native --to mri-template` resolves
through the blockface-to-b0 registration's entry (linc-register mri) and this one.
