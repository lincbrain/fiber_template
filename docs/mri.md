# Prepare a brain's MRI

**You need** one MRI volume of the brain, a b0 or a T2-weighted image (`docs/inputs.md`).

**What it does**
1. makes the brain mask with the MRI brain-mask network, or takes yours (`--mask`);
2. masks the brain;
3. corrects its bias field (N4).

**You get**, under `$LINC_DATA/sub-<ID>/mri/`: the image, its brain mask, and the masked and the bias-corrected brain, as NIfTI,
with a provenance JSON.

For a brain that is not one of the paper's, `cli/linc-mri prep` turns one MRI volume, a b0 or a T2-weighted image, into the inputs
of the registration (`docs/register.md`): the volume, its brain mask and the bias-corrected brain. It runs the paper's preprocessing
steps (Methods, "MRI preprocessing"), except the resampling: `linc-register` resamples onto the template's 0.5 mm frame.

Environment: `env/linc.yml`, with `ANTSPATH` set for ANTs 2.5.4. The brain mask is made by the MRI brain-mask network, as
`linc-register` runs it (`$LINC_MODELS`, `$LINC_SEGMENT_PYTHON`; `docs/models.md`).

```bash
$PY cli/linc-mri prep --subject <ID> --image <b0.nii.gz>                                  # a b0; its brain mask by the network
$PY cli/linc-mri prep --subject <ID> --image <b0.nii.gz> --mask <brain_mask.nii.gz>        # your own mask instead
$PY cli/linc-mri prep --subject <ID> --image <msme.nii.gz> --volume 4 --modality t2        # a T2-weighted image: the fifth echo of a multi-echo series
```

The steps:
1. **Stage** one volume under its own header; `--volume N` takes volume N of a 4-D series (default 0).
2. **Denoise**, only with `--denoise`: ANTs DenoiseImage, for Rician noise. It is off by default, because the paper's images were
   denoised before this step.
3. **Mask**: the mask given with `--mask` (on the image's grid, values 0 and 1), or the network's. The network runs on a GPU;
   `--device cpu` runs it on the CPU.
4. **Correct the bias field** with ANTs N4 inside the mask, with the parameters in `configs/mri_construction.yaml` (`prep.n4`);
   `--no-n4` skips it.
5. **Normalize**: the masked image, rescaled between its 1st and 99th percentiles inside the mask to (0, 4095].

The outputs go under `$LINC_DATA/sub-<ID>/mri/` (`--mri-root` gives another folder), written `$M` below:

| Input | Outputs |
|---|---|
| b0 | `$M/dwi/b0.nii.gz`, `$M/masks/dwi_brain_mask.nii.gz` (with a `.json` when the network made it), `$M/dwi/b0_brain.nii.gz`, `$M/dwi/b0_brain_n4.nii.gz` (and its `_bias`), `$M/dwi/b0_brain_n4.provenance.json` |
| T2-weighted | `$M/t2/echo.nii.gz`, `$M/masks/t2_brain_mask.nii.gz`, `$M/t2/echo_n4.nii.gz` (and its `_bias`), `$M/t2/echo_n4_brain.nii.gz` (with a `.provenance.json`) |

The provenance file records every step with its tool and version. Without ANTs, denoising and N4 are skipped with a message, which
the provenance records, and the run completes; the mask is the one step that cannot be skipped. ANTs' version is checked against the
paper's; `LINC_SKIP_VERSION_CHECK=1` runs with another version.

At the end, the run prints the next command, the first registration:

```bash
M=$LINC_DATA/sub-<ID>/mri
$PY cli/linc-register mri --subject <ID> --place-b0 $M/dwi/b0_brain_n4.nii.gz $M/masks/dwi_brain_mask.nii.gz --new-brain
```

For a T2-weighted image, give `$M/t2/echo_n4_brain.nii.gz` and `$M/masks/t2_brain_mask.nii.gz` there.

On one brain of the paper (`sub-MF283`, MF1 in the paper), with its hand-edited masks, the b0 and the T2-weighted paths reproduce the
paper's preprocessed files voxel for voxel. The b0 path took 19 s and the T2-weighted path 24 s of wall time (N4 about 4 CPU-minutes);
the network's mask added 3 s on a GPU, and `--denoise` 31 s. On the CPU, the network's mask of that 0.25 mm b0 took 40 s and
differed from the GPU's in 14 voxels.

## The MRI of the template's brains

We assume the inputs are each brain's preprocessed files, made for Gong et al. 2026 (doi:10.64898/2026.02.18.705310), as gzipped NIfTI
under `$LINC_MRI_PREPROCESSED/<ID>/`:
- `dwi/`, the 0.25 mm diffusion series (171 volumes, b up to 20,000 s/mm²):
  - `dwi_eddycorrect`, with `dwi_eddycorrect.ecclog`: denoised by MP-PCA (MRtrix3 `dwidenoise`), then corrected for eddy currents and
    motion (FSL `eddy_correct`, reference volume 0). The `.ecclog` holds the matrix of each volume;
  - `dwi_biascorr`: the same series after MRtrix3 `dwibiascorrect ants`. Its volume 0 is the b0;
  - `brain_mask`: the mask of the tensor's fit;
  - `dwi_orig.bvecs`, `dwi_orig.bvals`: the acquired table, in FSL's format.
- `msme/`, the 32-echo multi-echo spin-echo series (0.5 mm): `msme_denoised_las`, denoised by MP-PCA.
- `dt2/tortoise/`, the 0.5 mm multi-shell series (184 volumes, b up to 43,000 s/mm²):
  - `dwi_biascorr`: corrected for motion and eddy currents with TORTOISE, then `dwibiascorrect ants`;
  - `microstructure/SANDIdot/est.fracs.mean`, `est.fracs.std`: the SANDIdot fit (Microstructure.jl); volumes 0, 1, 2 are soma, neurite, dot.

We also assume each session's brain mask and the final brain mask (`docs/inputs.md`).

This repository adds (`docs/build_everything.md`):
- `stage`: the b0, the T2-weighted image (echo 5) and the microstructure session's b0, each corrected with N4 inside its brain mask; the
  fractions masked, clipped to [0, 1] and divided by their sum;
- `tensor`: the diffusion tensor, fitted with FSL `dtifit` on the 43 volumes with b up to 4800 s/mm², with the b-vectors rotated from the
  `.ecclog` and checked with `dwigradcheck`;
- `t2reg`, `microreg`, `nmt`: the registration of the T2-weighted image and the microstructure session to the b0, and of NMT v2 to the
  b0, whose SARM labels give the anterior and posterior commissures;
- `landmarks`, `frames`, `prepare`: each brain's canonical frame, and every channel resampled once onto it, cut to the final mask and
  mirrored;
- `assemble`, `mmorf`, `channels`, `links`: the groupwise construction, the template channels and each brain's transforms to the template.

Run from the paper's preprocessed files, step by step:
- `stage` and `tensor`, on all four brains: the 17 staged images of each brain, N4 included, and its tensor are voxel-identical to the
  paper's.

On MF1, each step on its own:
- `t2reg`, `microreg`: the fields differ from the paper's by a median of 0.019 mm (largest 0.52 mm) and 0.033 mm (largest 1.9 mm).
- `nmt`, `landmarks`, `frames`: the paper registered NMT v2 with an earlier version of the anatomix engine. With this release's, the SARM
  labels agree on 93.3 % of the brain's voxels, and the canonical frame differs by 0.04° and 0.16 mm (under 0.1° and 0.2 mm between runs).
- `prepare`: every channel is resampled through that frame, so no file is identical. Inside both masks, the Pearson correlation with the
  paper's is 0.968 for the T2-weighted image, 0.957 for the b0, 0.923 and 0.909 for the fractions and 0.995 for the tensor, and the Dice
  coefficient of the masks is 0.989. With the paper's frame, tensor, registrations and staged inputs, all 36 files are identical.

The time per brain is about 40 minutes. `microreg` needs about 22 GB free on the GPU.
