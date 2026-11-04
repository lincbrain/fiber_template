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

On one brain of the paper (`sub-MF283`, MF1 in the paper), with its hand-made masks, the b0 and the T2-weighted paths reproduce the
paper's preprocessed files voxel for voxel. The b0 path took 19 s and the T2-weighted path 24 s of wall time (N4 about 4 CPU-minutes);
the network's mask added 3 s on a GPU, and `--denoise` 31 s. On the CPU, the network's mask of that 0.25 mm b0 took 40 s and
differed from the GPU's in 14 voxels.

## The MRI of the template's brains, from the scanner's files

The MRI template build (`docs/build_everything.md`) starts each of its four brains from the Bruker sessions: it preprocesses the
diffusion, T2-weighted and SANDI images, registers NMT v2 to the b0, and registers the brain's sessions to each other.

The result matches the paper's files, with three small differences. On MF1, the eddy-current correction differs in one of 171
volumes, by 0.005 mm, because the release uses FSL 6.0.7.4 and the paper 6.0.6. TORTOISE's output is byte-identical to the
paper's for MF1–MF3; the paper's TORTOISE run of MF4 shared its volumes between two GPUs, and on one GPU 93 of its 184 volumes
register slightly differently (median relative difference 2e-3). The SANDI fractions agree with the paper's at a Pearson
correlation of 0.965 to 0.992 for every brain and fraction (median difference 0.06 at most), because of the paper's brain mask: in
the paper's microstructure processing of the four MRI brains (MF1–MF4), the mask of the motion- and eddy-corrected series
combined that run's `dwi2mask` with a `bet2` mask left over from an earlier run, as `bet2` stopped on a stale file in each brain's
final run. The release makes the mask as intended; given the paper's own inputs, its SANDI step reproduces the paper's fractions exactly. Given the paper's preprocessed files, the
staged inputs of the build are voxel-identical to the paper's, and the registrations agree to the precision of the GPU (the
canonical frame within 0.1° and 0.05 mm).
