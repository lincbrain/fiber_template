# MRI

Prepares a brain's MRI. Environment: `settings/env/linc.yml`. It needs ANTs 2.5.4.

## A new brain

```bash
$PY code/mri/linc-mri prep --subject <ID> --image <b0.nii.gz>
```

| Option | |
|---|---|
| `--mask <file>` | use your own brain mask |
| `--volume N` | take volume N of a 4-D series (default 0) |
| `--modality t2 --volume 4` | a T2-weighted image: the fifth echo of a multi-echo series |
| `--denoise` | denoise first (off by default: the paper's images were denoised before) |

What it does and writes, and the registration that follows: [01, Steps B1 and B2](../../docs/tutorials/01_mri.md#step-b1-prepare-the-b0).

## The four brains of the MRI template

We assume the inputs are each brain's preprocessed files, made for Gong et al. 2026 (doi:10.64898/2026.02.18.705310):

- the 0.25 mm diffusion series (171 volumes, b up to 20,000 s/mm2): denoised by MP-PCA (MRtrix3 dwidenoise), corrected for eddy currents and motion (FSL eddy_correct, reference volume 0), then bias-corrected (MRtrix3 dwibiascorrect ants); with its b-table and fit mask
- the 0.5 mm 32-echo multi-echo spin-echo series: denoised by MP-PCA
- the 0.5 mm multi-shell series (184 volumes, b up to 43,000 s/mm2): corrected for motion and eddy currents with TORTOISE, then bias-corrected; with the SANDIdot fit (Microstructure.jl)
- each session's brain mask, and the final brain mask of each brain

What this code does with them, step by step: [01, part A](../../docs/tutorials/01_mri.md#a-the-mri-template-on-the-papers-four-brains).

| | |
|---|---|
| `mri_prep.py` | stages and corrects each brain's images |
| `mri_new_brain.py` | the new-brain preparation above |
| `mri_register.py` | the registrations within each brain |
| `canonical.py` | each brain's canonical frame |
| `linc-mri` | the command |
