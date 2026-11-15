# MRI

`linc-mri prep` prepares a brain's MRI for its registration to the template: it takes one volume, makes its brain mask and corrects its bias.
It runs in `$PY` (`settings/env/linc.yml`) with ANTs 2.5.4, and [01, Step B1](../../docs/tutorials/01_mri.md#step-b1-prepare-the-b0) walks through it and lists its outputs.

## Commands

```bash
$PY code/mri/linc-mri prep --subject <ID> --image <b0.nii.gz>
```

| Option | |
|---|---|
| `--mask <file>` | use your own brain mask, on the image's grid, with only 0 and 1 |
| `--volume N` | take volume N of a 4D series (default 0) |
| `--modality t2 --volume 4` | a T2-weighted image: the fifth echo of a multi-echo series |
| `--denoise` | denoise first (off by default, because the paper's images were denoised before) |
| `--device cpu` | run the mask network on the CPU |

The input is one b0 (or T2-weighted image) as NIfTI.

## The four brains of the MRI template

Their inputs are each brain's preprocessed files, made for Gong et al. 2026 (doi:10.64898/2026.02.18.705310):

- the 0.25 mm diffusion series (171 volumes, b up to 20,000 s/mm2), denoised by MP-PCA (MRtrix3 dwidenoise), corrected for eddy currents and motion (FSL eddy_correct, reference volume 0) and then bias-corrected (MRtrix3 dwibiascorrect ants), with its b-table and fit mask;
- the 0.5 mm 32-echo multi-echo spin-echo series, denoised by MP-PCA;
- the 0.5 mm multi-shell series (184 volumes, b up to 43,000 s/mm2), corrected for motion and eddy currents with TORTOISE and then bias-corrected, with the SANDIdot fit (Microstructure.jl);
- each session's brain mask, and the final brain mask of each brain.

[01, part A](../../docs/tutorials/01_mri.md#a-the-mri-template-on-the-papers-four-brains) shows what this code does with them.

## Files

| | |
|---|---|
| `linc-mri` | the command |
| `mri_new_brain.py` | `prep` |
| `mri_prep.py` | stages and corrects the template brains' images |
| `mri_register.py` | the registrations within each template brain |
| `canonical.py` | each template brain's canonical frame |
