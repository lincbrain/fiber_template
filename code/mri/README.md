# MRI

Prepares a brain's MRI. Environment: `settings/env/linc.yml`. It needs ANTs 2.5.4.


## A new brain

One b0 or T2-weighted image becomes a brain mask and a bias-corrected image, ready for registration:

```bash
$PY code/mri/linc-mri prep --subject <ID> --image <b0.nii.gz>
```

|  |  |
|---|---|
| `--mask <file>`               | use your own brain mask |
| `--volume N`                  | take volume N of a 4-D series (default 0) |
| `--modality t2 --volume 4`    | a T2-weighted image: the fifth echo of a multi-echo series |
| `--denoise`                   | denoise first (off by default: the paper's images were denoised before) |

It makes the mask with the MRI brain-mask network, corrects the bias field with N4 inside the mask,
and rescales the masked image. It takes under a minute. It writes under `sub-<ID>/mri/`: `dwi/`
(the b0, the masked brain and the bias-corrected brain), `masks/` (the brain mask), and a provenance
file listing every step with its tool and version. It ends by printing the next command.


## The four brains of the MRI template

We assume the inputs are each brain's preprocessed files, made for Gong et al. 2026
(doi:10.64898/2026.02.18.705310):

- the 0.25 mm diffusion series (171 volumes, b up to 20,000 s/mm2): denoised by MP-PCA (MRtrix3
  dwidenoise), corrected for eddy currents and motion (FSL eddy_correct, reference volume 0),
  then bias-corrected (MRtrix3 dwibiascorrect ants); with its b-table and fit mask
- the 0.5 mm 32-echo multi-echo spin-echo series: denoised by MP-PCA
- the 0.5 mm multi-shell series (184 volumes, b up to 43,000 s/mm2): corrected for motion and eddy
  currents with TORTOISE, then bias-corrected; with the SANDIdot fit (Microstructure.jl)
- each session's brain mask, and the final brain mask of each brain

This code adds: the N4 correction; the diffusion tensor (FSL dtifit on the 43 volumes with b up to
4800 s/mm2, with the b-vectors rotated by the motion correction and checked with dwigradcheck); the
registrations within each brain and to NMT v2; each brain's canonical frame; and the template
(`code/template/`). About 40 minutes per brain. `tests/README.txt` says how close a rerun comes.

|  |  |
|---|---|
| `mri_prep.py`         | stages and corrects each brain's images |
| `mri_new_brain.py`    | the new-brain preparation above |
| `mri_register.py`     | the registrations within each brain |
| `canonical.py`        | each brain's canonical frame |
| `linc-mri`            | the command |
