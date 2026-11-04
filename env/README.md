# Environments

The pipeline runs in four Python environments, plus one optional one. Their packages conflict (TIRL needs its own stack, the
registration engine pins torch 2.13 + CUDA 12.6, the networks torch 2.6 + CUDA 12.4), so they stay separate. `bash env/install.sh`
builds them (`docs/install.md`), and `envs/linc_envs.sh`, which it writes, points the code at them. Each file's header says how to
create it by hand, what it runs, and which variable points the code at it. Every step of `cli/` and `reproduce/` runs in the
environment it ran in for the paper. Figures are named by content; `docs/paper_to_code.md` maps the paper's figure numbers to them.

| File | Runs | Key versions (as used in the paper) | Pointed to by |
|---|---|---|---|
| `linc.yml` | orchestration: `cli/linc-register` (linear stages, inputs, products, transform entries), `cli/linc-mri`, the MRI template construction (MMORF), the MRI records (variants, variability, LOBO, LGN/STN overlap and geometry, `lgnstn_*_r059`, `r057_mri_facts`), `linc/evaluation`, `reproduce/numbers`, the demons interpolation of labels between sections (`cli/linc-labels interpolate`), and the figures `reconstruction`, `bf_template`, `alignment_comparison`, `template`, `pipeline`, `variability`, `link`, `tracer`, `s_tracer_rules` | python 3.11.16, numpy 2.4.6, scipy 1.17.1, nibabel 5.4.2, matplotlib 3.11.1, zarr 3.1.5, scikit-image 0.26.0, opencv 4.13, SimpleITK 2.5.3, fsl-mmorf 1.0.1 (the FSL conda channel); FSL 6.0.7.4 (external) | `PY` (and `PY_MRI`, the same) |
| `deep.yml` | the deep-learning networks: the four segmentation models (`cli/linc-segment`; trained and validated here) and the caudal completion: training, inference (`cli/linc-complete`), the `d3_*` records, `r057_s1_synthesis`, `r057_missing_extent`, `lgnstn_budget_r059`, and the figures `s_synthesis_depth`, `s_synthesis_gallery`, `lgnstn` | torch 2.6.0+cu124 (CUDA 12.4), MONAI 1.4.0, numpy 1.26.4; matplotlib 3.10.6 and its dependencies at the versions of record (the three figures render pixel-identical) | `PY_DEEP` (Makefile); `LINC_SEGMENT_PYTHON` (linc-reconstruct, linc-register) |
| `anatomix.yml` | the feature-based deformable registration engine of every registration of record (anatomix features + FireANTs) | torch 2.13.0 (CUDA 12.6), anatomix @ 4c86aee (MIT license) and its FireANTs fork @ 21ae072 (Apache 2.0), both public commits, installed editable; pins every package the engine imported in a registration of record (traced) | `ANATOMIX_PYTHON` |
| `tirl.yml` | histology → blockface registration (TIRL 3.6.3, 12 threads per piece), the label push, the tube evaluation, `histo_pixel_sizes_r057`, the `histology` figure, and the Python steps of the blockface reconstruction | TIRL 3.6.3 and its three script packages (from source; the paper's runs used 3.6.2, which registers byte for byte the same), zarr 3.1.5; nifti-zarr 1.0.0rc6 (`zarr2nii` / `nii2zarr`, the reconstruction's exports, found on PATH) | `PY_TIRL` |
| `viz3d.yml` | optional: the 3D tracer scene and its rotation video | pyvista 0.44.2 on the headless vtk-osmesa 9.3.1 wheel (from wheels.vtk.org), imageio-ffmpeg 0.6.0 | `PY_VIZ3D` |

Removed on 3 Nov 2026: `niizarr.yml` (environment `linc-niizarr`, nifti-zarr 1.0.0rc6), where the paper's `zarr2nii` / `nii2zarr`
exports ran; nifti-zarr is now installed in `tirl.yml`, whose exports are byte-identical.

Former name: `deep.yml` was `env/completion.yml` (environment `linc-completion`) until 3 Nov 2026, with the same pins; the records
and model cards made before then name that file. `PY_COMPLETION` is still honored for `PY_DEEP`.

**External programs** (not pip-installable): FSL 6.0.7.4 and ANTs 2.5.4. MMORF 1.0.1 comes with `linc.yml`.

**Exact reproduction** (`docs/reproduce.md`): the deformable registrations are deterministic within a GPU session and differ by
floating point across sessions or GPU models (median 0.002–0.025 mm; single voxels up to 1.9 mm); TIRL must run with 12 threads; one
GPU record (`d3_domain_shift`) is reproducible only to its last digits.
