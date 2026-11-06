# From the paper to the code

Where each part of the paper lives in this repository. Section names are the paper's (Gonzales et al., *A multimodal macaque template
for aligning ex vivo MRI and microscopy*). Paths are relative to the repository root.

- `linc/`: the pipeline as a Python package.
- `configs/`: one YAML per pipeline. Each parameter's comment names the Methods subsection it comes from.
- `cli/`: entry points for new data.
- `reproduce/numbers/<record>.py`: each script rewrites one numbers record of the paper, `<record>.json`, which the paper's text and
  tables read. `reproduce/check_numbers.py` compares a rewritten record with the paper's.
- `reproduce/figures/<name>.py`: one script per figure, named by what the figure shows (the table below gives the paper's number).
  `reproduce/templates/`: how the templates were built, and a build driver for each (`docs/build_everything.md`).

Parts marked *(not in this snapshot)* are still being ported.

## Methods

| Paper (Methods) | Code | Parameters | Entry point | External tools |
|---|---|---|---|---|
| Brains | `docs/specimen_labels.json` (paper names ↔ Specimen IDs) | `configs/subjects/<ID>.yaml` | — | — |
| MRI template: preprocessing | `linc/construction/mri_prep.py`, `linc/construction/canonical.py` | `configs/mri_construction.yaml` | — | N4, FSL `dtifit`, MRtrix3 `dwigradcheck` |
| MRI template: image registration | `linc/registration/featurereg.py` (anatomix features + FireANTs, run in `env/anatomix.yml` through `anatomix_runner.py`); pull maps in `xfmtools.py` | `configs/registration.yaml` | `cli/linc-register` | FireANTs 1.5.1, anatomix |
| MRI template: symmetric groupwise construction | `linc/construction/mmorf_run.py`, `tools.py`; our change to the FSL template driver as a patch (`linc/construction/patches/`) | `configs/mri_construction.yaml` | `reproduce/templates/build_mri_channels.py` (recipe: `reproduce/templates/mri_channels.md`) | MMORF 1.0.1, FSL 6.0.7.4, ANTs 2.5.4 |
| MRI template: template channels | `linc/construction/channels.py` | `configs/mri_construction.yaml` | — | FSL |
| Evaluation of the MRI template: contribution of each MRI contrast; leave-one-brain-out | `linc/evaluation/variants.py`, `variability.py`, `lobo.py` | `configs/mri_construction.yaml` | — | MMORF |
| Evaluation of the MRI template: relation to published templates | `linc/link/atlas.py`, `ants.py` | `configs/link.yaml` (`atlas`) | — | ANTs |
| Evaluation of the MRI template: the template as a registration target | `linc/evaluation/targets.py` (the driver; engine `linc/link/symreg.py`, affine start `linc/link/ants.py`) | `configs/targets.yaml` (where to download NMT v2 and CIVM-R) | — | ANTs 2.5.4, anatomix + FireANTs |
| Blockface template: blockface reconstruction | `linc/blockface/` (`photos.py`, `jumps.py`, `prepare.py`, `segment3d.py`, `destep.py`, `reconstruct.py`, `spacing.py`, `curate.py`; `README.md`) | `configs/blockface.yaml`, `configs/subjects/<ID>.yaml` | `cli/linc-reconstruct` | the segmentation networks (`models/segmentation.md`), TIRL 3.6.3, nifti-zarr |
| Blockface template: blockface to MRI registration and synthesis of missing sections | `linc/registration/to_own_mri.py`, `direct.py` (the cross-modal alignment of a brain without MRI), `resample.py`; `linc/completion/` (`README.md`) | `configs/registration.yaml`, `configs/completion.yaml` | `cli/linc-register`, `cli/linc-complete` | weights: `models/completion.md` |
| Blockface template: blockface template | `linc/construction/blockface_build.py`, `blockface_mmorf.py`, `blockface_finish.py`; our change to the FSL template driver (`linc/construction/patches/`) | `configs/mri_construction.yaml` | `reproduce/templates/build_blockface_template.py` (recipe: `reproduce/templates/blockface.md`) | MMORF 1.0.1 |
| Blockface template: registration of the blockface template to the MRI template; aligning a new photographed brain | `linc/link/` (`register.py`, `symreg.py`, `estimate.py` for the composed registrations, `qc.py`); `linc/link/b0_to_template.py` (a brain's b0 → the MRI template, the second registration of the MRI-based alignment) | `configs/link.yaml` | `cli/linc-register mri-template`; `reproduce/templates/build_link.py` (recipe: `reproduce/templates/link.md`) | ANTs |
| Histology alignment: histology registration and label propagation | `linc/histology/` (`match.py`, `pieces.py`, `register.py`, `labels.py`, `interpolate.py`, `carry.py`) | `configs/histology.yaml`, `configs/labels.yaml` | `cli/linc-histology`, `cli/linc-labels` | TIRL 3.6.2, SimpleITK |
| Histology alignment: evaluation of the registration pipeline | `linc/evaluation/tubes.py`, `labels_agreement.py`, `direct_placement.py` (the cross-modal alignment), `route_test.py` (the alignment comparison), `regeval.py` | — | — | — |
| Supplementary S5: segmentation models | `linc/segment/` (`train.py`, `predict.py`, `folds.py`; `docs/models.md`) | each model's `spec.json` (`$LINC_MODELS/segment/<task>/`) | `cli/linc-segment` | PyTorch (`env/deep.yml`) |

## Results: numbers, tables and figures

Each record below is `reproduce/numbers/<record>.py` → `<record>.json` (two are written by a figure script, as noted). A record's
free-text `method` field is kept verbatim from the analysis as it ran, so its wording can differ from the paper's; the numbers are what
the paper reports.

| Paper (Results) | Numbers records | Tables |
|---|---|---|
| Brains (Methods) | `bf_cohort`, `bf_camera_exif`, `bf_spacing_used`, `bf_spacing_summary_r073` and `bf_step_groups_r073` (pixel size and section step over brains), `bf_tubes_per_brain_r073` (marker tubes per brain), `mm_mri_check_r054` (reference only without the ParaVision files, which are not in the data release), `mri_acq_params_r065` (the b0 series' TE, TR, volumes and b-values; reference only, not regenerated: it reads the ParaVision files) | Table 1 |
| The template | `bf_cohort` | — |
| Contribution of each MRI contrast | `table1_rebuild`, `variants_leave_self_out_rebuild_fa025` | Table 2 |
| Between-brain variability | `variability_four_brains_r061` (Fig. 4 and its text) | — |
| Leave-one-brain-out evaluation | `lobo_t2_rebuild` | — |
| Relation to published templates | `atlas_final_v2_summary` | — |
| The template as a registration target | `r060_target_comparison` (needs NMT v2 and CIVM-R, which you download), `lgnstn_targets_r060c` (the LGN and STN rows; `lgnstn_targets_r060b`, which it replaces, stays in `reproduce/reference/` as history) | Table 3 |
| The blockface template and its registration to the MRI template | `registration_params_r073` (the settings of the deformable registration to MRI), `bf_sanity`, `bf_template_metrics_d2b`, `precision_weights_b1_d2b_s8`, `r1_edge_summary`, `d3_methods`, `d3_domain_shift`, `d3_predicted_share`, `s5_link_d2bs8`, `s5_link_regions_d2bs8` | — |
| Histology registered to the blockface | `tube_offsets_r068`, `histo_tube_landmarks`, `tube_mf8_check_r068`, `registration_qc_histo2bf_r068` (made by `reproduce/numbers/mf8_s46_drop_r068.py` from `tube_offsets_r054`, `tube_mf8_check_r054` and `registration_qc_histo2bf_20261001`, with one MF8 component unmatched), `fig_histology_simple` | Table 4 |
| Agreement of LGN and STN labels across brains | `lgnstn_geometry_current`, `labels_overlap_current` | Table 5 |
| Blockface-based versus MRI-based alignment | `route_test_1c`, `fig_routes_1c_section` (written by `alignment_comparison.py`) | — |
| Tracer labels in the template | `tracer_sections_r058d` (the labeled sections), `tracer_template_v2_all`, `tracer_connected_v2c` (the connected pathway of each case and its inferred fraction), `tracer_alic_topography_v2c`, `tracer_placement_v2c`, `fig_tracer_sections3d_v2` (written by `tracer_sections3d.py`: the positions of Fig. 10's capsule slabs), `fig_tracer_r058` (written by `tracer.py`: the label classes of Supplementary S2) | — |
| Construction parameters (Methods) | `mri_recipe` | — |
| Supplementary S1: deep learning models of the blockface pipeline | `s1_segmentation_r057` (S1.1); `blockface_params_r073` and `s1_photographs_r068b` (the reconstruction's settings, S1.2); `r057_s1_synthesis`, `d3p_vs_d3_pseudo_prod`, `d3_methods`, `d3_predicted_share`, `d3_domain_shift` (S1.2); `r057_missing_extent`, `s1_perbrain_r057`, `s1_photographs_r068b` (the photographed sections of each brain) | Tables S1–S2 |
| Supplementary S2: tracer labels, checks | `tracer_label_volumes_v2c`, `tracer_interp_accounting_v2`, `tracer_connected_v2c`, `tracer_alic_topography_v2c`, `tracer_placement_v2c`, `tracer_sections_r058d` | Tables S3–S4 |
| Supplementary S3: agreement of the LGN and STN labels, an error budget | `lgnstn_budget_r059`, `lgnstn_geometry_current`, `lgnstn_midplane_planes_r059`, `lgnstn_b0_visibility_r059` | Table S5 |
| Supplementary S4: detection of the marker-tube holes | no record of its own: the rule is `linc/evaluation/tubes.py`, and the offsets it gives are `tube_offsets_r068` | — |
| Supplementary S5: segmentation models | `segmodels_mri_brainmask_2026-11`, `segmodels_bf_mask2d_2026-11`, `segmodels_bf_mask3d_2026-11`, `segmodels_bf_holes_2026-11` (from the training runs' records, on a CPU in `env/deep.yml`; the networks are not retrained) | Table S6 |

Table numbers follow the paper's current version. The incremental version numbers them 1 (brains), 2 (histology), 3 (LGN and STN
labels) and 4 (the seven templates), and has no table of the registration targets. Two facts records the Methods read,
`histo_pixel_sizes_r057` and `r057_mri_facts_b`, read raw data (`$LINC_RAW`).

### Figures

The paper's figure numbers follow its current version. The incremental version of the manuscript (the revision of the reviewed
draft) numbers six of them differently (in bold in the last column).

| Figure | Script (`reproduce/figures/`) | Make target | Environment | Incremental version |
|---|---|---|---|---|
| 1 The multimodal template | `template.py` | `template` | `env/linc.yml` | 1 |
| 2 Building the template and aligning histology to it | not drawn by this release yet: `pipeline.py` draws the figure with MM2 as the blockface example, the paper shows MM6 | `pipeline` | `env/linc.yml` | 2 |
| 3 Blockface reconstruction of one brain (MM2) | `reconstruction.py` | `reconstruction` | `env/linc.yml` | 3 |
| 4 Between-brain deviation after alignment to each template | `variability.py` | `variability` | `env/linc.yml` | **9** |
| 5 The blockface template | `bf_template.py` | `bf_template` | `env/linc.yml` | **4** |
| 6 The blockface template registered into the MRI template | `link.py` | `link` | `env/linc.yml` | **5** |
| 7 Histology registered to the blockface | `histology.py` (needs the histology source data: `HISTOLOGY_ARGS`) | `histology` | `env/tirl.yml` | **6** |
| 8 The LGN and STN of three brains in the template | `lgnstn.py` | `lgnstn` | `env/deep.yml` | **7** |
| 9 Blockface-based and MRI-based alignment | `alignment_comparison.py` | `alignment_comparison` | `env/linc.yml` | **8** |
| 10 Fiber bundles of three tracer cases in the template | not drawn by this release yet (the bundles as translucent surfaces); `tracer_sections3d.py` draws each charted section's labels and writes the record of the capsule slabs | `tracer_sections3d` | `env/viz3d.yml` (`PY_VIZ3D`) | 10 |
| S1 Registration of a blockface volume to its own MRI (MM2) | not drawn by this release yet | — | — | — |
| S2 Synthesis of withheld slabs | not drawn by this release yet: `s_synthesis_depth.py` draws panel (a) with four further methods beside the synthesis | `s_synthesis_depth` | `env/deep.yml` | S1 |
| S3 Blockface volumes with their synthesized tissue | not drawn by this release yet: `s_synthesis_gallery.py` draws each brain at the plane with the most synthesized tissue, the paper at the plane near it that crosses the fewest gaps of the brain's MRI mask | `s_synthesis_gallery` | `env/deep.yml` | S2 |
| S4 From the chart to the template, one section per case | `s_tracer_fidelity.py` (reads LINC dataset 000003 under `$LINC_RAW/linc_000003`; skipped without it) | `s_tracer_fidelity` | `env/tirl.yml` | S3 |

`tracer.py` (`--figure main`) draws the tracer figure of 30 Oct 2026 and writes `fig_tracer_r058`, whose label classes Supplementary S2
gives.

Record names keep the analysis project's names (for example `_r054`, `_rebuild`, `_current`), so each can be traced to the run that
made it. They do not name different methods: each record holds the final numbers the paper reports.
