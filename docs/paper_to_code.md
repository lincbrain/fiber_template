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
| Brains (Methods) | `bf_cohort`, `bf_species`, `bf_camera_exif`, `bf_spacing_used`, `mm_mri_check_r054` (reference only without the ParaVision files, which are not in the data release), `mri_acq_params_r065` (the b0 series' TE, TR, volumes and b-values; reference only, not regenerated: it reads the ParaVision files), `table_specimens_check_r054` | Table 1 |
| The template | `bf_cohort` | — |
| Contribution of each MRI contrast | `table1_rebuild`, `variants_leave_self_out_rebuild_fa025` | Table 2 |
| Between-brain variability | `variability_four_brains_r061` (Fig. 4 and its text), `variability_rebuild_rb_final` (the eight inputs) | — |
| Leave-one-brain-out evaluation | `lobo_t2_rebuild`; `lobo_multichannel_cost_r061` (the cost sentence in Methods; reference only, not regenerated: it is made from run logs that are not released) | — |
| Relation to published templates | `atlas_final_v2_summary` | — |
| The template as a registration target | `r060_target_comparison` (needs NMT v2 and CIVM-R, which you download), `lgnstn_targets_r060c` (the LGN and STN rows; `lgnstn_targets_r060b`, which it replaces, stays in `reproduce/reference/` as history) | Table 3 |
| The blockface template and its registration to the MRI template | `bf_sanity`, `bf_template_metrics_d2b`, `precision_weights_b1_d2b_s8`, `r1_edge_summary`, `d2_s2_validation_prod_s125`, `d3_methods`, `d3_domain_shift`, `d3_predicted_share`, `s5_link_d2bs8`, `s5_link_regions_d2bs8` | — |
| Histology registered to the blockface | `tube_offsets_r068`, `histo_tube_landmarks`, `tube_mf8_check_r068`, `registration_qc_histo2bf_r068` (made by `reproduce/numbers/mf8_s46_drop_r068.py` from `tube_offsets_r054`, `tube_mf8_check_r054` and `registration_qc_histo2bf_20261001`, with one MF8 component unmatched), `fig_histology_simple` | Table 4 |
| Agreement of LGN and STN labels across brains | `lgnstn_geometry_current`, `labels_overlap_current` | Table 5 |
| Blockface-based versus MRI-based alignment | `route_test_1c`, `fig_routes_1c_section` (written by `alignment_comparison.py`) | — |
| Tracer labels in the template | `tracer_sections_r058d` (the labeled sections), `tracer_template_v2_all`, `tracer_connected_v2c` (the connected pathway of each case and its inferred fraction), `tracer_alic_topography_v2c`, `tracer_placement_v2c`, `fig_tracer_sections3d_v2` (written by `tracer_sections3d.py`: Fig. 10's points per case, planes, hidden shares and capsule counts), `fig_tracer_3d_stills` (written by `tracer_3d.py --for-composite`: Supplementary Fig. S5's three 3D stills, their planes, window, translucent shell and each pathway's share left of the midline), `fig_tracer_3d` (the same scene rendered whole, with both rotation videos), `fig_tracer_pathways` (written by `tracer_pathways.py`: the composite of Supplementary Fig. S5, its slabs, the pathway voxels drawn per case with the inferred fraction and the longest join, the capsule counts of panel e), `fig_tracer_pathways_panels` (the four-panel form drawn without `PY_VIZ3D`); `tracer_sections_v2` (reference only until its script is ported: `configs/numbers_exceptions.yaml`); `tracer_side_check_r058d`, `tracer_fix_validation_r058` (reference only: the paper reads its check numbers from them, and the audit's registration arms are not release products) | — |
| Construction parameters (Methods) | `mri_recipe` | — |
| Supplementary S1: deep learning models of the blockface pipeline | `s1_segmentation_r057` (S1.1); `r057_s1_synthesis`, `d3p_vs_d3_pseudo_prod`, `d3_methods`, `d3_predicted_share`, `d3_domain_shift` (S1.2–S1.4); `r057_missing_extent`, `s1_perbrain_r057`, `s1_photographs_r068b` (the photographed sections of each brain) (S1.5) | Tables S1–S3 |
| Supplementary S2: tracer labels, checks | `tracer_label_volumes_v2c`, `tracer_interp_accounting_v2`, `tracer_connected_v2c`, `tracer_alic_topography_v2c`, `tracer_placement_v2c`; `tracer_match_audit_r058`, `tracer_side_check_r058d` (reference only, as above) | Tables S4–S5 |
| Supplementary S3: agreement of the LGN and STN labels, an error budget | `lgnstn_budget_r059` (the rows above the network rows of Table S6), `lgnstn_midplane_planes_r059`, `lgnstn_b0_visibility_r059`, `lgnstn_model_loso_r059`, `lgnstn_model_agreement_r059` | Table S6 |
| Supplementary S4: settings of the histology registration | `tirl_sweep_r058`, `tirl_sweep_r058b` (reference only, not regenerated: see `configs/numbers_exceptions.yaml`; the nine settings are in `configs/histology_sweep.yaml`, and `linc/histology/README.md` shows how to rerun one setting on one piece in `env/tirl.yml`) | Table S7 |
| Supplementary S5: segmentation models | `segmodels_mri_brainmask_2026-11`, `segmodels_bf_mask2d_2026-11`, `segmodels_bf_mask3d_2026-11`, `segmodels_bf_holes_2026-11` (from the training runs' records, on a CPU in `env/deep.yml`; the networks are not retrained) | Table S8 |

Table numbers follow the paper's current version. The incremental version numbers them 1 (brains), 2 (histology), 3 (LGN and STN
labels) and 4 (the seven templates), and has no table of the registration targets. Two facts records the Methods read,
`histo_pixel_sizes_r057` and `r057_mri_facts_b`, read raw data (`$LINC_RAW`).

### Figures

The paper's figure numbers follow its current version. The incremental version of the manuscript (the revision of the reviewed
draft) numbers six of them differently (in bold in the last column).

| Figure | Script (`reproduce/figures/`) | Make target | Environment | Incremental version |
|---|---|---|---|---|
| 1 The multimodal template | `template.py` | `template` | `env/linc.yml` | 1 |
| 2 Building the template and aligning histology to it | `pipeline.py` | `pipeline` | `env/linc.yml` | 2 |
| 3 Blockface reconstruction of one brain (MM2) | `reconstruction.py` | `reconstruction` | `env/linc.yml` | 3 |
| 4 Between-brain deviation after alignment to each template | `variability.py` | `variability` | `env/linc.yml` | **9** |
| 5 The blockface template | `bf_template.py` | `bf_template` | `env/linc.yml` | **4** |
| 6 The blockface template registered into the MRI template | `link.py` | `link` | `env/linc.yml` | **5** |
| 7 Histology registered to the blockface | `histology.py` (needs the histology source data: `HISTOLOGY_ARGS`) | `histology` | `env/tirl.yml` | **6** |
| 8 The LGN and STN of three brains in the template | `lgnstn.py` | `lgnstn` | `env/deep.yml` | **7** |
| 9 Blockface-based and MRI-based alignment | `alignment_comparison.py` (ported from the analysis project's `fig9_routes_1c_r061.py`, the script of record since panel (a) shows each alignment's values per pair; its predecessor `fig9_routes_1c_r057.py` is kept unchanged) | `alignment_comparison` | `env/linc.yml` | **8** |
| 10 Tracer labels of three cases in the template | `tracer_sections3d.py` (a: each charted section's labels where that section lands in the template, the cases in layers; b: the anterior limb of the internal capsule on three coronal slabs) | `tracer_sections3d` | `env/viz3d.yml` (`PY_VIZ3D`) | 10 |
| S1 Synthesis of withheld slabs | `s_synthesis_depth.py` | `s_synthesis_depth` | `env/deep.yml` | S1 |
| S2 Blockface volumes with their synthesized tissue | `s_synthesis_gallery.py` | `s_synthesis_gallery` | `env/deep.yml` | S2 |
| S3 From the chart to the template, one section per case | `s_tracer_fidelity.py` (reads LINC dataset 000003 under `$LINC_RAW/linc_000003`; skipped without it) | `s_tracer_fidelity` | `env/tirl.yml` | S3 |
| S4 Two rules for drawing a label | `s_tracer_rules.py` (runs `tracer.py --figure rules`) | `s_tracer_rules` | `env/linc.yml` | S4 |
| S5 The connected pathways of three cases in the template | `tracer_3d.py --for-composite` (a–c: the 3D scene from the front, above and right, from the right, and from the right with the brain as a translucent shell), `tracer_pathways.py` (d, e, the key and the composite) | `tracer_pathways` (renders the three stills, then the composite; without `PY_VIZ3D`, the four-panel form only) | `env/viz3d.yml` (a–c; optional, `PY_VIZ3D`), `env/linc.yml` (d, e) | S5 |
| Supplementary video: the scene of Fig. 10a, one turn (12 s) | `tracer_sections3d.py --video` | `tracer_sections3d` | `env/viz3d.yml` (`PY_VIZ3D`) | video |
| Supplementary videos: the tracer pathways' 3D scene, one turn each (10 s), with the template's planes and with the brain as a translucent shell | `tracer_3d.py` | `tracer_3d` (optional, not part of `make figures`) | `env/viz3d.yml` (optional; `PY_VIZ3D`) | videos |

`tracer.py`'s main figure (`--figure main`, record `fig_tracer_r058`) is the tracer figure of 30 Oct 2026, before the connected
pathways; Figs. 10 and S5 replace it, and it is kept as history.

Record names keep the analysis project's names (for example `_r054`, `_rebuild`, `_current`), so each can be traced to the run that
made it. They do not name different methods: each record holds the final numbers the paper reports.
