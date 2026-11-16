# Reproduce the paper's numbers and figures from the released products.
#
#   cd code/figures
#   make numbers        # every number record the paper reads (CPU), into $LINC_NUMBERS (default ../../outputs/numbers)
#   make numbers-gpu    # the two records that need a GPU (d3_domain_shift, d3_domain_shift_prod; CUDA card, settings/env/deep.yml)
#   make check          # compare the regenerated records with the paper's (reference/), exceptions as documented
#   make test           # the installation check: tests + toy smokes of the command lines, no data (selftest.sh, minutes)
#   make figures        # the paper's figures, into ../../outputs/figures/ (one target per figure, named by its content;
#                       # README.md maps the paper's figure numbers to them)
#   make all            # numbers + figures + check (two figure targets also write a numbers record)
#
# Set the roots first (see README.md): LINC_PRODUCTS, LINC_DATA, LINC_TEMPLATE, LINC_RAW, LINC_MODELS, LINC_NUMBERS.
# Each target runs in the environment it ran in for the paper (settings/env/README.md): PY = settings/env/linc.yml; PY_MRI = the same python (an alias kept for the MRI records) (the heavy MRI
# records and the figures FIGURES_MRI lists); PY_DEEP = settings/env/deep.yml (the d3_* records, Supplementary S1 and the LGN/STN
# budget, and their three figures); PY_TIRL = settings/env/tirl.yml (histology, s_tracer_fidelity;
# histo_pixel_sizes_r057).

PY ?= python
PY_MRI ?= $(PY)
# PY_COMPLETION: PY_DEEP's name until 3 Nov 2026 (env/completion.yml), still honored
PY_DEEP ?= $(or $(PY_COMPLETION),$(PY))
PY_TIRL ?= $(PY)
# PY_VIZ3D = settings/env/viz3d.yml (the 3D tracer scene, figures-3d; optional: install.sh --with-figures-3d)
PY_VIZ3D ?=
# raw data outside the release, needed by two targets (they are skipped, with a message, when unset):
#   MM_MRI_ARGS = --raw-bruker-root $$LINC_RAW/mri_mm --unpacked-root $$LINC_RAW/macaque_mri    (mm_mri_check_r054)
#   HISTOLOGY_ARGS = --jp2-root <LINC source data>/sourcedata                                  (the histology figure)
#   the LINC histology dataset 000003 at $$LINC_RAW/linc_000003                                  (s_tracer_fidelity)
#   LINC_PARTICIPANTS = the LINC participants.tsv                                           (bf_cohort)
MM_MRI_ARGS ?=
HISTOLOGY_ARGS ?=
LINC_PARTICIPANTS ?=
OUT ?= $(abspath ../../outputs)
export LINC_NUMBERS ?= $(OUT)/numbers
# scratch of the records that write intermediate files (code/shared/paths.py work_root; unset, it would be ./work inside code/figures/)
export LINC_WORK ?= $(OUT)/work
# never import packages from ~/.local: every target uses exactly its environment file's packages
export PYTHONNOUSERSITE ?= 1

NUMBERS_LINC := \
  bf_camera_exif bf_cohort bf_sanity bf_spacing_used bf_template_metrics_d2b bf_template_metrics_s9_015_s11 \
  precision_weights_b1_d2b_s8 r1_edge_summary route_test_1c \
  atlas_final_v2_summary fig_histology_simple histo_tube_landmarks \
  mm_mri_check_r054 mri_recipe registration_qc_histo2bf_20261001 s5_link_d2bs8 s5_link_regions_d2bs8 table1_rebuild \
  tracer_sections_r054 tracer_sections_r058d tube_mf8_check_r054 tube_offsets_r054 \
  s1_segmentation_r057 s1_perbrain_r057 s1_photographs_r068b bf_spacing_summary_r073 bf_step_groups_r073 bf_tubes_per_brain_r073 \
  blockface_params_r073 registration_params_r073 tracer_label_volumes_r058d tracer_alic_topography_r058d r060_target_comparison \
  lgnstn_targets_r060c mf8_s46_drop_r068 histology_h9 \
  tracer_sections_v2 tracer_connected_v2c tracer_label_volumes_v2c tracer_alic_topography_v2c tracer_interp_accounting_v2 \
  tracer_placement_v2c tracer_template_v2_all tracer_alic_topography_v5 lgnstn_label_volumes
NUMBERS_MRI := variants_leave_self_out_rebuild_fa025 variability_four_brains_r061 lobo_t2_rebuild \
  labels_overlap_current lgnstn_geometry_current \
  lgnstn_b0_visibility_r059 lgnstn_midplane_planes_r059 r057_mri_facts_b
NUMBERS_TIRL := histo_pixel_sizes_r057 bf_jump_pullback
NUMBERS_COMPLETION := d3_sigma2_pred d3_methods d3_predicted_share \
  d3p_vs_d3_pseudo d3p_vs_d3_pseudo_prod r057_missing_extent r057_s1_synthesis lgnstn_budget_r059
NUMBERS_ATLAS := gallery_plane_2026-11
NUMBERS_GPU := d3_domain_shift d3_domain_shift_prod
# the segmentation models' records (Supplementary S5, Table S8): CPU, settings/env/deep.yml (torch counts the parameters); bf_mask2d ~6 min
NUMBERS_SEGMENT := segmodels_mri_brainmask_2026-11 segmodels_bf_mask2d_2026-11 segmodels_bf_mask3d_2026-11 segmodels_bf_holes_2026-11

FIGURES_MRI := template pipeline variability link tracer
FIGURES_LINC := reconstruction bf_template alignment_comparison s_networks
FIGURES_COMPLETION := s_synthesis_depth s_synthesis_gallery
FIGURES := $(FIGURES_MRI) $(FIGURES_LINC) $(FIGURES_COMPLETION) lgnstn s_bf2mri_steps histology s_tracer_fidelity tracer_sections3d

.PHONY: all numbers numbers-gpu numbers-figures check test figures print-figures print-figure-records tracers-animation $(NUMBERS_LINC) $(NUMBERS_MRI) $(NUMBERS_TIRL) $(NUMBERS_COMPLETION) $(NUMBERS_SEGMENT) $(NUMBERS_GPU) $(NUMBERS_ATLAS) $(FIGURES)

all: numbers figures check

numbers: $(NUMBERS_LINC) $(NUMBERS_MRI) $(NUMBERS_TIRL) $(NUMBERS_COMPLETION) $(NUMBERS_SEGMENT) $(NUMBERS_ATLAS)

numbers-gpu: $(NUMBERS_GPU)

# the figures that also write a numbers record: `make check` needs them (tracer_sections3d: install.sh --with-figures-3d)
numbers-figures: tracer alignment_comparison tracer_sections3d

$(filter-out mm_mri_check_r054 bf_cohort s1_perbrain_r057 s1_segmentation_r057 r060_target_comparison lgnstn_targets_r060c,$(NUMBERS_LINC)):
	$(PY) paper_numbers/$@.py

bf_cohort:
	@if [ -z "$(LINC_PARTICIPANTS)" ]; then echo "SKIPPED bf_cohort: set LINC_PARTICIPANTS (the LINC participants.tsv)"; \
	else LINC_PARTICIPANTS="$(LINC_PARTICIPANTS)" $(PY) paper_numbers/$@.py; fi

# reads the regenerated bf_cohort, bf_camera_exif and bf_spacing_used, so it needs LINC_PARTICIPANTS as bf_cohort does
s1_perbrain_r057: bf_cohort bf_camera_exif bf_spacing_used
	@if [ -z "$(LINC_PARTICIPANTS)" ]; then echo "SKIPPED s1_perbrain_r057: set LINC_PARTICIPANTS (it reads bf_cohort)"; \
	else $(PY) paper_numbers/$@.py; fi

# reads the released blockface segmentation weights' folder (dataset.json) under LINC_MODELS
s1_segmentation_r057:
	@if [ -z "$(LINC_MODELS)" ]; then echo "SKIPPED s1_segmentation_r057: set LINC_MODELS (the network weights)"; \
	else $(PY) paper_numbers/$@.py; fi

# reads the published templates NMT v2 and CIVM-R, which are not redistributed: download them as settings/targets.yaml says, into
# settings/link.yaml inputs.atlas_reference (under LINC_PRODUCTS); skipped when they are not there. CPU, about 15 min at --jobs 4
R060_JOBS ?= 4
HAVE_ATLASES = $(PY) -c "import os, sys, yaml; r = os.path.join(os.environ.get('LINC_PRODUCTS', ''), yaml.safe_load(open('../../settings/link.yaml'))['inputs']['atlas_reference']); sys.exit(0 if all(os.path.isdir(os.path.join(r, d)) for d in ('NMT_v2.0_sym', 'CIVM-R_T2_DTI')) else 1)"
r060_target_comparison:
	@if $(HAVE_ATLASES); then $(PY) paper_numbers/$@.py --jobs $(R060_JOBS); \
	else echo "SKIPPED r060_target_comparison: NMT v2 and CIVM-R not found under settings/link.yaml inputs.atlas_reference (see settings/targets.yaml)"; fi

# the LGN and STN rows of the same comparison: carried onto the target grids that r060_target_comparison prepares in $$LINC_WORK/r060
# the gallery's axial plane (Supplementary S2): reads NMT v2's SARM from the same published templates
gallery_plane_2026-11:
	@if $(HAVE_ATLASES); then $(PY_DEEP) paper_numbers/$@.py; \
	else echo "SKIPPED gallery_plane_2026-11: NMT v2 not found under settings/link.yaml inputs.atlas_reference (see settings/targets.yaml)"; fi

lgnstn_targets_r060c: r060_target_comparison
	@if $(HAVE_ATLASES); then $(PY) paper_numbers/$@.py; \
	else echo "SKIPPED lgnstn_targets_r060c: it needs r060_target_comparison's prepared targets (NMT v2 and CIVM-R)"; fi

mm_mri_check_r054:
	@if [ -z "$(MM_MRI_ARGS)" ]; then echo "SKIPPED mm_mri_check_r054: set MM_MRI_ARGS (raw MRI outside the release)"; \
	else $(PY) paper_numbers/$@.py $(MM_MRI_ARGS); fi

# records that read another regenerated record from $$LINC_NUMBERS
tube_mf8_check_r054: tube_offsets_r054
lgnstn_geometry_current: labels_overlap_current
fig_histology_simple: registration_qc_histo2bf_20261001
bf_spacing_summary_r073: bf_spacing_used
mf8_s46_drop_r068: registration_qc_histo2bf_20261001     # writes registration_qc_histo2bf_r068, tube_offsets_r068, tube_mf8_check_r068
# histology_h9 reads only the products; it writes registration_qc_histo2bf_h9, tube_offsets_h9, tube_mf8_check_h9
r057_missing_extent: d3_methods d3_predicted_share
alignment_comparison: route_test_1c
s_synthesis_depth: r057_s1_synthesis
s_synthesis_gallery: d3_predicted_share gallery_plane_2026-11
s_networks: $(NUMBERS_SEGMENT) d3_methods
reconstruction: bf_jump_pullback

$(filter-out r057_mri_facts_b,$(NUMBERS_MRI)):
	$(PY_MRI) paper_numbers/$@.py

# these two read raw data under LINC_RAW (macaque_mri and microstructure; linc_000003's histology), skipped when it is unset
r057_mri_facts_b:
	@if [ -z "$(LINC_RAW)" ]; then echo "SKIPPED r057_mri_facts_b: set LINC_RAW (raw MRI outside the release)"; \
	else $(PY_MRI) paper_numbers/$@.py; fi

histo_pixel_sizes_r057:
	@if [ -z "$(LINC_RAW)" ]; then echo "SKIPPED histo_pixel_sizes_r057: set LINC_RAW (raw histology outside the release)"; \
	else $(PY_TIRL) paper_numbers/$@.py; fi

# the camera jump the reconstruction figure's panel (b) shows, registered again as the reconstruction's step 2 did (TIRL, a few minutes)
bf_jump_pullback:
	$(PY_TIRL) paper_numbers/$@.py

$(filter-out d3p_vs_d3_pseudo_prod d3_domain_shift_prod,$(NUMBERS_COMPLETION) $(NUMBERS_SEGMENT) $(NUMBERS_GPU)):
	$(PY_DEEP) paper_numbers/$@.py

# the stand-in test on the final stand-in b0s (the paper's): the same two scripts with --final
d3p_vs_d3_pseudo_prod d3_domain_shift_prod:
	$(PY_DEEP) paper_numbers/$(patsubst %_prod,%,$@).py --final

check:
	$(PY) ../../tests/check_numbers.py --ref ../../tests/reference --new $(LINC_NUMBERS)

# lists for the checks of the build plan (steps.yaml), so they follow this file: the figure targets, and the
# figure records that no numbers target writes (those that numbers-figures writes)
FIGURE_RECORDS := $(filter-out $(NUMBERS_LINC) $(NUMBERS_MRI) $(NUMBERS_TIRL) $(NUMBERS_COMPLETION) $(NUMBERS_SEGMENT) $(NUMBERS_GPU),\
  $(patsubst ../../tests/reference/%.json,%,$(wildcard ../../tests/reference/fig_*.json)))
print-figures:
	@echo $(FIGURES)
print-figure-records:
	@echo $(FIGURE_RECORDS)

# the installation check, no data needed (selftest.sh): the tests, each in its environment, and toy-data smokes of the command lines
test:
	PY="$(PY)" PY_DEEP="$(PY_DEEP)" PY_TIRL="$(PY_TIRL)" bash ../../tests/selftest.sh

# alignment_comparison also writes the fig_routes_1c_section record, tracer the fig_tracer_r058 record
figures: $(FIGURES)

histology: histology_h9
	@if [ -z "$(HISTOLOGY_ARGS)" ]; then echo "SKIPPED histology: set HISTOLOGY_ARGS (histology source data outside the release)"; \
	else $(PY_TIRL) histology.py $(HISTOLOGY_ARGS) --out $(OUT)/figures/$@.png; fi

# the chart-to-template figure (Supplementary S3): the darkfield sections and charts of the LINC histology dataset
s_tracer_fidelity:
	@if [ ! -d "$(LINC_RAW)/linc_000003/rawdata" ]; then echo "SKIPPED s_tracer_fidelity: put the LINC histology dataset 000003 at \$$LINC_RAW/linc_000003"; \
	else $(PY_TIRL) $@.py --out $(OUT)/figures/$@.png; fi

# Fig. 10, the tracer figure: each charted section's labels in 3D (a) and the internal capsule slabs (b), with its rotation video;
# renders headless in settings/env/viz3d.yml (record fig_tracer_sections3d_v6_h9_c, the cases of settings/figures.yaml tracer_sections3d): skipped with a message when PY_VIZ3D is unset
# the README's animation, docs/media/tracers_rotation.gif: Fig. 9a's two views (MM1, MM2, MM3 and MF8) turning together, +-45 degrees
# about the vertical axis (72 frames, 12 fps, 960 px, as the template's); settings/env/viz3d.yml, about 1.5 min and 1.6 GB of RAM
tracers-animation:
	@if [ -z "$(PY_VIZ3D)" ]; then echo "SKIPPED tracers-animation: set PY_VIZ3D (settings/env/viz3d.yml; install.sh --with-figures-3d)"; \
	else $(PY_VIZ3D) tracer_sections3d.py --animation ../../docs/media/tracers_rotation.gif; fi

tracer_sections3d:
	@if [ -z "$(PY_VIZ3D)" ]; then echo "SKIPPED tracer_sections3d: set PY_VIZ3D (settings/env/viz3d.yml; install.sh --with-figures-3d)"; \
	else $(PY_VIZ3D) $@.py --video --out $(OUT)/figures/$@.png; fi

$(filter-out pipeline,$(FIGURES_MRI)):
	$(PY_MRI) $@.py --out $(OUT)/figures/$@.png

# the pipeline figure prints the cohort's size from the regenerated bf_cohort: skipped with it when LINC_PARTICIPANTS is unset
pipeline: bf_cohort histo_tube_landmarks
	@if [ -z "$(LINC_PARTICIPANTS)" ]; then echo "SKIPPED pipeline: set LINC_PARTICIPANTS (it reads bf_cohort)"; \
	else $(PY_MRI) $@.py --out $(OUT)/figures/$@.png; fi

$(FIGURES_LINC):
	$(PY) $@.py --out $(OUT)/figures/$@.png

# these two write a PDF (the paper's) and a PNG beside it
$(FIGURES_COMPLETION):
	$(PY_DEEP) $@.py --out $(OUT)/figures/$@.pdf

# Fig. 8, the same: in settings/env/linc.yml, the environment the paper's figure was made in (its matplotlib draws other pixels than the deep one's)
lgnstn:
	$(PY) $@.py --out $(OUT)/figures/$@.pdf

# Supplementary: the blockface registered to its own MRI, step by step (MM2; settings/figures.yaml bf2mri_steps): a PDF and a PNG beside it
s_bf2mri_steps:
	$(PY) $@.py --out $(OUT)/figures/$@.pdf

# the README's animation, docs/media/template_coronal.gif: the template's six channels in a 2 x 3 grid (T2-weighted, b0, tensor;
# soma and neurite fractions, blockface), coronal planes 0.75 mm apart from front to back (84 frames, 10 fps, 960 px wide); reads
# $LINC_DATA_TREE/template/outputs, about 2 min and 1.1 GB of RAM on a CPU
.PHONY: template-animation
template-animation:
	@if [ -z "$(LINC_DATA_TREE)" ]; then echo "SKIPPED template-animation: set LINC_DATA_TREE (settings/roots.sh)"; \
	else $(PY) readme_animation.py --out ../../docs/media/template_coronal.gif; fi
