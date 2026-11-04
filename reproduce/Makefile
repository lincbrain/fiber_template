# Reproduce the paper's numbers and figures from the released products.
#
#   cd reproduce
#   make numbers        # every number record the paper reads (CPU), into $LINC_NUMBERS (default ../outputs/numbers)
#   make numbers-gpu    # the one record that needs a GPU (d3_domain_shift; CUDA card, env/deep.yml)
#   make check          # compare the regenerated records with the paper's (reference/), exceptions as documented
#   make test           # the installation check: tests + toy smokes of the command lines, no data (selftest.sh, minutes)
#   make figures        # the paper's figures, into ../outputs/figures/ (one target per figure, named by its content;
#                       # ../docs/paper_to_code.md maps the paper's figure numbers to them)
#   make all            # numbers + figures + check (two figure targets also write a numbers record)
#
# Set the roots first (see ../CONVENTIONS.md): LINC_PRODUCTS, LINC_DATA, LINC_TEMPLATE, LINC_RAW, LINC_MODELS, LINC_NUMBERS.
# Each target runs in the environment it ran in for the paper (env/README.md): PY = env/linc.yml; PY_MRI = the same python (an alias kept for the MRI records) (the heavy MRI
# records and the figures FIGURES_MRI lists); PY_DEEP = env/deep.yml (the d3_* records, Supplementary S1 and the LGN/STN
# budget, and their three figures); PY_TIRL = env/tirl.yml (histology, s_tracer_fidelity;
# histo_pixel_sizes_r057).

PY ?= python
PY_MRI ?= $(PY)
# PY_COMPLETION: PY_DEEP's name until 3 Nov 2026 (env/completion.yml), still honored
PY_DEEP ?= $(or $(PY_COMPLETION),$(PY))
PY_TIRL ?= $(PY)
# PY_VIZ3D = env/viz3d.yml (the 3D tracer scene, figures-3d; optional: install.sh --with-figures-3d)
PY_VIZ3D ?=
# raw data outside the release, needed by two targets (they are skipped, with a message, when unset):
#   MM_MRI_ARGS = --raw-bruker-root $$LINC_RAW/mri_mm --unpacked-root $$LINC_RAW/macaque_mri    (mm_mri_check_r054)
#   HISTOLOGY_ARGS = --jp2-root <LINC source data>/sourcedata                                  (the histology figure)
#   the LINC histology dataset 000003 at $$LINC_RAW/linc_000003                                  (s_tracer_fidelity)
#   LINC_PARTICIPANTS = the LINC participants.tsv                                           (bf_cohort)
MM_MRI_ARGS ?=
HISTOLOGY_ARGS ?=
LINC_PARTICIPANTS ?=
OUT ?= $(abspath ../outputs)
export LINC_NUMBERS ?= $(OUT)/numbers
# scratch of the records that write intermediate files (linc/paths.py work_root; unset, it would be ./work inside reproduce/)
export LINC_WORK ?= $(OUT)/work
# never import packages from ~/.local: every target uses exactly its environment file's packages
export PYTHONNOUSERSITE ?= 1

NUMBERS_LINC := \
  bf_camera_exif bf_cohort bf_sanity bf_spacing_used bf_species bf_template_metrics_d2b \
  d2_s2_validation_prod_s125 precision_weights_b1_d2b_s8 r1_edge_summary route_test_1c \
  atlas_final_v2_summary fig_histology_simple histo_tube_landmarks \
  mm_mri_check_r054 mri_recipe registration_qc_histo2bf_20261001 s5_link_d2bs8 table1_rebuild \
  table_specimens_check_r054 tracer_sections_r054 tracer_template_current_all tube_mf8_check_r054 tube_offsets_r054 \
  s1_segmentation_r057 s1_perbrain_r057 tracer_label_volumes_r058d tracer_alic_topography_r058d r060_target_comparison \
  lgnstn_targets_r060c \
  tracer_sections_v2 tracer_connected_v2c tracer_label_volumes_v2c tracer_alic_topography_v2c tracer_interp_accounting_v2 \
  tracer_placement_v2c tracer_template_v2_all
NUMBERS_MRI := variants_leave_self_out_rebuild_fa025 variability_rebuild_rb_final lobo_t2_rebuild \
  labels_overlap_current lgnstn_geometry_current \
  lgnstn_model_agreement_r059 lgnstn_model_loso_r059 lgnstn_b0_visibility_r059 lgnstn_midplane_planes_r059 r057_mri_facts_b
NUMBERS_TIRL := histo_pixel_sizes_r057
NUMBERS_COMPLETION := d3_sigma2_pred d3_methods d3_predicted_share \
  d3p_vs_d3_pseudo r057_missing_extent r057_s1_synthesis lgnstn_budget_r059
NUMBERS_GPU := d3_domain_shift
# the segmentation models' records (Supplementary S5, Table S8): CPU, env/deep.yml (torch counts the parameters); bf_mask2d ~6 min
NUMBERS_SEGMENT := segmodels_mri_brainmask_2026-11 segmodels_bf_mask2d_2026-11 segmodels_bf_mask3d_2026-11 segmodels_bf_holes_2026-11

FIGURES_MRI := template pipeline variability link tracer s_tracer_rules tracer_pathways
FIGURES_LINC := reconstruction bf_template alignment_comparison
FIGURES_COMPLETION := s_synthesis_depth s_synthesis_gallery lgnstn
FIGURES := $(FIGURES_MRI) $(FIGURES_LINC) $(FIGURES_COMPLETION) histology s_tracer_fidelity
FIGURES_3D := tracer_3d                                  # optional (PY_VIZ3D): not in `make figures`; `make tracer_3d`

.PHONY: all numbers numbers-gpu check test figures $(NUMBERS_LINC) $(NUMBERS_MRI) $(NUMBERS_TIRL) $(NUMBERS_COMPLETION) $(NUMBERS_SEGMENT) $(NUMBERS_GPU) $(FIGURES) $(FIGURES_3D)

all: numbers figures check

numbers: $(NUMBERS_LINC) $(NUMBERS_MRI) $(NUMBERS_TIRL) $(NUMBERS_COMPLETION) $(NUMBERS_SEGMENT)

numbers-gpu: $(NUMBERS_GPU)

$(filter-out mm_mri_check_r054 bf_cohort bf_species s1_perbrain_r057 s1_segmentation_r057 r060_target_comparison lgnstn_targets_r060c,$(NUMBERS_LINC)):
	$(PY) numbers/$@.py

bf_cohort:
	@if [ -z "$(LINC_PARTICIPANTS)" ]; then echo "SKIPPED bf_cohort: set LINC_PARTICIPANTS (the LINC participants.tsv)"; \
	else LINC_PARTICIPANTS="$(LINC_PARTICIPANTS)" $(PY) numbers/$@.py; fi

# reads the regenerated bf_cohort, so it needs LINC_PARTICIPANTS as bf_cohort does (skipped with it)
bf_species: bf_cohort
	@if [ -z "$(LINC_PARTICIPANTS)" ]; then echo "SKIPPED bf_species: set LINC_PARTICIPANTS (it reads bf_cohort)"; \
	else $(PY) numbers/$@.py; fi

# reads the regenerated bf_cohort, bf_camera_exif and bf_spacing_used, so it needs LINC_PARTICIPANTS as bf_cohort does
s1_perbrain_r057: bf_cohort bf_camera_exif bf_spacing_used
	@if [ -z "$(LINC_PARTICIPANTS)" ]; then echo "SKIPPED s1_perbrain_r057: set LINC_PARTICIPANTS (it reads bf_cohort)"; \
	else $(PY) numbers/$@.py; fi

# reads the released blockface segmentation weights' folder (dataset.json) under LINC_MODELS
s1_segmentation_r057:
	@if [ -z "$(LINC_MODELS)" ]; then echo "SKIPPED s1_segmentation_r057: set LINC_MODELS (the network weights)"; \
	else $(PY) numbers/$@.py; fi

# reads the published templates NMT v2 and CIVM-R, which are not redistributed: download them as configs/targets.yaml says, into
# configs/link.yaml inputs.atlas_reference (under LINC_PRODUCTS); skipped when they are not there. CPU, about 15 min at --jobs 4
R060_JOBS ?= 4
HAVE_ATLASES = $(PY) -c "import os, sys, yaml; r = os.path.join(os.environ.get('LINC_PRODUCTS', ''), yaml.safe_load(open('../configs/link.yaml'))['inputs']['atlas_reference']); sys.exit(0 if all(os.path.isdir(os.path.join(r, d)) for d in ('NMT_v2.0_sym', 'CIVM-R_T2_DTI')) else 1)"
r060_target_comparison:
	@if $(HAVE_ATLASES); then $(PY) numbers/$@.py --jobs $(R060_JOBS); \
	else echo "SKIPPED r060_target_comparison: NMT v2 and CIVM-R not found under configs/link.yaml inputs.atlas_reference (see configs/targets.yaml)"; fi

# the LGN and STN rows of the same comparison: carried onto the target grids that r060_target_comparison prepares in $$LINC_WORK/r060
lgnstn_targets_r060c: r060_target_comparison
	@if $(HAVE_ATLASES); then $(PY) numbers/$@.py; \
	else echo "SKIPPED lgnstn_targets_r060c: it needs r060_target_comparison's prepared targets (NMT v2 and CIVM-R)"; fi

mm_mri_check_r054:
	@if [ -z "$(MM_MRI_ARGS)" ]; then echo "SKIPPED mm_mri_check_r054: set MM_MRI_ARGS (raw MRI outside the release)"; \
	else $(PY) numbers/$@.py $(MM_MRI_ARGS); fi

# records that read another regenerated record from $$LINC_NUMBERS
tube_mf8_check_r054: tube_offsets_r054
lgnstn_geometry_current: labels_overlap_current
fig_histology_simple: registration_qc_histo2bf_20261001
r057_missing_extent: d3_methods d3_predicted_share
s_synthesis_depth: r057_s1_synthesis
s_synthesis_gallery: d3_predicted_share

$(filter-out r057_mri_facts_b,$(NUMBERS_MRI)):
	$(PY_MRI) numbers/$@.py

# these two read raw data under LINC_RAW (macaque_mri and microstructure; linc_000003's histology), skipped when it is unset
r057_mri_facts_b:
	@if [ -z "$(LINC_RAW)" ]; then echo "SKIPPED r057_mri_facts_b: set LINC_RAW (raw MRI outside the release)"; \
	else $(PY_MRI) numbers/$@.py; fi

histo_pixel_sizes_r057:
	@if [ -z "$(LINC_RAW)" ]; then echo "SKIPPED histo_pixel_sizes_r057: set LINC_RAW (raw histology outside the release)"; \
	else $(PY_TIRL) numbers/$@.py; fi

$(NUMBERS_COMPLETION) $(NUMBERS_SEGMENT) $(NUMBERS_GPU):
	$(PY_DEEP) numbers/$@.py

check:
	$(PY) check_numbers.py --ref reference --new $(LINC_NUMBERS)

# the installation check, no data needed (selftest.sh): the tests, each in its environment, and toy-data smokes of the command lines
test:
	PY="$(PY)" PY_DEEP="$(PY_DEEP)" PY_TIRL="$(PY_TIRL)" bash selftest.sh

# alignment_comparison also writes the fig_routes_1c_section record, tracer the fig_tracer_r058 record
figures: $(FIGURES)

histology: tube_offsets_r054
	@if [ -z "$(HISTOLOGY_ARGS)" ]; then echo "SKIPPED histology: set HISTOLOGY_ARGS (histology source data outside the release)"; \
	else $(PY_TIRL) figures/histology.py $(HISTOLOGY_ARGS) --out $(OUT)/figures/$@.png; fi

# the chart-to-template figure (Supplementary S3): the darkfield sections and charts of the LINC histology dataset
s_tracer_fidelity:
	@if [ ! -d "$(LINC_RAW)/linc_000003/rawdata" ]; then echo "SKIPPED s_tracer_fidelity: put the LINC histology dataset 000003 at \$$LINC_RAW/linc_000003"; \
	else $(PY_TIRL) figures/$@.py --out $(OUT)/figures/$@.png; fi

# the tracer figure (Fig. 10): the four-panel form always (record fig_tracer_pathways_panels); with PY_VIZ3D the three composite
# stills render (no in-scene key, large letters; record fig_tracer_3d_stills) and the composite follows (the paper's figure, record
# fig_tracer_pathways); `make tracer_3d` renders the full scene with its two rotation videos for the supplement
tracer_pathways:
	$(PY_MRI) figures/tracer_pathways.py --layout panels --out $(OUT)/figures/tracer_pathways_panels.png
	@if [ -z "$(PY_VIZ3D)" ]; then echo "SKIPPED the composite of tracer_pathways: set PY_VIZ3D (env/viz3d.yml) for the 3D stills; the four-panel form is drawn"; \
	else $(PY_VIZ3D) figures/tracer_3d.py --for-composite --views oblique lateral glass --no-video --record fig_tracer_3d_stills --out $(OUT)/figures/tracer_3d_stills && \
	$(PY_MRI) figures/tracer_pathways.py --stills $(OUT)/figures/tracer_3d_stills_oblique.png $(OUT)/figures/tracer_3d_stills_lateral.png $(OUT)/figures/tracer_3d_stills_glass.png --out $(OUT)/figures/tracer_pathways.png; fi

# the 3D tracer scene (Fig. 10 a, b and the rotation video) renders headless in env/viz3d.yml: skipped with a message when PY_VIZ3D is unset
tracer_3d:
	@if [ -z "$(PY_VIZ3D)" ]; then echo "SKIPPED tracer_3d: set PY_VIZ3D (env/viz3d.yml; install.sh --with-figures-3d)"; \
	else $(PY_VIZ3D) figures/tracer_3d.py --glass-video --out $(OUT)/figures/tracer_3d; fi

$(filter-out pipeline tracer_pathways,$(FIGURES_MRI)):
	$(PY_MRI) figures/$@.py --out $(OUT)/figures/$@.png

# the pipeline figure prints the cohort's size from the regenerated bf_cohort: skipped with it when LINC_PARTICIPANTS is unset
pipeline: bf_cohort
	@if [ -z "$(LINC_PARTICIPANTS)" ]; then echo "SKIPPED pipeline: set LINC_PARTICIPANTS (it reads bf_cohort)"; \
	else $(PY_MRI) figures/$@.py --out $(OUT)/figures/$@.png; fi

$(FIGURES_LINC):
	$(PY) figures/$@.py --out $(OUT)/figures/$@.png

# these three write a PDF (the paper's) and a PNG beside it
$(FIGURES_COMPLETION):
	$(PY_DEEP) figures/$@.py --out $(OUT)/figures/$@.pdf
