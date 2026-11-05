# Rebuilding the blockface channel of the template

The blockface channel of the template (in its own space `bftemplate/d2b/final_s8` under `$LINC_PRODUCTS`; in the MRI template's frame
`$LINC_TEMPLATE/blockface/linc_bf_*`, made by `build_link.py`) is a symmetric groupwise construction of the 21 photographed brains, each used twice (as reconstructed and exactly mirrored), with two
channels per input, the blockface luminance and the b0, built with MMORF 1.0.1 through the patched template-construction driver
(paper: Methods, "Blockface template"). A 43rd input, TPL, is the MRI template's b0, used once as a b0-only member; its warp is the
registration from the MRI template to this template.

**Driver.** `reproduce/templates/build_blockface_template.py` runs the commands below in order from the released products:
`--plan` prints every step with its environment, estimated time, inputs and outputs and runs nothing; `--step NAME` (or `--from` /
`--to`) runs steps; each step that runs writes `<work>/provenance/<NN>_<step>.json`, and a step whose outputs exist is not rerun
without `--force`. Its one step the commands below leave implicit is `bf-view` (the step-8 run without TPL, which the final blockface
average uses).

The build has five stages: **members → the first stage (7 steps at 0.5 mm) → step 8 (0.25 mm) → final average → publish.** MMORF
runs on a GPU, at 8 tasks at a time in the first stage and 4 in step 8 on one 48 GB card; everything else runs on CPU. The record ran the same tasks on three such cards (RTX 6000 Ada): the first stage in 4 h 24 min (44.6 task-hours), step 8 in 7 h 11 min (59.5 task-hours). On one card that is about 8 and 17 hours, estimated from those task times; the rerun below agrees with the first on a free card. The code is `linc/construction/blockface_build.py`,
`blockface_finish.py`, `blockface_mmorf.py` and `blockface_shim/`; every step below is `python -m linc.construction.blockface_build
<step> ...` from the repository root.

## Environment

- `env/linc.yml` for the members, the run trees and the finish. The patched driver, and the two steps that import it for its own
  averaging functions (marked `$PY_MRI` below), run in the same `env/linc.yml` (`PY_MRI` names its Python, as in
  `reproduce/Makefile`), which holds fsl-mmorf 1.0.1, fslpy, file-tree and fsl-sub (until 3 Nov 2026 they had their own
  `env/mri.yml`).
- The driver, edited (our change ships only as our own lines: `linc/construction/patches/README.md`):
  ```
  git clone https://git.fmrib.ox.ac.uk/cart/mm-template-construction.git && cd mm-template-construction && git checkout f4907f4
  python <this repo>/linc/construction/patches/apply_edits.py <this repo>/linc/construction/patches/mm-template-construction_v1_blockface.edits.json .
  export LINC_BLOCKFACE_DRIVER=$PWD/template_construction_v1    # its own variable: the MRI build's driver is another edit of the same file, and the build refuses it
  ```
  The patched file is the driver the template of record was built with (md5 `fa7771595c86ef23e797a1bb5016c07f`) up to the author tag
  of the local changes (17 lines: comments, one `--help` text, one docstring; no code change; patched md5
  `fd775b9b9e8b615826bc48459d65ded2`).
  The edits include the lab's local changes of `mm-template-construction_v1_linc.edits.json` (the MRI build's) except the tensor
  `--rotwarp` change, which came later and has no effect here (no tensor channel), plus two changes of its own, both marked in
  the file and off unless `LINC_MASKWEIGHTED=1`:
  - **mask-weighted construction:** each input's cost mask weights its affine FLIRT cost (`-inweight`) and is its weight in
    every average (affine templates, average warps, nonlinear templates): sum_i (w_i + 1e-3) I_i / sum_i (w_i + 1e-3);
  - **zero-weight inputs:** an input whose weight is zero everywhere (TPL's blockface) is left out of the intensity normalization
    region.
- FSL 6.0.7.4 (`FSLDIR`); MMORF 1.0.1 (`LINC_MMORF_EXE` = the `mmorf_cuda` of env/linc.yml), and `linc/construction/blockface_shim` first on
  `PATH` when the driver runs: the shim rewrites TPL's MMORF call only (blockface weight 0, a stand-in image and a ones mask, and a
  ones blockface bias field afterwards), because MMORF's cost is undefined for an all-zero moving image or mask.
- Inputs: each brain's volume with its missing sections synthesized and its synthesis records (`linc/completion`,
  `linc-complete`); the rigid part of the cross-modal alignment of the 8 brains without MRI (`linc-register direct-placement`);
  `$LINC_TEMPLATE/mri/linc_mri_b0` and its brain mask;
  the run-113 frame and the bridge to the canonical frame (`configs/products_blockface.yaml`).

## 1. Members

```bash
B="python -m linc.construction.blockface_build"
$B prep-frame --out frame2prep_0p5.nii.gz                         # frame1c grown 8 mm: no member can be clipped
$B members --frame frame2prep_0p5.nii.gz --out canon_prep --root <completion root> --cache <completion cache>
$B crop --canon canon_prep --pad 4 --mult 4 --out canon            # the build frame: the union + 4 mm, x-symmetric
$B tpl-canon --vox 0.5 --source-canon canon --out canon_tpl        # TPL beside symlinks to the 21
$B tpl-costmask --canons canon_tpl --record tpl_costmask.json      # TPL's b0 cost: 0 below the brainstem plane + 1 mm, cosine over 2 mm
```

Per brain and mirror: the luminance with synthesized tissue, color, photographed fraction and brain mask, resampled once onto the
frame (anti-aliased); the b0 the synthesis used, on one rule for all, clip(b0 / mhi, 0, 1) x brain; the blockface cost mask (1 for a
brain with MRI; for one without, 1 on its kept sections, 0 within 1 mm of its junction plane and a cosine to 1 over 2 mm, through
the rigid part of its cross-modal alignment); the b0 cost mask 1 everywhere.

## 2. The first stage: 7 steps at 0.5 mm

```bash
$B run-plan-a --canon canon_tpl --root runs --name d2b              # data tree, configs (7 steps, lambda x10, smoothing floors), standardspace
$B copy-affine --plan-a-affine <the affine stage> --run runs/d2b     # the 42 inputs' affine stage + TPL at the identity
$B tpl-affine --run runs/d2b --ref-template <affine b0 template>     # TPL's 12-dof affine, symmetrized
$PY_MRI -m linc.construction.blockface_build affine-templates --run runs/d2b   # the affine templates over all 43
$B tpl-standins --run runs/d2b --canon2 canon_tpl                    # TPL's stand-in blockface image and ones mask for its MMORF call
LINC_MASKWEIGHTED=1 FSLOUTPUTTYPE=NIFTI_GZ PATH=linc/construction/blockface_shim:$FSLDIR/share/fsl/bin:$PATH \
  $PY_MRI $($B driver-command --run runs/d2b -aff False -nln True -nres 7 -nit 3 -b 8)
```

The affine stage of the 42 inputs is the one of design 2's build (the same inputs; its FLIRT weight is the blockface cost, so
TPL is aligned by its own affine above). The analysis copied this stage from design 2's build; its copy is a product of record
(`bftemplate/d2b/plan_a_affine`).

## 3. Step 8: 0.25 mm

```bash
$B s8-frame --canon canon --out frame_canon_0p5.nii.gz
$B s8-inputs --canon canon --frame frame_canon_0p5.nii.gz --out canon2 -j 6     # the members on the frame's exact 2x lattice
$B tpl-canon --vox 0.25 --source-canon canon2 --out canon2_tpl && $B tpl-costmask --canons canon_tpl canon2_tpl --record tpl_costmask.json
$B step8-weights --canon2 canon2_tpl --out canon2_s8w            # the author's step-8 rule (below)
$B s8-box --template <the first stage's two step-6 templates> --canon2 canon2 --out box_0p25.nii.gz   # their support + 4 mm
$B run-step8 --canon2 canon2_s8w --plan-a runs/d2b --box box_0p25.nii.gz --root runs --name d2bs8
$B tpl-standins --run runs/d2bs8 --canon2 canon2_tpl
$B s8-mats --affine-dir runs/d2b/out/affine_step2 --canon canon_tpl --canon2 canon2_s8w --box box_0p25.nii.gz --run runs/d2bs8 \
    --standard05 runs/d2b/standardspace.nii.gz
$B s8-step7 --plan-a-it runs/d2b/out/nln_step_07/iteration_03 --canon canon_tpl --canon2 canon2_s8w --run runs/d2bs8 -j 6
$PY_MRI -m linc.construction.blockface_build step7-templates --run runs/d2bs8
LINC_MASKWEIGHTED=1 FSLOUTPUTTYPE=NIFTI_GZ PATH=linc/construction/blockface_shim:$FSLDIR/share/fsl/bin:$PATH \
  $PY_MRI $($B driver-command --run runs/d2bs8 -aff False -nln True --resume_from 8:1 -nres 8 -nit 3 -b 4)
```

MMORF takes no warp input: step 8 registers every input again, from its first-stage affine, to the step-7 template, at 0.25 mm knots
(9 levels; smoothing floored at each channel's own bandwidth: blockface 0.281 mm, b0 0.562 mm). **The b0 weight of step 8:**
at step 8 only, the 8 brains without MRI count less in never-photographed tissue, the same way their synthetic blockface already
does: their b0 cost mask equals their blockface cost mask wherever they were not fully photographed.

## 4. Finish and publish

```bash
$B finish --run runs/d2bs8_bfview --arm d2bs8 --canon canon2 --iteration runs/d2bs8/out/nln_step_08/iteration_03 \
    --precision-json <precision rule> --out final -j 6           # runs/d2bs8_bfview: config and out linked, run_spec.json without TPL
$B finish-b0 --run runs/d2bs8 --arm d2bs8 --canon canon2_tpl --iteration runs/d2bs8/out/nln_step_08/iteration_03 \
    --weights-canon canon2_s8w --out final -j 6
$B embed --run runs/d2bs8 --src final --dst final_run113
```

`<arm>_photo_share` is the photographed fraction (released as `linc_bf_photographed_fraction`). The published template is not MMORF's average: every input is resampled once from its 0.25 mm volume (photographed and synthesized
tissue) through its step-8 warp and averaged with precision weights, w = 1/(tau^2 + sigma^2) per member and part (photographed,
interpolated, synthesized; the synthesis of a brain without MRI has weight 0; tau^2 and the depth curve sigma^2_syn(d) from
`reproduce/numbers/precision_weights_b1_d2b_s8.py`), then symmetrized exactly, 0.5 (V + M(V)). The b0 channel uses the same warps
(TPL through the bridge). `-j 6` reproduces the products of record bit for bit (the partial sums are chunked by input).

## Rerun on one card (4-5 Nov 2026)

The commands above, in a new folder, from `members` for all 21 brains through the seven steps (`-nres 7 -nit 3 -b 8`), on one RTX 6000 Ada,
compared with the record file by file:

| Stage | Result |
|---|---|
| members, build frame, TPL | all 596 volumes identical |
| run folder | every MMORF .ini and .tree, the standard space and the subject list identical; `run_spec.json` differs in six text fields (paths and the wording of decisions), no parameter |
| the affine part | 181 of 182 volumes identical; the other is a backup of TPL's b0 cost mask from before the affine step, which nothing reads |
| nonlinear steps 1 to 5, 3 iterations each, 43 inputs | all 1,315 files of each step identical: warps, Jacobians, bias fields, templates |
| nonlinear step 6 | 71 of 1,315 files identical |
| nonlinear step 7 | 13 of 1,315 files identical |

The record ran the same tasks on three such cards, 24 at a time, so through step 5 neither the schedule nor the host changed the result. This run was also stopped twice and
resumed (`--resume_from`, `LINC_SKIP_DONE_MMORF=1`, `FSLSUB_PARALLEL=8`). Step 8 was not rerun.

**Steps 6 and 7.** The MMORF executable, its parameters and every input through step 5 are identical, both runs report one thread, and the cards are the same model. In step 6's first
iteration, 42 of the 43 task logs print the same costs through six levels and part at the seventh, the first level of the build with 1 mm knots and the first below the parameter
file's `hires = 1.9`; the last log prints the same costs throughout. The 43 warps differ by a median of 6e-8 mm. Each iteration's averages become the next one's target, so the difference is carried
forward: the average warp's median difference is 1e-8 mm at 6:1, then 0.0013, 0.0014, 0.0015, 0.0011 and 0.0014 mm from 6:2 to 7:3, so it does not keep growing. What makes
that level differ was not tested: this rerun does not show whether two runs on one card agree there.

The last template, 7:3, against the record, inside tissue (563,757 voxels above 10% of either template's 99th percentile):

| Compared | Difference |
|---|---|
| average warp | median 0.0008 mm, 99th percentile 0.004 mm, at most 0.04 mm; outside tissue at most 1.15 mm |
| warps of the 43 inputs | median 0.0004 mm; 99th percentile 0.005 mm for the median input, 0.12 mm for the worst; at most 1.4 mm at single voxels; at most 829 voxels (0.15%) of one input over 0.25 mm |
| blockface template | correlation 0.999999; absolute difference a median of 0.012% of the tissue median, 99th percentile 0.16%, at most 3.2% |
| b0 template | correlation 0.999996; absolute difference a median of 0.022% of the tissue median, 99th percentile 0.23%, at most 4.5% |
| blockface cost mask | 7 of 3,035,520 voxels cross its threshold of 0.1; the rescaling that follows changes the weight mask's every voxel slightly |

**Time.** Steps 2 to 7 took 12 h 12 min at low priority (1 h 32 min, 1 h 47 min, 2 h 01 min, 1 h 33 min, 2 h 34 min, 2 h 45 min), with other people's jobs on the card for most of the first ten hours:
about 25 GB until step 5, then about 5 GB in the second half of step 6 and the start of step 7. With the card free the rate matches the record: 36 min for step 6's first iteration and 50 min each for step 7's last two,
which is the record's task times at 8 at a time plus about 10 min per iteration outside MMORF. MMORF peaked at 8.2 GB for 8 tasks; the seven steps take 120 GB on disk.

## Validation (22 Oct 2026; piecewise, MMORF not rerun)

Every deterministic step was recomputed with this code from the products of record and compared file by file (arrays and headers);
MMORF's own registrations were not rerun here (GPU); the rerun above covers them.

| Step | Recomputed | Result |
|---|---|---|
| driver | upstream f4907f4 + `mm-template-construction_v1_blockface.edits.json` | the driver of record (md5 `fa7771595c86ef23e797a1bb5016c07f`) up to the author tag in 17 comment, help and docstring lines (1 Nov: md5 `2bb2aef696ca053e1e942ed171bde61f`) |
| frames | prep frame, step-8 frame, step-8 box | identical |
| members, 0.5 mm | MF6 (MRI) and MF10 (no MRI) on the prep frame | 14/14 files each identical |
| build frame | the crop of all 21 members (the 0.5 mm members of record) | 21 x 14 files identical, same grid |
| members, 0.25 mm | MF6 and MF10 on the step-8 lattice | 14/14 files each identical |
| TPL | its 0.25 mm member and its b0 cost mask | 8/8 files identical |
| step-8 rule | the b0 cost masks of MF8, MF10, MM14 (both mirrors) | identical |
| run folders | the first stage (7 steps) and step 8: every MMORF .ini and .tree, standardspace, subject list, the 172 linked inputs | identical |
| the first stage's affine part | TPL's 12-dof FLIRT affine (2 .mat) and its 4 resampled images; the 6 affine templates (the driver's averaging, in the MMORF environment) | identical |
| step 8's start | the 86 rewritten affines; the 43 inputs' step-7 pulls (258 files); the 6 step-7 templates and the standardspace (the driver's averaging) | identical |
| finish | the blockface and b0 channels from the step-8 warps, and the embed: the 15 published files and the 15 box products | identical |
