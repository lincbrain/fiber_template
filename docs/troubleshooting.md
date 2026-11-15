# Troubleshooting

What you may see, why, and what to do, for each part of the pipeline.
For a registration's check, see each tutorial's checks: [01](tutorials/01_mri.md#4-checks), [02](tutorials/02_blockface.md#4-checks), [03](tutorials/03_histology.md#4-checks).

## MRI

Steps in [01](tutorials/01_mri.md).

| What you see | Why | What to do |
|---|---|---|
| `linc-mri prep` prints that N4 was skipped, and exits 0 | ANTs is not on the path; the step went on without the bias correction, and the provenance file says "not found: skipped" | put ANTs 2.5.4 on the path and run the step again |
| `linc-mri prep` stops: the mask is "not on the image's grid", or has values other than 0 and 1 | `--mask` must be on the b0's own grid, with only 0 and 1 | resample or binarize the mask onto the b0's grid first |
| The network's mask takes in tissue outside the brain, or misses some | the network was trained on this collection's ex vivo b0 and T2-weighted images | correct the mask manually, save it in `brains/sub-<ID>/inputs/manual_annotations/`, and rerun with `--mask` |
| Step B2 stops: the transform exists | a registration of this brain is already there | add `--overwrite` (the old one is backed up), or `--registry <folder>` |
| The outline check fails anteriorly, the frontal pole pushed forward | the b0's mask took in the olfactory bulbs or tracts, which the template's mask does not have; it happened on two of the paper's brains during development | remove them from the mask and run B2 again |
| `meta.json` shows `fallback_to_similarity: true` | the affine fit stretched the b0 too much (the ratio of its scales above 1.5), so the registration started from the similarity transform; MR253, with the weakest b0, had 1.80 | nothing: check the outline verdict as usual |
| A b0 with a strong leftover intensity gradient | when the remaining bias field's range exceeds 1.5, the registration corrects it again with N4 inside the mask, for the deformable stage only (MR253 needed it) | nothing: it is automatic |
| The construction (A9) writes nothing for hours, the GPU idle on repeated samples | MMORF can hang with a tensor channel at 8 inputs per batch on the fine steps | stop it and resume at the step and iteration it reached (`-aff False -resume <step>:<iteration>` through the driver; [code/template](../code/template/README.md)) with 4 inputs per batch for steps 6 to 8; the batch size does not change the result |
| A GPU sample shows 0 % during the construction | MMORF alternates between CPU and GPU work on the fine steps | sample several times over a few seconds before deciding |
| A5 runs out of GPU memory | it needs 22 GB | use a 48 GB card |
| A7's comparison shows low correlations (a T2 r near 0.88) | the canonical frame of A6 can land on a slightly different plane (up to 0.8 degrees on MF286), and every image goes through it | compare after undoing the frame: the images agree |
| A build step does nothing | its outputs exist | add `--force` |

## Blockface

Steps in [02](tutorials/02_blockface.md).

| What you see | Why | What to do |
|---|---|---|
| A step between two sections in the volume | a camera jump that is not listed | add it to `jumps`; run `prepare` and the later steps again |
| A smeared or empty section | a damaged or missing photograph | add `[before, after]` to `missing` |
| The brain upside down | the photographs were taken upside down | `rotate180: true` |
| The brain upside down, and no marker tubes found | `rotate180: true` on photographs that were already upright: the tubes ended up on the right and were dropped | look at `00_slices/`, set `rotate180` to match, and reconstruct again |
| A photograph pushed off the canvas | a large camera jump | raise `stack_pad` |
| The mask or the tubes wrong | the network missed or added tissue | correct them manually ([02, Step 2](tutorials/02_blockface.md#step-2-correct-the-masks-manually-optional)) |
| Corrected tubes do not show in the volume | they were saved after `tubes`, which does not fill them again | run `volume` and `tubes` again, with the working folders (`--keep-work`; Step 2) |
| `prepare`, `mask` or `volume` cannot be run again | `volume` and `tubes` deleted the working folders | run the steps one at a time with `--keep-work`, or start again from `photos` |
| Corrected masks no longer line up after a change of `vz` | the manual-edit grid follows round(400 µm / `vz`) | pin `curation_f` to the value the corrections were made with |
| A run is killed with no Python error | out of memory: the full-resolution steps need about 90 GB for 1,400 photographs | use a machine with more free memory; to cap it, cap the resident memory, never with `ulimit -v`, which kills every TIRL run |
| The last sections darker than the rest | their illumination | `native_end_crop_mm` |
| A registration stops before it starts | the GPU has less memory free than it needs | it prints what it needs; use a card with more free memory, one registration at a time |
| "already exists" | a transform folder from an earlier run | `--overwrite` (the old one is backed up) |
| `qc` fails on scales | the pixel size or section step is off | check `vx` and `vz` |
| `linc-complete prepare` stops: "no plane from which the photographs are complete over ... mm" | the last photographed sections are torn or missing | with MRI, drop them with `native_end_crop_mm` and register again; otherwise move the caudal cutoff before them (`missing: [n, -1]`) and reconstruct again; then run `prepare` again |
| A brightness step at the junction plane of the completed brain | the 8 mm next to the junction holds damaged or darker sections | drop those sections as in the row above, then run the completion again |
| The template build's cost turns to `-nan` | a member has an all-zero image or mask, which MMORF cannot take | check that no member's channels or cost masks are empty |
| MMORF fails to allocate its memory | other jobs on the same card | run the template build alone on its card |
| "No space left on device" during the template build | the run folders are large | check the free disk first (120 GB for the first stage alone) |

## Histology

Steps in [03](tutorials/03_histology.md).

| What you see | Why | What to do |
|---|---|---|
| A registered section lies on the wrong anatomy, or its outline fits but its sulci do not | the match table pairs the component with the wrong photograph (neighboring photographs look alike) | correct its row in `match_table_edited.txt` and run `all` again into a new `--out` folder: a folder that already holds a registration is resumed, not redone |
| `propose` leaves a component unmatched (-1) | its score on its run's line was too low to guess | find its photograph by eye (`match_review/`) and write it into `match_table_edited.txt`, or leave it out |
| A registration differs slightly from the paper's on rerun | TIRL's result depends on the thread count | use `--threads 12` (the default) |
| A run stopped | the machine went down, or a component crashed | run the same command again: it resumes where it stopped |
| `all` stops at one component with "Matrix does not exist on the same manifold" | TIRL's rotation search failed on that component; it happened on one component of MF275, as in the paper's run, which has no result for it | run `linc-histology rgb` with the same options, to make the color images of the components that did register |
| The labels of one section sit about a millimeter off on one side, while the section's registration looks right | the drawn frame's fitted transform onto its component is wrong ([03, Step E](tutorials/03_histology.md#step-e-lgn-and-stn)); it happened on MR252's frame 36, with the left LGN and STN 1.1 mm off | fit that frame's transform again (`$H/structures/transforms/f<frame>_<piece>.tfm`, a SimpleITK transform) and push the labels again; the data folder's MR252 files carry the refit frame |
| A section's tissue mask takes in another object, or misses tissue | an unusual contrast: on one of MF191's sections, with the contrast saturated, the mask took in another object | check `tissue_masks/` by eye; if a mask is wrong, leave that component out of the table (-1) |
| No photograph for a section | the section was not photographed, or the volume stops before it | the section cannot be registered; leave it out of the table (-1) |
