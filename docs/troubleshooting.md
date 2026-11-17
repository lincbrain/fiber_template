# Troubleshooting

Problems we have run into, why they happen, and what to do.
To judge whether a registration is good, see each tutorial's checks: [01](tutorials/01_mri.md#4-checks), [02](tutorials/02_blockface.md#4-checks), [03](tutorials/03_histology.md#4-checks).

## Setup

The steps are in the README's [quick start](../README.md#quick-start) and in [00](tutorials/00_start_here.md#before-you-start).

**`source settings/roots.sh` says a folder is not the data folder.**
`LINC_DATA` must be the data folder itself, the one that holds `brains/`, `template/`, `models/` and `paper/`, not a folder inside it such as `brains/`.
Set `LINC_DATA` again and source the file again; the message names the folder that looks right when it is one or two levels up.

**`source settings/roots.sh` says `outputs/` was built from another data folder.**
The working folders in `outputs/` hold copies of one data folder's manual inputs, and your results from it, so they are not reused for another.
Move `outputs/` aside or delete it, then source the file again; or go back to the first folder.

**An error names a file under `outputs/in/`.**
`outputs/in/` holds one link per file of the data folder, under the name the code knows it by, so the data folder does not have that file.
`settings/files.tsv` says where it belongs: search it for the part of the name after `outputs/in/`.

**The self-test fails with import errors.**
It ran with the system's Python instead of the environments.
It takes them from `envs/linc_envs.sh`, which `install.sh` writes; if they were installed elsewhere (`--prefix`), source that folder's `linc_envs.sh` first.

## MRI

The steps are in [01](tutorials/01_mri.md).

**`linc-mri prep` says N4 was skipped, but exits 0.**
ANTs is not on the path, so the step went on without the bias correction, and the provenance file says "not found: skipped".
Put ANTs 2.5.4 on the path and run the step again.

**`linc-mri prep` stops: the mask is "not on the image's grid", or has values other than 0 and 1.**
`--mask` must be on the b0's own grid and hold only 0 and 1; resample or binarize it onto that grid first.

**The network's mask takes in tissue outside the brain, or misses some.**
The network was trained on this collection's ex vivo b0 and T2-weighted images.
Correct the mask manually, save it in `brains/sub-<ID>/inputs/manual_annotations/`, and rerun with `--mask`.

**Step B2 stops because the transform exists.**
This brain already has a registration; add `--overwrite`, which backs up the old one, or use `--registry <folder>`.

**The outline check fails anteriorly, with the frontal pole pushed forward.**
The b0's mask took in the olfactory bulbs or tracts, which the template's mask does not have; this happened on two of the paper's brains during development.
Remove them from the mask and run B2 again.

**`meta.json` shows `fallback_to_similarity: true`.**
The affine fit stretched the b0 too much (a ratio of its scales above 1.5), so the registration started from the similarity transform; MR253, with the weakest b0, had 1.80.
Nothing to do: check the outline verdict as usual.

**The b0 has a strong leftover intensity gradient.**
When the remaining bias field's range exceeds 1.5, the registration automatically corrects it again with N4 inside the mask, for the deformable stage only (MR253 needed this).

**The construction (A9) writes nothing for hours, and the GPU is idle on repeated samples.**
MMORF can hang with a tensor channel at 8 inputs per batch on the fine steps.
Stop it and resume at the step and iteration it reached (`-aff False -resume <step>:<iteration>` through the driver; [code/template](../code/template/README.md)), with 4 inputs per batch for steps 6 to 8; the batch size does not change the result.

**A GPU sample shows 0 % during the construction.**
MMORF alternates between CPU and GPU work on the fine steps, so sample several times over a few seconds before deciding.

**A5 runs out of GPU memory.**
It needs 22 GB; use a 48 GB card.

**A7's comparison shows low correlations, such as a T2 r near 0.88.**
A6's canonical frame can land on a slightly different plane (up to 0.8 degrees on MF286), and every image goes through it.
Compare after undoing the frame: the images agree.

**A build step does nothing.**
Its outputs exist; add `--force`, which replaces them (a registration's earlier transform is backed up in `$LINC_WORK/xfm_backup/`).

**A build step stopped partway and will not start again.**
The files it wrote before it stopped count as its outputs.
Fix what stopped it, then run the step again with `--force`.
In the staging (A1) the usual cause is a program that was not found: it needs FSL (`FSLDIR`) and ANTs 2.5.4 (`ANTSPATH`), and the tensor (A2) needs MRtrix (`MRTRIX`).

## Blockface

The steps are in [02](tutorials/02_blockface.md).

**There is a step between two sections in the volume.**
A camera jump is missing from the list; add it to `jumps`, then run `prepare` and the later steps again.

**A section is smeared or empty.**
Its photograph is damaged or missing; add `[before, after]` to `missing`.

**The brain is upside down.**
The photographs were taken upside down; set `rotate180: true`.

**The brain is upside down, and no marker tubes were found.**
`rotate180: true` was set on photographs that were already upright, so the tubes ended up on the right and were dropped.
Look at `00_slices/`, set `rotate180` to match, and reconstruct again.

**A photograph is pushed off the canvas.**
A large camera jump did this; raise `stack_pad`.

**The mask or the tubes are wrong.**
The network missed or added tissue; correct them manually ([02, Step 2](tutorials/02_blockface.md#step-2-correct-the-masks-manually-optional)).

**Corrected tubes do not show in the volume.**
They were saved after `tubes`, which does not fill them again.
Run `volume` and `tubes` again with the working folders (`--keep-work`; Step 2).

**`prepare`, `mask` or `volume` cannot be run again.**
`volume` and `tubes` deleted the working folders.
Run the steps one at a time with `--keep-work`, or start again from `photos`.

**Corrected masks no longer line up after a change of `vz`.**
The manual-edit grid follows round(400 µm / `vz`); pin `curation_f` to the value the corrections were made with.

**A run is killed with no Python error.**
It ran out of memory: the full-resolution steps need about 90 GB for 1,400 photographs.
Use a machine with more free memory; to cap memory, cap the resident memory, and never use `ulimit -v`, which kills every TIRL run.

**The last sections are darker than the rest.**
That is their illumination; set `native_end_crop_mm`.

**A registration stops before it starts.**
The GPU has less free memory than the registration needs, and it prints how much that is.
Use a card with more free memory, one registration at a time.

**A registration says the transform "already exists".**
A transform folder from an earlier run is there; add `--overwrite`, which backs up the old one.

**`qc` fails on scales.**
The pixel size or section step is off; check `vx` and `vz`.

**`linc-complete prepare` stops: "no plane from which the photographs are complete over ... mm".**
The last photographed sections are torn or missing.
With MRI, drop them with `native_end_crop_mm` and register again; without, move the caudal cutoff before them (`missing: [n, -1]`) and reconstruct again.
Then run `prepare` again.

**The completed brain has a brightness step at the junction plane.**
The 8 mm next to the junction hold damaged or darker sections; drop them as in the previous item and run the completion again.

**The template build's cost turns to `-nan`.**
A member has an all-zero image or mask, which MMORF cannot take; check that no member's channels or cost masks are empty.

**MMORF fails to allocate its memory.**
Other jobs are on the same card; run the template build alone on its card.

**"No space left on device" during the template build.**
The run folders are large; check the free disk first (120 GB for the first stage alone).

## Histology

The steps are in [03](tutorials/03_histology.md).

**A registered section lies on the wrong anatomy, or its outline fits but its sulci do not.**
The match table pairs the component with the wrong photograph; neighboring photographs look alike.
Correct its row in `match_table_edited.txt` and run `all` again into a new `--out` folder, because a folder that already holds a registration is resumed, not redone.

**`propose` leaves a component unmatched (-1).**
Its score on its run's line was too low to guess.
Find its photograph by eye (`match_review/`) and write it into `match_table_edited.txt`, or leave the component out.

**A registration differs slightly from the paper's when rerun.**
TIRL's result depends on the thread count; use `--threads 12`, the default.

**A run stopped.**
The machine went down or a component crashed; run the same command again and it resumes where it stopped.

**`all` stops at one component with "Matrix does not exist on the same manifold".**
TIRL's rotation search failed on that component.
This happened on one component of MF275, as in the paper's run, which has no result for it.
Run `linc-histology rgb` with the same options to make the color images of the components that did register.

**One section's labels sit about a millimeter off on one side, although its registration looks right.**
The drawn frame's fitted transform onto its component is wrong ([03, Step E](tutorials/03_histology.md#step-e-lgn-and-stn)); on MR252's frame 36, this put the left LGN and STN 1.1 mm off.
Fit that frame's transform again (`$H/structures/transforms/f<frame>_<piece>.tfm`, a SimpleITK transform) and push the labels again; the data folder's MR252 files carry the refit frame.

**A section's tissue mask takes in another object, or misses tissue.**
The section has an unusual contrast; on one of MF191's sections, with the contrast saturated, the mask took in another object.
Check `tissue_masks/` by eye, and leave any component with a wrong mask out of the table (-1).

**A section has no photograph.**
It was not photographed, or the volume stops before it, so it cannot be registered; leave it out of the table (-1).
