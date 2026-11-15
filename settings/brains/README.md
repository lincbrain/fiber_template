# One file per brain

`<ID>.yaml` holds a brain's photographs, pixel size and section step, camera jumps and missing sections, plus some settings only the paper's brains need.
It is read by `linc-reconstruct`, `linc-register` and `linc-histology`.
For a new brain, copy `TEMPLATE.yaml` to `<ID>.yaml` and fill in the required fields ([02, Step 6](../../docs/tutorials/02_blockface.md#step-6-adding-a-new-brain-its-settings-file)); `TEMPLATE.yaml` explains every field in a comment.

## Every field

| Field | Required? | Default | When to set it |
|---|---|---|---|
| `subject` | required | | the brain's ID, as in the file's name and in `sub-<ID>` |
| `raw` | required | | the photographs' folder, relative to `$LINC_RAW` |
| `raw_glob` | required | | the photographs' file pattern, such as `'*.JPG'` |
| `raw_regex` | required | | a regular expression whose first group is the photograph's number, such as `'Img(\d+)\.JPG'` |
| `vx` | required | | the photographs' pixel size, in µm |
| `vz` | required | | the section step, in µm |
| `idx_range` | optional | every photograph found | the first and last photograph numbers to use; a number missing in the range becomes a black frame, which keeps the order |
| `rotate180` | optional | `false` | `true` if the photographs are upside down (the brain must end up dorsal side up); look at `00_slices/` first ([02, Step 6](../../docs/tutorials/02_blockface.md#step-6-adding-a-new-brain-its-settings-file)) |
| `reduce` | optional | `mean` | how the coarser resolutions are computed; leave it |
| `jumps` | optional | none | the camera jumps: each `[before, after]` is the last section before the jump and the first after it, counted from 0 in `00_slices/` |
| `missing` | optional | none | `[before, after]` around missing or damaged sections, which are filled in between; `[-1, k]` drops frames 0 to k-1 that are not brain (a name card, a calibration grid); `[n, -1]` is the caudal cutoff, keeping sections 0 to n |
| `stack_pad` | optional | 200 pixels | the stack's margin on each side; raise it when a large camera jump pushes a photograph off the canvas |
| `jump_mode` | optional | `affine` | the only mode released |
| `ccs`, `cce` | optional | `0`, `0` | the first and last histology section where the corpus callosum joins the hemispheres ([03](../../docs/tutorials/03_histology.md)); `0`, `0` when there is none or no histology |
| `native_end_crop_mm` | optional | `0` | drops this many mm of the last photographed sections before `linc-register mri` resamples, when a brain's own last sections are darker. It acts only on a brain registered to its own b0, and from there on its completion. MF282 sets it (3.0 mm); MF204 (1.62 mm) and MR263 (3.5 mm), which have no MRI, keep it as a record of what acted when the paper's blockface template was built |
| `curation_f` | optional | from the manual-edit grid | sections per slice of the 0.40 mm manual-edit grid. Unset, it is read from that grid's header if one exists, otherwise round(400 µm / `vz`), at least 1. The paper's brains pin it (8 for 50 µm sections, 1 for 400 µm or more) so that a rebuild lines up with their manual corrections; a new brain leaves it out |
| `reconstruction_spacing_um` | the paper's brains | `vx`, `vz` | the spacing their volumes were reconstructed at, before the spacing calibrated against each brain's MRI was written into the headers |
| `registration_to_own_mri` | the paper's brains | | the b0 mask, fixed b0 and start pose of the 13 brains with MRI; without it, `linc-register mri` estimates the start pose |
| `series_alignment` | the paper's brains | | MF191 and MF196 only: a camera that moved on nearly every frame, corrected by replaying a recorded alignment to an external series |
| `shading` | the paper's brains | `poly2` | MR263 only: the illumination correction its volume was made with |
| `stack_origin_frames` | the paper's brains | the stack's own count | MR261 only: its paper stack held frames that are not its photographs |
| `reconstruction_mask` | the paper's brains | the final mask | MR254 only: the earlier mask its volume was reconstructed with |
| `mri`, `masks_of_record`, `histo_vx`, `histo_vz`, `histo_divisor` | records | | kept as a record only; no code reads them, and a new brain leaves them out |

## What a person makes once per brain

|  |  |
|---|---|
| pixel size and section step | `vx`, `vz` in this file, from a ruler and the microtome |
| camera jumps, missing sections | `jumps`, `missing` in this file, by eye from the stacked photographs |
| corrected masks (optional) | [02, Step 2](../../docs/tutorials/02_blockface.md#step-2-correct-the-masks-manually-optional) |
| match table | [03, Step B](../../docs/tutorials/03_histology.md#step-b-the-match-table) |
| section map | which section each image in the histology stack is, checked manually |
| labels | drawn on the sections by an anatomist |
