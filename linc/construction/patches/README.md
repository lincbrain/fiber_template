# Our change to the template-construction driver

The MRI template was built with the FMRIB multimodal template-construction driver (Arthofer et al., *Imaging Neuroscience*,
doi:10.1162/imag_a_00361), `run_template_construction.py` of
[mm-template-construction](https://git.fmrib.ox.ac.uk/cart/mm-template-construction), commit `f4907f4` (3 Jul 2024),
folder `template_construction_v1`. The driver is not part of the FSL 6.0.7.4 installation (FSL provides the tools it calls:
flirt, fslmaths, applywarp, convertwarp, invwarp, vecreg, and `fsl_sub`); MMORF 1.0.1 is installed separately.

The driver's repository carries no licence (no LICENSE file; its files say only "Copyright: FMRIB 2021/2022"), so this repository
ships none of its text: our changes are scripted edits that hold only our own lines. Each `*.edits.json` lists, per change, the range
of the driver's lines it replaces (in the clean file), the sha256 of those lines (not the lines) and our new lines;
`apply_edits.py` checks the clean file's md5 first and the result's md5 after:

```
git clone https://git.fmrib.ox.ac.uk/cart/mm-template-construction.git && git -C mm-template-construction checkout f4907f4
python <this repo>/linc/construction/patches/apply_edits.py <this repo>/linc/construction/patches/mm-template-construction_v1_linc.edits.json mm-template-construction
```

The edited file (md5 `f70f7da56b9cafceafebaa540fbd9328`) is the driver the template of record was built with (md5
`d47cf996be5e5ceefde47aa17ba6afbf`) with differences in comments and help text only: the author tag of the local changes reads
`(fiber_template)` where the driver of record has a username (15 lines: 14 comments and one `--help` text), and two comment lines
name the date and the records instead of internal notes. The code is token-for-token identical apart from that help text. Checked
3 Nov 2026: the edits applied to a clean checkout of `f4907f4` give the code tokens of the driver of record. (Until 3 Nov 2026 these
changes shipped as unified-diff patches; those carried context lines of the driver and were replaced by the edit files.)

What the edits change (each change is marked `LOCAL CHANGE` or `LINC PATCH` in the file):

| Change | Effect on results |
|---|---|
| Tensors are warped with `vecreg -w <field> --rotwarp=<field / voxel size>`. FSL 6.0.7.4 `vecreg -w` alone applies the field's rotation times the reference voxel size in mm, i.e. 25 % of it on the 0.25 mm grid | Yes: full tensor reorientation at every iteration |
| `--quant_scalar_mods`: 1-based scalar channels that are physical quantities (the SANDI fractions) are averaged without bias division and without `fslmaths -inm 1000` | Yes: the fraction template stays a fraction that sums to 1 |
| Bias-field existence checked when the average runs, not when the command is generated (the old check silently dropped bias correction under an asynchronous scheduler) | None with the `fsl_sub` shell plugin |
| Tensor modality tree tag indexed by `i`, not `0` (upstream bug; matters only with two or more tensor channels) | None with one tensor channel |
| `--batch`: concurrent MMORF tasks per step (`fsl_sub` `array_limit`); `submitJob` threads default `None` so the shell plugin can run array tasks in parallel | None (scheduling only) |
| `--resume_from STEP[:IT]`, and `LINC_SKIP_DONE_MMORF=1` to keep the finished MMORF tasks of the interrupted iteration | None (restarts only) |

Environment the driver ran in: Python 3.11 with `fsl-sub` 2.11 (shell plugin), `fslpy` 3.29.1, `file-tree` 1.6.1,
`file-tree-fsl` 0.2.3, `pandas`, `numpy`, `nibabel`; `mmorf` 1.0.1 (`fsl-mmorf-cuda`) first on `PATH`, then FSL 6.0.7.4.
The command line for a run is `linc.construction.mmorf_run.driver_command()` (see `reproduce/templates/mri_channels.md`).

## The blockface template's driver

`mm-template-construction_v1_blockface.edits.json` (standalone against `f4907f4`; used by `linc/construction/blockface_build.py`): the
driver the blockface template was built with. Applied to a clean `f4907f4` it gives the driver of record (md5
`fa7771595c86ef23e797a1bb5016c07f`) up to the same author tag (17 lines: 15 comments, one `--help` text and one docstring; edited md5
`fd775b9b9e8b615826bc48459d65ded2`; checked 3 Nov 2026).
It carries the same local changes as the MRI edits above
except the tensor `--rotwarp` change (the blockface copy was made before it, and the blockface build has no tensor channel). It adds two
changes of its own, both off unless `LINC_MASKWEIGHTED=1`:

| Change | Effect on results |
|---|---|
| Mask-weighted construction: each input's cost mask weights its FLIRT cost and every average | Yes (the blockface template of record used it) |
| An input with zero weight everywhere (TPL's blockface channel) is left out of the intensity normalization | Yes (the blockface template of record used it) |

See `reproduce/templates/blockface.md`.
