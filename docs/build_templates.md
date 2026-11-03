# Build the templates

Three drivers rebuild the template from the released per-brain products, with the recipe and the parameters of the paper's build:
- the MRI channels (`reproduce/templates/build_mri_channels.py`);
- the blockface channel (`reproduce/templates/build_blockface_template.py`);
- the link between the two templates, with the mappings to published templates (`reproduce/templates/build_link.py`).

Each driver lists its steps with `--plan`, runs one step with `--step`, or a range of steps with `--from` and `--to`. Every step
writes a provenance JSON beside its output, and a step whose output exists is not rerun without `--force`. The outputs go to a work
folder (`--work`, by default under `$LINC_WORK`); nothing is written into `$LINC_DATA`. The parameters come from `configs/`; the
drivers restate none of them. The recipe of each build, with the reasons for its choices, is in `reproduce/templates/`.

```bash
$PY reproduce/templates/build_mri_channels.py --plan
$PY reproduce/templates/build_blockface_template.py --plan
$PY reproduce/templates/build_link.py --plan
```

`--plan` runs nothing: it prints every step with its environment, its estimated time, its command, its inputs and its outputs. The
drivers run in `env/linc.yml`. The steps marked `[mri]` run with `$PY_MRI`, which `envs/linc_envs.sh` sets to the same Python, the steps marked `[anatomix]` with
`$ANATOMIX_PYTHON` on a GPU, and the steps marked `[driver]` through the patched template-construction driver, with MMORF on a GPU.

The full builds take days of GPU time and were not rerun for this release. Their CPU steps were rerun, and they reproduce the
paper's files (below).

## The MRI channels

The recipe is `reproduce/templates/mri_channels.md`. The steps, with the times of the paper's run:

| Steps | Environment | Time |
|---|---|---|
| per brain of the MRI template (`sub-MF283` to `sub-MF286`, MF1–MF4 in the paper): `tensor-<ID>`, `landmarks-<ID>`, `frames-<ID>`, `prepare-<ID>` | `[mri]`, CPU | about 8 min, seconds, 1–2 min and 2.5 min per brain |
| `assemble`: the MMORF run folder | `[mri]`, CPU | 17 s |
| `mmorf`: 8 steps × 3 iterations | `[driver]`, one 48 GB GPU | 3 days 12 hours |
| `channels`: the template's MRI channels from the last step's warps | `[mri]`, CPU | 10.5 min |

It needs `PY_MRI`, `FSLDIR` (FSL 6.0.7.4), `ANTSPATH` (ANTs 2.5.4, for `prepare`) and `MRTRIX` (MRtrix3 3.0.2, for `tensor`); the
`mmorf` step also needs `LINC_TEMPLATE_DRIVER`, the patched driver's folder (`linc/construction/patches/README.md`), and
`LINC_MMORF_EXE`. The inputs are each brain's MRI products under `$LINC_DATA/sub-<ID>/`, which `--plan` lists; the output
`channels/` is the template's MRI folder.

```bash
$PY reproduce/templates/build_mri_channels.py --step prepare-MF283
```

On MF1, the frames are exact, the 36 construction inputs of `prepare` are identical voxel for voxel to the paper's, and `assemble`
writes the paper's run configuration. On the paper's MMORF run, the `channels` step reproduces all 20 template channel files voxel
for voxel.

## The blockface channel

The recipe is `reproduce/templates/blockface.md`. Its 26 steps form four stages:

| Stage | Steps | GPU stage and its time |
|---|---|---|
| members | `prep-frame` to `tpl-costmask-0p5` | — |
| 7 MMORF steps at 0.5 mm | `run-plan-a` to `plan-a-mmorf` | `plan-a-mmorf` `[driver]`, about 5 hours |
| the last MMORF step, at 0.25 mm | `s8-frame` to `step8-mmorf` | `step8-mmorf` `[driver]`, about 8 hours |
| finish and publish | `bf-view`, `finish`, `finish-b0`, `embed` | — |

The other steps run on a CPU, in minutes (`--plan` gives each estimate). The build needs `PY_MRI`, `LINC_TEMPLATE_DRIVER`,
`LINC_MMORF_EXE` and `FSLDIR`, and one 48 GB GPU for the MMORF stages. The inputs are under `$LINC_PRODUCTS`, as the product maps
in `configs/` (`products_*.yaml`) name them: the completed blockface volumes, the frames, the MRI template's b0 and mask, and the
affine stage of the paper's build. The published files are written to `final_run113/` in the work folder.

```bash
$PY reproduce/templates/build_blockface_template.py --step prep-frame
```

This step writes the frame identical, byte for byte, to the paper's, and refuses to run again without `--force`. `bf-view`, on the
paper's run of the last MMORF step, writes the paper's run description. The MMORF stages were not rerun.

## The link and the mappings to published templates

The recipe is `reproduce/templates/link.md`. Its 18 steps form three stages:

| Stage | Steps | GPU steps |
|---|---|---|
| the link | `link-prepare`, `link-solve`, `link-finish`, `link-export` | `link-solve` `[anatomix]`, about 2 min |
| the construction estimate and its comparison with the link | `greedy-link`, `compose`, `compose-tpl`, `compare` | — |
| the mappings to NMT v2, INIA19 and CIVM-R | per atlas: `atlas-affine-<atlas>`, `atlas-solve-<atlas>`, `atlas-score-<atlas>`; then `atlas-summary` | `atlas-solve-<atlas>` `[anatomix]`, about 10 min each |

It needs `ANATOMIX_PYTHON` (one GPU: about 5 GB for the link, more for the mappings at 0.25 mm) and ANTs 2.5.4. The inputs are
named in `configs/link.yaml` (`inputs`): the two templates, the frame bridge and the member warps under `$LINC_PRODUCTS`, and your own
copies of CIVM-R and ONPRC18 (Data, in the README). One input is not rebuilt: the one-way link that `compare`
reads beside the symmetric one is linked from the paper's product. `atlas-summary` writes its numbers record to `$LINC_NUMBERS`.

```bash
$PY reproduce/templates/build_link.py --step link-prepare
```

This step took 184 s on a CPU and writes the link's prepared inputs identical to the paper's; the 22 fields its record shares with
the paper's (the similarity and affine fits, the grid, the brainstem cut and the start of the registration, among others) are all
equal. The GPU steps were not rerun.
