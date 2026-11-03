# The link, its construction estimate and the mappings to published templates

Paper: Methods, "Registration of the blockface template into the MRI template" and "Relation to published templates"; Results,
"The blockface template in the MRI template" and "Relation to published templates". Code: `linc/link/`. Parameters:
`configs/link.yaml`. Numbers: `reproduce/numbers/s5_link_d2bs8.py`, `reproduce/numbers/atlas_final_v2_summary.py`.

**Driver.** `reproduce/templates/build_link.py` runs the commands below in order: `--plan` prints every step with its environment,
estimated time, inputs and outputs and runs nothing; `--step NAME` (or `--from` / `--to`) runs steps, each writing
`<work>/provenance/<NN>_<step>.json`; a step whose outputs exist is not rerun without `--force`. The engine steps run with
`$ANATOMIX_PYTHON`. The one input the commands below do not build, the one-way (greedy) link that `compare` reads, is linked from the
product of record by its `greedy-link` step.

## Setup

```bash
export LINC_PRODUCTS=/path/to/products   # inputs listed in configs/link.yaml `inputs` (the templates, the frame bridge, the member warps)
export LINC_DATA=/path/to/data           # per-brain registrations (linc/xfm.py registry), read by the construction estimate
export LINC_TEMPLATE=/path/to/template
export LINC_NUMBERS=$PWD/numbers LINC_WORK=$PWD/work
```

Environment: `env/linc.yml`. The two engine steps (`solve`) need the registration extras (torch, the FireANTs fork, anatomix;
`env/README.md`) and one GPU (about 5 GB for the link at 0.35 mm; more for the 0.25 mm atlas mappings). Everything else runs on CPU.
ANTs 2.5.4 must be on `PATH` (or in `$ANTSPATH`) for the atlas affines; `linc/link/ants.py` refuses other versions.

## 1. The link (blockface template b0 -> MRI template b0)

```bash
python -m linc.link.register prepare --out work/link/sym                 # CPU, ~2 min: linear start, working grid, engine inputs
python -m linc.link.register solve   --out work/link/sym --device cuda:0 # GPU, ~2 min: the symmetric engine -> work/link/sym/sym/
python -m linc.link.register finish  --out work/link/sym                 # CPU: link_warp, moved b0 / mask / grey, s5_link.json
python -m linc.link.register export  work/link/sym work/link/maps        # CPU: the two full pull maps + round trips (export.json)
```

`run` does the first three in one call. Products of record: `bftemplate/cv/s5_direct_d2bs8/sym/` (link) and
`bftemplate/cv/template2template/d2bs8_s5sym/` (maps). The greedy (one-way) link `s5_direct_d2bs8/greedy/`, used only as a second
direct estimate in the comparison below, comes from the feature-based engine of `linc/registration`.

## 2. The construction estimate and the comparison

```bash
python -m linc.link.estimate compose --out work/link/s5_compose_d2bs8_r05_guard -j 8          # all brain members (TPL excluded)
#   add --order-from <record>/s5_compose.json to add the members in the record's order (bit-exact; otherwise ~1e-7 mm)
python -m linc.link.estimate compose --out work/link/s5_tpl_d2bs8_r05_guard --only TPL        # the MRI template member alone
python -m linc.link.estimate compare work/link/s5_compose_d2bs8_r05_guard work/link work/link/s5_compare_d2bs8.json
```

`compare` reads `<link root>/greedy` and `<link root>/sym` and the TPL estimate next to the composition folder. CPU only: about
3 min with 8 processes on the full 0.5 mm grid (42 members). Products of record: `bftemplate/cv/s5_compose_d2bs8_r05_guard/`,
`bftemplate/cv/s5_tpl_d2bs8_r05_guard/`, `bftemplate/cv/s5_compare_d2bs8.json`.

## 3. Mappings to NMT v2, INIA19 and CIVM-R

```bash
for A in NMT_v2.0_sym INIA19 CIVM-R; do
  python -m linc.link.atlas affine $A --out work/atlas_v2           # CPU (ANTs)
  python -m linc.link.atlas solve  $A --out work/atlas_v2           # GPU
  python -m linc.link.atlas score  $A --out work/atlas_v2           # CPU -> score.json + $LINC_NUMBERS/atlas_final_v2_<A>.json
done
python reproduce/numbers/atlas_final_v2_summary.py                  # with LINC_PRODUCTS pointing at the folder holding atlas_v2/
```

The CIVM-R and ONPRC18 images are read from your own copies (configs/link.yaml `inputs.atlas_reference`); only transforms and numbers
are written. Products of record: `mri/tc/atlas_v2/<atlas>/`.

## Validation against the records (21 Oct 2026, CPU, 8 threads)

Run on the analysis project's products of record (`LINC_PRODUCTS` = the project, `LINC_DATA` = its `data/`), with the record's
Python stack (numpy 2.4.6, scipy 1.17.1, nibabel 5.4.2):

| Check | Result |
|---|---|
| `register export` of the run of record vs `template2template/d2bs8_s5sym` | both maps bit-identical; `export.json` round trips identical |
| `register prepare` (similarity + affine fits, working grid, brainstem cut) + the engine's halfway step (`symreg.halfway`) | Pst, fits, grid, cut identical; H, Hinv, mid_A/B and their masks bit-identical |
| `register finish` on the record's engine run vs `s5_direct_d2bs8/sym` | link_warp, moved b0, moved mask, carried grey bit-identical; `s5_link.json` identical (except wall time) |
| `estimate compose`, full 0.5 mm grid, all brain members and TPL alone (`--order-from` the record) | link_ras_disp, spread, carried mask bit-identical; `s5_compose.json` identical |
| `estimate compose`, default member order | displacement and mask bit-identical; 2 spread voxels differ by 6e-8 mm |
| `estimate compare`, on the record's folders and on the recomputed ones, vs `s5_compare_d2bs8.json` | identical |
| `atlas`: S from the record's ANTs affines, `score INIA19` on the record's engine run, the numbers record | identical |
| `atlas affine INIA19` (ANTs rerun) | not reproducible: ANTs gives a different affine on every run (no random seed in the recipe); S moves template-brain points by 0.08 mm (median), 0.26 mm (max) |

Not rerun: the engine (`solve`) of the link and of the atlas mappings, which needs a GPU. The S of the atlas mappings needs the
pip `scipy-openblas` build to be bit-exact (a conda LAPACK changes it by 1e-14).
