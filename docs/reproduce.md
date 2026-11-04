# Reproduce the paper

**You need** the paper's products (`$LINC_PRODUCTS`, with the data release) and the environments.

**What it does**
1. rewrites every numbers record the paper reads;
2. draws every figure;
3. compares the rewritten records with the paper's.

**You get** the records in `$LINC_NUMBERS`, the figures in `outputs/figures/`, and the comparison's report.

The paper's text and tables read numbers records, each rewritten by one script, and each figure has one script. To reproduce them
you need the paper's intermediate products (`$LINC_PRODUCTS`), which come with the data release. `reproduce/Makefile` runs them all:

```bash
make numbers        # every numbers record the paper reads (CPU), into $LINC_NUMBERS (default outputs/numbers)
make numbers-gpu    # the one record that needs a GPU (env/deep.yml)
make figures        # the paper's figures, into outputs/figures (two of them also write a numbers record)
make check          # compare the recomputed records with the paper's (reference/), counts exact, floats to 1e-9
```

`make check` passes only when every record is there: run `make numbers-gpu` as well, because `make numbers` leaves out the one GPU
record, and a record skipped for lack of its inputs (for example NMT v2 and CIVM-R) shows as `MISSING`. To check a subset, name its
records: `$PY reproduce/check_numbers.py --ref reproduce/reference <record> ...`.

- **The Python each target uses:** each record and figure runs in the environment it ran in for the paper: `PY` (env/linc.yml),
  `PY_DEEP` (env/deep.yml) and `PY_TIRL` (env/tirl.yml). `source envs/linc_envs.sh` sets them; or give them on the command line,
  for example `make figures PY=<linc env>/bin/python`.
- **Targets that need more than the products:** they are skipped, with a message, when their input is not given:
  - `LINC_PARTICIPANTS`, the LINC participants table (`rawdata/participants.tsv` of the data release): `bf_cohort`, and
    `bf_species`, `s1_perbrain_r057` and the `pipeline` figure, which read it;
  - `LINC_MODELS`, the network weights: `s1_segmentation_r057`;
  - your copies of NMT v2 and CIVM-R (`configs/targets.yaml`): `r060_target_comparison` and `lgnstn_targets_r060c`;
  - raw data outside the release: `r057_mri_facts_b` and `histo_pixel_sizes_r057` (`LINC_RAW`), `mm_mri_check_r054`
    (`MM_MRI_ARGS`) and the `histology` figure (`HISTOLOGY_ARGS`).
- **One record or figure at a time.** Some records read others (for example, `bf_species` reads `bf_cohort`); the Makefile runs
  them in order, so run a record alone only after the records it reads are in `$LINC_NUMBERS`:

  ```bash
  $PY reproduce/numbers/<record>.py                                    # rewrite one record into $LINC_NUMBERS
  $PY reproduce/check_numbers.py --ref reproduce/reference <record>    # compare it with the paper's
  $PY reproduce/figures/pipeline.py --out figures/pipeline.png             # one figure (scripts are named by content)
  ```

- **Where things live:** `docs/paper_to_code.md` lists which records belong to which part of the paper, and which script makes each
  figure (by the paper's figure number).
  `reproduce/templates/` documents how the MRI channels and the blockface template were built.

## Checking the installation

`make test` (or `bash reproduce/selftest.sh`) runs the package's tests, each in the environment its code runs in, then a run of
each command line on toy data written on the fly: `linc-info`, `linc-segment`, `linc-complete`, `linc-reconstruct` and
`linc-register`. It needs no data, no network weights and no GPU. It runs in the environments `source envs/linc_envs.sh` points at:

```bash
make test
```

To test other interpreters, give them on the command line: `make test PY=<linc env>/bin/python PY_DEEP=<deep env>/bin/python
PY_TIRL=<tirl env>/bin/python`.

It prints PASS or FAIL for each step, with its time; a failed step shows the end of its log, and the exit status is the number
of failed steps. A step whose environment lacks a package fails with the import error, which names what to install. On one
machine it took about 3 minutes.

## Exact reproduction

Three settings decide whether a rerun matches the paper's products bit for bit:
- **GPU model and session.** The deformable registration (anatomix features + FireANTs) is deterministic within a session on one
  GPU: two runs on the same card give the same bytes, from the release's command or from the scripts the products were made with,
  and from a freshly built environment or the one of record. Across sessions, the GPU's convolution kernels can be chosen
  differently, and the warps then differ by floating point. The b0-to-template registrations of record (made 4 Oct 2026) are
  reproduced today on the same RTX 6000 Ada to a median of 0.002–0.025 mm, a 99th percentile of 0.04–0.47 mm and at most 1.9 mm
  at single voxels (three brains, 0.35 mm working voxel), with the linear start and the brain-mask overlap identical; a
  registration made on 22 Oct reproduced bit for bit. The code, the environment and the inputs are not what differs. On another
  GPU model, the differences are of the same order. From identical inputs, the warps differed:
  - by 0.00016 mm on average (maximum 0.029 mm) for a cross-modal alignment (RTX 4000 Ada vs RTX 6000 Ada);
  - by 0.005 mm on average (maximum 0.23 mm; brain-mask Dice 0.9998) for a registration to a blockface template (RTX 6000 Ada vs
    RTX 3090);
  - by a median of 0.005 mm (maximum 0.45 mm) for MF6, and 0.001 mm (maximum 0.37 mm) for MM2, at the template's brain voxels, for
    the registration of a b0 to the MRI template (RTX 6000 Ada vs RTX 3090).

  Whether two models agree depends on the registration: MF6's registration to its own MRI, made on an RTX 4000 Ada, reproduced bit
  for bit on an RTX 6000 Ada.
- **TIRL threads.** Histology registration depends on the thread count; use 12 threads per component (the default).
- **Software versions.** The versions in each environment file. With another matplotlib version, a figure's text can be laid out
  slightly differently; the data drawn are the same.
  The figures `reconstruction`, `bf_template` and `alignment_comparison` were made with matplotlib 3.10.8; under `env/linc.yml`'s
  3.11.1, their text is laid out slightly differently, and the data drawn are the same. The other figures are byte-identical to the
  paper's files.
