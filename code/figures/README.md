# The paper's figures and numbers

This folder remakes every figure and number in the paper and checks the numbers against the paper's records.
Each record and figure runs in the same environment it ran in for the paper.

## Commands

|  |  |
|---|---|
| `make -C code/figures numbers`            | the numbers records that run on a CPU |
| `make -C code/figures numbers-gpu`        | the one record that needs a GPU |
| `make -C code/figures numbers-figures`    | the figures that also write a record |
| `make -C code/figures figures`            | every figure |
| `make -C code/figures check`              | compares the records with the paper's |
| `make -C code/figures template-animation`, `tracers-animation` | the README's animations ([docs/media](../../docs/media/README.md)) |

`check` passes only after all three numbers targets have run.
A target whose input is missing skips with a message; the inputs it may miss are your own copies of NMT v2 and CIVM-R (`settings/targets.yaml`) and raw data outside the data folder.
The figures take about 10 minutes on a CPU.

To make one figure: `$PY code/figures/<script>.py --out <file.png>`.

## Figures

| Figure | Script |
|---|---|
| 1 | `template.py` |
| 2 | `pipeline.py` |
| 3 | `reconstruction.py` |
| 4 | `variability.py` |
| 5 | `bf_template.py` |
| 6 | `link.py` |
| 7 | `histology.py` |
| 8 | `lgnstn.py` |
| 9 | not in this release: drawn by the analysis from the tracer labels without the light fiber bundle class; the numbers it prints are the references `fig9_coronal_slabs3m_allclasses_s11wr17`, `fig9_rowa_virtual_sections_s11wr17nl` and `fig9_rowa_style_r3_s11wr17nl` (not regenerated, `tests/numbers_exceptions.yaml`) |
| S1 | `s_bf2mri_steps.py` |
| S2 | `s_synthesis_gallery.py` |
| S3 | `s_tracer_fidelity.py` |
| S4 | not in this release: each bundle's distance from the internal capsule's ventral end, drawn by the analysis with Fig. 9 |
| S5 | `s_networks.py` |
| not in the paper | `alignment_comparison.py` (it still writes a numbers record), `s_synthesis_depth.py` (its numbers are in the Supplement's text), `tracer_sections3d.py` (an earlier form of the tracer figure) |

`figstyle.py` holds the shared figure style, and the font is Liberation Sans.

## Files

|  |  |
|---|---|
| `paper_numbers/`    | one script per numbers record |
| `evaluation/`       | the measures the paper reports |
