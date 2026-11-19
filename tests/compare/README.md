# Compare a result with the paper's

|  |  |
|---|---|
| `compare_reconstruction.py`    | a blockface reconstruction |
| `compare_registration.py`      | a registration |
| `compare_histology.py`         | a histology registration |
| `compare_mri_build.py`         | a step of the MRI template's build |
| `compare_nifti.py`             | any NIfTI files, array by array |

Each prints one line per item and exits 0 when all agree, 1 when one differs or is missing, and 3, with a line saying "nothing to compare
with", when the data folder holds none of the paper's files for it (a data folder of inputs only); `run.py` then reports the step as
passed, ran to its end, like a step that has no check. `--help` lists its options.
`steps.yaml` gives the exact check for each step.
