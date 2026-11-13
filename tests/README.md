# Tests

```text
bash tests/selftest.sh     the tests and a toy run of each command; no data, no GPU, a few minutes
$PY tests/check_numbers.py --ref tests/reference <record>     a numbers record against the paper's
$PY tests/verify_weights.py     checks the networks' weights ([settings/models](../settings/models/README.md))
```

|  |  |
|---|---|
| `smoke/`        | the toy runs |
| `reference/`    | the paper's numbers records |
| `compare/`      | compares a result of yours with the paper's |


## What decides whether a rerun matches the paper bit for bit

GPU model and session. The deformable registration is exact within one GPU session. Across sessions or
GPU models the warps can differ by floating point: a median of 0.002 to 0.025 mm and at most 1.9 mm at
single voxels.

TIRL threads. Histology registration depends on the thread count. Use 12 per component (the default).

N4 threads. The bias correction depends on the number of threads.

MMORF. The blockface template's first five nonlinear steps reproduce byte for byte on another card of
the same model. Later steps differ slightly, and the templates correlate at 0.999996 or higher.

Software versions. The ones in `settings/env/`. With another matplotlib version, a figure's text can be
laid out slightly differently; the data drawn are the same.


## What was and was not rerun

A step's note in `steps.yaml` says how it was checked. The optional steps were rerun only in part: of
the MRI template's construction, the assembly, the affine stage, the first nonlinear step and the
channels; of the networks' training, mri_brainmask and the synthesis network. The code and parameters
of the rest are the paper's.
