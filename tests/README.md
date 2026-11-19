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

**GPU model and session.**
The deformable registration is exact within one GPU session.
Across sessions or GPU models the warps can differ by floating point.
In the lab's rerun of the 13 b0-to-template registrations (6 Nov 2026, on the paper's GPU model), they differed by medians of 0.002 to 0.060 mm and 99th percentiles of 0.04 to 0.47 mm, and by at most 1.9 mm at single voxels.

**TIRL threads.**
The histology registration depends on the thread count; use 12 per component, the default.

**N4 threads.**
The bias correction depends on the number of threads.

**MMORF.**
The blockface template's first five nonlinear steps reproduce byte for byte on another card of the same model.
Later steps differ slightly, and the templates correlate at 0.999996 or higher.

**Software versions.**
Use the ones in `settings/env/`.
With another matplotlib version, a figure's text can be laid out slightly differently, but the data drawn are the same.

## What was and was not rerun

A step's note in `steps.yaml` says how it was checked.
The optional steps were rerun only in part: of the MRI template's construction, the assembly, the affine stage, the first nonlinear step and the channels; of the networks' training, mri_brainmask and the synthesis network.
For the rest, the code and parameters are those used for the paper.
