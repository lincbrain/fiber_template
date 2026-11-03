# Synthesize the missing caudal tissue

**You need** the brain's blockface volume and its registration (`docs/register.md`); for a brain without MRI, the stand-in b0 its
registration makes.

**What it does**
1. `prepare`: resamples the registered photographs onto the b0's frame and detects the junction plane, caudal to which the
   photographs no longer cover the brain;
2. predicts the blockface over the whole brain from the b0;
3. matches the prediction's brightness to the photographs near the cut and blends the two over 2 mm;
4. adds color.

**You get**, under the `--out` folder: `<ID>/<ID>_bf_completed.nii.gz`, its color version and the maps of what was predicted, as
NIfTI on the brain's 0.25 mm grid, with a provenance JSON.

Environment: `env/deep.yml`.

No brain was photographed to its caudal end. `cli/linc-complete` synthesizes the missing tissue from the brain's b0, or from a
stand-in b0 for a brain without MRI (`--stand-in`). It runs in two steps: `prepare` makes the brain's folder from its own
products, then the networks synthesize:

```bash
$PY_DEEP cli/linc-complete prepare --subject <ID> --out $LINC_WORK/prepared
$PY_DEEP cli/linc-complete --brain $LINC_WORK/prepared/<ID> --out $LINC_WORK/complete [--stand-in]
```

`prepare` writes `<ID>/` (`seam.json`, `coarse/`, `fine/`) from the brain's products under `$LINC_DATA/sub-<ID>/`, on a CPU, in
about a minute. A brain without MRI needs its stand-in b0 and registration inputs as flags (`--stand-in-b0 … --lo0`, with
`--nonlin`, `--matrix`, `--brain-mask`); `prepare --help` and [`linc/completion/README.md`](../linc/completion/README.md) list them
and how each is made. On MF5 (with MRI) and MM13 (without), `prepare` rebuilds the paper's folders array for array. The
synthesis takes about 20 s per brain on a GPU. Three of the paper's prepared
roots link into a sibling root; `python -m linc.completion.relocate <root> <new root>` makes a copy that works on another machine.
