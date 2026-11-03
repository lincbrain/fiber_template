# Synthesize the missing caudal tissue

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
