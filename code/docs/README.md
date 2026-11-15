# Docs

`tutorial_figures.py` draws the figures of the step-by-step tutorials in [docs/tutorials/](../../docs/tutorials/README.md).

```text
$PY_TIRL code/docs/tutorial_figures.py --tree $LINC_DATA_TREE          every figure
$PY_TIRL code/docs/tutorial_figures.py --tree $LINC_DATA_TREE --only <name> ...
$PY_TIRL code/docs/tutorial_figures.py --list                          the figures, in order
```

Each figure is one function, drawn in the paper's figure style (`code/figures/figstyle.py`) from the data folder only, and written to `docs/tutorials/figures/<name>.png` (200 dpi, 178 mm wide).
Its provenance, `<name>.provenance.json`, goes to `outputs/docs/figures_provenance/` and stays local, because it records this machine's paths.
`--out <folder>` writes a trial render elsewhere, with the provenance beside it.
The names start with the page they belong to: overview (00), mri (01), blockface (02) or histology (03).
The histology figures read the JPEG 2000 sections and need `$PY_TIRL`; the others also run with `$PY`.
