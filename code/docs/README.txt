DOCS

The figures of the step-by-step tutorials, which are in docs/tutorials/ of the data folder.

    $PY_TIRL code/docs/tutorial_figures.py --tree $LINC_DATA_TREE          every figure
    $PY_TIRL code/docs/tutorial_figures.py --tree $LINC_DATA_TREE --only <name> ...
    $PY_TIRL code/docs/tutorial_figures.py --list                          the figures, in order

Each figure is one function in tutorial_figures.py, drawn in the paper's figure style
(code/figures/figstyle.py) from the data folder only, and written to docs/tutorials/figures/ there
as <name>.png (200 dpi, 178 mm wide), with <name>.provenance.json beside it. The names start with
the tutorial they belong to: overview (00), mri (01), blockface (02), histology (03). $PY_TIRL is
needed for the histology figures, which read the JPEG 2000 sections; the others run with $PY too.

    tutorial_figures.py   the figures
