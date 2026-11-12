DOCS

The code behind the documentation: the figures of the step-by-step tutorials, which are in
docs/tutorials/ of this repository, and the plain-text copy of every README.

    $PY_TIRL code/docs/tutorial_figures.py --tree $LINC_DATA_TREE          every figure
    $PY_TIRL code/docs/tutorial_figures.py --tree $LINC_DATA_TREE --only <name> ...
    $PY_TIRL code/docs/tutorial_figures.py --list                          the figures, in order

Each figure is one function in tutorial_figures.py, drawn in the paper's figure style
(code/figures/figstyle.py) from the data folder only, and written to docs/tutorials/figures/ of
this repository as <name>.png (200 dpi, 178 mm wide). Its provenance, <name>.provenance.json,
goes to outputs/docs/figures_provenance/, which stays local: it records this machine's paths.
--out <folder> writes a trial render elsewhere, with the provenance beside it. The names start with
the tutorial they belong to: overview (00), mri (01), blockface (02), histology (03). $PY_TIRL is
needed for the histology figures, which read the JPEG 2000 sections; the others run with $PY too.

Every README.md of this repository is the source of the README.txt beside it, the same text in plain
form. After editing a README.md, write its README.txt again:

    python3 code/docs/readme_txt.py            checks that every README.txt matches its README.md
    python3 code/docs/readme_txt.py --write    writes them

    tutorial_figures.py   the figures
    readme_txt.py         the README.txt files, from the README.md files
