# Docs

`tutorial_figures.py` draws the figures of the step-by-step tutorials in [docs/tutorials/](../../docs/tutorials/README.md).

```text
$PY_TIRL code/docs/tutorial_figures.py --tree $LINC_DATA_TREE          every figure
$PY_TIRL code/docs/tutorial_figures.py --tree $LINC_DATA_TREE --only <name> ...
$PY_TIRL code/docs/tutorial_figures.py --list                          the figures, in order
```

Each figure is one function, drawn in the paper's figure style (`code/figures/figstyle.py`) from the data folder only, and written to `docs/tutorials/figures/<name>.png` (200 dpi, 178 mm wide).
Its provenance, `<name>.provenance.json`, goes to the repository's `outputs/docs/figures_provenance/` and stays local, because it records this machine's paths.
`--out <folder>` writes a trial render elsewhere, with the provenance beside it.
The names start with the page they belong to: overview (00), mri (01), blockface (02) or histology (03).
The histology figures read the JPEG 2000 sections and need `$PY_TIRL`; the others also run with `$PY`.

## The documentation site

`mkdocs.yml`, at the top of the repository, builds the pages of `docs/` into a site as they are: a home page, the overview, the tutorials, the how-to pages, the brain identifiers, the FAQ and troubleshooting, with a search box.

```text
pip install mkdocs-material      in any Python environment; the pipeline's environments do not need it
mkdocs build --strict            writes site/; stops at a broken link or a link to a heading that does not exist
mkdocs serve                     shows the site on this machine while you edit
```

The pages keep their relative links, so they read the same on GitHub.
`site_links.py` sends the links that leave `docs/`, to the code, the settings or the tests, to the same file on GitHub.
A new page goes into the `nav` list of `mkdocs.yml`; `tests/test_docs_site.py` says so if it is missing.
