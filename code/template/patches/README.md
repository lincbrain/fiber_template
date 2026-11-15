# Our changes to the template-construction driver

The templates were built with the FMRIB multimodal template-construction driver (Arthofer et al., Imaging Neuroscience, doi:10.1162/imag_a_00361), from https://git.fmrib.ox.ac.uk/cart/mm-template-construction at commit f4907f4.
That repository carries no license, so this one ships none of its text.
Each `.edits.json` holds only our own lines, and `apply_edits.py` applies them to a clean checkout:

```bash
git clone https://git.fmrib.ox.ac.uk/cart/mm-template-construction.git
git -C mm-template-construction checkout f4907f4
python code/template/patches/apply_edits.py \
    code/template/patches/mm-template-construction_v1_linc.edits.json mm-template-construction
```

`mm-template-construction_v1_linc.edits.json` is for the MRI template, and it changes results in two ways.
Tensors are warped with the full rotation (on a 0.25 mm grid, FSL's vecreg -w alone applies only a fraction of it), and the SANDI fractions are averaged as physical quantities, so they still sum to 1.
Its other changes affect only scheduling and restarts.

`mm-template-construction_v1_blockface.edits.json` is for the blockface template: it adds mask-weighted construction and leaves a zero-weight channel out of the intensity normalization.
