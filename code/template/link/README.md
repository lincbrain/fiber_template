# The template link

Registers the blockface template to the MRI template, checks the result, and registers the template to NMT v2, INIA19 and CIVM-R (Methods, "Registration of the blockface template to the MRI template" and "Relation to published templates").
Run it with `code/template/build_link.py`: about 1 hour on a GPU. Settings: `settings/link.yaml`. ANTs 2.5.4 gives the affine starts.

It also holds the registration of one brain's b0 to the template (`linc-register mri-template`, [01, Step B2](../../../docs/tutorials/01_mri.md#step-b2-register-the-b0-to-the-template)):

1. a rigid placement of the b0 on the template's frame
2. a new bias correction, only when the b0 needs it
3. a linear start: 7 parameters, then 12 (rejected when too anisotropic)
4. one resampling onto a centered 0.35 mm grid
5. the moving image cut below the template brain's lowest slice
6. the deformable registration

## Files

| | |
|---|---|
| `register.py`, `symreg.py`, `estimate.py` | the template link |
| `b0_to_template.py` | a brain's b0 to the template |
| `atlas.py`, `ants.py` | the registrations to the published templates |
| `qc.py` | the outline check |
