# The template link

This code registers the blockface template to the MRI template, checks the result, and registers the template to NMT v2, INIA19 and CIVM-R (Methods, "Registration of the blockface template to the MRI template" and "Relation to published templates").
Run it with `code/template/build_link.py`; it takes about 1 hour on a GPU, reads `settings/link.yaml`, and uses ANTs 2.5.4 for the affine starts.

It also holds the registration of one brain's b0 to the template (`linc-register mri-template`, [01, Step B2](../../../docs/tutorials/01_mri.md#step-b2-register-the-b0-to-the-template)), which runs:

1. a rigid placement of the b0 on the template's frame;
2. a new bias correction, only when the b0 needs it;
3. a linear start, 7 parameters and then 12 (rejected when too anisotropic);
4. one resampling onto a centered 0.35 mm grid;
5. a cut of the moving image below the template brain's lowest slice;
6. the deformable registration.

## Files

| | |
|---|---|
| `register.py`, `symreg.py`, `estimate.py` | the template link |
| `b0_to_template.py` | a brain's b0 to the template |
| `atlas.py`, `ants.py` | the registrations to the published templates |
| `qc.py` | the outline check |
