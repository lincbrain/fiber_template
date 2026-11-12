# Registering the templates

Registers the blockface template to the MRI template, checks the result, and registers the template
to NMT v2, INIA19 and CIVM-R (Methods, "Registration of the blockface template to the MRI template"
and "Relation to published templates"). Run it with `code/template/build_link.py`. About 1 hour on a
GPU. Settings: `settings/link.yaml`. ANTs 2.5.4 gives the affine starts.

It also holds the MRI-based alignment of one brain, its own b0 to the MRI template
(`linc-register` mri-template):

1. a rigid placement of the b0 on the template's frame
2. a new bias correction, only when the b0 needs it
3. a linear start: 7 parameters, then 12 (rejected when too anisotropic)
4. one resampling onto a centered 0.35 mm grid
5. the moving image cut below the template brain's lowest slice
6. the deformable registration

|  |  |
|---|---|
| `register.py`, `symreg.py`, `estimate.py`    | the registration between the templates |
| `b0_to_template.py`                      | a brain's b0 to the MRI template |
| `atlas.py`, `ants.py`                      | the registrations to published templates |
| `qc.py`                                  | the outline check |
