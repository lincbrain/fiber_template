BUILDING THE TEMPLATES

Three builds, one driver each. Each reads its brains, grids and parameters from its settings file.

    build_mri_channels.py         the MRI template (settings/mri_construction.yaml)
    build_blockface_template.py   the blockface template (settings/blockface_template.yaml)
    build_link.py                 the two templates registered (settings/link.yaml)

    $PY code/template/build_mri_channels.py --plan               every step; runs nothing
    $PY code/template/build_mri_channels.py --step <name>        one step
    $PY code/template/build_mri_channels.py --from <a> --to <b>  a range

A step whose outputs exist is not run again without --force. Each step writes a provenance file.

THE MRI TEMPLATE

Four Macaca fascicularis brains and their mirror images, one symmetric groupwise registration driven
by T2, the tensor and the two SANDI fractions together (MMORF 1.0.1). About 40 minutes per brain, then
3.5 days on a 48 GB GPU.

THE BLOCKFACE TEMPLATE

The 21 photographed brains, each used twice (as reconstructed and mirrored), with two channels per
input: the blockface luminance and the b0. The MRI template's b0 joins as one more, b0-only member.
About 25 hours on one 48 GB GPU (estimated; the paper's run took 11 h 35 min on three cards).

Both builds run through the template-construction driver of Arthofer et al. with our edits. See
patches/README.txt.

    mmorf_run.py, channels.py, tools.py                        the MRI template
    blockface_build.py, blockface_mmorf.py, blockface_finish.py   the blockface template
    link/                                                      the registration between the two
