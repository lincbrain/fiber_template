SETTINGS

Every parameter of the paper, with a comment naming the Methods section it comes from. The code reads
these files and does not repeat their values.

    roots.sh               set LINC_DATA, then source this file in every new shell
    brains/                one file per brain
    <stage>.yaml           one file per stage (blockface, registration, completion, histology,
                           labels, link, mri_construction, blockface_template, targets, figures)
    files.tsv              every file of the data tree: where it is, what it is, what makes it
    specimen_labels.json   each brain's Specimen ID and its name in the paper
    photograph_counts.json the number of photographs of each brain, for Table S1
    segment/               the segmentation networks' training tasks
    tirl/                  the TIRL settings of the camera-jump registrations
    env/                   the Python environments
    models/                the networks' checksums
    internal/              maps from the code's names to the data tree
