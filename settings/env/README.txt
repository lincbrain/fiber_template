PYTHON ENVIRONMENTS

The pipeline needs separate environments because their packages conflict. install.sh builds them.
Each file's header says how to build it by hand. Every step runs in the environment it ran in for the
paper.

    linc.yml       registration, MRI preparation, MRI template, labels, evaluation, numbers, figures
    deep.yml       the segmentation networks and the synthesis
    anatomix.yml   the deformable registration engine (anatomix features and FireANTs)
    tirl.yml       histology registration and the blockface reconstruction (TIRL 3.6.3)
    viz3d.yml      optional, for the 3D tracer figure (install.sh --with-figures-3d)

Other programs, at these versions: FSL 6.0.7.4 (FSLDIR), ANTs 2.5.4 (ANTSPATH) and, to rebuild the
MRI template, MRtrix3 3.0.2 (MRTRIX). MMORF 1.0.1 comes with linc.yml. TIRL and its script packages
are not yet public: ask their authors for the sources and set LINC_TIRL_SRC before install.sh. The
code checks each version before running. Set PYTHONNOUSERSITE=1, so that packages in ~/.local do not
replace the pinned ones.
