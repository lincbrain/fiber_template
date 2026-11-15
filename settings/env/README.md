# Python environments

The pipeline needs several Python environments because their packages conflict.
`install.sh` builds them, `settings/roots.sh` points the variables below to them, and each file's header says how to build it manually.
Every step runs in the environment it ran in for the paper.

| Variable | Environment | Runs |
|---|---|---|
| `$PY` | `linc.yml` | MRI preparation, registration, the MRI template, the labels after `push`, evaluation, the numbers and figures |
| `$PY_DEEP` | `deep.yml` | the segmentation networks and the synthesis |
| `$PY_TIRL` | `tirl.yml` | the blockface reconstruction, the histology registration and `linc-labels push` (TIRL 3.6.3) |
| (called by `linc-register`) | `anatomix.yml` | the deformable registration engine: anatomix features and FireANTs |
| `$PY_VIZ3D` | `viz3d.yml` | optional: the 3D tracer figure (`install.sh` `--with-figures-3d`) |

You also need FSL 6.0.7.4 (FSLDIR) and ANTs 2.5.4 (ANTSPATH), and MRtrix3 3.0.2 (MRTRIX) to rebuild the MRI template; MMORF 1.0.1 comes with `linc.yml`.
The code checks each version before running.
TIRL and its script packages are not public yet: ask their authors for the sources and set LINC_TIRL_SRC before running `install.sh`.
Set PYTHONNOUSERSITE=1, so that packages in `~/.local` cannot replace the pinned ones.
