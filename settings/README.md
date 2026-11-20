# Settings

Every parameter of the paper is here, with a comment naming the Methods section it comes from.
The code reads these files and does not repeat their values.

|  |  |
|---|---|
| `roots.sh`                | set LINC_DATA_TREE to the data folder, then source this file in every new shell |
| `outputs_README.txt`     | what each folder under the repository's `outputs/` is; `roots.sh` writes it there as `README.txt` |
| `brains/`                 | one file per brain |
| `<stage>.yaml`            | one file per stage (blockface, registration, completion, histology, labels, link, mri_construction, blockface_template, targets, figures) |
| `files.tsv`               | every file of the data folder: where it is, what it is, what makes it |
| `results.tsv`             | the finished files of a run that `code/shared/linc-results` shows in `outputs/brains/sub-<ID>/outputs/`, each under the data folder's name: the blockface volume's light copy, its brain mask and marker tubes, the completion, the tracer labels, the MRI-template brains' channels and, with `--transforms`, the complete transforms |
| `specimen_labels.json`    | each brain's Specimen ID and its name in the paper |
| `photograph_counts.json`  | the number of photographs of each brain, for Table S1 |
| `manual_inputs.tsv`       | every file a person drew or edited for the paper's products, in the data folder, with its md5 |
| `segment/`                | the segmentation networks' training tasks |
| `tirl/`                   | the TIRL settings of the camera-jump registrations |
| `env/`                    | the Python environments |
| `models/`                 | the networks' checksums (the weights are in `models/` at the top) |
| `internal/`               | maps from the code's names to the data folder |
