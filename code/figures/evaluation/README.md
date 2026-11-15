# The evaluation

These scripts compute the measures the paper reports; `code/figures/paper_numbers/` runs them and writes the records.

|  |  |
|---|---|
| `variants.py`             | compares the versions of the MRI template, channel by channel |
| `variability.py`          | voxelwise variability between the inputs of a template |
| `lobo.py`                 | leave one brain out for the MRI template |
| `targets.py`              | the template as a registration target, against NMT v2 and CIVM-R |
| `route_test.py`           | the blockface-based and MRI-based alignments, compared |
| `direct_placement.py`     | the accuracy of the cross-modal alignment |
| `regeval.py`              | how well a registration aligns the inside of the brain |
| `tubes.py`                | marker-tube offsets, a measure of histology registration error |
| `labels_agreement.py`     | agreement of the LGN and STN labels in the template |
| `plan.py`                 | what an evaluation script would do, without running it |
