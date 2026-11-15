# Brain identifiers

A brain can have up to three names.
The paper calls them MF1 to MF10 and MM1 to MM15.
The laboratory's files and folders, and this repository, use the Specimen ID, such as MR243, which is also the brain's name on the LINC data platform.
LINC's identifier page assigns published IDs separately, and so far only MF283 has one, M1.

| Paper ID | Specimen ID | LINC published ID | Role in the paper |
|---|---|---|---|
| MF1 | MF283 | M1 | MRI template |
| MF2 | MF284 | not listed | MRI template |
| MF3 | MF285 | not listed | MRI template |
| MF4 | MF286 | not listed | MRI template |
| MF5 | MF278 | not listed | blockface template (with MRI); histology |
| MF6 | MF275 | not listed | blockface template (with MRI); histology |
| MF7 | MF282 | internal only | blockface template (with MRI) |
| MF8 | MF191 | not listed | blockface template (no MRI); histology; tracer labels |
| MF9 | MF196 | not listed | blockface template (no MRI) |
| MF10 | MF204 | not listed | blockface template (no MRI) |
| MM1 | MR243 | not listed | blockface template (with MRI); histology; tracer labels; LGN and STN labels |
| MM2 | MR252 | not listed | blockface template (with MRI); histology; tracer labels; LGN and STN labels |
| MM3 | MR256 | not listed | blockface template (with MRI); histology; tracer labels; LGN and STN labels |
| MM4 | MR249 | not listed | blockface template (with MRI) |
| MM5 | MR250 | not listed | blockface template (with MRI) |
| MM6 | MR251 | not listed | blockface template (with MRI) |
| MM7 | MR253 | not listed | blockface template (with MRI) |
| MM8 | MR254 | not listed | blockface template (with MRI) |
| MM9 | MR257 | not listed | blockface template (with MRI) |
| MM10 | MR261 | not listed | blockface template (with MRI) |
| MM11 | MR244 | not listed | blockface template (no MRI) |
| MM12 | MR247 | not listed | blockface template (no MRI) |
| MM13 | MR248 | not listed | blockface template (no MRI) |
| MM14 | MR263 | not listed | blockface template (no MRI) |
| MM15 | MR273 | not listed | blockface template (no MRI) |

The MF brains are *Macaca fascicularis*, and the MM brains (Specimen IDs MR...) *Macaca mulatta*.

The paper IDs come from [settings/specimen_labels.json](../settings/specimen_labels.json), and the data folder's `brains.tsv` gives the same names.
The roles come from the brain sets in [steps.yaml](../steps.yaml), the members in [settings/blockface_template.yaml](../settings/blockface_template.yaml) and the inputs in [settings/mri_construction.yaml](../settings/mri_construction.yaml); for the LGN and STN, they are the brains of Fig. 8 ([code/figures/lgnstn.py](../code/figures/lgnstn.py)).
The LINC published IDs come from the LINC documentation's [identifier mapping](https://docs.lincbrain.org/data/identifiers) (accessed 14 November 2026), where MF283 is M1, MF282 is internal only, and the other brains are not listed.
