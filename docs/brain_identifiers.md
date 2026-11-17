# Brain identifiers

A brain can have up to three names.
The paper calls them MF1 to MF10 and MM1 to MM15.
The laboratory's files and folders, and this repository, use the Specimen ID, such as MR243, which is also the brain's name on the LINC data platform.
The four brains of the MRI template also have published IDs, M1 to M4, from DANDI dataset 001372, where part of their MRI is public.

| Paper ID | Specimen ID | Published ID | Role in the paper |
|---|---|---|---|
| MF1 | MF283 | M1 | MRI template |
| MF2 | MF284 | M2 | MRI template |
| MF3 | MF285 | M3 | MRI template |
| MF4 | MF286 | M4 | MRI template |
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
M1 to M4 are the subject names in DANDI dataset [001372](https://dandiarchive.org/dandiset/001372) (Gong et al. 2026, doi:10.64898/2026.02.18.705310). Its high-res tensor series are byte-identical to ours for MF283 to MF286, in that order; `MRI_INPUTS.txt` (in the data folder and in LINC dataset 000059) lists the matching files, and the data folder's `brains.tsv` gives the same key.
The LINC documentation's [identifier mapping](https://docs.lincbrain.org/data/identifiers) (accessed 14 November 2026) lists MF283 as M1 and MF282 as internal only, and does not list the other brains.
