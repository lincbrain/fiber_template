# Brain identifiers

Every brain has up to three names.
The paper uses its own names, MF1 to MF10 and MM1 to MM15.
The laboratory's files and folders, and this repository, use the Specimen ID, such as MR243 (also the brain's name on the LINC data platform).
LINC's identifier page assigns published IDs separately; today only MF283 has one, M1.

| Paper ID | Specimen ID | LINC published ID | Species | Role in the paper |
|---|---|---|---|---|
| MF1 | MF283 | M1 | *M. fascicularis* | MRI template |
| MF2 | MF284 | not listed | *M. fascicularis* | MRI template |
| MF3 | MF285 | not listed | *M. fascicularis* | MRI template |
| MF4 | MF286 | not listed | *M. fascicularis* | MRI template |
| MF5 | MF278 | not listed | *M. fascicularis* | blockface template (with MRI); histology |
| MF6 | MF275 | not listed | *M. fascicularis* | blockface template (with MRI); histology |
| MF7 | MF282 | internal only | *M. fascicularis* | blockface template (with MRI) |
| MF8 | MF191 | not listed | *M. fascicularis* | blockface template (no MRI); histology; tracer labels |
| MF9 | MF196 | not listed | *M. fascicularis* | blockface template (no MRI) |
| MF10 | MF204 | not listed | *M. fascicularis* | blockface template (no MRI) |
| MM1 | MR243 | not listed | *M. mulatta* | blockface template (with MRI); histology; tracer labels; LGN and STN labels |
| MM2 | MR252 | not listed | *M. mulatta* | blockface template (with MRI); histology; tracer labels; LGN and STN labels |
| MM3 | MR256 | not listed | *M. mulatta* | blockface template (with MRI); histology; tracer labels; LGN and STN labels |
| MM4 | MR249 | not listed | *M. mulatta* | blockface template (with MRI) |
| MM5 | MR250 | not listed | *M. mulatta* | blockface template (with MRI) |
| MM6 | MR251 | not listed | *M. mulatta* | blockface template (with MRI) |
| MM7 | MR253 | not listed | *M. mulatta* | blockface template (with MRI) |
| MM8 | MR254 | not listed | *M. mulatta* | blockface template (with MRI) |
| MM9 | MR257 | not listed | *M. mulatta* | blockface template (with MRI) |
| MM10 | MR261 | not listed | *M. mulatta* | blockface template (with MRI) |
| MM11 | MR244 | not listed | *M. mulatta* | blockface template (no MRI) |
| MM12 | MR247 | not listed | *M. mulatta* | blockface template (no MRI) |
| MM13 | MR248 | not listed | *M. mulatta* | blockface template (no MRI) |
| MM14 | MR263 | not listed | *M. mulatta* | blockface template (no MRI) |
| MM15 | MR273 | not listed | *M. mulatta* | blockface template (no MRI) |

MF: *Macaca fascicularis*. MM (Specimen IDs MR...): *Macaca mulatta*.

Sources:

- the paper IDs: [settings/specimen_labels.json](../settings/specimen_labels.json);
- the roles: the brain sets of [steps.yaml](../steps.yaml), the members of [settings/blockface_template.yaml](../settings/blockface_template.yaml), the inputs of [settings/mri_construction.yaml](../settings/mri_construction.yaml), and for the LGN and STN the brains of Fig. 8 ([code/figures/lgnstn.py](../code/figures/lgnstn.py));
- the LINC published IDs: the LINC documentation's [identifier mapping](https://docs.lincbrain.org/data/identifiers), read on 14 Nov 2026, where MF283 is M1, MF282 is internal only, and the other brains are not listed.

The data folder's `brains.tsv` gives the same paper names.
