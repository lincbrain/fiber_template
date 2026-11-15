# FAQ

Short answers, each with a link to the page that has the full answer.
The terms are defined in the [overview's glossary](overview.md#glossary).

## Getting started

### Where do I start?

Install with the README's [quick start](../README.md#quick-start), then follow [00_start_here.md](tutorials/00_start_here.md).
To see how the parts fit together first, read the [overview](overview.md).

### I only want the template. Do I need the code?

No.
The template is in LINC dataset 000059, under `derivatives/template/` (its blockface channel in `micr/`), and in the data folder's `template/outputs/`.
Any NIfTI viewer, such as FSLeyes or ITK-SNAP, opens it.
The dataset's `derivatives/template/xfms/README.txt` says how to apply its mappings to NMT v2, INIA19 and CIVM-R without this code.

### What do I need to run the code?

Linux, a GPU, FSL 6.0.7.4, ANTs 2.5.4 and the TIRL sources ([settings/env](../settings/env/README.md)).
`install.sh` takes about 25 minutes and 23 GB.

### TIRL is not public. What can I run without it?

Ask TIRL's authors for its sources.
Without them, `install.sh` builds the TIRL environment without TIRL, and the steps that run in it refuse to run: the blockface reconstruction, the histology registration and `linc-labels push` ([overview, environments](overview.md#the-environments)).

### Can I run it without a GPU?

In part.
The histology registration and the labels use no GPU, and `linc-mri prep` runs on the CPU with `--device cpu`.
The registrations, the reconstruction's networks, the synthesis and the template builds run on a GPU.

### How long does each step take, and how much GPU memory does it need?

`$PY run.py --plan` prints every step with what it needs, from [steps.yaml](../steps.yaml) (`gpu_gb`, `ram_gb`, `hours`).
Some typical figures, from the tutorials:

| Step | GPU memory free | Time |
|---|---|---|
| Prepare a b0 | none (5 GB for the mask network, or the CPU) | under a minute |
| Register a b0 to the template | 15 GB | about 12 minutes |
| Reconstruct 1,400 photographs | 10 GB, and about 90 GB of RAM | about 10 hours |
| Register a blockface volume to its own b0 | 15 to 25 GB | about 15 minutes |
| Register a blockface volume to the blockface template | about 19 GB | about 6 minutes (MF191) |
| Synthesize the missing tissue | 2 GB | about a minute to prepare, 20 seconds to predict |
| Register one brain's histology | none; 24 cores, 32 GB of RAM | about 3 hours (MR243) |
| Carry one brain's tracer labels | none | under an hour |
| Build the MRI template | a 48 GB card | 3.5 days |
| Build the blockface template | 10, then 12 GB | about 8.5 hours, then about 17 hours (estimated) plus 2.5 hours of CPU steps |

## Names and files

### What do MR243, MM1 and M1 mean?

MR243 is the brain's Specimen ID, which every file and folder uses.
MM1 is its name in the paper.
M1 is a LINC published ID, which only MF283 has so far.
[brain_identifiers.md](brain_identifiers.md) lists all 25 brains.

### Where is ...?

| What | Where |
|---|---|
| the template | the data folder's `template/outputs/`; LINC dataset 000059 |
| a paper brain's transform to the template, as one file | `brains/sub-<ID>/outputs/<ID>_blockface_to_template.nii.gz` and `<ID>_b0_to_template.nii.gz` in the data folder |
| a brain's settings | `settings/brains/<ID>.yaml` |
| every other parameter | `settings/<stage>.yaml` |
| every step of the paper, with its command | [steps.yaml](../steps.yaml) |
| the networks' weights | `models/` (joined into `$LINC_MODELS`) |
| the networks' cards | the data folder's `models/<network>/card.txt` |
| the paper's numbers | `tests/reference/<record>.json`, made by `code/figures/paper_numbers/` |
| the scripts of the paper's figures | [code/figures](../code/figures/README.md) |
| the results of your runs | `$LINC_DATA/sub-<ID>/` and `$LINC_WORK/` ([overview](overview.md#the-data-folder)) |
| a `run.py` step's log | `outputs/build_everything/steps/<step>/` |

### Why does `$LINC_DATA` point inside the repository after `source settings/roots.sh`?

The code never writes into the data folder.
`settings/roots.sh` keeps the data folder as `$LINC_DATA_TREE` and points `$LINC_DATA` to a working folder, `outputs/local_data/data`, whose links lead to the data folder's files ([overview](overview.md#the-data-folder)).

### How does the code find a file in the data folder?

Not by its name there.
`settings/files.tsv` gives each file the name the code knows it by, and `settings/roots.sh` makes a link for each row under `outputs/in/`.
A new file needs a row there or a link where the code looks ([how-to, step 2](how-to/new_brain.md#2-make-them-visible-to-the-code)).

## The method

### Why are mirror images used?

So that each template is symmetric about the midsagittal plane: every brain and its mirror image are separate members of the construction.

### Why does a brain's MRI reach the template through its b0?

The b0 is the one MRI contrast that all the sectioned brains share.
Its transform to the template carries every image already aligned with it ([01, part B](tutorials/01_mri.md#b-a-new-brains-b0)).

### Why is there a blockface template?

Eight of the 21 photographed brains have no MRI.
They register their blockface volume to a template of the same contrast, and follow the template link into the template ([02, Step 4](tutorials/02_blockface.md#step-4-register-the-volume-to-the-blockface-template-a-brain-without-mri)).

### Why synthesize tissue that was never photographed?

No brain was photographed to its caudal end, so an average of the photographed tissue alone would end where the shortest brain ends.
The synthesis network predicts the rest from the b0; every file marks the synthesized voxels, and they count less in the template's average ([02, Step 5](tutorials/02_blockface.md#step-5-synthesize-the-tissue-that-was-never-photographed)).

### Why are histology sections registered to the blockface photographs and not to the MRI?

The photograph shows the section's tissue before it was cut, mounted and stained, and it is a slice of the blockface volume, which already has its transform to the template ([03](tutorials/03_histology.md)).

### How accurate is it?

The paper reports the accuracy, measured with data that no registration used: marker tubes hidden from the histology registration, and the LGN and STN carried through the whole chain.
Each tutorial's checks give the numbers of its own step.

## Running

### How do I rerun one of the paper's steps?

`$PY run.py --plan --only <step>` prints it, `--run` runs it with its check, and `--status` shows the result ([00, Two ways to run a step](tutorials/00_start_here.md#two-ways-to-run-a-step)).

### How do I remake the paper's numbers and figures?

`make -C code/figures numbers`, `numbers-gpu`, `numbers-figures`, `figures`, then `check` ([code/figures](../code/figures/README.md)).

### My rerun differs slightly from the paper's. Is something wrong?

Not if its check passes.
GPU registrations differ slightly from one session to the next, and TIRL and N4 depend on their thread counts ([tests/README.md](../tests/README.md)).

### A registration failed its check. What now?

A WARN asks for a look, not a rerun ([01, Checks](tutorials/01_mri.md#4-checks)).
For a FAIL, [troubleshooting](troubleshooting.md) lists the causes met so far, for [MRI](troubleshooting.md#mri), [blockface](troubleshooting.md#blockface) and [histology](troubleshooting.md#histology): among them, a brain mask that takes in the olfactory bulbs, a wrong pixel size or section step, and a wrong row in the match table.

### A command stops because its output already exists.

A transform: add `--overwrite`, which backs up the old one.
A template build step: add `--force`.
A histology registration resumes where it stopped; to redo it, give a new `--out` folder.

### How do I bring a new brain into the template?

Follow [Bring a new brain into the template](how-to/new_brain.md).
It aligns the brain to the template and leaves the template unchanged.

### How do I add a brain into the template itself?

That is a rebuild of the template, and everything aligned to it must then be derived again ([00](tutorials/00_start_here.md#adding-a-new-brain-two-different-things)).

### What has been tested on a brain that is not one of the paper's?

- The b0 to the template: once, on a paper brain's b0 treated as new, with the network's mask; it lay a median of 0.2 mm from the paper's registration.
- A blockface volume to the blockface template: MF191 treated as new passed its check.
- A blockface volume to its own b0: MR257 treated as new, through the placement, the estimated start pose and the linear stage; the deformable stage has not yet been run on a brain treated as new.
- Not tried yet: a brain that is not one of the paper's, with its own settings file and its photographs made visible through a new link or new rows in `settings/files.tsv`.

### Can I move images to NMT v2, INIA19 or CIVM-R?

Yes, with your own copy of that template: `shared.xfm apply` with `atlas:NMT_v2.0_sym`, `atlas:INIA19` or `atlas:CIVM-R` ([01, Step B3](tutorials/01_mri.md#step-b3-bring-images-into-the-template)).
[settings/targets.yaml](../settings/targets.yaml) says where to get them; CIVM-R's license does not allow its images to be redistributed.

### Can I align a new brain's T2-weighted image, or move a tensor?

No: see [what the package cannot do](overview.md#what-the-package-cannot-do).

## Sharing, citing and help

### What may I share, and where?

The template is in LINC dataset 000059 and the histology sections in 000003.
A brain's MRI results do not go to 000059; if they are shared at all, they go to that brain's own LINC dataset.
The photographs, the blockface volumes and everything computed from them stay local.

### How do I cite it, and under which license?

Cite the paper ([CITATION.cff](../CITATION.cff)).
The code is under the Apache License 2.0 and the template (LINC dataset 000059) under CC BY 4.0 ([README](../README.md#cite-and-license)).

### How do I report a problem?

Run `$PY code/shared/linc-info` and attach what it prints: the repository's commit, the environments, the GPUs and the external programs' versions.
