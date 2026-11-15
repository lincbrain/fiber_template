# FAQ

Short answers, with links to the full ones.
Terms are explained in the [overview](overview.md) and its [glossary](overview.md#glossary), and the overview's [data folder](overview.md#the-data-folder) section says where files live.

## Getting started

### I only want the template. Do I need the code?

No.
The template is in LINC dataset 000059, under `derivatives/template/` (its blockface channel is in `micr/`), and in the data folder's `template/outputs/`.
Any NIfTI viewer, such as FSLeyes or ITK-SNAP, opens it.
To apply its mappings to NMT v2, INIA19 and CIVM-R without this code, see the dataset's `derivatives/template/xfms/README.txt`.

### What can I run without TIRL?

Ask TIRL's authors for its sources, since it is not public yet.
Without them, the steps that run in the TIRL environment refuse to start: the blockface reconstruction, the histology registration and `linc-labels push` ([overview](overview.md#the-environments)).

### Can I run it without a GPU?

Partly.
The histology registration and the labels use no GPU, and `linc-mri prep` runs on the CPU with `--device cpu`.
The registrations, the reconstruction's networks, the synthesis and the template builds need a GPU.

### How long does each step take, and how much GPU memory does it need?

`$PY run.py --plan` prints every step with its needs, taken from [steps.yaml](../steps.yaml) (`gpu_gb`, `ram_gb`, `hours`).
Each step in the tutorials ([01](tutorials/01_mri.md), [02](tutorials/02_blockface.md), [03](tutorials/03_histology.md)) also gives its time, and its GPU memory when that is 24 GB or more.
The longest are the template builds, which only a rebuild needs ([00](tutorials/00_start_here.md#adding-a-new-brain-two-different-things)).

## Names and files

### What do MR243, MM1 and M1 mean?

MR243 is a Specimen ID, which every file and folder uses, and MM1 is the same brain's name in the paper.
M1 is a LINC published ID; so far only MF283 has one.
[brain_identifiers.md](brain_identifiers.md) lists all 25 brains.

### Why does `$LINC_DATA` point inside the repository?

Because the code never writes into the data folder: `$LINC_DATA_TREE` is the data folder, and `$LINC_DATA` a working folder whose links lead to its files ([overview](overview.md#the-data-folder)).
The links follow `settings/files.tsv`, so a new file needs a row there, or a link where the code looks ([how-to, step 2](how-to/new_brain.md#2-make-them-visible-to-the-code)).

## The method

### Why does a brain's MRI reach the template through its b0?

The b0 is the one MRI contrast that all the sectioned brains share.
Its transform to the template carries along every image already aligned with it ([01, part B](tutorials/01_mri.md#b-a-new-brains-b0)).

### Why is there a blockface template?

Eight of the 21 photographed brains have no MRI.
Their blockface volumes are registered to a template of the same contrast and then follow the template link into the template ([02, Step 4](tutorials/02_blockface.md#step-4-register-the-volume-to-the-blockface-template-a-brain-without-mri)).

### Why synthesize tissue that was never photographed?

No brain was photographed to its caudal end, so an average of photographed tissue alone would stop where the shortest brain stops.
The synthesis network predicts the rest from the b0.
Every file marks the synthesized voxels, and they count less in the template's average ([02, Step 5](tutorials/02_blockface.md#step-5-synthesize-the-tissue-that-was-never-photographed)).

### Why register histology to the blockface photographs rather than to the MRI?

The photograph shows the section's tissue before it was cut, mounted and stained.
It is also a slice of the blockface volume, which already has its transform to the template ([03](tutorials/03_histology.md)).

### How accurate is it?

The paper reports the accuracy, measured with data that no registration used: marker tubes hidden from the histology registration, and the LGN and STN carried through the whole chain.
Each tutorial's checks give the numbers for its own step.

## Running

### How do I rerun the paper's steps, numbers and figures?

`$PY run.py --plan --only <step>` prints a step, `--run` runs it with its check, and `--status` shows the result ([00, Two ways to run a step](tutorials/00_start_here.md#two-ways-to-run-a-step)).
The numbers and figures have their own targets: `make -C code/figures numbers`, `numbers-gpu`, `numbers-figures`, `figures`, then `check` ([code/figures](../code/figures/README.md)).

### My rerun differs slightly from the paper's. Is something wrong?

Not if its check passes.
GPU registrations differ slightly from one session to the next, and TIRL and N4 depend on their thread counts ([tests/README.md](../tests/README.md)).

### A registration failed its check. What now?

A WARN asks you to look, not to rerun ([01, Checks](tutorials/01_mri.md#4-checks)).
For a FAIL, [troubleshooting](troubleshooting.md) lists the causes we have met for [MRI](troubleshooting.md#mri), [blockface](troubleshooting.md#blockface) and [histology](troubleshooting.md#histology), among them a brain mask that takes in the olfactory bulbs, a wrong pixel size or section step, and a wrong row in the match table.

### How do I bring a new brain into the template?

Follow [Bring a new brain into the template](how-to/new_brain.md), which aligns the brain without changing the template.
Adding a brain into the template itself is a rebuild ([00](tutorials/00_start_here.md#adding-a-new-brain-two-different-things)).

### What has been tested on a brain that is not one of the paper's?

- The b0 to the template: once, on a paper brain's b0 treated as new, with the network's mask; it lay a median of 0.2 mm from the original registration.
- A blockface volume to the blockface template: MF191, treated as new, passed its check.
- A blockface volume to its own b0: MR257, treated as new, through the placement, the estimated start pose and the linear stage.
  The deformable stage has not yet been run on a brain treated as new.
- Not tried yet: a brain that is not one of the paper's, with its own settings file and its photographs made visible through a new link or new rows in `settings/files.tsv`.

### Can I move images to NMT v2, INIA19 or CIVM-R?

Yes, if you have your own copy of that template: use `shared.xfm apply` with `atlas:NMT_v2.0_sym`, `atlas:INIA19` or `atlas:CIVM-R` ([01, Step B3](tutorials/01_mri.md#step-b3-bring-images-into-the-template)).
[settings/targets.yaml](../settings/targets.yaml) says where to get them; CIVM-R's license does not allow its images to be redistributed.

## Sharing and help

### What may I share, and where?

The template is in LINC dataset 000059 and the histology sections are in 000003.
A brain's MRI results do not go into 000059; if they are shared at all, they go into that brain's own LINC dataset.
The photographs, the blockface volumes and everything computed from them stay local.

### How do I report a problem?

Run `$PY code/shared/linc-info` and attach its output, which lists the repository's commit, the environments, the GPUs and the external programs' versions.
