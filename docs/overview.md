# Overview

This page explains how the parts fit together.
The commands are in the [tutorials](tutorials/00_start_here.md), and the terms are defined in the [glossary](#glossary).

## The package

The package has three parts.
The template is published as LINC dataset 000059, and the code is this repository.
The data folder, `fiber_template_data`, holds the inputs, the results and another copy of the template (in `template/outputs/`), and stays with the laboratory.

![The chain from a histology section to the template, with the two routes of a blockface volume](tutorials/figures/overview_01_chain.png)

*Figure 1. The whole chain, shown on MR243 (MM1).
Each tile shows an image with its space in gray, and each arrow names the command that makes the transform.
A histology section is registered to the blockface photograph of its own section ([03](tutorials/03_histology.md)).
On the MRI route, the blockface volume is registered to the brain's own b0 ([02](tutorials/02_blockface.md)), and the b0 to the MRI template ([01](tutorials/01_mri.md)).
On the blockface route, it is registered to the blockface template instead, and the template link carries it into the MRI template (02); a brain without MRI has only this route.
The tracer labels and the LGN and STN take the blockface route in every brain; these brains helped build the blockface template, so for them that route is the transforms that built it, which for a brain with MRI start from its registration to its own b0. The paper also reports them on the MRI route.*

## The data

Each brain's inputs are in the data folder's `brains/sub-<ID>/inputs/`:

- `mri/`;
- `photographs/`, one per section;
- `histology/`, the scanned sections and the tracer labels drawn from their charts;
- `manual_annotations/`, whatever a person drew or corrected, such as brain masks, fiducial markers (the round holes in the block face), match tables and the LGN and STN.
  For the paper's brains it also holds the photograph masks made by an earlier network whose weights stay with the laboratory, most of them unchanged, and `settings/manual_inputs.tsv` lists every file with its md5.
  Two match tables of record were edited by hand after a check of the tracer labels and are in the brain's `intermediate/histology/`: MR243's (`num_MR243_r058c.txt`, one row added for section 3's left half) and MF191's (`num_MF191_r058.txt`, section 4 split into its two lobes).

Four brains with MRI only, MF283 to MF286, built the MRI template ([01](tutorials/01_mri.md)).
The 21 photographed brains built the blockface template ([02](tutorials/02_blockface.md)); 13 of them have a b0, such as MR243, and 8 have no MRI, such as MF191.
Six of these also have histology sections, four have tracer labels and three have the LGN and STN ([03](tutorials/03_histology.md)).
The 11 brains photographed at every section, with few interpolated sections and steady lighting, also sharpened the blockface template at 0.15 mm.

Files name a brain by its Specimen ID, such as MR243, which the paper calls MM1; [brain_identifiers.md](brain_identifiers.md) lists all 25 brains and what each was used for.
MR brains are *Macaca mulatta* and MF brains *Macaca fascicularis*.

## Spaces and transforms

Every image lives in a named space, and every registration writes one transform from one space to another.

| Space | What it is | Its transform toward the template |
|---|---|---|
| `histo-piece:<piece>` | one component of a histology section | to `bf-native`, by `linc-histology all` (03) |
| `bf-native` | a brain's blockface volume | to `b0-native` by `linc-register mri`, or to `bf-template` by `linc-register blockface-template` (02) |
| `b0-native` | a brain's b0, as acquired | to `mri-template`, by `linc-register mri-template` (01) |
| `t2-native`, `micro-native` | the other MRI acquisitions of the four MRI-template brains | to `b0-native`, by `build_mri_channels.py` (01) |
| `bf-template` | the blockface template, in its own space | to `mri-template`, the template link, by `build_link.py` (02) |
| `mri-template` | the template, at 0.25 mm, in the symmetric frame of NMT v2 | to each published template, by `build_link.py` |
| `atlas:<name>` | a published template: `atlas:NMT_v2.0_sym`, `atlas:INIA19` or `atlas:CIVM-R` | |

A brain without MRI also has a transform from `bf-native` straight to `mri-template`, the cross-modal alignment (`linc-register direct-placement`).
It gives the brain a stand-in b0, and a chain uses it only when nothing else connects the two spaces.

Each transform is a folder, `$LINC_DATA/sub-<ID>/xfm/<from>__to__<to>/`, holding its files and a `meta.json` with its check's verdict.
Coordinates are RAS, in millimeters.
`shared.xfm` chains the transforms between two spaces and applies the chain in one resampling:

```bash
$PY -m shared.xfm chain --from bf-native --to mri-template --subject <ID>      # lists the links it uses
$PY -m shared.xfm apply --from bf-native --to mri-template --subject <ID> \
    --ref $LINC_DATA_TREE/template/outputs/template_T2w.nii.gz <image> <output>   # add --nearest for a label image
```

`<image>` is any image on the grid of the brain's blockface volume, such as its brain mask, `$LINC_DATA/sub-<ID>/blockface/<ID>_mask.nii.gz`, and `<output>` is the file to write, that image on the template's grid (`--ref`).

An inverted link is solved at every voxel of the target grid, so use a grid of 0.3 mm or coarser.
For the paper's brains, the data folder also keeps each whole transform as one file, such as `brains/sub-MR243/outputs/MR243_blockface_to_template.nii.gz`, and that folder's `README.txt` shows how to apply it with ANTs alone.

## The templates

**The MRI template** was built from MF283 to MF286 and their mirror images, eight members, by one symmetric groupwise registration: MMORF 1.0.1, run through the template-construction driver of Arthofer et al. with [our edits](../code/template/patches/README.md).
The T2-weighted image, the diffusion tensor and the SANDI soma and neurite fractions drove it together.
The mirror images make the template symmetric about the midsagittal plane ([01, part A](tutorials/01_mri.md#a-the-mri-template-on-the-papers-four-brains)).

**The blockface template** was built the same way from the 21 photographed brains, each with its synthesized tissue, and their mirror images.
Each member had two channels, its blockface grayscale and its b0 (its own or the stand-in b0), and the MRI template's b0 joined as one extra, b0-only member.
In the average, photographed tissue counts more than synthesized tissue, and tissue synthesized from a stand-in b0 not at all.
The construction gives the template the mean shape of its members, so it has a space of its own ([02, Step 7](tutorials/02_blockface.md#step-7-the-blockface-template-and-its-link-to-the-mri-template)).
A last step sharpened it at 0.15 mm with 11 brains (photographed at every section, with few interpolated sections and steady lighting) and their mirror images: each was registered to the template as a whole, with anatomix features and FireANTs, in three rounds, and they were averaged as before.
That step keeps the template's space, so the link below still applies.

**The template link** registers the blockface template's b0 to the MRI template's b0.
The blockface template moved through it becomes the template's blockface channel, `template/outputs/template_blockface_*`.
The blockface template itself, in its own space, is in `template/outputs/blockface_template/`.
`build_link.py` also registers the template to NMT v2, INIA19 and CIVM-R.

## The networks

| Network | What it makes | Used by |
|---|---|---|
| `mri_brainmask` | the brain mask of an ex vivo b0 or T2-weighted image | `linc-mri prep`, and `linc-register` when no mask is given |
| `bf_mask2d` | the brain mask of each blockface photograph | `linc-reconstruct photos` |
| `bf_mask3d` | the brain mask of a blockface volume | `linc-reconstruct mask` |
| `bf_holes` | the fiducial markers of a blockface volume | `linc-reconstruct tubes` |
| `d3_full_A_s0` | the synthesis network: blockface grayscale from a brain's own b0 | `linc-complete` |
| `d3p_full_A_s0` | the same, from a stand-in b0 | `linc-complete --stand-in` |

The weights are in this repository's `models/segment/` and `models/completion/` ([settings/models](../settings/models/README.md)).
Each segmentation network's card, with its training data, held-out results and limits, is `card.txt` in the data folder's `models/<network>/`.
The deformable registrations also use anatomix, a pretrained feature network.
To retrain, see [code/shared/segment](../code/shared/segment/README.md) and [code/blockface/synthesis](../code/blockface/synthesis/README.md).

## The data folder

The files live in two places.

**The data folder**, `$LINC_DATA_TREE`, holds the paper's data. The code reads it and never writes into it.
- `brains/sub-<ID>/inputs/`: what an instrument or a person made.
- `brains/sub-<ID>/intermediate/`: everything in between, from the paper's run.
- `brains/sub-<ID>/outputs/`: the paper's finished files.
- `template/`: the template, split the same way; `models/`: the trained networks; `paper/`: what only the paper's numbers and figures read.

**The repository's `outputs/`**, made by `settings/roots.sh`, holds what your own runs write.
- `brains/sub-<ID>/`: each brain's results in the data folder's three parts, as links: `inputs` (the brain's inputs in the data folder), `intermediate/` (one link per place where a run wrote for the brain) and `outputs/` (the finished files your run made that `settings/results.tsv` lists, under the data folder's names); `settings/roots.sh`, `run.py` after each step and every tool at its end refresh it, and `$PY code/shared/linc-results` does it on request.
- `in/`: links to the data folder's files, under the names the code uses; nothing of yours.
- `local_data/`: `data/`, the working folder (`$LINC_DATA`: one `sub-<ID>/` per brain, with `blockface/`, `mri/`, `xfm/`, `histology/` and `labels/`), `raw/` (`$LINC_RAW`) and `template/` (`$LINC_TEMPLATE`).
- `local_rebuild/` and `local_mri/`: the working folders for reconstructing a paper brain from its photographs (`$LINC_REBUILD_DATA`) and for the MRI template's build (`$LINC_MRI_DATA`); `local_mri/` is not made when the data folder lacks the four template brains' masks.
- `work/` (`$LINC_WORK`), `models/` (`$LINC_MODELS`) and `build_everything/` (`run.py`'s commands, logs and checks).

The repository's `outputs/` is not a brain's `outputs/`.
A brain's `outputs/` is in the data folder and holds the paper's finished files.
Each brain's results are laid out in `outputs/brains/sub-<ID>/` in those three parts, as links, with the finished files under the data folder's names; the other folders of `outputs/` are where the code works.

The data folder's top level also holds:

| File | What it is |
|---|---|
| `brains.tsv` | every brain: its ID, its name in the paper, its species and its data |
| `MRI_INPUTS.txt` | each MRI input, what the code does to it and what comes out |
| `checksums.md5` | checksums of the inputs |

Each of its folders, and each of a brain's three parts, has a `README.txt` describing what is in it.
`settings/files.tsv` has one row per file of the data folder, saying where it is and what name the code knows it by.
`source settings/roots.sh` turns each row into a link under the repository's `outputs/in/` and sets the [names](#names) below; you set only `LINC_DATA_TREE` ([00](tutorials/00_start_here.md#before-you-start)).
A data folder may hold less than the whole, for example the inputs alone: a step that needs a missing file then says which.
After you add files to it, `source settings/roots.sh refresh` links them.
To keep a new brain's results, copy them into its folder in the data folder, `brains/sub-<ID>/` ([how-to, step 5](how-to/new_brain.md#5-keep-the-results)).

## Names

Every name the documents use, what it is and who sets it.
`settings/roots.sh` sets the folders and the environments in every shell where you source it; [settings/env](../settings/env/README.md) says which environment runs what.

| Name | What it is | Who sets it |
|---|---|---|
| `$LINC_DATA_TREE` | the data folder, read only: `brains/`, `template/`, `models/`, `paper/` | you, before `source settings/roots.sh` |
| `$LINC_DATA` | the working folder, `outputs/local_data/data`, one `sub-<ID>/` per brain; the commands read and write here | `settings/roots.sh` |
| `$LINC_REBUILD_DATA` | the working folder for reconstructing a paper brain from its photographs, `outputs/local_rebuild/data` | `settings/roots.sh`; a command works there when it starts with `LINC_DATA=$LINC_REBUILD_DATA` |
| `$LINC_MRI_DATA` | the working folder of the MRI template's build, `outputs/local_mri/data` | `settings/roots.sh`; a command works there when it starts with `LINC_DATA=$LINC_MRI_DATA` |
| `$LINC_TEMPLATE` | the template's files, under the names the code uses, `outputs/local_data/template` | `settings/roots.sh` |
| `$LINC_RAW` | the raw inputs, under the names the code uses, `outputs/local_data/raw` | `settings/roots.sh` |
| `$LINC_WORK` | working results, `outputs/work`: registrations' working files and checks, completions, template builds, images moved into the template | `settings/roots.sh` |
| `$LINC_MODELS` | the networks' weights, joined and checked, `outputs/models` | `settings/roots.sh` |
| `$PY` | the main Python environment (`linc.yml`): MRI preparation, registration, the MRI template, the labels after `push`, evaluation, the numbers and figures | `settings/roots.sh`, from `envs/linc_envs.sh`, which `install.sh` writes |
| `$PY_DEEP` | the environment of the segmentation networks and the synthesis (`deep.yml`) | `settings/roots.sh`, from `envs/linc_envs.sh` |
| `$PY_TIRL` | the environment of the blockface reconstruction, the histology registration and `linc-labels push` (`tirl.yml`) | `settings/roots.sh`, from `envs/linc_envs.sh` |
| `$ANATOMIX_PYTHON` | the environment of the deformable registrations, anatomix features and FireANTs (`anatomix.yml`); `linc-register` calls it, and the blockface template's final step runs in it | `settings/roots.sh`, from `envs/linc_envs.sh` |
| `$PY_VIZ3D` | optional: the environment of the 3D tracer figure (`viz3d.yml`) | `settings/roots.sh`, from `envs/linc_envs.sh`, when `install.sh` ran with `--with-figures-3d` |
| `FSLDIR`, `ANTSPATH`, `MRTRIX` | where FSL 6.0.7.4, ANTs 2.5.4 and MRtrix3 3.0.2 are installed | you ([settings/env](../settings/env/README.md)) |
| `LINC_TIRL_SRC` | the folder with the TIRL sources | you, before `bash install.sh` |
| `LINC_RESULTS_VIEW` | set to `0` to stop `settings/roots.sh`, `run.py` and the tools from refreshing `outputs/brains/` by themselves | you, optional |

A command may start with a name and its value, as in `LINC_DATA=$LINC_REBUILD_DATA`: that value holds for that command only.
The tutorials also set short names for one block of commands, such as `M=$LINC_DATA/sub-<ID>/mri`; each is set in the block that uses it.

## Settings, steps and checks

Every parameter is in [settings/](../settings/README.md), with a comment naming the Methods section it comes from, and the code does not repeat the values.
There is one file per brain (`settings/brains/<ID>.yaml`) and one per stage (`settings/<stage>.yaml`).

[steps.yaml](../steps.yaml) lists every step of the paper.
For each, it gives the command, inputs, outputs, check and the steps it needs, the GPU memory, RAM, cores, disk and time it takes, and a note.
`run.py` runs these steps ([00, Two ways to run a step](tutorials/00_start_here.md#two-ways-to-run-a-step)) and checks each rerun against the paper's file.
A CPU step must be identical.
A registration must be identical or within a set distance of the original, given in each tutorial's checks, because GPU results differ slightly from one session to the next and TIRL's last stage can round off differently.
[tests/README.md](../tests/README.md) explains what decides whether a rerun matches bit for bit, and which optional steps were rerun only in part.
`bash tests/selftest.sh` checks the installation without data, and `make -C code/figures check` checks the paper's numbers.

## The environments

`install.sh` builds the Python environments, and `settings/roots.sh` points `$PY`, `$PY_DEEP`, `$PY_TIRL`, `$ANATOMIX_PYTHON` and `$PY_VIZ3D` to them ([names](#names)); [settings/env](../settings/env/README.md) says which runs what, and which other programs you need.
TIRL is not public yet: without its sources, `install.sh` builds the TIRL environment without TIRL, and the steps that need it refuse to run.

## What the package cannot do

- **MRI.** A brain's MRI reaches the template only through its b0, and another contrast follows only if it is already aligned with the b0.
  You can prepare a new brain's T2-weighted image (`linc-mri prep --modality t2`), but nothing registers it.
- **Tensors.** `shared.xfm apply` resamples images; it does not rotate tensors (the template's own build does, in `code/template/channels.py`).
- **Networks.** They were trained on this collection's brains, cameras and MRI, so check their masks on other data.
- **Camera.** A camera that moves on nearly every frame cannot be corrected as camera jumps; MF191 and MF196 replay a recorded alignment to an external series instead.
- **Synthesized tissue.** It is estimated from the b0, not measured, and every file that has it marks it.
  From a stand-in b0 it shows the template's anatomy: in MR244 the cerebellar folia stop at the junction plane.
- **Histology.** A section is registered only through the photograph of its own section.
  Between sections, the histology and the labels are interpolated (`provenance_L2.nii.gz` marks the slices; interpolated labels end in `_demons`).
  A component that fails its check stays in, flagged.
- **Labels.** The single-fiber strokes (`Fibers`) are too sparse for the template's solid-label rule, so MR243's template file for them is empty.
- **Reruns.** `run.py` reruns only the paper's brains; a new brain is run one command at a time.
- **Templates.** Aligning a brain leaves the template unchanged.
  Adding one is a rebuild, after which everything aligned to or carried into the old template must be derived again ([00](tutorials/00_start_here.md#adding-a-new-brain-two-different-things)).
- **Sharing.** The photographs, the blockface volumes and everything computed from them stay with the laboratory; the per-brain data the paper reports are available on request ([contact](../README.md#contact)).

What has been tested on a brain treated as new is in the [FAQ](faq.md#what-has-been-tested-on-a-brain-that-is-not-one-of-the-papers).

## Glossary

| Term | Meaning |
|---|---|
| b0 | the diffusion MRI volume without diffusion weighting; the one MRI contrast all sectioned brains share |
| blockface photograph | a photograph of the frozen block from a fixed camera, taken as each section is cut |
| blockface volume | a brain's photographs stacked into 3D (`bf-native`) |
| blockface-based alignment | blockface volume to the blockface template, then the template link; for the 21 brains that built the blockface template, the transforms that built it |
| camera jump | a shift of the camera between two sections, corrected in the reconstruction |
| completion | joining the synthesized tissue to the photographs (`linc-complete`), which makes the completed volume |
| component | the part of a histology section registered on its own: a hemisphere, or the cerebellum with the brainstem; the code calls it a piece |
| fiducial markers | the round holes in the block face, seen in the blockface photographs and used as fiducials (files named `tubes` or `holes`) |
| junction plane | where a brain's photographs end and its synthesized tissue begins |
| match table | which photograph each histology component goes with |
| member | one input of a template: a brain or its mirror image |
| MRI-based alignment | blockface volume to the brain's own b0, then the b0 to the template |
| outline check | how far a moved brain outline lies from the target's, in six directions: PASS, WARN or FAIL |
| regions volume | the template's left, right and cerebellum regions on a brain's blockface volume, plus its fiducial markers |
| section step | the distance between two photographs (`vz`) |
| stand-in b0 | for a brain without MRI, the template's b0 moved into the brain by the cross-modal alignment |
| synthesized tissue | tissue the photographs do not cover, predicted from the b0 by the synthesis network (`_synthesized_mask`) |
| template | the MRI template with its blockface channel, in `mri-template` |
