# Start here

These tutorials show, step by step and on the paper's brains, how the template was made and how to bring a new brain into it.
They go with the paper "A multimodal macaque template for aligning ex vivo MRI and microscopy" (Gonzales et al., in preparation).
The [overview](../overview.md) explains the parts they use: the data, the spaces, the templates, the networks and the data folder.

| Tutorial | What it covers |
|---|---|
| [01_mri.md](01_mri.md) | How the MRI template was built, and how a new brain's b0 is registered to it. |
| [02_blockface.md](02_blockface.md) | From blockface photographs to a 3D volume in the template, with or without the brain's MRI; the synthesized tissue; the blockface template. |
| [03_histology.md](03_histology.md) | Histology sections registered to their blockface photographs, and the labels drawn on them carried into the template. |

Each builds on the one before it.
A brain with MRI reaches the template through 01's b0 registration, 02 starts from it, and 03 starts from 02's blockface volume and its registration.

## Before you start

You need Linux, a GPU, FSL 6.0.7.4 and ANTs 2.5.4.
TIRL, which the blockface reconstruction and the histology registration use, is not yet public: ask its authors for the sources.

Once, from the top of the repository:

```bash
export LINC_TIRL_SRC=<folder with the TIRL sources>
bash install.sh                  # about 25 minutes and 23 GB
bash tests/selftest.sh           # a few minutes, no data, no GPU
```

In every new shell, from the top of the repository:

```bash
export LINC_DATA=<the data folder>
source settings/roots.sh
```

This sets `$PY`, `$PY_TIRL` and `$PY_DEEP`, the Python environments that `install.sh` built, and every command in the tutorials assumes it.
After it, `$LINC_DATA_TREE` is the data folder you gave, and `$LINC_DATA` is the repository's own working folder, `outputs/local_data/data`: the code reads the data folder and never writes into it ([overview](../overview.md#the-data-folder)).

## Two ways to run a step

**On the paper's brains, with checks.**
`run.py` runs the steps of [steps.yaml](../../steps.yaml) and checks each result against the paper's file.

```bash
$PY run.py --plan --only register-mri-template-MR243      # prints the step, its exact command and its check; runs nothing
$PY run.py --run --only register-mri-template-MR243       # runs it, then its check
$PY run.py --status                                       # one line per step: passed, differs or failed
```

`--run` also runs the steps that a step needs, when they have not passed yet; `--plan` lists them.
`--only` takes a step (such as `histology-MR243`), a group (`setup`, `mri-template`, `blockface`, `registration`, `synthesis`, `blockface-template`, `link`, `histology`, `labels`, `paper`, `training`), or a brain.
Each step's log is in `outputs/build_everything/steps/<step>/`.

To rerun everything (the plan lists the GPUs it needs: 48 GB for the templates):

```bash
mkdir -p outputs
nohup $PY run.py --run > outputs/build_everything.log 2>&1 &
$PY run.py --status
```

The optional steps (building the MRI template, training the networks) run only when you name them with `--only`.

**On a new brain, one command at a time.**
The tutorials give the commands, which are the programs `run.py` runs; `run.py` itself covers only the paper's brains.
Every command prints what it needs, what it does and what it writes with `--help`.

The paper's numbers and figures have their own commands: [code/figures](../../code/figures/README.md).

## Adding a new brain: two different things

**(a) Aligning a new brain to the template** is the normal use.
The brain's images are registered to the template and moved into it, where they can be compared with every other brain.
The template does not change.
Each tutorial shows how for its kind of data, and [the how-to](../how-to/new_brain.md) lists the steps in order.

**(b) Adding a brain into the template** means building the template again, with the brain as a member.
The result is a new template, with new links to the blockface template and to the published templates.
A new MRI template also means a new blockface template, because the MRI template's b0 is one of its members.
Everything aligned to the current template must then be derived again: each brain's transform to the template, the template's blockface channel, and the labels carried into it.
The members are listed in `settings/mri_construction.yaml` and `settings/blockface_template.yaml`, and 01 and 02 show the builds.
It is a decision, not a routine step:

| Rebuild | Time |
|---|---|
| The MRI template's groupwise construction | 3.5 days on one 48 GB card (the paper's run) |
| The blockface template's first stage | about 8.5 hours on a free card |
| The blockface template's last step, at 0.25 mm | about 17 hours on one card (estimated), plus about 2.5 hours of CPU steps |
