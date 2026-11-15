# Start here

These tutorials walk through how the template was made and how to bring a new brain into it, using the brains from the paper "A multimodal macaque template for aligning ex vivo MRI and microscopy" (Gonzales et al., in preparation).
The [overview](../overview.md) explains the parts they use.

- [01_mri.md](01_mri.md) shows how the MRI template was built and how to register a new brain's b0 to it.
- [02_blockface.md](02_blockface.md) goes from blockface photographs to a 3D volume in the template, with or without the brain's MRI, and covers the synthesized tissue and the blockface template.
- [03_histology.md](03_histology.md) registers histology sections to their blockface photographs and carries the labels drawn on them into the template.

Each one builds on the one before.
A brain with MRI reaches the template through the b0 registration in 01, 02 starts from there, and 03 starts from the blockface volume and registration made in 02.

## Before you start

Install the repository and its requirements as described in the README's [quick start](../../README.md#quick-start).
Then, in every new shell, run this from the top of the repository:

```bash
export LINC_DATA=<the data folder>
source settings/roots.sh
```

It sets `$PY`, `$PY_TIRL` and `$PY_DEEP` to the Python environments that `install.sh` built, and every command in the tutorials assumes them.
It also points `$LINC_DATA` to a working folder inside the repository, since the code never writes into the data folder ([overview](../overview.md#the-data-folder)).

## Two ways to run a step

**On the paper's brains, with checks.**
`run.py` runs the steps listed in [steps.yaml](../../steps.yaml) and checks each result against the paper's file.

```bash
$PY run.py --plan --only register-mri-template-MR243      # prints the step, its exact command and its check; runs nothing
$PY run.py --run --only register-mri-template-MR243       # runs it, then its check
$PY run.py --status                                       # one line per step: passed, differs or failed
```

`--run` also runs any steps the step needs that have not passed yet, and `--plan` lists them.
`--plan` also prints each step's exact needs from steps.yaml (GPU memory, RAM, cores, disk and time), with notes on measured runs; the tutorials mostly give only the time, and the GPU memory when it is 24 GB or more.
`--only` takes a step (such as `histology-MR243`), a group (`setup`, `mri-template`, `blockface`, `registration`, `synthesis`, `blockface-template`, `link`, `histology`, `labels`, `paper`, `training`) or a brain.
Each step's log goes to `outputs/build_everything/steps/<step>/`.

To rerun everything (the plan lists the GPUs this needs, including 48 GB for the templates):

```bash
mkdir -p outputs
nohup $PY run.py --run > outputs/build_everything.log 2>&1 &
$PY run.py --status
```

The optional steps, building the MRI template and training the networks, run only when you name them with `--only`.

**On a new brain, one command at a time.**
`run.py` covers only the paper's brains, so for a new brain you run the commands the tutorials give, which are the same programs `run.py` calls.
Each command explains what it needs, what it does and what it writes with `--help`.

The paper's numbers and figures have their own commands, described in [code/figures](../../code/figures/README.md).

## Adding a new brain: two different things

**(a) Aligning a new brain to the template** is the normal use.
You register the brain's images to the template and move them into it, where you can compare them with every other brain, and the template stays as it is.
Each tutorial shows how to do this for its kind of data, and [the how-to](../how-to/new_brain.md) lists the steps in order.

**(b) Adding a brain into the template** means building the template again with that brain as a member.
The result is a new template, with new links to the blockface template and to the published templates.
A new MRI template also means a new blockface template, because the MRI template's b0 is one of the blockface template's members.
Everything aligned to the current template then has to be derived again: each brain's transform to the template, the template's blockface channel, and the labels carried into it.
The members are listed in `settings/mri_construction.yaml` and `settings/blockface_template.yaml`, and 01 and 02 show the builds.

This is a decision, not a routine step.
The MRI template's groupwise construction took 3.5 days on one 48 GB card in the paper's run.
The blockface template's first stage takes about 8.5 hours on a free card, and its last step, at 0.25 mm, about 17 hours on one card (estimated) plus about 2.5 hours of CPU steps.
