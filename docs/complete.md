# Synthesize the missing caudal tissue

**You need** the brain's blockface volume and its registration (`docs/register.md`); for a brain without MRI, the stand-in b0 its
registration makes.

**What it does**
1. `prepare`: resamples the registered photographs onto the b0's frame and detects the junction plane, caudal to which the
   photographs no longer cover the brain;
2. predicts the blockface over the whole brain from the b0;
3. matches the prediction's brightness to the photographs near the cut and blends the two over 2 mm;
4. adds color.

**You get**, under the `--out` folder: `<ID>/<ID>_bf_completed.nii.gz`, its color version and the maps of what was predicted, as
NIfTI on the brain's 0.25 mm grid, with a provenance JSON.

Environment: `env/deep.yml`.

No brain was photographed to its caudal end. `cli/linc-complete` (the name stands for this synthesis, as do `_bf_completed` and
`completion/`) synthesizes the missing tissue from the brain's b0, or from a
stand-in b0 for a brain without MRI (`--stand-in`). It runs in two steps: `prepare` makes the brain's folder from its own
products, then the networks synthesize:

```bash
$PY_DEEP cli/linc-complete prepare --subject <ID> --out $LINC_WORK/prepared
$PY_DEEP cli/linc-complete --brain $LINC_WORK/prepared/<ID> --out $LINC_WORK/complete [--stand-in]
```

`prepare` writes `<ID>/` (`seam.json`, `coarse/`, `fine/`) on a CPU, in about a minute, and needs no paths. For the paper's 21 brains,
its inputs are listed in `configs/completion.yaml` (`prepare.inputs`). A new brain's come from its own registration
(`$LINC_WORK/register/<ID>`, from `linc-register mri`) or, without MRI, from `$LINC_DATA/sub-<ID>/mri/stand_in/` (from
`linc-register direct-placement`). For a new brain with MRI, it also detects the tissue only the MRI shows, and writes it as
`<ID>_mri_only_tissue.nii.gz` beside the folder. On MM2 (with MRI) and MF8 (without), it rebuilds the paper's folders array for array. The
synthesis takes about 20 s per brain on a GPU, with about 2 GB of its memory. Three of the paper's prepared
roots link into a sibling root; `$PY_DEEP -m linc.completion.relocate <root> <new root>` makes a copy that works on another machine.

## Train the network

The weights come with the data release. To train the network again, on the 13 brains with MRI that `configs/completion.yaml` lists
(`training`), from their prepared folders; the targets are each brain's own photographs, so there are no hand labels:

```bash
$PY_DEEP cli/linc-complete train --root <prepared folders> --plan     # the brains and any missing files, without running
$PY_DEEP cli/linc-complete train --root <prepared folders>
```

About 1 hour on a GPU, with about 8 GB free. It writes `last.pt`, which `linc-complete --model <folder>` reads, under
`$LINC_WORK/completion/runs/`. `--protocol lso` trains the network of the withheld-slab evaluation. The stand-in network trains on folders made with
`prepare --subject <ID> --out <folder> --targets-from <the prepared folders>`, from each brain's stand-in b0. A retrained network has the same recipe, not the
same bits. Measured once, on the network for brains with MRI on the same GPU model: over the last 10 epochs, its total loss was 0.06 % from the
paper's run while the weights differ, and on one brain (MF5) the synthesized tissue of the two networks correlated with a Pearson r of 0.95.

