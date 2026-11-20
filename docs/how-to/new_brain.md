# Bring a new brain into the template

These are the steps, in order, for a brain that is not one of the paper's; each links to the tutorial step that shows it on a paper brain.
This aligns the brain to the template and leaves the template unchanged; adding a brain into the template is a rebuild ([00](../tutorials/00_start_here.md#adding-a-new-brain-two-different-things)).

## 1. Put its files in the data folder

Copy everything you receive into the data folder, under `brains/sub-<ID>/inputs/`:

| What | Where |
|---|---|
| the blockface photographs | `inputs/photographs/` |
| the MRI | `inputs/mri/` |
| the histology sections and the labels drawn on them | `inputs/histology/` |
| anything drawn or corrected manually | `inputs/manual_annotations/` |

Then, from the top of the data folder, give each folder a `README.txt` saying what it is a copy of, add the brain to `brains.tsv` (columns: brain, paper_name, linc_name, species, data), and add the inputs' checksums:

```bash
cd $LINC_DATA_TREE
find brains/sub-<ID>/inputs -type f ! -name README.txt -exec md5sum {} + >> checksums.md5
```

## 2. Make them visible to the code

The code reads the data folder through `settings/files.tsv` and the links that `settings/roots.sh` makes ([overview](../overview.md#the-data-folder)).
You give the MRI and the histology sections to their commands as paths, and the labels and the section map are read from the brain's histology folder ([03, Step D](../tutorials/03_histology.md#step-d-tracer-labels)).
The photographs are read from `$LINC_RAW/<raw>`, so one link makes them visible:

```bash
ln -s $LINC_DATA_TREE/brains/sub-<ID>/inputs/photographs $LINC_RAW/<ID>    # then raw: <ID> in its settings file
```

This way of adding a brain's photographs has not been tried yet ([FAQ](../faq.md#what-has-been-tested-on-a-brain-that-is-not-one-of-the-papers)).

## 3. Run the steps for the data it has

| It has | Do |
|---|---|
| a b0 | [01, Steps B1 and B2](../tutorials/01_mri.md#step-b1-prepare-the-b0), then [B3](../tutorials/01_mri.md#step-b3-bring-images-into-the-template) for any image on the b0 |
| blockface photographs | its settings file ([02, Step 6](../tutorials/02_blockface.md#step-6-adding-a-new-brain-its-settings-file)), then [02, Steps 1 and 2](../tutorials/02_blockface.md#step-1-reconstruct-the-volume) |
| photographs and a b0 | the two rows above, then [02, Step 3](../tutorials/02_blockface.md#step-3-register-the-volume-to-the-brains-own-b0-a-brain-with-mri) |
| photographs and no MRI | the photographs row, then [02, Step 4](../tutorials/02_blockface.md#step-4-register-the-volume-to-the-blockface-template-a-brain-without-mri-or-with-histology): `blockface-template`, and `direct-placement` for the stand-in b0 that Step 5 needs. With histology, also the refinement at the end of Step 4 |
| photographs, MRI and histology | the first two rows, then [02, Step 3](../tutorials/02_blockface.md#step-3-register-the-volume-to-the-brains-own-b0-a-brain-with-mri), whose registration to the brain's own b0 Step 5 needs, then [02, Step 4](../tutorials/02_blockface.md#step-4-register-the-volume-to-the-blockface-template-a-brain-without-mri-or-with-histology): `blockface-template` and the refinement at its end (0.15 mm), the route the histology takes into the template, as in the paper. The route through the brain's own b0 is the other one the paper reports |
| a blockface volume that stops before the end of the brain | [02, Step 5](../tutorials/02_blockface.md#step-5-synthesize-the-tissue-that-was-never-photographed) |
| histology sections with their photographs | [03, Steps A to C](../tutorials/03_histology.md#step-a-the-regions-volume) |
| labels drawn on those sections | [03, Steps D and E](../tutorials/03_histology.md#step-d-tracer-labels) |

## 4. Check each result

Each tutorial's checks describe what a good and a bad result look like: [01](../tutorials/01_mri.md#4-checks), [02](../tutorials/02_blockface.md#4-checks), [03](../tutorials/03_histology.md#4-checks).
A new brain's registrations are unchecked until you check them yourself.

## 5. Keep the results

The commands write under the repository's `outputs/`, never into the data folder.
`outputs/brains/sub-<ID>/` shows the brain's results laid out like its folder in the data folder, as links ([00](../tutorials/00_start_here.md#where-the-files-are)): `outputs/` holds the finished files your run made, under the data folder's names, and `intermediate/` one link per place where the run wrote for the brain.
To keep the finished files, copy the files the links point to into the data folder, under the same names; from the top of the repository:

```bash
$PY code/shared/linc-results --subject <ID> --transforms      # also the complete transforms to the template, about 15 s each
mkdir -p $LINC_DATA_TREE/brains/sub-<ID>/outputs
cp -L outputs/brains/sub-<ID>/outputs/* $LINC_DATA_TREE/brains/sub-<ID>/outputs/
```

`--transforms` composes the complete transforms from your run's own registrations, into `$LINC_WORK/transforms/<ID>/`; leave it out if you do not want them.
`settings/results.tsv` lists the finished files the view can show; the histology stacked on the blockface, the LGN and STN composites, the labels' `.tsv` tables and the viewer copy of the blockface store are not among them, because no step of the repository writes them.
Copy what else you want to keep from `outputs/brains/sub-<ID>/intermediate/` into the brain's `intermediate/` in the data folder, and give each folder a `README.txt` (any paper brain's `outputs/README.txt` in the data folder lists the names of the finished files).
