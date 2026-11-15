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
| photographs and no MRI | [02, Step 4](../tutorials/02_blockface.md#step-4-register-the-volume-to-the-blockface-template-a-brain-without-mri) |
| a blockface volume that stops before the end of the brain | [02, Step 5](../tutorials/02_blockface.md#step-5-synthesize-the-tissue-that-was-never-photographed) |
| histology sections with their photographs | [03, Steps A to C](../tutorials/03_histology.md#step-a-the-regions-volume) |
| labels drawn on those sections | [03, Steps D and E](../tutorials/03_histology.md#step-d-tracer-labels) |

## 4. Check each result

Each tutorial's checks describe what a good and a bad result look like: [01](../tutorials/01_mri.md#4-checks), [02](../tutorials/02_blockface.md#4-checks), [03](../tutorials/03_histology.md#4-checks).
A new brain's registrations are unchecked until you check them yourself.

## 5. Keep the results

The commands write under the repository's `outputs/`, never into the data folder.
To keep a result, copy it into the data folder under `brains/sub-<ID>/`: the finished files in `outputs/` and everything else in `intermediate/`, named like the paper's brains' files (any brain's `outputs/README.txt` lists the names), with a `README.txt` in each folder.
