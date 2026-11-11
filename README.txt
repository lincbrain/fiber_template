FIBER_TEMPLATE

Code for "A multimodal macaque template for aligning ex vivo MRI and microscopy" (Gonzales et al.,
in preparation). It builds the template from blockface photographs and preprocessed MRI, brings new
brains into it, and rewrites the paper's numbers and figures.

If you only want the template, you do not need this code: open template/outputs/ in the data folder.
The MRI template is also in LINC dataset 000059, under derivatives/template/.


THE DATA

The data folder has brains/ (one folder per brain), template/, models/ and paper/. A brain and the
template each have inputs/ (what an instrument or a person made), outputs/ (the finished files: open
these) and intermediate/ (everything else; nobody needs to open it). brains.tsv lists the brains.


WHAT IS HERE

    install.sh    builds the Python environments
    run.py        runs the whole method step by step and checks each step
    steps.yaml    every step: what it reads and writes, its command and its check
    code/         the pipeline, one folder per stage
    settings/     one file per brain and one per stage
    tests/        checks of the installation and of the results

Every folder has a README.txt saying what is in it.


INSTALL

You need Linux, a GPU, FSL 6.0.7.4 and ANTs 2.5.4. TIRL (Tensor Image Registration Library) is not
yet public: ask its authors for the sources.

    export LINC_TIRL_SRC=<folder with the TIRL sources>
    bash install.sh                          # about 25 minutes and 23 GB

The cloud data are LINC dataset 000059 (the MRI template, each brain's MRI results and the MRI
brain-mask network) and LINC dataset 000003 (the histology). Blockface data and anything computed
from them stay local. In every new shell, point the code at your copy:

    export LINC_DATA=<your copy of the data>
    source settings/roots.sh

After that, $LINC_DATA is the folder your results go to, and $LINC_DATA_TREE is the copy you gave.
To check the installation (a few minutes, no data, no GPU):

    bash tests/selftest.sh
    python3 code/shared/linc-info            # prints what this installation runs with


IF YOU HAVE X, DO Y

Find what you have. Each case says what you need, what to run and what you get.

    You have                                                      Case
    only the template                                               1
    an MRI of a new brain, no photographs                           2
    blockface photographs of a new brain                            3
    a blockface volume and an MRI of the same brain                 4
    a blockface volume and no MRI                                   5
    a blockface volume that stops before the end of the brain       6
    histology sections with their blockface photographs             7
    labels drawn on those histology sections                        8
    one image, and you only want its brain mask                     9
    the paper's data, and you want to rebuild it                   10

Blockface photographs are pictures of the frozen tissue block, taken as each section is cut.
A b0 is the diffusion MRI volume with no diffusion weighting.

Before cases 2 to 10, install the code (above).

<ID> is your brain's name, used in every folder and file (for example MR252). Add --help to any
command to see what it needs, what it does and what it writes.

The cloud version of the data (LINC dataset 000059) has the MRI template and the MRI brain-mask
network. Cases 3, 5, 6 and 10, the blockface template in case 1 and the photograph network in case 9
also need data that stay local, with the laboratory. So do the commands of case 1, which read files
that stay local; with the cloud data alone, apply the parts yourself, as
derivatives/template/xfms/README.txt in the cloud dataset says.


1. ONLY THE TEMPLATE

To look at it, you need no code: open any .nii.gz file in template/outputs/ of the data folder, in a
viewer such as FSLeyes or ITK-SNAP.

To move an image between the template and NMT v2, INIA19 or CIVM-R (bring your own copy of
those; add --nearest for a label image):

    $PY -m shared.xfm apply --from mri-template --to atlas:NMT_v2.0_sym --ref <NMT image> \
        <image in the template> <output>
    $PY -m shared.xfm apply --from atlas:NMT_v2.0_sym --to mri-template \
        --ref $LINC_TEMPLATE/mri/linc_mri_b0.nii.gz <image in NMT> <output>

The other names are atlas:INIA19, atlas:CIVM-R and bf-template (the blockface template).
You get: the image on the grid of --ref. It takes seconds, with no GPU.


2. AN MRI OF A NEW BRAIN, NO PHOTOGRAPHS

You need: one preprocessed b0 as NIfTI (from a 4-D series it takes volume 0, or add --volume N)
and a GPU with 14 GB free.

    $PY code/mri/linc-mri prep --subject <ID> --image <b0.nii.gz>
    M=$LINC_DATA/sub-<ID>/mri
    $PY code/blockface/registration/linc-register mri-template --subject <ID> \
        --b0 $M/dwi/b0_brain_n4.nii.gz --mask $M/masks/dwi_brain_mask.nii.gz \
        --place-b0 $LINC_TEMPLATE/frames/mri_template_0p5.nii.gz \
        --template-b0 $LINC_TEMPLATE/mri/linc_mri_b0.nii.gz \
        --template-mask $LINC_TEMPLATE/mri/linc_mri_brainmask.nii.gz

You get: its brain mask and bias-corrected b0 (in $M) and its transform to the template
($LINC_DATA/sub-<ID>/xfm/b0-native__to__mri-template/). About 15 minutes on an RTX 4000 Ada. Tested
once, on the b0 of one of the paper's brains treated as a new brain: a median of 0.2 mm from the
paper's registration.
To bring any image on the b0 into the template:

    $PY -m shared.xfm apply --from b0-native --to mri-template --subject <ID> \
        --ref $LINC_TEMPLATE/mri/linc_mri_b0.nii.gz <image> <output>


3. BLOCKFACE PHOTOGRAPHS OF A NEW BRAIN

You need: the photographs, one JPEG per section, in a folder under $LINC_RAW; their pixel size
and section step; a GPU with 10 GB free; about 90 GB of memory for 1,400 photographs.

Copy the settings file and fill in its REQUIRED fields:

    cp settings/brains/TEMPLATE.yaml settings/brains/<ID>.yaml

Put the photographs in order and stack them:

    $PY_TIRL code/blockface/linc-reconstruct photos --subject <ID>
    $PY_TIRL code/blockface/linc-reconstruct stack --subject <ID>

Go through the numbered photographs in $LINC_DATA/sub-<ID>/blockface/00_slices/. Write the
camera jumps and the missing or damaged sections in <ID>.yaml. Then:

    for s in prepare mask volume tubes spacing; do
        $PY_TIRL code/blockface/linc-reconstruct $s --subject <ID>
    done

You get, in $LINC_DATA/sub-<ID>/blockface/: the 3D volume (<ID>_blockface.ome.zarr), its brain
mask (<ID>_mask.nii.gz) and its marker tubes (<ID>_holes.nii.gz). About 10 hours.


4. A BLOCKFACE VOLUME AND AN MRI OF THE SAME BRAIN

You need: the volume from case 3; the b0 prepared as in the first command of case 2; a GPU with
25 GB free.

    M=$LINC_DATA/sub-<ID>/mri
    $PY code/blockface/registration/linc-register mri --subject <ID> --new-brain \
        --place-b0 $M/dwi/b0_brain_n4.nii.gz $M/masks/dwi_brain_mask.nii.gz
    $PY code/blockface/registration/linc-register mri-template --subject <ID> \
        --template-b0 $LINC_TEMPLATE/mri/linc_mri_b0.nii.gz \
        --template-mask $LINC_TEMPLATE/mri/linc_mri_brainmask.nii.gz

You get, in $LINC_DATA/sub-<ID>/xfm/: bf-native__to__b0-native/ and b0-native__to__mri-template/,
the two transforms from the blockface to the template. About 20 minutes.
To bring an image on the blockface volume into the template (add --nearest for a label image):

    $PY -m shared.xfm apply --from bf-native --to mri-template --subject <ID> \
        --ref $LINC_TEMPLATE/mri/linc_mri_b0.nii.gz <image> <output>

Check each registration before you use it: code/blockface/registration/linc-register qc --help.


5. A BLOCKFACE VOLUME AND NO MRI

You need: the volume from case 3 and a GPU with 19 GB free.

    $PY code/blockface/registration/linc-register blockface-template --subject <ID>

You get: $LINC_DATA/sub-<ID>/xfm/bf-native__to__bf-template/. About 15 minutes.
The command that brings an image into the template is the same as in case 4.


6. A BLOCKFACE VOLUME THAT STOPS BEFORE THE END OF THE BRAIN

No brain was photographed to its caudal end. This predicts the missing tissue and marks it.

You need: the brain registered as in case 4 or 5, and a GPU with 2 GB free. Without MRI, also
run this first (17 GB free, 5 minutes):

    $PY code/blockface/registration/linc-register direct-placement --subject <ID> \
        --template-b0 $LINC_TEMPLATE/mri/linc_mri_b0.nii.gz \
        --template-mask $LINC_TEMPLATE/mri/linc_mri_brainmask.nii.gz

Then (add --stand-in to the second command if the brain has no MRI):

    $PY_DEEP code/blockface/synthesis/linc-complete prepare --subject <ID> --out $LINC_WORK/prepared
    $PY_DEEP code/blockface/synthesis/linc-complete --brain $LINC_WORK/prepared/<ID> \
        --out $LINC_WORK/complete

You get, in $LINC_WORK/complete/<ID>/: the whole brain (<ID>_bf_completed.nii.gz, and _rgb in
color) and the mask of the predicted tissue (<ID>_synth_mask.nii.gz). About 3 minutes.


7. HISTOLOGY SECTIONS WITH THEIR BLOCKFACE PHOTOGRAPHS

Histology reaches the template through the photograph of its own section. Sections without
blockface photographs cannot be used here.

You need: the sections, one JPEG 2000 per section; the brain registered as in case 4 or 5; 24
cores. No GPU.

Carry the template's regions (left, right, cerebellum) into your brain:

    H=$LINC_DATA/sub-<ID>/histology; mkdir -p $H
    $PY -m shared.xfm apply --from mri-template --to bf-native --subject <ID> --nearest \
        --ref $LINC_DATA/sub-<ID>/blockface/<ID>_blockface_L2.nii.gz \
        $LINC_TEMPLATE/labels/regions_in_t2.nii.gz $H/regions.nii.gz

Set the voxels of the left-hemisphere marker tubes in regions.nii.gz to 4. Then propose which
photograph goes with each section:

    STORE=$LINC_DATA/sub-<ID>/blockface/<ID>_blockface.ome.zarr
    $PY_TIRL code/histology/linc-histology propose --subject <ID> --store $STORE \
        --histology <sections folder> --regions $H/regions.nii.gz --out $H/registration

Check the proposal row by row, starting with the flagged ones (match_review.csv). Copy
$H/match_table_proposed.txt to $H/match_table_edited.txt and correct it there. Then:

    $PY_TIRL code/histology/linc-histology all --subject <ID> --store $STORE --threads 12 --jobs 2 \
        --histology <sections folder> --regions $H/regions.nii.gz --out $H/registration

You get, in $H/registration/: each section's parts (hemispheres, cerebellum with brainstem)
registered to their photographs, and the registered images. About 2 hours.


8. LABELS DRAWN ON THOSE HISTOLOGY SECTIONS

You need: case 7 done; the labels, one OME-Zarr per class, in $H/annotations/<class>.ome.zarr;
and $H/section_map.tsv, which says which section each image in the stack is.

    $PY_TIRL code/labels/linc-labels push        --subject <ID>
    $PY      code/labels/linc-labels interpolate --subject <ID>
    $PY      code/labels/linc-labels carry       --subject <ID>
    $PY      code/labels/linc-labels connect     --subject <ID>

You get, in $LINC_DATA/sub-<ID>/labels/: the labels on the blockface (blockface/) and in the
template (template/), with the bundles joined into one pathway. About 40 minutes.


9. ONE IMAGE, AND YOU ONLY WANT ITS BRAIN MASK

You need: a GPU with 5 GB free.

    $PY_DEEP code/shared/linc-segment predict --model mri_brainmask --out <mask.nii.gz> <b0.nii.gz>
    $PY_DEEP code/shared/linc-segment predict --model bf_mask2d --native --in <folder> --out <masks>

The first is for one MRI image (a b0 or a T2-weighted image). The second is for a folder of
blockface photographs, named <case>_0000.png. You get the masks, on the grid of the input.


10. THE PAPER'S DATA, AND YOU WANT TO REBUILD IT

You need: the data tree and the GPUs the plan lists (48 GB for the templates).

    $PY run.py --plan        # every step; runs nothing
    mkdir -p outputs
    nohup $PY run.py --run > outputs/build_everything.log 2>&1 &
    $PY run.py --status      # one line per step

For one part, add --only <part> to --plan or --run: a step, a brain, or one of setup, mri-template,
blockface, registration, synthesis, blockface-template, link, histology, labels, paper, training.
Optional steps (building the MRI template, training the networks) run only when you name them.

You get: every result, each checked against the paper's file and marked passed, differs or
failed. Each step's commands and logs are in outputs/build_everything/steps/<step>/.

To build a template from other brains, change the list of brains in
settings/mri_construction.yaml or settings/blockface_template.yaml, then see what will run:

    $PY code/template/build_mri_channels.py --plan
    $PY code/template/build_blockface_template.py --plan


THE NUMBERS AND FIGURES OF THE PAPER

    make -C code/figures numbers             # also numbers-gpu and numbers-figures
    make -C code/figures figures
    make -C code/figures check               # compares the numbers with the paper's

code/figures/README.txt says which script makes each figure, and tests/README.txt says what decides
whether a rerun matches the paper bit for bit.


THE PUBLISHED TEMPLATES

The transforms between the template and NMT v2, INIA19 and CIVM-R are in the data. Their images are
not: download NMT v2 and CIVM-R yourself. settings/targets.yaml gives the sources and the folders.
CIVM-R's license does not allow its images to be redistributed.


CITE AND LICENSE

Gonzales RA, Habte E, Wu J, Gong T, Maffei C, Celestine MR, Dey N, Huszar IN, París G, Shao J,
Bell E, Lehman JF, Haber SN, Balbastre Y, Yendiki A. A multimodal macaque template for aligning
ex vivo MRI and microscopy. In preparation.

The code is under the Apache License 2.0 (LICENSE, NOTICE). The data are under CC BY 4.0.


CONTACT

Ricardo A. Gonzales, Athinoula A. Martinos Center for Biomedical Imaging, Massachusetts General
Hospital (rgonzalesvera@mgh.harvard.edu).
