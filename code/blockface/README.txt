BLOCKFACE RECONSTRUCTION

Makes a brain's 3D color volume from its blockface photographs (Methods, "Blockface reconstruction").
Environment: settings/env/tirl.yml. Settings: settings/blockface.yaml, and settings/brains/<ID>.yaml.

    $PY_TIRL code/blockface/linc-reconstruct all --subject <ID>

all runs these steps in order. Each can run alone:

    photos    copies the photographs in cutting order; the network makes their brain masks
    stack     stacks the photographs and their masks
    prepare   corrects the camera jumps (TIRL), fills missing sections, evens the illumination
    mask      makes the 3D brain mask
    volume    writes the volume, with the marker tubes filled
    tubes     detects the marker tubes
    spacing   writes the calibrated pixel size and section step into the headers

You get, in the brain's blockface folder: <ID>_blockface.ome.zarr, <ID>_mask.nii.gz (brain mask),
<ID>_holes.nii.gz (marker tubes), <ID>_blockface_L2.nii.gz (a light copy) and <ID>.info.

It takes about 10 hours per brain; prepare is the longest step. The GPU is used only by the networks
(10 GB free for mask and tubes, 1.5 GB for photos). Memory grows with the stack: about 90 GB for
1,400 photographs, 9 GB for 77.

A NEW BRAIN

Copy settings/brains/TEMPLATE.yaml to settings/brains/<ID>.yaml and fill it in. Camera jumps and
missing sections are listed manually, from the stacked photographs.

MANUAL CORRECTIONS (optional)

    3D brain mask     after mask: edit curated/<ID>_mask.nii.gz, save as curated/<ID>_mask_edited.nii.gz
    marker tubes      after volume: save curated/<ID>_holes_edited.nii.gz the same way
    photograph masks  before prepare: linc-reconstruct edit-photo-masks export, edit, then import

A manually corrected mask always wins over a network's.

TWO THINGS ABOUT THE PAPER'S BRAINS

The paper's volumes were reconstructed at each brain's nominal spacing and then given the spacing
calibrated against its own MRI, by changing the headers only (the spacing step). For a new brain
the two are the same. MF8 and MF9 had a camera that moved on nearly every frame: each replays a
recorded alignment to an external series instead (series.py).

    reconstruct.py, prepare.py, photos.py, jumps.py, spacing.py, curate.py   the steps above
    subject.py                                                                reads a brain's settings
    linc-reconstruct                                                          the command
