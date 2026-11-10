REGISTRATION OF A BLOCKFACE VOLUME

Registers a brain's blockface volume to its own MRI or to the template. Environment:
settings/env/linc.yml, with FSL and the anatomix environment. Settings: settings/registration.yaml.

    linc-register mri                  the blockface volume to the brain's own b0
    linc-register mri-template         the brain's b0 to the MRI template
    linc-register blockface-template   a brain without MRI, to the blockface template
    linc-register direct-placement     a brain without MRI, to the MRI template's b0 (gives its
                                       stand-in b0, which code/blockface/synthesis/ reads)
    linc-register qc                   checks one registration

A brain with MRI uses the first two. A brain without MRI uses the last two. The README.txt at the
top of this repository gives the full commands.

GPU MEMORY

mri needs 15 to 25 GB free, depending on the brain. mri-template needs 12 to 14 GB, direct-placement
15 to 17 GB and blockface-template about 19 GB. Each prints what it needs before it starts, and stops
if the card is short. Run one at a time on a card.

WHAT YOU GET

Each registration writes one folder, <from>__to__<to>/, with the warp, its matrices and meta.json.
An existing folder stops the run unless you give --overwrite, which backs up the old one.
shared/xfm.py finds these folders and chains them.

CHECK A REGISTRATION

A new brain's registrations are unchecked until you check them. qc tests the linear part (scales and
rotations), the correlation of the two images, the folds of the warp, the overlap of the brain masks
and the distance between the outlines. It records pass or fail in the registration's folder, and
writes a three-plane picture and a JSON row. The presets are bf2b0, direct, b02template,
bf2bftemplate and template2template. --help lists the inputs; the limits are in
settings/registration.yaml.

    linc-register              the command
    to_own_mri.py, direct.py   the registrations
    featurereg.py              the deformable engine (anatomix features and FireANTs)
    qc.py                      the checks
