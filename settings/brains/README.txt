ONE FILE PER BRAIN

<ID>.yaml holds a brain's photographs, pixel size and section step, camera jumps, missing sections and
masks of record. For a new brain, copy TEMPLATE.yaml to <ID>.yaml and fill it in: every field is
explained there.

WHAT A PERSON MAKES ONCE PER BRAIN

    pixel size and section step     vx, vz in this file; from a ruler and the microtome
    camera jumps, missing sections  jumps, missing in this file; by eye, from the stacked photographs
    corrected masks (optional)      see code/blockface/README.txt
    match table                     see code/histology/README.txt
    section map                     which section each image in the histology stack is; checked by hand
    labels                          drawn on the sections by an anatomist
