SHARED TOOLS

    paths.py             finds every input and output from the variables settings/roots.sh sets
    xfm.py               the transforms: finds them, chains them and applies them once
    provenance.py        records each product's inputs, command and tool versions
    linc-segment         runs and trains the segmentation networks (segment/)
    linc-info            prints what this installation runs with; attach it to a bug report
    link_data.py         gives the code its working names for the files of the data tree
    make_local_data.sh   builds the working folders from the data tree
    subjects_overview.py prints all the brains' settings as one table
    run_until.py         runs a command until a file appears, then stops it

SPACES AND TRANSFORMS

    bf-native                         a brain's blockface volume
    b0-native, t2-native, micro-native  a brain's MRI as acquired
    mri-template                      the MRI template, 0.25 mm
    bf-template                       the blockface template in its own space
    histo-piece:<piece>               one part of a histology section
    atlas:<name>                      a published template (NMT_v2.0_sym, INIA19, CIVM-R)

A transform <from>__to__<to> moves images from the first space to the second. Its folder holds the
transform files and meta.json. World coordinates are RAS, in millimeters. A chain is applied with one
interpolation. An inverted link is solved at every voxel of the target grid, so use 0.3 mm or coarser.

    $PY -m shared.xfm chain --from bf-native --to mri-template --subject <ID>   lists the links
    $PY -m shared.xfm apply --from <a> --to <b> [--subject <ID>] --ref <grid> [--nearest] <in> <out>
