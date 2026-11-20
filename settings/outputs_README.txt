OUTPUTS: WHAT YOUR OWN RUNS WRITE

This folder is what your own runs write. settings/roots.sh makes it from the data folder (the folder
LINC_DATA_TREE names); nothing here is written back into the data folder.

The working area is this folder, not the whole repository. A data folder may sit inside the
repository too, beside outputs/; it is still the data folder, and the tools only read it.

The paper's finished files are in the data folder, brains/sub-<ID>/outputs. brains/ here shows your
run's own results in the same layout.

    brains/                each brain's results in the data folder's layout, as links: sub-<ID>/inputs (the data
                           folder's inputs), sub-<ID>/intermediate/ (each place your run wrote for the brain:
                           local_data, local_rebuild, local_mri and work_<kind>) and sub-<ID>/outputs/ (the
                           finished files your run made, named as in the data folder; settings/results.tsv
                           lists them). code/shared/linc-results updates it; every tool does too, at its end.
                           Your run's own file is one that physically lives in the working folder or work
                           folder it was found in; a link to the paper's file, or a file inside a linked
                           folder, is not, wherever it points
    in/                    links to the data folder's files, under the names the code uses; nothing of yours
    local_data/data        $LINC_DATA, the working folder: one sub-<ID>/ per brain, with blockface/, mri/,
                           xfm/ (your runs' registrations), histology/ and labels/; a paper brain also has
                           xfm_record/, the paper's registrations
    local_data/raw         $LINC_RAW: the raw inputs (photographs, MRI, histology), as links
    local_data/template    $LINC_TEMPLATE: the template's files, as links
    local_rebuild/data     $LINC_REBUILD_DATA: the working folder for reconstructing a paper brain from its
                           photographs; its blockface/ holds only the manual inputs, as copies
    local_mri/data         $LINC_MRI_DATA: the working folder of the MRI template's build; not made when the
                           data folder lacks the brain masks of the four template brains (MF283 to MF286)
    local_mri/preprocessed $LINC_MRI_PREPROCESSED: links to those four brains' preprocessed MRI, which
                           their build reads
    work/                  $LINC_WORK: working results: registrations' working files and checks, completions,
                           template builds, images moved into the template
    numbers/               $LINC_NUMBERS: the paper's numbers, as your runs compute them again
    models/                $LINC_MODELS: the networks' weights, joined and checked
    build_everything/      run.py's steps: each one's command, log and check
    local_data.log         what settings/roots.sh did, with its errors

work/, numbers/ and build_everything/ appear with the first command that writes there. brains/ is made
when settings/roots.sh builds outputs/ or refreshes it, or by code/shared/linc-results.
settings/roots.sh writes this file again with `source settings/roots.sh refresh`; it changes nothing
else of yours.
