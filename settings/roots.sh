# The one root: point LINC_DATA at the data tree and source this file, from any folder, after install.sh:
#
#     export LINC_DATA=<the data tree>
#     source settings/roots.sh
#
# The histology steps also read LINC dataset 000003: the data tree holds the files they read (brains/sub-<ID>/inputs/histology).
# To read a full copy of the dataset instead, set LINC_000003 to it.
#
# It sets
#   - the environments' interpreters ($PY, $PY_DEEP, $PY_TIRL), from envs/linc_envs.sh, which install.sh writes;
#   - outputs/in/: the code's working names for the data tree's files, one link per row of settings/files.tsv (code/shared/link_data.py);
#   - the folders the pipeline works in, under outputs/, whose files link to outputs/in. Nothing is written into the data tree;
#   - every internal root the code reads ($LINC_PRODUCTS, $LINC_TEMPLATE, $LINC_RAW, $LINC_MODELS, $LINC_WORK, $LINC_NUMBERS, ...).
#     Afterwards LINC_DATA itself is the internal per-brain folder, outputs/local_data/data, and the data tree is in $LINC_DATA_TREE.
#     Sourcing it again changes nothing.

_FT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ -z "${LINC_DATA_TREE:-}" ]; then
  if [ -z "${LINC_DATA:-}" ] || [ ! -d "$LINC_DATA" ]; then echo "settings/roots.sh: set LINC_DATA to the data tree first" >&2; unset _FT; return 1; fi
  export LINC_DATA_TREE=$(cd "$LINC_DATA" && pwd)
fi
if [ -f "$_FT/envs/linc_envs.sh" ]; then
  . "$_FT/envs/linc_envs.sh"
else
  echo "settings/roots.sh: no envs/linc_envs.sh yet: run 'bash install.sh', then source this file again" >&2
fi
export PYTHONNOUSERSITE=1

_O=$_FT/outputs; _IN=$_O/in; _L=$_O/local_data; _LOG=$_O/local_data.log; mkdir -p "$_O"
"${PY:-python3}" "$_FT/code/shared/link_data.py" "$LINC_DATA_TREE" "$_IN" >> "$_LOG" 2>&1 || echo "settings/roots.sh: linking the data tree FAILED, see $_LOG" >&2
# the path prefixes recorded inside the products, kept with the data (settings/internal/path_placeholders.yaml explains them)
[ -f "$LINC_DATA_TREE/paper/recorded_path_prefixes.yaml" ] && cp -f "$LINC_DATA_TREE/paper/recorded_path_prefixes.yaml" "$_FT/settings/internal/path_placeholders.local.yaml"

# the trained networks, from the repository's models/: files over 100 MB are kept there in parts; joined once into outputs/models and
# checked against settings/models/*.sha256 (code/shared/join_weights.sh)
bash "$_FT/code/shared/join_weights.sh" "$_O/models" >> "$_LOG" 2>&1 || echo "settings/roots.sh: putting the network weights together FAILED, see $_LOG" >&2
export LINC_RECORDS=$_IN LINC_PRODUCTS=$_IN LINC_MODELS=$_O/models
export LINC_COMPLETION_PREPARED=$_IN/isbi2027/data/canon_d3                    # the 13 brains with MRI
export LINC_COMPLETION_PREPARED_STANDIN=$_IN/isbi2027/data/canon_d3n_prod_lo0  # the 8 brains without MRI (stand-in b0)

# the working folders, built once: local_data (the products: registration, histology, labels), local_rebuild (the hand inputs only, to
# reconstruct from the photographs) and local_mri (the MRI build of the four template brains)
_RAW="--raw blockface_curated=$_IN/raw/blockface_curated --raw blockface_segmentation=$_IN/raw/blockface_segmentation
      --raw macaque_mri=$_IN/raw/macaque_mri --raw microstructure=$_IN/raw/microstructure
      --raw mri_mm=$_IN/raw/mri_mm --raw linc_000003=$_IN/raw/linc_000003"
_LINKS="--link MR252:histology/annotations=$_IN/raw/linc_000003/rawdata/sub-MR252/micr/masks
        --link MR252:histology/section_map.tsv=$_IN/labels/section_maps/MR252_FS.tsv
        --link MR252:histology/match_table.txt=$_IN/labels/match_tables/num_MR252_plus.txt
        --link MR252:histology/regions.nii.gz=$_IN/labels/regions_t2_final/MR252_regions_t2.nii.gz"
if [ ! -e "$_L/.complete" ]; then
  echo "settings/roots.sh: building $_L (once; log: $_LOG)" >&2
  "$_FT/code/shared/make_local_data.sh" "$_L" MR252 MF191 MF283 $_RAW $_LINKS >> "$_LOG" 2>&1 || echo "settings/roots.sh: FAILED, see $_LOG" >&2
fi
if [ ! -e "$_O/local_rebuild/.complete" ]; then
  echo "settings/roots.sh: building $_O/local_rebuild (once; log: $_LOG)" >&2
  "$_FT/code/shared/make_local_data.sh" "$_O/local_rebuild" --rebuild MR252 --rebuild MF191 --rebuild MR263 $_RAW >> "$_LOG" 2>&1 || echo "settings/roots.sh: FAILED, see $_LOG" >&2
fi
if [ ! -e "$_O/local_mri/.complete" ]; then
  _M=""
  for _S in MF283 MF284 MF285 MF286; do
    for _K in dwi t2 micro; do _M="$_M --copy $_S:mri/masks/${_K}_brain_mask.nii.gz=$_IN/mri/data/sub-$_S/masks/${_K}_brain_mask.nii.gz"; done
    _M="$_M --copy $_S:mri/masks/final_brain_mask.nii.gz=$_IN/mri/data/sub-$_S/canonical_rebuild_manual_annotations_frac/model_mask_source_uint8.nii.gz"
  done
  echo "settings/roots.sh: building $_O/local_mri (once; log: $_LOG)" >&2
  "$_FT/code/shared/make_local_data.sh" "$_O/local_mri" MF283 MF284 MF285 MF286 $_M >> "$_LOG" 2>&1 || echo "settings/roots.sh: FAILED, see $_LOG" >&2
  unset _M _S _K
fi
[ -e "$_L/raw/match_tables_manual_annotations" ] || ln -s "$_IN/raw/match_tables_manual_annotations" "$_L/raw/match_tables_manual_annotations"   # the manual match tables (num_<ID>.txt) of MM1 and MM3
export LINC_DATA=$_L/data LINC_TEMPLATE=$_L/template LINC_RAW=$_L/raw
export LINC_PARTICIPANTS=$LINC_RAW/linc_000003/participants.tsv
export LINC_REBUILD_DATA=$_O/local_rebuild/data      # for linc-reconstruct: LINC_DATA=$LINC_REBUILD_DATA code/blockface/linc-reconstruct ...
export LINC_MRI_DATA=$_O/local_mri/data              # for the MRI build: LINC_DATA=$LINC_MRI_DATA $PY code/template/build_mri_channels.py ...
export LINC_MRI_PREPROCESSED=$_O/local_mri/preprocessed   # <ID>/{dwi,msme,dt2}: what stage-<ID> reads
for _S in MF283 MF284 MF285 MF286; do
  _P=$LINC_MRI_PREPROCESSED/$_S; mkdir -p "$_P"
  [ -e "$_P/dwi" ] || ln -s "$LINC_RAW/macaque_mri/Preprocessing/$_S/CS_DSI" "$_P/dwi"
  [ -e "$_P/msme" ] || ln -s "$LINC_RAW/microstructure/data/$_S/msme.n32" "$_P/msme"
  [ -e "$_P/dt2" ] || ln -s "$LINC_RAW/microstructure/data/$_S/dmri.dt2" "$_P/dt2"
done
unset _S _P
export LINC_WORK=$_O/work LINC_NUMBERS=$_O/numbers
export HISTOLOGY_ARGS="--jp2-root $LINC_RAW/linc_000003/sourcedata"                  # the histology figure reads the JPEG 2000 sections
export MM_MRI_ARGS="--raw-bruker-root $_L/raw/mri_mm --unpacked-root $_L/raw/macaque_mri"   # the numbers record mm_mri_check_r054 reads the raw MRI
unset _FT _O _IN _L _LOG _RAW _LINKS
