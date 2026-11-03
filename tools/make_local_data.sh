#!/bin/bash
# Make a writable data tree for one user from the project's read-only products: tools/make_local_data.sh <dest> [ID ...]
# Every folder is a real folder of yours and every file a link to the record, so whatever you run writes beside the record, never into it.
# Set LINC_RECORDS first (env/lab.sh does); then point LINC_DATA, LINC_TEMPLATE and LINC_RAW at <dest>/data, <dest>/template, <dest>/raw.
#
# Details:
#   <dest>/data/sub-<ID>/   (LINC_DATA)   from $LINC_RECORDS/data/sub-<ID>/: the folders blockface/, mri/, histology/ and labels/
#                                         re-created as real folders; files inside are links; an OME-Zarr store is one link (read
#                                         only). xfm/ (the registry your runs write) starts EMPTY; xfm_record/ holds the registry
#                                         entries of record (R1 of record, direct placement of record), each entry's meta.json a
#                                         copy and its transform files links. A brain without products of record gets empty
#                                         folders. Default IDs: MR252 MF191 (the walkthrough brains).
#   --rebuild ID          for a brain you will reconstruct yourself: its blockface/ gets only the hand inputs, as copies
#                         (curated/<ID>_*_edited.*, 01_brainmask/01_final/, and for MF8 / MF9 the recorded series alignment that
#                         configs/subjects/<ID>.yaml series_alignment names), and no reconstruction of record
#   <dest>/template/        (LINC_TEMPLATE) links to $LINC_RECORDS/template/*, plus frames/mri_template_0p5.nii.gz (the frame
#                                         linc-register --place-b0 reads)
#   <dest>/raw/             (LINC_RAW)    a link NAME -> PATH for every --raw NAME=PATH (the photographs: blockface_curated,
#                                         blockface_segmentation; the MRI and histology sources; env/lab.sh passes them)
#   --link ID:PATH=TARGET  <dest>/data/sub-ID/PATH links to TARGET: an input of record kept outside data/sub-ID (e.g. the label
#                          steps' histology/annotations, section_map.tsv, match_table.txt; env/lab.sh passes them)
# Safe to rerun: what exists is left alone, missing parts are added; nothing is ever deleted or overwritten.
set -eu
[ $# -ge 1 ] && [ "$1" != "-h" ] && [ "$1" != "--help" ] || { sed -n '2,4p' "$0"; exit 0; }
DEST=$1; shift
REC=${LINC_RECORDS:?set LINC_RECORDS to the project folder holding data/ and template/ (env/lab.sh sets it)}
REPO=$(cd "$(dirname "$0")/.." && pwd)
IDS=(); REBUILD=(); RAWS=(); LINKS=()
while [ $# -gt 0 ]; do
  case $1 in
    --rebuild) REBUILD+=("$2"); IDS+=("$2"); shift 2;;
    --raw) RAWS+=("$2"); shift 2;;
    --link) LINKS+=("$2"); shift 2;;
    *) IDS+=("$1"); shift;;
  esac
done
[ ${#IDS[@]} -gt 0 ] || IDS=(MR252 MF191)
mkdir -p "$DEST/data" "$DEST/template/frames" "$DEST/raw"
n_link=0; n_copy=0

link() { [ -e "$2" ] || [ -L "$2" ] || { ln -s "$1" "$2"; n_link=$((n_link + 1)); }; }
copy() { [ -e "$2" ] || { cp -p "$1" "$2"; chmod u+w "$2"; n_copy=$((n_copy + 1)); }; }

farm() {                                  # farm <record dir> <local dir> [xfm]: real folders, linked files (zarr stores linked whole)
  local src=$1 dst=$2 mode=${3:-}
  mkdir -p "$dst"
  local e name
  for e in "$src"/* "$src"/.[!.]*; do
    [ -e "$e" ] || [ -L "$e" ] || continue
    name=$(basename "$e")
    if [ -d "$e" ] && [[ "$name" != *.zarr ]]; then farm "$e" "$dst/$name" "$mode"
    elif [ "$mode" = xfm ] && [ "$name" = meta.json ]; then copy "$e" "$dst/$name"
    else link "$e" "$dst/$name"; fi
  done
}

is_rebuild() { local r; for r in "${REBUILD[@]:-}"; do [ "$r" = "$1" ] && return 0; done; return 1; }

for S in "${IDS[@]}"; do
  src="$REC/data/sub-$S"; dst="$DEST/data/sub-$S"
  if [ ! -d "$src" ]; then                # no products of record for this brain (e.g. one prepared from raw MRI): empty folders
    for sub in blockface mri histology labels xfm; do mkdir -p "$dst/$sub"; done
    echo "sub-$S: no $src: empty folders"; continue
  fi
  mkdir -p "$dst"
  for sub in mri histology labels; do
    if [ -d "$src/$sub" ]; then farm "$src/$sub" "$dst/$sub"; else mkdir -p "$dst/$sub"; fi
  done
  # the registry: xfm/ is yours and starts empty (your runs write their entries there); xfm_record/ holds the entries of record for
  # comparison and for chains through steps you do not rerun: the plain xfm/ entries, with the blockface -> own b0 entry taken from
  # the R1 variant of record (configs/subjects/<ID>.yaml registration_to_own_mri.variant_of_record; the plain one is an older run) and,
  # for a brain without MRI, the direct placement of record (xfm_d2s2_prod_s125/)
  mkdir -p "$dst/xfm"
  v=$(sed -n 's/^ *variant_of_record: *\([A-Za-z0-9_]*\).*/\1/p' "$REPO/configs/subjects/$S.yaml" 2>/dev/null | head -1)
  if [ -d "$src/xfm" ]; then
    mkdir -p "$dst/xfm_record"
    for e in "$src/xfm"/*/; do
      e=${e%/}; name=$(basename "$e")
      if [ "$name" = "bf-native__to__b0-native" ] && [ -n "$v" ] && [ -d "$src/xfm_$v/$name" ]; then e="$src/xfm_$v/$name"; fi
      farm "$e" "$dst/xfm_record/$name" xfm
    done
  fi
  if [ -z "$v" ] && [ -d "$src/xfm_d2s2_prod_s125" ]; then
    for e in "$src/xfm_d2s2_prod_s125"/*/; do e=${e%/}; farm "$e" "$dst/xfm_record/$(basename "$e")" xfm; done
  fi
  if is_rebuild "$S"; then                # the hand inputs only, as copies (linc-reconstruct writes the rest)
    mkdir -p "$dst/blockface/curated" "$dst/blockface/01_brainmask/01_final"
    for f in "$src/blockface/curated/${S}"_*_edited.*; do [ -e "$f" ] && copy "$f" "$dst/blockface/curated/$(basename "$f")"; done
    for f in "$src/blockface/01_brainmask/01_final"/*; do [ -e "$f" ] && copy "$f" "$dst/blockface/01_brainmask/01_final/$(basename "$f")"; done
    al=$(awk '/^series_alignment:/{f=1; next} f && /^[^ #]/{f=0} f && /^ *align:/{print $2; exit}' "$REPO/configs/subjects/$S.yaml" 2>/dev/null)
    if [ -n "$al" ] && [ -e "$src/blockface/$al" ]; then          # MF8, MF9: the recorded alignment to their external series
      mkdir -p "$(dirname "$dst/blockface/$al")"; copy "$src/blockface/$al" "$dst/blockface/$al"
    fi
  else
    farm "$src/blockface" "$dst/blockface"
  fi
  echo "sub-$S: $( is_rebuild "$S" && echo "rebuild (hand inputs copied)" || echo "products of record linked" )"
done
for e in "$REC/template"/*; do link "$e" "$DEST/template/$(basename "$e")"; done
link "$REC/template/mri_template_0p5.nii.gz" "$DEST/template/frames/mri_template_0p5.nii.gz"
for kv in "${RAWS[@]:-}"; do [ -n "$kv" ] && link "${kv#*=}" "$DEST/raw/${kv%%=*}"; done
for kv in "${LINKS[@]:-}"; do                # ID:PATH=TARGET -> <dest>/data/sub-ID/PATH links to TARGET (an input kept elsewhere)
  [ -n "$kv" ] || continue
  id=${kv%%:*}; rest=${kv#*:}; rel=${rest%%=*}; tgt=${rest#*=}
  [ -e "$tgt" ] || { echo "--link $kv: no $tgt"; continue; }
  mkdir -p "$(dirname "$DEST/data/sub-$id/$rel")"; link "$tgt" "$DEST/data/sub-$id/$rel"
done
echo "made $DEST: $n_link links, $n_copy copies"
echo "  export LINC_DATA=$DEST/data LINC_TEMPLATE=$DEST/template LINC_RAW=$DEST/raw"
