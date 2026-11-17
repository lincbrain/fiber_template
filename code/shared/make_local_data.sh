#!/bin/bash
# Make your own writable data tree from the project's data of record (read only), so what you run never writes into the record.
#     code/shared/make_local_data.sh <dest> [ID ...] [--rebuild ID] [--raw NAME=PATH] [--link ID:PATH=TARGET] [--copy ID:PATH=SOURCE]
#
# You need
#   LINC_RECORDS: the project folder holding data/ and template/ (settings/lab.sh sets it, and runs this script for you).
# What it does
#   For each brain (default MR252 MF191, the walkthrough brains) it re-creates $LINC_RECORDS/data/sub-<ID>/ under <dest>/data as
#   real folders of yours, with every file inside a link to the record (an OME-Zarr store is one link). Safe to rerun: what exists
#   is left alone, missing parts are added; nothing is deleted or overwritten.
#   --rebuild ID          for a brain you will reconstruct yourself: blockface/ gets only the manual inputs, as copies
#                         (curated/<ID>_mask_edited.nii.gz and _holes_edited.nii.gz, 01_brainmask/01_final/ (MF8 / MF9: the
#                         masks from before their series alignment), and for MF8 / MF9 the recorded series alignment that
#                         settings/brains/<ID>.yaml series_alignment names; for MR254 the earlier mask its reconstruction_mask
#                         names), and no reconstruction of record
#   --raw NAME=PATH       <dest>/raw/NAME links to PATH (the photographs, the MRI and histology sources)
#   --link ID:PATH=TARGET <dest>/data/sub-ID/PATH links to TARGET (an input of record kept outside data/sub-ID, e.g. the label
#                         steps' histology/annotations, section_map.tsv, match_table.txt)
#   --copy ID:PATH=SOURCE <dest>/data/sub-ID/PATH is a copy of SOURCE (a manually made input, e.g. a brain's masks for the MRI build)
# You get
#   <dest>/data/sub-<ID>/  (point LINC_DATA here)      blockface/, mri/, histology/, labels/ as above; xfm/ (the registry your
#                          runs write) EMPTY; xfm_record/ the registry entries of record (the blockface-to-b0 registration; for a
#                          brain without MRI, the cross-modal alignment), each meta.json a copy, the transform files
#                          links. A brain without products of record gets
#                          empty folders.
#   <dest>/template/       (point LINC_TEMPLATE here)  links to $LINC_RECORDS/template/*, plus frames/mri_template_0p5.nii.gz (the
#                          frame linc-register --place-b0 reads)
#   <dest>/raw/            (point LINC_RAW here)       the --raw links
#   <dest>/.complete       written last, when the tree is complete (settings/lab.sh builds again when it is missing)
set -eu
[ $# -ge 1 ] && [ "$1" != "-h" ] && [ "$1" != "--help" ] || { sed -n '2,28p' "$0" | cut -c3-; exit 0; }
DEST=$1; shift; ARGS="$*"
REC=${LINC_RECORDS:?set LINC_RECORDS to the project folder holding data/ and template/ (settings/lab.sh sets it)}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
IDS=(); REBUILD=(); RAWS=(); LINKS=(); COPIES=()
while [ $# -gt 0 ]; do
  case $1 in
    --rebuild) REBUILD+=("$2"); IDS+=("$2"); shift 2;;
    --raw) RAWS+=("$2"); shift 2;;
    --link) LINKS+=("$2"); shift 2;;
    --copy) COPIES+=("$2"); shift 2;;
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
  # the R1 variant of record (settings/brains/<ID>.yaml registration_to_own_mri.variant_of_record; the plain one is an older run) and,
  # for a brain without MRI, the direct placement of record (xfm_d2s2_prod_s125/) and its route through the blockface template (xfm_member/)
  mkdir -p "$dst/xfm"
  v=$(sed -n 's/^ *variant_of_record: *\([A-Za-z0-9_]*\).*/\1/p' "$REPO/settings/brains/$S.yaml" 2>/dev/null | head -1)
  if [ -d "$src/xfm_record" ]; then       # a data tree: the entries of record are already laid out (settings/files.tsv)
    farm "$src/xfm_record" "$dst/xfm_record" xfm
  elif [ -d "$src/xfm" ]; then
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
  if [ -d "$src/xfm_member" ]; then       # a brain without MRI: its route through the template it helped build (construction warp, rigid placement)
    for e in "$src/xfm_member"/*/; do e=${e%/}; farm "$e" "$dst/xfm_record/$(basename "$e")" xfm; done
  fi
  if is_rebuild "$S"; then                # the manual inputs only, as copies (linc-reconstruct writes the rest)
    mkdir -p "$dst/blockface/curated" "$dst/blockface/01_brainmask/01_final"
    for k in mask holes; do                # the manual edits the reconstruction reads (curated/<ID>_mask_edited, _holes_edited)
      f="$src/blockface/curated/${S}_${k}_edited.nii.gz"; [ -e "$f" ] && copy "$f" "$dst/blockface/curated/$(basename "$f")"
    done
    rm0=$(sed -n 's/^reconstruction_mask: *\([^ #]*\).*/\1/p' "$REPO/settings/brains/$S.yaml" 2>/dev/null | head -1)
    if [ -n "$rm0" ] && [ -e "$src/blockface/$rm0" ]; then   # the earlier mask the volume of record was reconstructed with
      copy "$src/blockface/$rm0" "$dst/blockface/$rm0"
    fi
    al=$(awk '/^series_alignment:/{f=1; next} f && /^[^ #]/{f=0} f && /^ *align:/{print $2; exit}' "$REPO/settings/brains/$S.yaml" 2>/dev/null)
    m="$src/blockface/01_brainmask/01_final"  # the accepted photograph masks; for MF8 / MF9 those from before the series alignment,
    [ -n "$al" ] && [ -d "$src/blockface/01_final_prealign" ] && m="$src/blockface/01_final_prealign"   # which the rebuild applies
    for f in "$m"/*; do [ -e "$f" ] && copy "$f" "$dst/blockface/01_brainmask/01_final/$(basename "$f")"; done
    if [ -n "$al" ] && [ -e "$src/blockface/$al" ]; then          # MF8, MF9: the recorded alignment to their external series
      mkdir -p "$(dirname "$dst/blockface/$al")"; copy "$src/blockface/$al" "$dst/blockface/$al"
    fi
  else
    farm "$src/blockface" "$dst/blockface"
  fi
  echo "sub-$S: $( is_rebuild "$S" && echo "rebuild (manual inputs copied)" || echo "products of record linked" )"
done
for e in "$REC/template"/*; do [ -e "$e" ] || [ -L "$e" ] || continue; link "$e" "$DEST/template/$(basename "$e")"; done
link "$REC/template/mri_template_0p5.nii.gz" "$DEST/template/frames/mri_template_0p5.nii.gz"
for kv in "${RAWS[@]:-}"; do [ -n "$kv" ] && link "${kv#*=}" "$DEST/raw/${kv%%=*}"; done
for kv in "${LINKS[@]:-}"; do                # ID:PATH=TARGET -> <dest>/data/sub-ID/PATH links to TARGET (an input kept elsewhere)
  [ -n "$kv" ] || continue
  id=${kv%%:*}; rest=${kv#*:}; rel=${rest%%=*}; tgt=${rest#*=}
  [ -e "$tgt" ] || { echo "--link $kv: no $tgt"; continue; }
  mkdir -p "$(dirname "$DEST/data/sub-$id/$rel")"; link "$tgt" "$DEST/data/sub-$id/$rel"
done
for kv in "${COPIES[@]:-}"; do               # ID:PATH=SOURCE -> <dest>/data/sub-ID/PATH is a copy of SOURCE
  [ -n "$kv" ] || continue
  id=${kv%%:*}; rest=${kv#*:}; rel=${rest%%=*}; src=${rest#*=}
  [ -r "$src" ] || { echo "--copy $kv: cannot read $src"; exit 1; }
  mkdir -p "$(dirname "$DEST/data/sub-$id/$rel")"; copy "$src" "$DEST/data/sub-$id/$rel"
done
echo "$(date '+%F %T') code/shared/make_local_data.sh $DEST $ARGS" > "$DEST/.complete"   # written last: settings/lab.sh rebuilds a tree without it
echo "made $DEST: $n_link links, $n_copy copies"
echo "  export LINC_DATA=$DEST/data LINC_TEMPLATE=$DEST/template LINC_RAW=$DEST/raw"
