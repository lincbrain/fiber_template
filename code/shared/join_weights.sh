#!/usr/bin/env bash
# Put the trained networks of the repository's models/ together in one folder, the one $LINC_MODELS names (settings/roots.sh):
#
#     bash code/shared/join_weights.sh [<folder>]          # default outputs/models
#
# A weight file over GitHub's 100 MB limit is kept in models/ as <file>.part1, <file>.part2, ...; it is joined here, in order.
# Every other file is linked. Then every file is checked against settings/models/*.sha256 (tests/verify_weights.py), and the
# folder is marked complete. Run again, it does nothing; delete the folder to build it again.
set -euo pipefail
FT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SRC=$FT/models
DST=${1:-$FT/outputs/models}
[ -e "$DST/.complete" ] && exit 0
mkdir -p "$DST"
(cd "$SRC" && find . -type f | sort) | while read -r rel; do
  rel=${rel#./}; out=$DST/$rel; mkdir -p "$(dirname "$out")"
  case "$rel" in
    *.part1) base=${rel%.part1}; i=1; : > "$DST/$base.joining"
             while [ -e "$SRC/$base.part$i" ]; do cat "$SRC/$base.part$i" >> "$DST/$base.joining"; i=$((i + 1)); done
             mv "$DST/$base.joining" "$DST/$base" ;;
    *.part[0-9]*) ;;
    *) ln -sfn "$SRC/$rel" "$out" ;;
  esac
done
PYV=${PY:-python3}
"$PYV" "$FT/tests/verify_weights.py" "$FT/settings/models/segmentation.sha256" --root "$DST/segment" > "$DST/verify.log"
"$PYV" "$FT/tests/verify_weights.py" "$FT/settings/models/completion_public.sha256" --root "$DST/completion" >> "$DST/verify.log"
touch "$DST/.complete"
