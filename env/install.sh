#!/bin/bash
# Build the repository's environments, and check that they work.
#
# You need
#   conda (or mamba) and git on PATH; about 23 GB; network access during the install only: the conda channels, PyPI, the PyTorch and VTK wheel
#   indexes, GitHub (anatomix and FireANTs) and huggingface.co (anatomix's weights, 105 MB).
#   TIRL's sources, for the histology registration and the reconstruction's Python steps: TIRL is not publicly distributed yet, so give a local folder
#   with its four source trees (--tirl-src DIR, or LINC_TIRL_SRC=DIR): tirl 3.6.3, tirlscripts-oxford-scriptutils 1.1.1, tirlscripts-stanford-core 1.2.2 and
#   tirlscripts-mgh-linc 1.0.1, each found as <package>, _src_<package> or <package>-<version> (env/tirl.yml). Without them the tirl environment is created
#   without TIRL, those steps refuse, and make test skips them. The folder is only read: each tree is copied to DIR/src/<package> and built there.
#
# What it does  (expected duration on a cluster node with a fast network; most of it is downloading)
#   linc      env/linc.yml        ~3 min   orchestration, the MRI construction (MMORF), registration's linear stages,
#                                          the labels' interpolation, the numbers and most figures   -> PY (= PY_MRI), MMORF
#   deep      env/deep.yml        ~6 min   segmentation models and tissue synthesis (torch 2.6 + CUDA 12.4)        -> PY_DEEP
#   anatomix  env/anatomix.yml    ~9 min   the deformable registration engine (torch 2.13 + CUDA 12.6; anatomix and
#                                          its FireANTs fork from their repositories at the pinned commits, and the
#                                          network's weights at the pinned revision, checked)    -> ANATOMIX_PYTHON, HF_HOME
#   tirl      env/tirl.yml        ~3 min   histology registration, reconstruction's Python steps (+ TIRL sources)   -> PY_TIRL
#   then `make test` (~2 min), which checks each environment on toy data. Every step prints what it runs, in a clean shell with PYTHONNOUSERSITE=1 and
#   PIP_USER=0, so packages in ~/.local never enter or shadow an environment. `pip check` in linc-anatomix lists three requirements the FireANTs fork
#   declares but the engine never imports (hydra-core, pytest, typing), as in the environment of record: expected, not a failure.
#   Safe to rerun: what this script built from the same env file is skipped, and a rerun adds what is missing. An environment built otherwise is
#   reported and left alone (nothing is ever deleted: remove it yourself to rebuild).
#
# You get  (in DIR, by default envs/ in this clone)
#   DIR/linc, linc-deep, linc-anatomix, linc-tirl; DIR/src, the source checkouts and the downloads; DIR/linc_envs.sh: `source` it to point the pipeline at them.
#   About 20-25 min and 23 GB in all (measured from fresh clones on this project's cluster, 3 Nov 2026: 18.5, 20.3 and 21.1 min; linc 1.3 GB, deep 5.5 GB,
#   anatomix 7.1 GB, tirl 2.7 GB, sources 0.4 GB). About 5 GB of that is downloads: conda's package cache in DIR/src/conda_pkgs (set CONDA_PKGS_DIRS to use
#   another one) and pip's in DIR/src/pip_cache; both can be removed once the build is done. A shared conda's own package folder is not written, and
#   runs need no network: linc_envs.sh keeps anatomix's weights in DIR/src/hf_home (HF_HOME, HF_HUB_OFFLINE=1).
#   Outside DIR it writes only small files: conda's list of environments and notices (~/.conda/environments.txt, ~/.cache/conda), matplotlib's font list
#   (~/.cache/matplotlib), the GPU driver's compute cache (~/.nv), and make test's logs in $TMPDIR (default /tmp). TIRL's setup.py writes $HOME/.config/tirl*.yml,
#   which would reset your account's TIRL script registry, so HOME points into DIR/src/tirl_home for that step; the pipeline needs neither file at run time.
#
#   bash env/install.sh [--prefix DIR] [--tirl-src DIR] [--with-figures-3d] [--only NAME[,NAME...]]
#
#   --prefix DIR       where the environments go (default: envs/ in this clone)
#   --tirl-src DIR     TIRL's source trees (above); on the lab cluster env/lab.sh sets LINC_TIRL_SRC
#   --with-figures-3d  also the 3D tracer figure's environment (env/viz3d.yml: pyvista on a headless VTK; ~1 min, 1.1 GB)      -> PY_VIZ3D
#   --only NAMES       just these (linc, deep, anatomix, tirl, figures-3d), without make test
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
PREFIX="$REPO/envs"; TIRL_SRC="${LINC_TIRL_SRC:-}"; VIZ3D=0; ONLY=""   # LINC_TIRL_SRC: the TIRL sources (env/lab.sh)
while [ $# -gt 0 ]; do
  case $1 in
    --prefix) PREFIX=$2; shift 2;; --tirl-src) TIRL_SRC=$2; shift 2;; --with-figures-3d) VIZ3D=1; shift;;
    --only) ONLY=$2; shift 2;;
    -h|--help) awk 'NR > 1 && /^#/ {print; next} NR > 1 {exit}' "$0"; exit 0;;
    *) echo "unknown argument: $1 (see --help)"; exit 2;;
  esac
done
CONDA=$(command -v conda) || { CONDA=$(command -v mamba) && "$CONDA" --version > /dev/null 2>&1; } || { echo "conda (or a working mamba) not found on PATH"; exit 2; }   # conda first: some mamba builds are broken
command -v git > /dev/null || { echo "git not found on PATH"; exit 2; }
mkdir -p "$PREFIX/src"; PREFIX=$(cd "$PREFIX" && pwd)
PKGS="${CONDA_PKGS_DIRS:-$PREFIX/src/conda_pkgs}"          # conda's package cache: in the prefix unless you set CONDA_PKGS_DIRS
CLEAN=(env -i HOME="$HOME" PATH="$(dirname "$CONDA"):/usr/bin:/bin" PYTHONNOUSERSITE=1 PIP_USER=0 LANG=C.UTF-8
       PIP_CACHE_DIR="$PREFIX/src/pip_cache" CONDA_PKGS_DIRS="$PKGS" TMPDIR="${TMPDIR:-/tmp}")   # downloads stay in the prefix (not ~/.cache/pip or a shared conda)
fails=0
say() { echo "[$(date +%H:%M:%S)] $*"; }
run() { say "  \$ $*"; "${CLEAN[@]}" "$@" || { say "  FAILED: $*"; return 1; }; }
want() { [ -z "$ONLY" ] || [[ ",$ONLY," == *",$1,"* ]]; }
say "prefix $PREFIX; conda package cache $PKGS; pip cache $PREFIX/src/pip_cache"

# create <short name> <env file>: DIR/<the file's name:> with a stamp (the file's md5) so a rerun can tell it built this one
create() {
  local short=$1 file="$REPO/env/$2" name; name=$(sed -n 's/^name: *//p' "$file")
  local p="$PREFIX/$name" md5; md5=$(md5sum "$file" | cut -d' ' -f1)
  if [ -d "$p" ]; then
    if [ "$(cat "$p/.linc_env_file_md5" 2>/dev/null)" = "$md5" ]; then say "$short: $p exists, built from this env/$2: skipped"; return 0; fi
    if [ -f "$p/.linc_env_file_md5" ]; then
      say "$short: OUTDATED: $p was built by this script from an older version of env/$2: remove it (rm -rf $p) and run install.sh again"
    else
      say "$short: $p exists but was not built by this script: left alone; to rebuild it, remove it (rm -rf $p) and run install.sh again"
    fi
    return 1
  fi
  say "$short: creating $p from env/$2"
  run "$CONDA" env create -y -f "$file" -p "$p" || return 1
  run "$p/bin/python" -m pip install --no-deps -e "$REPO" || return 1
  echo "$md5" > "$p/.linc_env_file_md5"
}
py() { echo "$PREFIX/$(sed -n 's/^name: *//p' "$REPO/env/$1")/bin/python"; }

if want linc; then create linc linc.yml || fails=$((fails + 1)); fi
if want deep; then create deep deep.yml || fails=$((fails + 1)); fi
if want anatomix; then
  if create anatomix anatomix.yml; then
    A="$PREFIX/src/anatomix"; B="$A/anatomix/registration/registration_backend/fireants"; P=$(py anatomix.yml)
    if ! "${CLEAN[@]}" "$P" -c "import anatomix, fireants" 2> /dev/null; then         # the engine at the commits of record
      [ -d "$A" ] || run git clone https://github.com/neel-dey/anatomix.git "$A"
      run git -C "$A" checkout 4c86aeeca5e4bf712d9e5775a01fadb37b30c13a && run "$P" -m pip install --no-deps -e "$A" &&
      { [ -d "$B" ] || run git clone https://github.com/neel-dey/FireANTs "$B"; } &&
      run git -C "$B" checkout 21ae072 && run "$P" -m pip install --no-deps -e "$B" || fails=$((fails + 1))
    else say "anatomix: anatomix and fireants already installed: skipped"; fi
    # the network's weights, once, at the revision of record (env/anatomix.yml's header): linc_envs.sh then keeps runs offline, reading them here
    H="$PREFIX/src/hf_home"; rev=$(grep -o 'revision [0-9a-f]\{40\}' "$REPO/env/anatomix.yml" | cut -d' ' -f2); sha=$(grep -o 'sha256 [0-9a-f]\{64\}' "$REPO/env/anatomix.yml" | cut -d' ' -f2)
    if [ ! -f "$H/hub/models--neeldey--anatomix/refs/main" ]; then
      say "anatomix: the weights anatomix-dev-vit.pth at revision $rev"
      f=$("${CLEAN[@]}" HF_HOME="$H" "$P" -c "from huggingface_hub import hf_hub_download as d; print(d('neeldey/anatomix', 'anatomix-dev-vit.pth', revision='$rev'))" | tail -n 1)
      if [ "$(sha256sum "$f" 2>/dev/null | cut -d' ' -f1)" = "$sha" ]; then
        mkdir -p "$H/hub/models--neeldey--anatomix/refs" && printf "%s" "$rev" > "$H/hub/models--neeldey--anatomix/refs/main" || fails=$((fails + 1))   # no newline: the hub reads it verbatim
      else say "  FAILED: the weights ($f) do not match the sha256 in env/anatomix.yml"; fails=$((fails + 1)); fi
    else say "anatomix: the weights are already in $H: skipped"; fi
  else fails=$((fails + 1)); fi
fi
if want tirl; then
  if create tirl tirl.yml; then
    P=$(py tirl.yml)
    if "${CLEAN[@]}" "$P" -c "import tirl" 2> /dev/null; then say "tirl: TIRL already installed: skipped"
    elif [ -n "$TIRL_SRC" ]; then
      for s in tirl tirlscripts-oxford-scriptutils tirlscripts-stanford-core tirlscripts-mgh-linc; do
        d=""; for c in "$TIRL_SRC/$s" "$TIRL_SRC/_src_$s" "$TIRL_SRC/$s"-[0-9]*; do [ -d "$c" ] && { d=$c; break; }; done   # exact names only
        [ -n "$d" ] || { say "tirl: no $s, _src_$s or $s-<version> under $TIRL_SRC"; fails=$((fails + 1)); continue; }
        c="$PREFIX/src/$s"                                          # build from a copy: --tirl-src stays read-only (a build writes
        [ -d "$c" ] || run cp -a "$d" "$c" || { fails=$((fails + 1)); continue; }   # build/, *.egg-info and generated C into its tree)
        mkdir -p "$PREFIX/src/tirl_home"                         # TIRL's setup.py copies tirl.yml and an empty tirlscripts.yml into
        say "  (HOME=$PREFIX/src/tirl_home for this step: TIRL's setup.py writes \$HOME/.config/tirl*.yml)"   # $HOME/.config
        "${CLEAN[@]}" HOME="$PREFIX/src/tirl_home" "$P" -m pip install --no-deps --no-build-isolation "$c" \
          || { say "  FAILED: pip install $c"; fails=$((fails + 1)); }
      done
      # no `tirl register`: the pipeline calls TIRL's scripts by their path (linc/histology/register.py), so nothing is written to
      # the account-wide ~/.config/tirlscripts.yml
    else say "tirl: set LINC_TIRL_SRC to a TIRL 3.6.3 source tree; TIRL is not publicly distributed yet (the tirl environment exists without TIRL: the reconstruction and histology steps refuse, make test skips them)"; fi
  else fails=$((fails + 1)); fi
fi
if [ $VIZ3D = 1 ] && want figures-3d; then
  if create figures-3d viz3d.yml; then                                # env/viz3d.yml's header: the headless VTK wheel, then pyvista
    P=$(py viz3d.yml)
    if ! "${CLEAN[@]}" "$P" -c "import vtk, pyvista" 2> /dev/null; then
      run "$P" -m pip install --no-cache-dir "vtk-osmesa==9.3.1" --extra-index-url https://wheels.vtk.org &&
        run "$P" -m pip install --no-deps "pyvista==0.44.2" || fails=$((fails + 1))
    else say "figures-3d: vtk and pyvista already installed: skipped"; fi
  else fails=$((fails + 1)); fi
fi
# where the pipeline looks for each environment (env/README.md)
E="$PREFIX/linc_envs.sh"
{
  echo "# written by env/install.sh $(date +%Y-%m-%d): the environments of $REPO"
  echo "export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1"            # progress lines reach a log file as they happen
  [ -x "$(py linc.yml)" ] && echo "export PY=$(py linc.yml)"
  [ -x "$(py linc.yml)" ] && echo "export PY_MRI=$(py linc.yml)"    # one environment since 3 Nov 2026 (env/linc.yml)
  b="$(dirname "$(py linc.yml)")"                                     # MMORF from the FSL channel: the executables themselves (the
  if command -v nvidia-smi > /dev/null && [ -x "$b/mmorf_cuda" ]; then echo "export MMORF=$b/mmorf_cuda LINC_MMORF_EXE=$b/mmorf_cuda"
  elif [ -x "$b/mmorf_cpu" ]; then echo "export MMORF=$b/mmorf_cpu LINC_MMORF_EXE=$b/mmorf_cpu"; fi   # `mmorf` wrapper needs fslpython)
  [ -x "$(py deep.yml)" ] && echo "export PY_DEEP=$(py deep.yml) LINC_SEGMENT_PYTHON=$(py deep.yml)"
  [ -x "$(py anatomix.yml)" ] && echo "export ANATOMIX_PYTHON=$(py anatomix.yml)"
  [ -f "$PREFIX/src/hf_home/hub/models--neeldey--anatomix/refs/main" ] && echo "export HF_HOME=$PREFIX/src/hf_home HF_HUB_OFFLINE=1"   # anatomix's weights: no network at run time
  [ -x "$(py tirl.yml)" ] && echo "export PY_TIRL=$(py tirl.yml)"
  [ -x "$(py viz3d.yml)" ] && echo "export PY_VIZ3D=$(py viz3d.yml)"
} > "$E"
say "wrote $E (source it before running the pipeline)"; sed 's/^/    /' "$E"

if [ -z "$ONLY" ]; then
  say "make test"
  ( set -a; . "$E"; set +a; cd "$REPO/reproduce" && "${CLEAN[@]}" PY="${PY:-python3}" PY_DEEP="${PY_DEEP:-${PY:-python3}}" \
      PY_TIRL="${PY_TIRL:-${PY:-python3}}" make -s test ) || fails=$((fails + 1))
fi
say "done: $fails failed step(s)"
exit $fails
