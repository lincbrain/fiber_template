#!/bin/bash
# Create the environments of this repository from env/*.yml, run the source installs their headers describe, install the package
# into each, and check the result with `make test`. Needs conda (or mamba) and git on PATH; nothing else.
#
#   bash env/install.sh [--prefix DIR] [--tirl-src DIR] [--with-nnunet] [--with-figures-3d] [--only NAME[,NAME...]] [--no-test]
#
#   --prefix DIR      where the environments go (default: envs/ in this clone): DIR/<the env file's name>, i.e. linc, linc-deep,
#                     linc-anatomix, linc-tirl (linc-nnunet, linc-viz3d when asked); DIR/src holds the source checkouts
#   --tirl-src DIR    (or LINC_TIRL_SRC=DIR; on the lab cluster env/lab.sh sets it) a local folder with the four TIRL source trees
#                     (TIRL is not publicly distributed yet):
#                     tirl 3.6.3, tirlscripts-oxford-scriptutils 1.1.1, tirlscripts-stanford-core 1.2.2, tirlscripts-mgh-linc 1.0.1
#                     (not on PyPI, see env/tirl.yml; on this project's cluster LINC_template/env/tirl_src holds the same trees);
#                     each found as <package>, _src_<package> or <package>-<version>. Without it the tirl environment is created
#                     but TIRL itself is not installed (histology and reconstruction steps then refuse). Read only: each tree is
#                     copied to DIR/src/<package> and built there. TIRL's setup.py writes $HOME/.config/tirl*.yml, which would
#                     reset your account's TIRL script registry: install.sh points HOME into the prefix (DIR/src/tirl_home) for
#                     that step; the pipeline needs neither file at run time
#   --with-nnunet     also the earlier blockface networks' environment (env/nnunet.yml; only to reproduce their masks of record)
#   --with-figures-3d also the 3D tracer figure's environment (env/viz3d.yml: pyvista on a headless VTK)
#   --only NAMES      just these (linc, deep, anatomix, tirl, nnunet, figures-3d)
#
# In order (expected duration on a cluster node with a fast network; most of it is downloading):
#   linc      env/linc.yml        ~4 min   orchestration, the MRI construction (MMORF), registration's linear stages,
#                                          the labels' interpolation, the numbers and most figures   -> PY (= PY_MRI), MMORF
#   deep      env/deep.yml        ~8 min   segmentation models and caudal completion (torch 2.6 + CUDA 12.4)       -> PY_DEEP
#   anatomix  env/anatomix.yml    ~8 min   the deformable registration engine (torch 2.13 + CUDA 12.6; anatomix and
#                                          its FireANTs fork from their repositories at the pinned commits)    -> ANATOMIX_PYTHON
#   tirl      env/tirl.yml        ~6 min   histology registration, reconstruction's Python steps (+ TIRL sources)   -> PY_TIRL
#   nnunet    env/nnunet.yml     ~15 min   optional (--with-nnunet)                                                 -> NNUNET_BIN
#   figures-3d env/viz3d.yml     ~5 min   optional (--with-figures-3d): the 3D tracer scene                        -> PY_VIZ3D
# then `make test` (~3-10 min). Every step prints what it runs. Measured on this project's cluster, 3 Nov 2026 (empty package
# caches): linc 2 min 30 s to 3 min 30 s, deep 4 min 25 s to 7 min 40 s, anatomix 8 min 15 s, tirl 1 min 30 s to 5 min 34 s; the others are
# estimates until measured.
# In all (linc, deep, anatomix, tirl, then make test): about 25 min and about 17 GB in the prefix (measured 3 Nov 2026: linc 1.3 GB,
# deep 5.5 GB, anatomix 7.2 GB, tirl 2.8 GB, sources 0.3 GB; figures-3d adds 1.1 GB), plus the downloads, kept in the prefix:
# conda's package cache in DIR/src/conda_pkgs (unless you set CONDA_PKGS_DIRS, e.g. to reuse a cache of your own) and pip's in
# DIR/src/pip_cache; both are removable once the build is done. A shared conda's own package folder is never written. Network
# access is needed (conda channels, PyPI, the PyTorch and VTK wheel indexes, and GitHub for anatomix and FireANTs); nothing else
# outside the clone is read or written, except conda's own list of environments (~/.conda/environments.txt).
# `pip check` in linc-anatomix lists three requirements the FireANTs fork declares but the engine never imports (hydra-core,
# pytest, typing), as in the environment of record: expected, not a failure.
#
# Safe to rerun: an environment that exists and was built by this script from the same env file is skipped; one that exists but
# was built otherwise is reported and left alone (nothing is ever deleted: remove it yourself to rebuild). Every step runs in a
# clean shell with PYTHONNOUSERSITE=1 and PIP_USER=0, so packages in ~/.local never enter or shadow an environment.
# At the end it writes DIR/linc_envs.sh: `source` it to point the pipeline at these environments.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
PREFIX="$REPO/envs"; TIRL_SRC="${LINC_TIRL_SRC:-}"; NNUNET=0; VIZ3D=0; ONLY=""; TEST=1   # LINC_TIRL_SRC: the TIRL sources (env/lab.sh)
while [ $# -gt 0 ]; do
  case $1 in
    --prefix) PREFIX=$2; shift 2;; --tirl-src) TIRL_SRC=$2; shift 2;; --with-nnunet) NNUNET=1; shift;; --with-figures-3d) VIZ3D=1; shift;;
    --only) ONLY=$2; shift 2;; --no-test) TEST=0; shift;;
    -h|--help) awk 'NR > 1 && /^#/ {print; next} NR > 1 {exit}' "$0"; exit 0;;
    *) echo "unknown argument: $1 (see --help)"; exit 2;;
  esac
done
CONDA=$(command -v conda) || { CONDA=$(command -v mamba) && "$CONDA" --version > /dev/null 2>&1; } || { echo "conda (or a working mamba) not found on PATH"; exit 2; }   # conda first: some mamba builds are broken
command -v git > /dev/null || { echo "git not found on PATH"; exit 2; }
mkdir -p "$PREFIX/src"; PREFIX=$(cd "$PREFIX" && pwd)
PKGS="${CONDA_PKGS_DIRS:-$PREFIX/src/conda_pkgs}"          # conda's package cache: in the prefix unless you set CONDA_PKGS_DIRS
CLEAN=(env -i HOME="$HOME" PATH="$(dirname "$CONDA"):/usr/bin:/bin" PYTHONNOUSERSITE=1 PIP_USER=0 LANG=C.UTF-8
       PIP_CACHE_DIR="$PREFIX/src/pip_cache" CONDA_PKGS_DIRS="$PKGS")   # downloads stay in the prefix (not ~/.cache/pip or a shared conda)
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
if [ $NNUNET = 1 ] && want nnunet; then
  if create nnunet nnunet.yml; then
    P=$(py nnunet.yml); SP=$("${CLEAN[@]}" "$P" -c "import site; print(site.getsitepackages()[0])")
    if [ ! -f "$PREFIX/linc-nnunet/.linc_nnunet_patched" ]; then
      run "$P" -m pip install --no-deps seaborn==0.13.2 && run patch -p1 -d "$SP" -i "$REPO/env/nnunetv2-2.5.1_record.patch" &&
        touch "$PREFIX/linc-nnunet/.linc_nnunet_patched" || fails=$((fails + 1))
    else say "nnunet: seaborn and the record patch already applied: skipped"; fi
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
  echo "export PYTHONNOUSERSITE=1"
  [ -x "$(py linc.yml)" ] && echo "export PY=$(py linc.yml)"
  [ -x "$(py linc.yml)" ] && echo "export PY_MRI=$(py linc.yml)"    # one environment since 3 Nov 2026 (env/linc.yml)
  b="$(dirname "$(py linc.yml)")"                                     # MMORF from the FSL channel: the executables themselves (the
  if command -v nvidia-smi > /dev/null && [ -x "$b/mmorf_cuda" ]; then echo "export MMORF=$b/mmorf_cuda LINC_MMORF_EXE=$b/mmorf_cuda"
  elif [ -x "$b/mmorf_cpu" ]; then echo "export MMORF=$b/mmorf_cpu LINC_MMORF_EXE=$b/mmorf_cpu"; fi   # `mmorf` wrapper needs fslpython)
  [ -x "$(py deep.yml)" ] && echo "export PY_DEEP=$(py deep.yml) LINC_SEGMENT_PYTHON=$(py deep.yml)"
  [ -x "$(py anatomix.yml)" ] && echo "export ANATOMIX_PYTHON=$(py anatomix.yml)"
  [ -x "$(py tirl.yml)" ] && echo "export PY_TIRL=$(py tirl.yml)"
  [ -x "$(py nnunet.yml)" ] && echo "export NNUNET_BIN=$(dirname "$(py nnunet.yml)")"
  [ -x "$(py viz3d.yml)" ] && echo "export PY_VIZ3D=$(py viz3d.yml)"
} > "$E"
say "wrote $E (source it before running the pipeline)"; sed 's/^/    /' "$E"

if [ $TEST = 1 ]; then
  say "make test"
  ( set -a; . "$E"; set +a; cd "$REPO/reproduce" && "${CLEAN[@]}" PY="${PY:-python3}" PY_DEEP="${PY_DEEP:-${PY:-python3}}" \
      PY_TIRL="${PY_TIRL:-${PY:-python3}}" make -s test ) || fails=$((fails + 1))
fi
say "done: $fails failed step(s)"
exit $fails
