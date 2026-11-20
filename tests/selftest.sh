#!/bin/bash
# Check that the installation works, before downloading any data.
#
# You need
#   the environments' interpreters (settings/env/README.md). It takes them from envs/linc_envs.sh, which install.sh writes, unless PY
#   is already set (a shell that sourced linc_envs.sh or settings/roots.sh, or your own choice):
#     PY       settings/env/linc.yml        (default: python3 on PATH)
#     PY_DEEP  settings/env/deep.yml        (default: PY)
#     PY_TIRL  settings/env/tirl.yml        for the reconstruction and histology steps; without it or without TIRL those two steps SKIP and name the
#                                  install command, because PY would test the wrong environment
#   No release data, no GPU, no network; the weights checked are the repository's own (models/). A few minutes.
#
# What it does
#   1. runs the repository's tests, each in the environment its code runs in (tests/run.py, no pytest needed)
#   2. runs each command line on toy data written on the fly, including one training epoch of each network. The toy results mean nothing:
#      this checks the installation, not the models
#
# You get
#   PASS / FAIL / SKIP per step and a summary; a step whose environment lacks a package fails with the import error, which names what to install.
#   The exit status is the number of failed steps (skips do not count). The logs go to a folder under $TMPDIR, which it names.
#
#   bash tests/selftest.sh                       # or: make -C code/figures test
set -u
cd "$(dirname "$0")/.."
if [ -z "${PY:-}" ]; then                       # the environments install.sh built, unless the shell already names them
  if [ -f envs/linc_envs.sh ]; then . envs/linc_envs.sh
  else echo "linc self-test: PY is not set and there is no envs/linc_envs.sh: run bash install.sh first, or source <prefix>/linc_envs.sh. Trying python3 on PATH"; fi
fi
PY=${PY:-python3}; PY_DEEP=${PY_DEEP:-$PY}; PY_TIRL=${PY_TIRL:-}
export PYTHONNOUSERSITE=1                       # packages in ~/.local must not shadow the environment's own (settings/env/README.md)
LOG=${TMPDIR:-/tmp}/linc_selftest_$$; mkdir -p "$LOG"
fails=0; n=0; skips=0
step() {                                        # step <name> <command ...>; exit status 77 = SKIP (an optional environment absent)
  local name=$1; shift; n=$((n + 1)); local t0=$SECONDS rc=0
  "$@" > "$LOG/$name.log" 2>&1 || rc=$?
  if [ $rc -eq 0 ]; then echo "PASS  $name ($((SECONDS - t0)) s)"
  elif [ $rc -eq 77 ]; then skips=$((skips + 1)); echo "SKIP  $name: $(head -n 1 "$LOG/$name.log")"
  else fails=$((fails + 1)); echo "FAIL  $name ($((SECONDS - t0)) s): see $LOG/$name.log"; tail -n 5 "$LOG/$name.log" | sed 's/^/        /'; fi
}

echo "linc self-test: $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout'); logs in $LOG"
echo "  PY=$PY  PY_DEEP=$PY_DEEP  PY_TIRL=${PY_TIRL:-(not set)}"
tirl_py() {                                     # the settings/env/tirl.yml steps: run in PY_TIRL with its bin (tirl, zarr2nii) on PATH
  if [ -z "$PY_TIRL" ] || ! "$PY_TIRL" -c "import tirl, zarr" 2> /dev/null; then   # SKIP (77), naming the fix: never a silent pass
    echo "PY_TIRL (${PY_TIRL:-not set}) has no TIRL: set LINC_TIRL_SRC to the TIRL sources and run bash install.sh --only tirl, then source <prefix>/linc_envs.sh"; return 77; fi
  env PATH="$(dirname "$PY_TIRL"):$PATH" "$PY_TIRL" "$@"
}

# --- the tests (tests/), each in its environment
step test_paths            "$PY"      tests/run.py tests/test_paths.py tests/test_subjects_overview.py tests/test_labels_structures.py tests/test_labels_scale.py tests/test_refine_entry.py tests/test_run_machines.py tests/test_xfm_inverse.py tests/test_scale_warning.py tests/test_register_inputs.py
step weights               bash -c "$PY tests/verify_weights.py settings/models/segmentation.sha256 --root models/segment && $PY tests/verify_weights.py settings/models/completion_public.sha256 --root models/completion"
step test_segment          "$PY_DEEP" tests/run.py tests/test_segment.py
step test_completion       "$PY_DEEP" tests/run.py tests/test_completion.py
step test_blockface_rules  "$PY_DEEP" tests/run.py tests/test_blockface_rules.py
step test_data_folder      "$PY"      tests/run.py tests/test_link_data.py tests/test_data_folder.py
step test_build_force      "$PY"      tests/run.py tests/test_build_force.py tests/test_compare_nothing.py
step test_construction     "$PY"      tests/run.py tests/test_construction_inputs.py
step test_docs_site        "$PY"      tests/run.py tests/test_docs_site.py
step test_docs_names       "$PY"      tests/run.py tests/test_docs_names.py
step test_results_view     "$PY"      tests/run.py tests/test_results_view.py

# --- the command lines on toy data
step linc-info             "$PY"      code/shared/linc-info
step linc-segment          "$PY_DEEP" tests/smoke/segment_toy.py
step linc-complete         "$PY_DEEP" tests/smoke/complete_toy.py
# (blockface reconstruction, registration and histology smokes: added by their owners below)
step linc-reconstruct      tirl_py tests/smoke/reconstruct_toy.py
step linc-register         "$PY"      tests/smoke/register_toy.py
step linc-histology        tirl_py tests/smoke/histology_toy.py

echo "$((n - fails - skips)) of $n steps passed$( [ $skips -gt 0 ] && echo ", $skips skipped (see SKIP above)")"
exit $fails
