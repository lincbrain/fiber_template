"""The runner counts a machine's cores whatever OMP_NUM_THREADS says (run.py, PROBE; CPU, seconds, no data).

    python -m pytest tests/test_run_machines.py      or      python tests/run.py tests/test_run_machines.py

You need
  settings/env/linc.yml and bash.
What it does
  It runs the runner's probe of this machine three times: as the shell is, with OMP_NUM_THREADS=1 and with OMP_THREAD_LIMIT=1. `nproc`
  alone returns those variables in place of the cores, and the runner would then hold back every step that needs more than that.
You get
  pytest's pass / fail.
"""
import importlib.util
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_cores_do_not_follow_omp_num_threads():
    spec = importlib.util.spec_from_file_location("linc_run", str(REPO / "run.py"))
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    cores = len(os.sched_getaffinity(0))
    for extra in ({}, {"OMP_NUM_THREADS": "1"}, {"OMP_THREAD_LIMIT": "1"}):
        r = subprocess.run(["bash", "-c", run.PROBE], env=dict(os.environ, **extra), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        assert int(r.stdout.splitlines()[0]) == cores, (extra, r.stdout.splitlines()[:1], cores)
