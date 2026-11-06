#!/usr/bin/env python3
"""Run a command until a file appears, then stop it with everything it started; for example, the MRI template's construction up
to the end of its first nonlinear step.

    $PY tools/run_until.py <file> -- <command> [<argument> ...]

You need the command and the file whose appearance means the part you want is done (for the construction's step 1: the run's
out/scripts/021_nlnT_mmorf.sh, the first task of step 2).
What it does: starts the command, looks for the file every 5 s, and when it is there stops the command and every process under it
(SIGTERM, then SIGKILL after 30 s).
You get the exit status 0 when the file appeared, else the command's own status (1 when it ended with 0 but without the file); the last
line says which.
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def descendants(pid):
    """Every process under pid, from /proc (the construction's driver starts its tasks in their own process groups)."""
    kids = {}
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                ppid = int((d / "stat").read_text().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
            kids.setdefault(ppid, []).append(int(d.name))
    out, todo = [], [pid]
    while todo:
        for c in kids.get(todo.pop(), []):
            out.append(c)
            todo.append(c)
    return out


def stop(pid):
    procs = [pid] + descendants(pid)
    for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 0)):
        for p in procs:
            try:
                os.kill(p, sig)
            except ProcessLookupError:
                pass
        t0 = time.time()
        while time.time() - t0 < wait and any(Path(f"/proc/{p}").exists() for p in procs):
            time.sleep(1)


def main():
    if len(sys.argv) < 4 or sys.argv[2] != "--":
        sys.exit(__doc__)
    target, cmd = Path(sys.argv[1]), sys.argv[3:]
    proc = subprocess.Popen(cmd)
    while proc.poll() is None:
        if target.exists():
            time.sleep(5)                                                # let the writer finish the file
            stop(proc.pid)
            proc.wait()
            print(f"run_until: {target} appeared; the command and its processes were stopped")
            sys.exit(0)
        time.sleep(5)
    if target.exists():
        print(f"run_until: the command ended ({proc.returncode}) and {target} exists")
        sys.exit(0)
    print(f"run_until: the command ended ({proc.returncode}) without {target}")
    sys.exit(proc.returncode or 1)


if __name__ == "__main__":
    main()
