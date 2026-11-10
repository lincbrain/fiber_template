"""The small framework the template build drivers share (code/template/build_*.py), so every build runs, resumes and records its
steps the same way.

You need
  a build's ordered list of steps: each one command of the recipe of record (code/template/*.md) with its environment,
  inputs, outputs and estimated time.
What it does
  --plan              prints every step (environment, estimated time, command, inputs, outputs) and runs nothing
  --step NAME [...]   runs these steps, in recipe order
  --from A --to B     runs the steps from A through B, in recipe order
  --force             runs a step whose outputs already exist (by default it refuses: products are never overwritten silently)
You get
  for each step that ran, <work>/provenance/<NN>_<name>.json (the command, the environment variables it needs, start, end and wall
  time, the repository commit, every input and output with its size and, for files, md5; code/shared/provenance.py's record).
"""
import argparse
import datetime
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))


class Step:
    def __init__(self, name, stage, env, argv, outputs, inputs=(), est="", note="", setenv=None, gpu=False, fn=None, plan_argv=None, driver_var="LINC_TEMPLATE_DRIVER"):
        """env: "linc" (this script's python, settings/env/linc.yml), "mri" ($PY_MRI, settings/env/linc.yml's python), "driver" (the patched template
        driver, settings/env/linc.yml's python, MMORF on a GPU) or "anatomix" ($ANATOMIX_PYTHON, settings/env/anatomix.yml: the feature-based registration engine). argv: the command after the python (a list), or a callable returning it (evaluated at run time,
        e.g. the driver's command line, which reads the run tree). fn: a Python function run in-process instead of a command. driver_var: the variable naming the driver's folder (the blockface build has its own)."""
        self.name, self.stage, self.env, self.argv, self.outputs, self.inputs = name, stage, env, argv, list(outputs), list(inputs)
        self.est, self.note, self.setenv, self.gpu, self.fn, self.plan_argv = est, note, dict(setenv or {}), gpu, fn, plan_argv
        self.driver_var = driver_var

    def command(self, plan=False):
        if self.fn is not None:
            return [f"(in-process) {self.fn.__name__}"]
        argv = self.argv
        if callable(argv):
            try:
                argv = argv()
            except Exception:                                                 # noqa: BLE001
                if not plan:
                    raise
                argv = self.plan_argv or ["<computed at run time from the run tree>"]
        py = (sys.executable if self.env == "linc" else os.environ.get("ANATOMIX_PYTHON", "$ANATOMIX_PYTHON") if self.env == "anatomix"
              else os.environ.get("PY_MRI", "$PY_MRI"))
        return [py, *[str(a) for a in argv]]


def _record(p):
    p = Path(p)
    if p.is_dir():
        n = sum(1 for q in p.rglob("*") if q.is_file())
        return dict(path=str(p), kind="dir", files=n)
    if p.exists():
        from shared.provenance import file_record
        return dict(kind="file", **file_record(p, md5=p.stat().st_size < (2 << 30)))
    return dict(path=str(p), kind="missing")


def _commit():
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:                                                         # noqa: BLE001
        return None


def _check_env(step):
    need = []
    if step.env in ("mri", "driver") and not os.environ.get("PY_MRI"):
        need.append("PY_MRI (the python of settings/env/linc.yml)")
    if step.env == "anatomix" and not os.environ.get("ANATOMIX_PYTHON"):
        need.append("ANATOMIX_PYTHON (the python of settings/env/anatomix.yml)")
    if step.env == "driver":
        for v, why in ((step.driver_var, "the patched template_construction_v1 folder"), ("LINC_MMORF_EXE", "MMORF 1.0.1's mmorf_cuda"),
                       ("FSLDIR", "FSL 6.0.7.4")):
            if not os.environ.get(v):
                need.append(f"{v} ({why})")
    if need:
        raise SystemExit(f"step {step.name} needs: " + "; ".join(need))


def _short(text, work):
    """<work>/... and $LINC_PRODUCTS/... instead of the long absolute prefixes, python for this interpreter (plan display only)."""
    text = text.replace(str(work), "<work>").replace(sys.executable, "python").replace(str(REPO) + "/", "")
    for var in ("LINC_PRODUCTS", "LINC_TEMPLATE", "LINC_DATA"):
        v = os.environ.get(var, "").rstrip("/")
        if v:
            text = text.replace(v + "/", f"${var}/")
    return text


def print_plan(title, steps, work):
    print(f"{title}\nwork folder (<work>): {work}\n")
    stage = None
    for i, s in enumerate(steps, 1):
        if s.stage != stage:
            stage = s.stage; print(f"== {stage}")
        print(f"{i:2d}. {s.name}  [{s.env}{', GPU' if s.gpu else ''}]  ~{s.est}")
        print("    $ " + _short(" ".join(shlex.quote(c) for c in s.command(plan=True)), work))
        if s.setenv:
            print("    env: " + _short(" ".join(f"{k}={v}" for k, v in s.setenv.items()), work))
        if s.inputs:
            print("    in:  " + _short("; ".join(str(p) for p in s.inputs), work))
        print("    out: " + _short("; ".join(str(p) for p in s.outputs), work))
        if s.note:
            print("    note: " + s.note)


def run_step(i, step, work, force=False):
    _check_env(step)
    done = [p for p in step.outputs if Path(p).exists()]
    if done and not force:
        raise SystemExit(f"step {step.name}: outputs exist ({', '.join(map(str, done))}); pass --force to run it again")
    missing = [p for p in step.inputs if not Path(p).exists()]
    if missing:
        raise SystemExit(f"step {step.name}: inputs missing: {', '.join(map(str, missing))}")
    cmd = step.command()
    env = dict(os.environ, **{k: v.replace("$PATH", os.environ.get("PATH", "")) for k, v in step.setenv.items()})
    t0, start = time.time(), datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"== {step.name}\n  $ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    if step.fn is not None:
        step.fn()
    else:
        subprocess.run(cmd, check=True, cwd=str(REPO), env=env)
    wall = round(time.time() - t0, 1)
    from shared.provenance import provenance_dict
    rec = provenance_dict(step.name, cmd=" ".join(shlex.quote(c) for c in cmd), wall_s=wall,
                          extra=dict(step=step.name, stage=step.stage, env=step.env, setenv=step.setenv, started=start,
                                     repo_commit=_commit(), inputs=[_record(p) for p in step.inputs],
                                     outputs=[_record(p) for p in step.outputs]), md5=False)
    pd = Path(work) / "provenance"; pd.mkdir(parents=True, exist_ok=True)
    (pd / f"{i:02d}_{step.name}.json").write_text(json.dumps(rec, indent=1, default=str))
    print(f"  done in {wall} s -> {pd / f'{i:02d}_{step.name}.json'}", flush=True)


def main(title, make_steps, default_work, argv=None, doc=None):
    """doc: the build's own docstring for --help (else the title and this framework's description)."""
    ap = argparse.ArgumentParser(description=doc or (title + "\n\n" + __doc__), formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=None, help=f"the build's folder (default {default_work})")
    ap.add_argument("--plan", action="store_true"); ap.add_argument("--step", nargs="+", default=[])
    ap.add_argument("--from", dest="frm"); ap.add_argument("--to"); ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    from shared import paths
    work = Path(a.work) if a.work else Path(os.environ.get("LINC_WORK", "work")) / default_work
    steps = make_steps(work.resolve(), paths)
    names = [s.name for s in steps]
    if a.plan or not (a.step or a.frm or a.to):
        print_plan(title, steps, work.resolve())
        if not a.plan:
            print("\n(nothing run: give --step NAME, or --from / --to)")
        return
    bad = [n for n in a.step + [x for x in (a.frm, a.to) if x] if n not in names]
    if bad:
        raise SystemExit(f"unknown step(s) {bad}; the steps are: {', '.join(names)}")
    if a.step:
        sel = [i for i, s in enumerate(steps) if s.name in a.step]
    else:
        lo = names.index(a.frm) if a.frm else 0; hi = names.index(a.to) if a.to else len(steps) - 1
        sel = list(range(lo, hi + 1))
    work.mkdir(parents=True, exist_ok=True)
    for i in sel:
        run_step(i + 1, steps[i], work, a.force)
