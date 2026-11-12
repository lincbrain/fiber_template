#!/usr/bin/env python
"""Run the whole method step by step, on one machine or several, and check each step against the paper's files.

You need
  the environments and the data roots, set in the shell you start it from (README.md), and the plan,
  steps.yaml: for each step its command, the check that compares its result with the paper's, what it reads
  and writes, and the machine it needs. docs/tutorials/ describe the same steps in words.
What it does
  --plan              prints the plan and runs nothing
  --run               tests the installation on each machine, then runs every step whose inputs are ready on a machine with room
                      for it, and the step's check after it. It returns when nothing is left to run. Each step is a process of its
                      own: it goes on if this command stops, and the next --run takes it up. A step that passed is not run again; a
                      step that failed is.
  --status            prints one line per step
  --wait [MINUTES]    returns when a step fails or differs, when the run is over, or after MINUTES (default 30); prints what changed
  --stop              stops the running steps and the --run that started them
  --only NAME ...     with --plan or --run: these steps, groups or brains, and the steps they need. A step marked optional runs only
                      when it is named here.
  --hosts "A B"       the machines to use. Each must see this folder at the same path and accept ssh without a password
                      (default: $LINC_HOSTS, or this machine alone). A step starts on a machine only when its card, memory, cores
                      and the disk have room beside what already runs there.
  --force             with --run --only: run the named steps again although they passed
  --redo NAME ...     with --run: run these steps again although they passed or differ (after a fix to their command or check),
                      and everything else as usual
You get
  outputs/build_everything/status.json    every step with its command, state, machine, times and the last line of its check
  outputs/build_everything/steps/<step>/  the script that ran, run.log and check.log
  A step ends as passed (it ran and its check holds), differs (it ran and its check does not hold) or failed (its command failed).

  $PY run.py --plan
  nohup $PY run.py --run > outputs/build_everything.log 2>&1 &
  $PY run.py --status
"""
import argparse
import concurrent.futures
import getpass
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent
PLAN = REPO / "steps.yaml"
HERE = socket.gethostname().split(".")[0]
OUT = REPO / "outputs" / "build_everything"

# a step's fields in the plan and their defaults (None: required)
FIELDS = dict(name=None, group=None, title=None, run=None, reads="", writes="", check="", expect="", needs=(), gpu_gb=0,
              gpu_model="", gpu_alone=False, ram_gb=4, cores=1, disk_gb=0, hours=0, timed="", optional=False, note="")
TEXT = ("title", "reads", "writes", "run", "check", "expect", "note")
PUBLIC = ("name", "base", "brain", "label", "group") + tuple(k for k in FIELDS if k not in ("name", "group"))
RAN = ("passed", "differs")             # a step that needs another one starts once that one has run to its end
OVER = RAN + ("failed",)
RECENT = 20 * 60                        # s: a step this young may not yet hold the memory it will use
DISK_SPARE = 20                         # GB left free on the disk whatever runs
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]


def die(msg):
    sys.exit(f"build_everything: {msg}")


def say(msg=""):
    print(msg, flush=True)


def rel(p):
    try:
        return str(Path(p).relative_to(REPO))
    except ValueError:
        return str(p)


def clock(t=None):
    return time.strftime("%d %b %H:%M", time.localtime(t))


def span(seconds):
    m = int(round(seconds / 60))
    return f"{int(seconds)} s" if seconds < 120 else f"{m} min" if m < 90 else f"{m // 60} h {m % 60:02d} min"


# ---------------------------------------------------------------------------------------------------------------- the plan
def load_plan(path=PLAN):
    """The plan as (steps, entries): entries are the YAML items, steps the same with each `for` written out (one step per brain),
    in an order in which every step comes after the steps it needs."""
    doc = yaml.safe_load(Path(path).read_text())
    if set(doc) - {"sets", "steps"}:
        die(f"{path.name}: unknown top-level keys {sorted(set(doc) - {'sets', 'steps'})}")
    sets = doc.get("sets") or {}
    labels = json.loads((REPO / "settings" / "specimen_labels.json").read_text())["labels"]
    steps, bases, entries = [], {}, []
    for e in doc["steps"]:
        where = f"{path.name}, step {e.get('name', '?')!r}"
        if set(e) - set(FIELDS) - {"for"}:
            die(f"{where}: unknown fields {sorted(set(e) - set(FIELDS) - {'for'})}")
        if [k for k, d in FIELDS.items() if d is None and not e.get(k)]:
            die(f"{where}: missing {[k for k, d in FIELDS.items() if d is None and not e.get(k)]}")
        ids = e.get("for")
        if isinstance(ids, str):
            if ids not in sets:
                die(f"{where}: `for: {ids}` is not one of the sets {sorted(sets)}")
            ids = sets[ids]
        entries.append(dict(e, ids=list(ids or [])))
        for b in ids or [None]:
            s = {k: e.get(k, d) for k, d in FIELDS.items()}
            s.update(base=e["name"], brain=b or "", label=labels.get(b, b) if b else "", needs=list(e.get("needs") or []))
            if b:
                s["name"] = f"{e['name']}-{b}"
                for k in TEXT:
                    s[k] = str(s[k]).replace("{id}", b)
            for k in TEXT:                # commands keep their spacing (a pattern may need two spaces); prose is tidied
                s[k] = str(s[k]).strip() if k in ("run", "check") else " ".join(str(s[k]).split())
            bases.setdefault(e["name"], []).append(s["name"])
            steps.append(s)
    by = {}
    for s in steps:
        if s["name"] in by:
            die(f"{path.name}: two steps are named {s['name']}")
        by[s["name"]] = s
    for s in steps:                       # a need is a step's name, or a per-brain step's base name: then the same brain's, or all
        needs = []
        for n in s["needs"]:
            n = n.replace("{id}", s["brain"])
            if n in by:
                needs.append(n)
            elif n in bases:
                same = f"{n}-{s['brain']}"
                needs.extend([same] if s["brain"] and same in by else bases[n])
            else:
                die(f"{path.name}: {s['name']} needs {n!r}, which is not a step")
        s["needs"] = needs
    order, state = [], {}

    def visit(n, chain):
        if state.get(n) == 1:
            die(f"{path.name}: the steps need each other in a circle: {' -> '.join(chain + [n])}")
        if n not in state:
            state[n] = 1
            for d in by[n]["needs"]:
                visit(d, chain + [n])
            state[n] = 2
            order.append(by[n])
    for s in steps:
        visit(s["name"], [])
    return order, entries


def select(steps, only):
    """The names --only asks for (default: every step not marked optional), with the steps they need."""
    by = {s["name"]: s for s in steps}
    if not only:
        named = [s["name"] for s in steps if not s["optional"]]
    else:
        named = []
        for n in only:
            hit = ([n] if n in by else [s["name"] for s in steps if s["base"] == n]
                   or [s["name"] for s in steps if n in (s["group"], s["brain"], s["label"]) and not s["optional"]])
            if not hit:
                die(f"--only {n}: no step, group or brain of that name (see --plan)")
            named += hit
    seen, todo = set(named), list(named)
    while todo:
        for d in by[todo.pop()]["needs"]:
            if d not in seen:
                seen.add(d)
                todo.append(d)
    return [s["name"] for s in steps if s["name"] in seen], set(named)


def unset_variables(steps):
    """{variable: a step that names it} for the shell variables the steps' commands read that this shell does not set."""
    missing = {}
    for s in steps:
        for line in (s["run"], s["check"]):
            bare = re.sub(r"'[^']*'", "", line)
            local = set(re.findall(r"(?:^|[\s;&|(])([A-Za-z_][A-Za-z0-9_]*)=", bare)) | set(re.findall(r"\bfor\s+([A-Za-z_]\w*)\s+in\b", bare))
            for v, default in re.findall(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)(:?[-=+])?", bare):
                if not default and v not in local and v not in os.environ and v not in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "D"):
                    missing.setdefault(v, s["name"])
    return missing


def machine(s):
    parts = ([f"GPU {s['gpu_gb']:g} GB" + (f" ({s['gpu_model']})" if s["gpu_model"] else "") + (", the card to itself" if s["gpu_alone"] else "")]
             if s["gpu_gb"] else ["no GPU"])
    parts += [f"RAM {s['ram_gb']:g} GB", f"{s['cores']} core" + ("s" if s["cores"] != 1 else "")]
    if s["disk_gb"]:
        parts.append(f"disk {s['disk_gb']:g} GB")
    return ", ".join(parts)


def cmd_plan(steps, entries, names):
    chosen = set(names)
    group = None
    for e in entries:
        mine = [s for s in steps if s["base"] == e["name"] and s["name"] in chosen and (not e["ids"] or s["brain"] in e["ids"])]
        if not mine:
            continue
        if e["group"] != group:
            group = e["group"]
            say(f"\n== {group} ==")
        s = {k: " ".join(str(e.get(k, d)).split()) if k in TEXT else e.get(k, d) for k, d in FIELDS.items()}
        many = f"   [{len(mine)} brain{'s' if len(mine) > 1 else ''}: {' '.join(m['brain'] for m in mine)}]" if e["ids"] else ""
        say(f"\n{e['name']}{'-{id}' if e['ids'] else ''}{'   (optional)' if s['optional'] else ''}{many}\n    {s['title']}")
        for key, text in (("reads", s["reads"]), ("writes", s["writes"]), ("run", s["run"]), ("check", s["check"]),
                          ("proves", s["expect"]), ("needs", ", ".join(e.get("needs") or [])),
                          ("machine", machine(s) + (f"; about {s['hours']:g} h" + (" each" if e["ids"] else "") + (f" ({s['timed']})" if s["timed"] else "") if s["hours"] else "")),
                          ("note", s["note"])):
            if text:
                say(f"    {key + ':':9}{text}")
    picked = [s for s in steps if s["name"] in chosen]
    say(f"\n{len(picked)} steps, about {sum(s['hours'] for s in picked):.0f} hours of machine time if run one after another.")
    missing = unset_variables(picked)
    if missing:
        say("Not set in this shell (source the environment first): " + ", ".join(f"{v} ({n})" for v, n in sorted(missing.items())))


# ------------------------------------------------------------------------------------------------------- state, from files
def _read(p):
    try:
        return Path(p).read_text().strip()
    except OSError:
        return None


def _tail(p, n=1, width=300):
    try:
        with open(p, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 8192))
            lines = [x.strip() for x in f.read().decode("utf-8", "replace").splitlines() if x.strip()]
    except OSError:
        return ""
    return " | ".join(lines[-n:])[-width:]


def snapshot(steps):
    """{step: state, machine, start, end, evidence}, read from the files each step's script writes, so it is right whether or not
    a --run is alive. States: waiting (for the steps it needs), ready, running, checking, passed, differs, failed, blocked."""
    rows = {}
    for s in steps:
        d = OUT / "steps" / s["name"]
        r = dict(state="waiting", host="", start=0, end=0, evidence="", gpu=-1)
        launched = _read(d / "launched")
        if launched:
            host, t0, gpu = (launched.split() + ["-1"])[:3]
            r.update(host=host, start=int(t0), gpu=int(gpu))
            rc, ck, lost = _read(d / "run.exit"), _read(d / "check.exit"), _read(d / "lost")
            if rc is None or (rc == "0" and s["check"] and ck is None):
                if lost:
                    r.update(state="failed", evidence=lost, end=int((d / "lost").stat().st_mtime))
                else:
                    r["state"] = "running" if rc is None else "checking"
                    if s["hours"] and time.time() - r["start"] > max(3 * s["hours"] * 3600, 7200):
                        r["evidence"] = f"running for {span(time.time() - r['start'])}, over three times its usual time: look at its run.log"
            elif rc != "0":
                r.update(state="failed", evidence=f"exit {rc}: " + _tail(d / "run.log", 3))
            elif not s["check"]:
                r.update(state="passed", evidence="ran to its end (no file of record to compare with)")
            elif ck == "0":
                r.update(state="passed", evidence=_tail(d / "check.log"))
            else:
                r.update(state="differs", evidence=_tail(d / "check.log", 2))
            if r["state"] in OVER and not r["end"]:
                done = d / ("check.exit" if (d / "check.exit").exists() else "run.exit")
                r["end"] = int(_read(d / "end") or done.stat().st_mtime)
        rows[s["name"]] = r
    for s in steps:                       # in the plan's order, every step comes after the steps it needs
        r = rows[s["name"]]
        if r["state"] == "waiting":
            needs = {n: rows[n]["state"] for n in s["needs"]}
            stuck = [n for n, x in needs.items() if x in ("failed", "blocked")]
            if stuck:
                r.update(state="blocked", evidence="needs " + ", ".join(stuck))
            elif all(x in RAN for x in needs.values()):
                r["state"] = "ready"
    return rows


def commit():
    r = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def run_id():
    f = OUT / "run_id"
    if not f.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        f.write_text(f"{getpass.getuser()}-{time.strftime('%Y%m%d-%H%M')}\n")
    return _read(f)


def scheduler():
    """(machine, pid) of the live --run, or None."""
    text = _read(OUT / "scheduler")
    if not text:
        return None
    host, pid = text.split()
    try:
        alive = sh(host, f"kill -0 {pid}", timeout=20).returncode == 0
    except subprocess.TimeoutExpired:
        alive = True                      # the machine does not answer: do not start a second one beside it
    return (host, pid) if alive else None


def write_status(steps, rows, hosts=None, notes=None):
    f = OUT / "status.json"
    old = json.loads(_read(f) or "{}")
    live = scheduler()
    doc = dict(run=run_id(), user=getpass.getuser(), repo=str(REPO), commit=commit(), updated=int(time.time()),
               scheduler=live[0] if live else "", hosts=old.get("hosts", {}) if hosts is None else hosts, steps=[])
    if notes is None:
        notes = json.loads(_read(OUT / "notes.json") or "{}")
    for s in steps:
        r = dict(rows[s["name"]])
        if r["state"] == "ready" and s["name"] in notes:
            r["evidence"] = notes[s["name"]]
        doc["steps"].append({**{k: s[k] for k in PUBLIC}, **r})
    tmp = f.with_suffix(f".{os.getpid()}.part")
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, f)
    return doc


def counts(rows, names=None):
    c = {}
    for n, r in rows.items():
        if names is None or n in names:
            c[r["state"]] = c.get(r["state"], 0) + 1
    order = ("passed", "differs", "failed", "running", "checking", "ready", "waiting", "blocked")
    return ", ".join(f"{k} {c[k]}" for k in order if k in c) + f" ({sum(c.values())} steps)"


# ------------------------------------------------------------------------------------------------------------ the machines
def sh(host, command, timeout=40):
    argv = ["bash", "-c", command] if host == HERE else SSH + [host, command]
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)


PROBE = ("nproc; cut -d' ' -f1 /proc/loadavg; awk '/MemTotal/{t=$2}/MemAvailable/{a=$2}END{print t, a}' /proc/meminfo; "
         "nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null; echo ---; "
         "nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null")


def probe(host):
    """What a machine has and what is free on it now; None when it does not answer."""
    try:
        lines = sh(host, PROBE).stdout.strip().splitlines()
        cut = lines.index("---")
        total, free = (int(x) / 1048576 for x in lines[2].split())
        gpus = []
        for line in lines[3:cut]:
            i, uuid, name, mem, used = (x.strip() for x in line.split(","))
            gpus.append(dict(index=int(i), uuid=uuid, name=name, total=int(mem) / 1024, used=int(used) / 1024, jobs=0))
        for line in lines[cut + 1:]:
            for g in gpus:
                g["jobs"] += g["uuid"] == line.strip()
        return dict(cores=int(lines[0]), load=float(lines[1]), ram=total, ram_free=free, gpus=gpus)
    except (subprocess.TimeoutExpired, ValueError, IndexError):
        return None


def room(s, m, mine, now):
    """The card to run step s on (-1 for a step without GPU) when machine m has room for it beside what runs there, else None.
    mine: the steps this run already has on m. What they will use is counted even before they use it."""
    young = [x for x in mine if now - x["start"] < RECENT]
    if min(m["ram_free"] - sum(x["ram_gb"] for x in young), 0.9 * m["ram"] - sum(x["ram_gb"] for x in mine)) < s["ram_gb"]:
        return None
    if m["load"] + sum(x["cores"] for x in young) + s["cores"] > 0.9 * m["cores"]:
        return None
    if not s["gpu_gb"]:
        return -1
    best = None
    for g in m["gpus"]:
        on = [x for x in mine if x["gpu"] == g["index"]]
        if s["gpu_model"] and s["gpu_model"].lower() not in g["name"].lower():
            continue
        if any(x["gpu_alone"] for x in on) or (s["gpu_alone"] and (on or g["jobs"])):
            continue
        free = min(g["total"] - g["used"] - sum(x["gpu_gb"] for x in on if now - x["start"] < RECENT),
                   g["total"] - sum(x["gpu_gb"] for x in on))
        if free >= s["gpu_gb"] + 1 and (best is None or free > best[0]):
            best = (free, g["index"])
    return best[1] if best else None


def fits(s, m):
    """Whether machine m could ever hold step s."""
    return (s["ram_gb"] <= 0.9 * m["ram"] and s["cores"] <= 0.9 * m["cores"] and
            (not s["gpu_gb"] or any(g["total"] >= s["gpu_gb"] + 1 and (not s["gpu_model"] or s["gpu_model"].lower() in g["name"].lower())
                                    for g in m["gpus"])))


def test_machine(host):
    """The installation test on one machine, once per commit: {ok, summary}."""
    f = OUT / "machines" / f"{host}.json"
    old = json.loads(_read(f) or "{}")
    if old.get("ok") and old.get("commit") == commit():
        return old
    try:
        r = sh(host, f"cd {shlex.quote(str(REPO))} && . {shlex.quote(str(OUT / 'env.sh'))} && nice -n 10 bash tests/selftest.sh", timeout=3600)
        lines = [x for x in (r.stdout + r.stderr).strip().splitlines() if x.strip()]
        new = dict(ok=r.returncode == 0, summary=(lines[-1] if lines else "no answer")[:200], commit=commit(), t=int(time.time()))
        if not new["ok"]:
            new["failed"] = [x[:200] for x in lines if x.startswith("FAIL")]
    except subprocess.TimeoutExpired:
        new = dict(ok=False, summary="the installation test did not end within an hour", commit=commit(), t=int(time.time()))
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(new, indent=1))
    return new


_SKIP = re.compile(r"(_|PWD|OLDPWD|SHLVL|HOSTNAME|HOST|DISPLAY|TMPDIR|TMUX.*|STY|WINDOW.*|TERM.*|COLORTERM|LS_COLORS|MAIL|LESSOPEN|"
                   r"SSH_.*|XDG_.*|DBUS_.*|SESSION_MANAGER|KRB5.*|GPG_.*|VSCODE_.*|CUDA_VISIBLE_DEVICES|"
                   r".*(TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|API_?KEY|COOKIE).*)", re.I)


def write_env():
    """The shell environment --run was started from, as a file every step sources: so a step runs in the same environment on
    every machine. Variables of this login session, and anything that looks like a secret, are left out."""
    lines = [f"export {k}={shlex.quote(v)}" for k, v in sorted(os.environ.items())
             if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k) and not _SKIP.fullmatch(k)]
    f = OUT / "env.sh"
    f.touch(mode=0o600)
    f.chmod(0o600)
    f.write_text("# written by run.py --run: the environment every step runs in\n" + "\n".join(lines) + "\n")


def launch(s, host, gpu):
    d = OUT / "steps" / s["name"]
    d.mkdir(parents=True, exist_ok=True)
    if (d / "launched").exists():         # an earlier attempt: its files move to previous/
        shutil.rmtree(d / "previous", ignore_errors=True)
        (d / "previous").mkdir()
        for f in d.iterdir():
            if f.name != "previous":
                f.rename(d / "previous" / f.name)
    q = shlex.quote
    lines = ["#!/bin/bash", f"# {s['name']}: {s['title']}", "# written by run.py; ./launched names the machine it runs on",
             f"D={q(str(d))}", f"cd {q(str(REPO))} || exit 1", f". {q(str(OUT / 'env.sh'))}"]
    if gpu >= 0:
        lines.append(f"export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={gpu}")
    lines += ["echo $$ > $D/pid", f"( set -o pipefail; {s['run']} ) > $D/run.log 2>&1 < /dev/null",
              "rc=$?; echo $rc > $D/run.exit.part; mv $D/run.exit.part $D/run.exit"]
    if s["check"]:
        lines += ["if [ $rc -eq 0 ]; then", f"  ( set -o pipefail; {s['check']} ) > $D/check.log 2>&1 < /dev/null",
                  "  echo $? > $D/check.exit.part; mv $D/check.exit.part $D/check.exit", "fi"]
    lines.append("date +%s > $D/end")
    (d / "step.sh").write_text("\n".join(lines) + "\n")
    (d / "launched").write_text(f"{host} {int(time.time())} {gpu}\n")
    try:
        r = sh(host, f"nohup setsid nice -n 10 bash {q(str(d / 'step.sh'))} > /dev/null 2>&1 < /dev/null &")
        problem = r.stderr.strip()[-200:] if r.returncode else ""
    except subprocess.TimeoutExpired:
        problem = "no answer"
    if problem:
        (d / "lost").write_text(f"could not start it on {host}: {problem}\n")


def gone(name, host):
    """Whether a step's earlier process is gone for certain (so that it can be started again)."""
    pid = _read(OUT / "steps" / name / "pid")
    if not pid or not host:
        return True
    try:
        return sh(host, f"kill -0 {pid} 2>/dev/null && echo alive; echo answered").stdout.split() == ["answered"]
    except subprocess.TimeoutExpired:
        return False


def find_lost(steps, rows, dead):
    """Mark as failed the running steps whose process is gone without a result (killed, or the machine restarted). A step is
    marked at the second look, so that a result written a moment ago over the network is not missed."""
    by_host = {}
    for s in steps:
        r = rows[s["name"]]
        if r["state"] in ("running", "checking"):
            pid = _read(OUT / "steps" / s["name"] / "pid")
            if pid:
                by_host.setdefault(r["host"], {})[pid] = s["name"]
            elif time.time() - r["start"] > 300:
                (OUT / "steps" / s["name"] / "lost").write_text(f"it did not start on {r['host']}\n")
    for host, pids in by_host.items():
        try:
            alive = set(sh(host, "for p in " + " ".join(pids) + "; do kill -0 $p 2>/dev/null && echo $p; done; echo answered").stdout.split())
        except subprocess.TimeoutExpired:
            continue
        if "answered" not in alive:       # the machine did not answer: say nothing about its steps
            continue
        for pid, name in pids.items():
            if pid in alive:
                dead.pop(name, None)
            elif name in dead:
                d = OUT / "steps" / name
                if not (d / "end").exists():
                    (d / "lost").write_text(f"its process on {host} ended without a result (killed, or the machine restarted)\n")
            else:
                dead[name] = time.time()


# ---------------------------------------------------------------------------------------------------------------- commands
def cmd_run(a, steps):
    names, named = select(steps, a.only)
    redo = set()
    if a.redo:
        more, redo = select(steps, a.redo)
        names = [s["name"] for s in steps if s["name"] in set(names) | set(more)]
    by = {s["name"]: s for s in steps}
    missing = unset_variables([by[n] for n in names])
    if missing:
        die("these variables are not set in this shell (source the environment first): " + ", ".join(f"{v} ({n})" for v, n in sorted(missing.items())))
    OUT.mkdir(parents=True, exist_ok=True)
    live = scheduler()
    if live:
        die(f"a --run is already going on {live[0]} (process {live[1]}); --status shows it, --stop ends it")
    (OUT / "scheduler").write_text(f"{HERE} {os.getpid()}\n")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        _run(a, steps, names, (set(named) if a.force else set()) | redo, by)
    finally:
        (OUT / "scheduler").unlink(missing_ok=True)
        write_status(steps, snapshot(steps))


def _run(a, steps, names, forced, by):
    write_env()
    hosts = list(dict.fromkeys(HERE if h == "localhost" else h.split(".")[0] for h in (a.hosts or os.environ.get("LINC_HOSTS") or HERE).split()))
    say(f"{clock()} run {run_id()}, commit {commit() or 'unknown'}: {len(names)} steps on {len(hosts)} machine(s)")
    say("testing the installation on each machine (a few minutes, once per machine)")
    with concurrent.futures.ThreadPoolExecutor(len(hosts)) as pool:
        tests = dict(zip(hosts, pool.map(test_machine, hosts)))
    for h in hosts:
        say(f"  {h}: {'ok' if tests[h]['ok'] else 'NOT USED'}  {tests[h]['summary']}")
    good = [h for h in hosts if tests[h]["ok"]]
    if not good:
        die("no machine passed the installation test (bash tests/selftest.sh shows why)")
    tried, known, notes, dead, last_probe, last_look, last_say = set(), {}, {}, {}, 0, time.time(), 0
    (OUT / "notes.json").write_text("{}")
    while True:
        rows, now = snapshot(steps), time.time()
        mine = {h: [] for h in good}
        for s in steps:
            r = rows[s["name"]]
            if r["state"] in ("running", "checking"):
                mine.setdefault(r["host"], []).append({**s, **r})
        ready = [by[n] for n in names if n not in tried and (rows[n]["state"] == "ready" or (
            (rows[n]["state"] == "failed" or (n in forced and rows[n]["state"] in RAN)) and all(rows[d]["state"] in RAN for d in by[n]["needs"])))]
        ready = [s for s in ready if not set(s["needs"]) & (forced - tried)]     # a step waits for the redo of a step it needs
        never = [s for s in ready if len(known) == len(good) and not any(fits(s, m) for m in known.values())]
        for s in never:
            notes[s["name"]] = "no machine in the list can hold it: " + machine(s)
        ready = [s for s in ready if s not in never]
        active = [n for n in names if rows[n]["state"] in ("running", "checking")]
        if not ready and not active:
            break
        if ready and now - last_probe > 30:
            last_probe = now
            with concurrent.futures.ThreadPoolExecutor(len(good)) as pool:
                info = {h: m for h, m in zip(good, pool.map(probe, good)) if m}
            known.update(info)
            disk = shutil.disk_usage(REPO).free / 2 ** 30 - sum(x["disk_gb"] for xs in mine.values() for x in xs)
            for s in ready:
                if rows[s["name"]]["state"] != "ready" and not gone(s["name"], rows[s["name"]]["host"]):
                    notes[s["name"]] = f"its earlier process is still alive on {rows[s['name']]['host']}, or that machine does not answer"
                    continue
                if s["disk_gb"] + DISK_SPARE > disk:
                    notes[s["name"]] = f"waiting for disk: it writes {s['disk_gb']:g} GB, and {max(disk, 0):.0f} GB are free beside the running steps"
                    continue
                for h in sorted(info, key=lambda h: (max([g["total"] for g in info[h]["gpus"]] or [0]), len(mine[h]), good.index(h))):
                    gpu = room(s, info[h], mine[h], now)
                    if gpu is not None:
                        launch(s, h, gpu)
                        tried.add(s["name"])
                        notes.pop(s["name"], None)
                        mine[h].append({**s, "start": now, "gpu": gpu, "host": h})
                        disk -= s["disk_gb"]
                        say(f"{clock()} started {s['name']} on {h}")
                        break
                else:
                    notes[s["name"]] = "waiting for room on a machine: " + machine(s)
            rows = snapshot(steps)
        if now - last_look > 300:
            last_look = now
            find_lost(steps, rows, dead)
        shown = {h: dict(gpu="; ".join(f"{g['name'].replace('NVIDIA ', '')} {g['total']:.0f} GB, {g['used']:.0f} used" for g in m["gpus"]),
                         ram_free=round(m["ram_free"]), ram=round(m["ram"]), cores=m["cores"], load=m["load"],
                         steps=[x["name"] for x in mine.get(h, [])]) for h, m in known.items()}
        (OUT / "notes.json").write_text(json.dumps(notes))
        write_status(steps, rows, hosts=shown, notes=notes)
        if now - last_say > 1800:
            last_say = now
            say(f"{clock()} {counts(rows, set(names))}")
        time.sleep(15)
    rows = snapshot(steps)
    say(f"{clock()} nothing left to run: {counts(rows, set(names))}")
    for state in ("failed", "differs", "blocked"):
        for n in names:
            if rows[n]["state"] == state:
                say(f"  {state:8}{n}: {rows[n]['evidence']}")
    for n, text in notes.items():
        if rows[n]["state"] == "ready":
            say(f"  not run {n}: {text}")


def cmd_status(steps):
    rows = snapshot(steps)
    doc = (json.loads(_read(OUT / "status.json") or "{}") if scheduler() else {}) or write_status(steps, rows)
    notes = {s["name"]: s["evidence"] for s in doc.get("steps", []) if s["state"] == "ready" and s["evidence"]}
    say(f"run {doc['run']}, commit {doc['commit'] or 'unknown'}; " + (f"--run is alive on {doc['scheduler']}" if doc["scheduler"] else "no --run is alive"))
    say(counts(rows))
    for s in steps:
        r = rows[s["name"]]
        if r["state"] == "ready" and s["name"] in notes:
            say(f"  {'ready':9}{s['name']:34}{'':10}{'':>12}  {notes[s['name']]}")
        elif r["state"] not in ("waiting", "ready"):
            took = span((r["end"] or time.time()) - r["start"]) if r["start"] else ""
            say(f"  {r['state']:9}{s['name']:34}{r['host']:10}{took:>12}  {r['evidence'][:150]}")


def cmd_wait(a, steps):
    f = OUT / "synced.json"
    synced = json.loads(_read(f) or "{}")
    t0, dead, last_look = time.time(), {}, time.time()
    while True:
        rows, live = snapshot(steps), scheduler()
        if not live and time.time() - last_look > 300:
            last_look = time.time()
            find_lost(steps, rows, dead)
        now = {n: [r["state"], r["host"], r["start"], r["end"], r["evidence"][:160]] for n, r in rows.items()
               if r["state"] not in ("waiting", "ready", "blocked")}
        new = {n: v for n, v in now.items() if synced.get(n) != v}
        active = [n for n, v in now.items() if v[0] in ("running", "checking")]
        if [n for n, v in new.items() if v[0] in ("failed", "differs")] or (not live and not active) or time.time() - t0 >= a.wait * 60:
            break
        time.sleep(20)
    doc = (json.loads(_read(OUT / "status.json") or "{}") if live else {}) or write_status(steps, rows)
    say(f"{clock()} {counts(rows)}; " + (f"--run is alive on {live[0]}" if live else "no --run is alive"))
    if new:
        say("changed since the last look:")
        for n, (state, host, start, end, evidence) in new.items():
            took = f", {span(end - start)}" if end else ""
            log = f"  -> {rel(OUT)}/steps/{n}/{'run' if state == 'failed' else 'check'}.log" if state in ("failed", "differs") else ""
            say(f"  {state.upper() if state in ('failed', 'differs') else state:9}{n} ({host}{took}): {evidence}{log}")
        k = len(list((OUT / "events").glob("*.json"))) + 1 if (OUT / "events").exists() else 1
        (OUT / "events").mkdir(exist_ok=True)
        event = dict(n=k, t=int(time.time()), run=doc["run"], user=doc["user"], commit=doc["commit"], scheduler=doc["scheduler"],
                     hosts={h: [m["gpu"], m["ram_free"], m["load"], len(m["steps"])] for h, m in doc["hosts"].items()}, steps=new)
        (OUT / "events" / f"{k:04d}.json").write_text(json.dumps(event, separators=(",", ":")))
        synced.update(new)
        f.write_text(json.dumps(synced))
        say(f"event {rel(OUT)}/events/{k:04d}.json  (document events/{doc['run']}-{k:04d})")
    else:
        say("nothing changed since the last look")
    if not live and not active:
        left = [n for n, r in rows.items() if r["state"] in ("ready", "blocked", "failed")]
        say("OVER: every step that could run has run" + (f"; {len(left)} did not pass or did not run (--status lists them; --run tries the failed ones again)" if left else ""))


def cmd_stop(steps):
    live = scheduler()
    if live:
        host, pid = live
        sh(host, f"pkill -TERM -P {pid}; kill -TERM {pid}; for i in 1 2 3 4 5 6 7 8 9 10; do kill -0 {pid} 2>/dev/null || break; sleep 1; done; kill -KILL {pid} 2>/dev/null; true")
        (OUT / "scheduler").unlink(missing_ok=True)
        say(f"stopped the --run on {host}")
    rows = snapshot(steps)
    for s in steps:
        r = rows[s["name"]]
        if r["state"] in ("running", "checking"):
            d = OUT / "steps" / s["name"]
            pid = _read(d / "pid")
            if pid:
                sh(r["host"], f"pkill -TERM -s {pid}; kill -TERM -- -{pid} 2>/dev/null; true")
            (d / "lost").write_text("stopped by --stop\n")
            say(f"stopped {s['name']} on {r['host']}")
    write_status(steps, snapshot(steps))


def main(argv=None):
    global OUT
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    do = ap.add_mutually_exclusive_group(required=True)
    for flag in ("--plan", "--run", "--status", "--stop"):
        do.add_argument(flag, action="store_true")
    do.add_argument("--wait", nargs="?", type=float, const=30, metavar="MINUTES")
    ap.add_argument("--only", nargs="+", default=[], metavar="NAME")
    ap.add_argument("--hosts", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--redo", nargs="+", default=[], metavar="NAME")
    ap.add_argument("--plan-file", default=str(PLAN), help=argparse.SUPPRESS)
    ap.add_argument("--out", default=str(OUT), help="the run's folder (default outputs/build_everything)")
    a = ap.parse_args(argv)
    OUT = Path(a.out).resolve()
    steps, entries = load_plan(Path(a.plan_file))
    if a.plan:
        cmd_plan(steps, entries, select(steps, a.only)[0])
    elif a.run:
        cmd_run(a, steps)
    elif a.status:
        cmd_status(steps)
    elif a.stop:
        cmd_stop(steps)
    else:
        cmd_wait(a, steps)


if __name__ == "__main__":
    main()
