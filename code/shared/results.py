"""A brain's results in the data folder's layout, made of links: outputs/brains/sub-<ID>/ (standard library only).

    sub-<ID>/inputs         a link to the data folder's brains/sub-<ID>/inputs (none when the data folder has no such folder)
    sub-<ID>/intermediate/  a link to each place where the run wrote for this brain:
                              local_data      its working folder, $LINC_DATA/sub-<ID>
                              local_rebuild   its folder in $LINC_REBUILD_DATA, when there is one
                              local_mri       its folder in $LINC_MRI_DATA, when there is one
                              work_<kind>     each $LINC_WORK/<kind>/<ID>
    sub-<ID>/outputs/       the finished files the run made itself, named as the data folder's brains/sub-<ID>/outputs names them:
                            settings/results.tsv gives, for each name, where the run writes it. A file is the run's own only when it
                            physically lives in the working root it was found under ($LINC_DATA, $LINC_REBUILD_DATA, $LINC_MRI_DATA or
                            $LINC_WORK): its real path is inside that root's real path. A link to the paper's file, or a file inside
                            a linked folder, is not, wherever it points (also when the data folder itself is made of links).

update() writes only links, only under outputs/brains/; it never writes into the data folder. It is idempotent (a link already right
is left alone), safe when two run at once (each link is made beside its place and renamed over it), and it removes a link of its own
whose source is gone. It never touches a file or folder there that is not a link it made. code/shared/linc-results runs it by hand;
settings/roots.sh, run.py after each step and every tool at its end (code/shared/paths.py) run it quietly.
"""
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TABLE = REPO / "settings" / "results.tsv"
WORKING = (("local_data", "LINC_DATA"), ("local_rebuild", "LINC_REBUILD_DATA"), ("local_mri", "LINC_MRI_DATA"))
VAR = re.compile(r"\$\{?(LINC_[A-Z_]+)\}?")


def table(path=TABLE):
    """the rows of settings/results.tsv: name, run_file (a list of candidates), made_by"""
    rows, head = [], None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            cells = line.split("\t")
            if head is None:
                head = cells
                continue
            r = dict(zip(head, cells))
            r["run_file"] = [c.strip() for c in r["run_file"].split("|") if c.strip()]
            rows.append(r)
    return rows


def roots(outputs, env=None):
    """the folders the table's names stand for: the three working folders under `outputs` (as settings/roots.sh makes them, whatever
    LINC_DATA is set to for one command), and $LINC_WORK (default outputs/work)"""
    env = os.environ if env is None else env
    r = {var: str(Path(outputs) / name / "data") for name, var in WORKING}
    r["LINC_WORK"] = env.get("LINC_WORK") or str(Path(outputs) / "work")
    return r


def _inside(p, folder):
    return p == folder or p.startswith(folder.rstrip(os.sep) + os.sep)


def _own(p, root_real):
    """a file the run made: it exists, and its real path is inside the real path of the working root it was found under"""
    return os.path.isfile(p) and _inside(os.path.realpath(p), root_real)


def wanted(sid, outputs, tree, rows, env=None):
    """{relative path under sub-<ID>/: the link's target} for one brain"""
    R, want = roots(outputs, env), {}
    real = {k: os.path.realpath(v) for k, v in R.items()}
    inp = os.path.join(tree, "brains", f"sub-{sid}", "inputs")
    if os.path.isdir(inp):
        want["inputs"] = inp
    for name, var in WORKING:
        p = os.path.join(R[var], f"sub-{sid}")
        if os.path.isdir(p):
            want[f"intermediate/{name}"] = p
    W = R["LINC_WORK"]
    try:
        kinds = sorted(e.name for e in os.scandir(W) if e.is_dir())
    except OSError:
        kinds = []
    for k in kinds:
        p = os.path.join(W, k, sid)
        if os.path.isdir(p):
            want[f"intermediate/work_{k}"] = p
    for row in rows:                                                    # no test of ownership for the intermediate places: they are
        for cand in row["run_file"]:                                    # the run's folders themselves
            m = VAR.match(cand)
            if not m or m.group(1) not in R:
                continue
            p = VAR.sub(lambda m: R.get(m.group(1), "\0"), cand).replace("<ID>", sid)
            if "\0" not in p and _own(p, real[m.group(1)]):
                want["outputs/" + row["name"].replace("<ID>", sid)] = p
                break
    return want


def _link(path, target):
    """make `path` a link to `target`, by a rename over it; never replaces what is not a link"""
    if os.path.islink(path):
        if os.readlink(path) == target:
            return False
    elif os.path.lexists(path):
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = os.path.join(os.path.dirname(path), f".{os.path.basename(path)}.{os.getpid()}.tmp")
    try:
        os.remove(tmp)
    except FileNotFoundError:
        pass
    os.symlink(target, tmp)
    os.replace(tmp, path)
    return True


def brains(outputs, view):
    """the brains of the three working folders, and those the view already holds"""
    ids = set()
    for name, _ in WORKING:
        try:
            ids |= {e.name[4:] for e in os.scandir(Path(outputs) / name / "data") if e.name.startswith("sub-") and e.is_dir()}
        except OSError:
            pass
    try:
        ids |= {e.name[4:] for e in os.scandir(view) if e.name.startswith("sub-") and e.is_dir(follow_symlinks=False)}
    except OSError:
        pass
    return sorted(ids)


def update(subjects=None, outputs=None, tree=None, table_path=TABLE, env=None, view=None):
    """bring outputs/brains/sub-<ID>/ (or `view`/sub-<ID>/) up to date for `subjects` (default: every brain); [(ID, finished files,
    intermediate places, inputs linked)]. Needs the data folder (`tree`, default $LINC_DATA_TREE)."""
    env = os.environ if env is None else env
    outputs = str(outputs or REPO / "outputs")
    tree = str(tree or env.get("LINC_DATA_TREE") or "")
    if not tree or not os.path.isdir(tree):
        raise SystemExit("linc-results: LINC_DATA_TREE is not set to the data folder: source settings/roots.sh first")
    view = str(view or os.path.join(outputs, "brains"))
    os.makedirs(view, exist_ok=True)                                    # from now on every tool keeps it current (at_exit)
    rows = table(table_path)
    present = brains(outputs, view)
    out = []
    for sid in (subjects or present):
        want = wanted(sid, outputs, tree, rows, env) if sid in present or subjects else {}
        d = os.path.join(view, f"sub-{sid}")
        for rel, target in sorted(want.items()):
            _link(os.path.join(d, rel), target)
        if want:                                                        # the three parts are there from the start, outputs/ empty
            for sub in ("intermediate", "outputs"):                     # until the run has made a finished file
                os.makedirs(os.path.join(d, sub), exist_ok=True)
        for sub in ("", "intermediate", "outputs"):                     # links of its own that are no longer wanted: their source is gone
            try:
                entries = list(os.scandir(os.path.join(d, sub)))
            except OSError:
                continue
            for e in entries:
                rel = os.path.join(sub, e.name) if sub else e.name
                if e.is_symlink() and not e.name.startswith(".") and rel not in want:
                    try:
                        os.remove(e.path)
                    except FileNotFoundError:
                        pass
        if not want:                                                     # a brain gone from the working folders: its empty folders go
            for sub in ("intermediate", "outputs", ""):
                try:
                    os.rmdir(os.path.join(d, sub))
                except OSError:
                    pass
            continue
        n_out = sum(k.startswith("outputs/") for k in want)
        n_int = sum(k.startswith("intermediate/") for k in want)
        out.append((sid, n_out, n_int, "inputs" in want))
    return out


SPACES = (("b0-native", "b0"), ("bf-native", "blockface"), ("t2-native", "t2"), ("micro-native", "micro"))


def _own_entries(xfm_dir, root_real):
    """the registry entries (folders <src>__to__<dst>) of a working folder's xfm/, when every one is the run's own (its meta.json
    physically in that working folder: its real path inside the folder's real path); else None"""
    try:
        names = [e.name for e in os.scandir(xfm_dir) if e.is_dir() and "__to__" in e.name]
    except OSError:
        return None
    if not names or not all(_inside(os.path.realpath(os.path.join(xfm_dir, n, "meta.json")), root_real) for n in names):
        return None
    return set(names)


def transforms(subjects=None, outputs=None, tree=None, env=None, python=None):
    """--transforms: the complete transforms a brain's outputs/ holds in the data folder, <ID>_{b0,blockface,t2,micro}_to_template.nii.gz,
    composed from the run's own registry entries (a working folder's sub-<ID>/xfm/) by `python -m shared.xfm compose --ref
    <the template's b0 grid> --format itk`, as the data folder's were. Written to $LINC_WORK/transforms/<ID>/ (about 200 MB each), and
    again only when an entry is newer. The blockface one only for a brain with its own blockface-to-b0 registration and no registration
    to the blockface template (the data folder's route). [(ID, file, 'made' | 'up to date' | the error's last line)]"""
    import subprocess
    import sys
    env = dict(os.environ if env is None else env)
    outputs = str(outputs or REPO / "outputs")
    tree = str(tree or env.get("LINC_DATA_TREE") or "")
    if not tree or not os.path.isdir(tree):
        raise SystemExit("linc-results: LINC_DATA_TREE is not set to the data folder: source settings/roots.sh first")
    R, done = roots(outputs, env), []
    ref = os.path.join(env.get("LINC_TEMPLATE") or os.path.join(outputs, "local_data", "template"), "mri", "linc_mri_b0.nii.gz")
    for sid in (subjects or brains(outputs, os.path.join(outputs, "brains"))):
        made = set()
        for _, var in WORKING:
            data = R[var]
            x = os.path.join(data, f"sub-{sid}", "xfm")
            ents = _own_entries(x, os.path.realpath(data))
            if not ents:
                continue
            starts = {n.split("__to__")[0] for n in ents}
            for space, short in SPACES:
                if space not in starts or short in made:
                    continue
                if space == "bf-native" and ("bf-native__to__b0-native" not in ents or "bf-native__to__bf-template" in ents):
                    continue
                made.add(short)
                d = os.path.join(R["LINC_WORK"], "transforms", sid)
                out = os.path.join(d, f"{sid}_{short}_to_template.nii.gz")
                newest = max(os.path.getmtime(os.path.join(x, n, "meta.json")) for n in ents if os.path.exists(os.path.join(x, n, "meta.json")))
                if os.path.exists(out) and os.path.getmtime(out) >= newest:
                    done.append((sid, out, "up to date"))
                    continue
                os.makedirs(d, exist_ok=True)
                tmp = os.path.join(d, f"{sid}_{short}_to_template_tmp{os.getpid()}.nii.gz")     # its provenance: <stem>.provenance.json
                r = subprocess.run([python or sys.executable, "-m", "shared.xfm", "compose", "--from", space, "--to", "mri-template",
                                    "--subject", sid, "--ref", ref, "--format", "itk", tmp], cwd=str(REPO / "code"),
                                   env=dict(env, LINC_DATA=data), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
                if r.returncode == 0 and os.path.exists(tmp):
                    os.replace(tmp, out)
                    prov = tmp[:-len(".nii.gz")] + ".provenance.json"
                    if os.path.exists(prov):
                        os.replace(prov, out[:-len(".nii.gz")] + ".provenance.json")
                    done.append((sid, out, "made"))
                else:
                    for f in (tmp, tmp[:-len(".nii.gz")] + ".provenance.json"):
                        if os.path.exists(f):
                            os.remove(f)
                    done.append((sid, out, (r.stdout.strip().splitlines() or ["failed"])[-1]))
    return done


def at_exit():
    """run at the end of every tool (code/shared/paths.py): quiet, links only, and never an error or a changed exit status. Only in a
    shell that sourced settings/roots.sh for this clone (LINC_DATA_TREE set, outputs/in made, LINC_DATA a working folder of this clone),
    only when outputs/brains/ exists (settings/roots.sh or linc-results makes it: the first time takes a few seconds, an update a few
    hundredths), and not in a worker process; LINC_RESULTS_VIEW=0 turns it off."""
    try:
        env = os.environ
        if env.get("LINC_RESULTS_VIEW", "1") == "0" or not env.get("LINC_DATA_TREE") or not env.get("LINC_DATA"):
            return
        import sys
        mp = sys.modules.get("multiprocessing")
        if mp is not None and getattr(mp, "parent_process", lambda: None)() is not None:
            return
        outputs = str(REPO / "outputs")
        if not (os.path.isdir(os.path.join(outputs, "brains")) and os.path.isdir(os.path.join(outputs, "in"))
                and _inside(os.path.realpath(env["LINC_DATA"]), os.path.realpath(outputs))):
            return
        update()
    except BaseException:
        pass
