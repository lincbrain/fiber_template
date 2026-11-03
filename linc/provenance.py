"""Provenance stamps for products and numbers (ported from the analysis project's timing.py).

Every product carries `provenance.json` (or `<name>.provenance.json` beside a file):
    {"product", "created", "host", "user", "cmd", "cwd", "script", "script_md5", "python",
     "inputs": [{"path", "size", "mtime", "md5"}], "outputs": [...], "wall_s", "extra"}
Every number of the paper is written by `write_numbers(name, data, inputs)` as $LINC_NUMBERS/<name>.json =
{"data": ..., "_provenance": {...}}, with one index line per write in $LINC_NUMBERS/provenance.json.
Standard library only.
"""
import contextlib
import datetime as _dt
import fcntl
import getpass
import hashlib
import json
import os
import socket
import sys
from pathlib import Path

MD5_MAX_BYTES = 8 * 1024 ** 3          # hash inputs up to 8 GB; larger ones record size + mtime only


def _now():
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _host():
    return socket.gethostname().split(".")[0]


@contextlib.contextmanager
def _locked(path, mode="a"):
    """Open `path` with an exclusive POSIX lock (lockf works over NFS via the lock manager)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode) as fh:
        try:
            fcntl.lockf(fh, fcntl.LOCK_EX)
        except OSError:                 # no lock manager: fall back to O_APPEND atomicity
            pass
        try:
            yield fh
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            try:
                fcntl.lockf(fh, fcntl.LOCK_UN)
            except OSError:
                pass


# --------------------------------------------------------------------------- provenance

def file_record(path, md5=True):
    """{path, size, mtime, md5} for a file; directories (e.g. OME-Zarr) get a file count and the
    newest mtime inside instead of a hash."""
    p = Path(path)
    rec = {"path": str(p.resolve()) if p.exists() else str(p)}
    if not p.exists():
        rec["missing"] = True
        return rec
    if p.is_dir():
        n, newest, size = 0, 0.0, 0
        for root, _, files in os.walk(p):
            for f in files:
                st = os.stat(os.path.join(root, f))
                n += 1; size += st.st_size; newest = max(newest, st.st_mtime)
        rec.update({"kind": "dir", "n_files": n, "size": size,
                    "mtime": _dt.datetime.fromtimestamp(newest).isoformat(timespec="seconds")})
        return rec
    st = p.stat()
    rec.update({"size": st.st_size,
                "mtime": _dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="microseconds")})
    if md5 and st.st_size <= MD5_MAX_BYTES:
        h = hashlib.md5()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        rec["md5"] = h.hexdigest()
    return rec


def _script_record():
    main = sys.modules.get("__main__")
    f = getattr(main, "__file__", None)
    if not f:
        return None, None
    r = file_record(f)
    return r["path"], r.get("md5")


def provenance_dict(product, inputs=(), outputs=(), *, cmd=None, wall_s=None, extra=None, md5=True):
    script, script_md5 = _script_record()
    return {"product": str(product), "created": _now(), "host": _host(),
            "user": getpass.getuser(), "cmd": cmd if cmd is not None else " ".join(sys.argv),
            "cwd": os.getcwd(), "script": script, "script_md5": script_md5,
            "python": sys.executable,
            "inputs": [file_record(p, md5) for p in inputs],
            "outputs": [file_record(p, md5) for p in outputs],
            "wall_s": wall_s, "extra": extra or {}}


def provenance_path(product):
    p = Path(product)
    return p / "provenance.json" if p.is_dir() else p.with_name(p.name.split(".")[0] + ".provenance.json")


def write_provenance(product, inputs=(), outputs=(), *, cmd=None, wall_s=None, extra=None, md5=True):
    """Write `provenance.json` into a product directory, or `<stem>.provenance.json` beside a file."""
    d = provenance_dict(product, inputs, outputs, cmd=cmd, wall_s=wall_s, extra=extra, md5=md5)
    out = provenance_path(product)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(d, indent=1, default=_json_default))
    os.replace(tmp, out)
    return out


def check_provenance(product_or_json, md5=False):
    """Return the list of recorded inputs that changed (size/mtime, or md5 if md5=True) or vanished.
    Consumers call this on an upstream product before using it (final-version rule, §0.4)."""
    p = Path(product_or_json)
    j = p if p.suffix == ".json" and p.name.endswith("provenance.json") else provenance_path(p)
    if not j.exists():
        return [{"path": str(j), "problem": "no provenance.json"}]
    d = json.loads(j.read_text())
    bad = []
    for rec in d.get("inputs", []):
        now = file_record(rec["path"], md5=md5 and "md5" in rec)
        if now.get("missing"):
            bad.append({"path": rec["path"], "problem": "missing"})
        elif now.get("size") != rec.get("size") or now.get("mtime") != rec.get("mtime"):
            if md5 and rec.get("md5") and now.get("md5") == rec.get("md5"):
                continue                 # touched but identical content
            bad.append({"path": rec["path"], "problem": "changed",
                        "was": {k: rec.get(k) for k in ("size", "mtime", "md5")},
                        "now": {k: now.get(k) for k in ("size", "mtime", "md5")}})
    return bad


def write_numbers(name, data, inputs=(), *, cmd=None, extra=None, md5=True, root=None):
    """Write paper/numbers/<name>.json = {"data": data, "_provenance": {...}} and update the index
    $LINC_NUMBERS/provenance.json (root= overrides $LINC_NUMBERS)."""
    from . import paths
    NUMBERS = Path(root) if root else paths.numbers_root()
    NUMBERS.mkdir(parents=True, exist_ok=True)
    out = NUMBERS / f"{name}.json"
    prov = provenance_dict(out, inputs, (), cmd=cmd, extra=extra, md5=md5)
    tmp = out.with_suffix(f".json.tmp{os.getpid()}")
    tmp.write_text(json.dumps({"data": data, "_provenance": prov}, indent=1, default=_json_default))
    os.replace(tmp, out)
    idx = NUMBERS / "provenance.json"
    with _locked(idx, "a+") as fh:
        fh.seek(0)
        txt = fh.read()
        index = json.loads(txt) if txt.strip() else {}
        index[name] = {k: prov[k] for k in ("created", "host", "script", "script_md5", "cmd")}
        index[name]["inputs"] = [{k: r.get(k) for k in ("path", "mtime", "md5")} for r in prov["inputs"]]
        fh.seek(0); fh.truncate()
        fh.write(json.dumps(index, indent=1, sort_keys=True))
    return out


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


