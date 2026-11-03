#!/usr/bin/env python3
"""Compare regenerated number records with the records of the paper.

Every script in reproduce/numbers writes $LINC_NUMBERS/<name>.json = {"data": ..., "_provenance": ...}. This checker compares the
"data" block with a reference record of the same name: counts and strings must be identical, floats equal to a relative and
absolute tolerance of 1e-9. Provenance blocks are ignored.

    python reproduce/check_numbers.py --ref <dir with reference records> [--new $LINC_NUMBERS] [name ...]

Exit code 0 when every compared record matches; the first differences of each mismatching record are printed.

Documented exceptions (configs/numbers_exceptions.yaml): a record that cannot regenerate identically because its inputs changed
after it was written, for reasons the paper reports elsewhere, is compared only on the fields the paper reads:
    <name>:
      reason: <one line>
      compare: [<path>, ...]     # dotted paths into "data"; "*" expands every element of a list or every value of a dict;
                                 # a trailing "|len" compares the length of what the path selects
      tol: <float>               # optional: absolute and relative tolerance for these fields (default 1e-9), for values that are
                                 # not bitwise reproducible (GPU nondeterminism); the reason must say how the paper prints them
"""
import argparse
import json
import math
import sys
from pathlib import Path


def diff(a, b, path="data", out=None, tol=1e-9):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: only in {'reference' if k in a else 'new'}")
            else:
                diff(a[k], b[k], f"{path}.{k}", out, tol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out, tol)
    elif isinstance(a, bool) or isinstance(b, bool) or not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        if a != b:
            out.append(f"{path}: {a!r} vs {b!r}")
    elif isinstance(a, int) and isinstance(b, int):
        if a != b:
            out.append(f"{path}: {a} vs {b}")
    else:
        if not ((math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=tol, abs_tol=tol)):
            out.append(f"{path}: {a!r} vs {b!r}")
    return out


def _select(obj, path):
    """Values at a dotted path ("*" = every element / value; "|len" suffix = the length of the selection)."""
    want_len = path.endswith("|len")
    parts = [p for p in path[:-4 if want_len else None].split(".") if p]
    cur = [obj]
    for p in parts:
        nxt = []
        for o in cur:
            if p == "*":
                nxt += list(o.values()) if isinstance(o, dict) else list(o) if isinstance(o, list) else []
            elif isinstance(o, dict) and p in o:
                nxt.append(o[p])
            elif isinstance(o, list) and p.isdigit() and int(p) < len(o):
                nxt.append(o[int(p)])
        cur = nxt
    if want_len:
        return len(cur[0]) if len(cur) == 1 and isinstance(cur[0], (list, dict)) else len(cur)
    return cur


def _exceptions():
    cfg = Path(__file__).resolve().parents[1] / "configs" / "numbers_exceptions.yaml"
    if not cfg.exists():
        return {}
    import yaml
    return yaml.safe_load(cfg.read_text()) or {}


def _placeholders():
    """(prefix, placeholder) pairs, longest prefix first. The table maps analysis-machine prefixes; this machine's roots map to the
    same placeholders, and a root that lies inside $LINC_PRODUCTS maps to "$LINC_PRODUCTS/<relative path>", so a file reached through
    two nested roots scrubs to the same string on both sides."""
    import os
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from linc import paths
    pairs = list(paths.placeholder_table())        # configs/path_placeholders.local.yaml when present (see linc.paths)
    prod = os.environ.get("LINC_PRODUCTS", "").rstrip("/")
    for var in ("LINC_PRODUCTS", "LINC_DATA", "LINC_TEMPLATE", "LINC_RAW", "LINC_WORK", "LINC_MODELS"):
        root = os.environ.get(var, "").rstrip("/")
        if not root:
            continue
        if prod and var in ("LINC_DATA", "LINC_TEMPLATE") and (root + "/").startswith(prod + "/"):
            pairs.append((root, "$LINC_PRODUCTS" + root[len(prod):]))
        else:
            pairs.append((root, "$" + var))
    return sorted(set(pairs), key=lambda ab: -len(ab[0]))


def _scrub(o, pairs):
    if isinstance(o, str):
        for a, b in pairs:
            o = o.replace(a, b)
        return o
    if isinstance(o, list):
        return [_scrub(x, pairs) for x in o]
    if isinstance(o, dict):
        return {_scrub(k, pairs): _scrub(v, pairs) for k, v in o.items()}
    return o


def _data(rec):
    """The "data" block; a record written without that wrapper (older figure records) counts as data minus its stamp fields."""
    if "data" in rec:
        return rec["data"]
    return {k: v for k, v in rec.items() if k not in ("created", "script", "_provenance", "host", "cmd")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--new", type=Path, default=None)
    ap.add_argument("names", nargs="*")
    a = ap.parse_args()
    if a.new is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from linc import paths
        a.new = paths.numbers_root()
    names = a.names or sorted(p.stem for p in a.new.glob("*.json") if p.name != "provenance.json")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from linc import paths
    if not paths.PLACEHOLDERS_LOCAL.exists():
        print(f"NOTE     no configs/{paths.PLACEHOLDERS_LOCAL.name}: paths recorded on the machine that made the products are not "
              "mapped to placeholders, so records that hold them differ (copy configs/path_placeholders.yaml to that name and fill "
              "in the prefixes)")
    bad = 0
    exc = _exceptions()
    for n in names:
        r, w = a.ref / f"{n}.json", a.new / f"{n}.json"
        if not w.exists():
            print(f"MISSING  {n} (no new record)"); bad += 1; continue
        if not r.exists():
            print(f"NOREF    {n} (no reference record)"); bad += 1; continue
        ph = _placeholders()
        ref_d, new_d = (_scrub(_data(json.loads(x.read_text())), ph) for x in (r, w))
        if n in exc:
            d = []
            tol = float(exc[n].get("tol", 1e-9))
            for pth in exc[n]["compare"]:
                d += diff(_select(ref_d, pth), _select(new_d, pth), f"data.{pth}", tol=tol)
            print(("OK*      " if not d else "DIFFERS* ") + f"{n}  (exception: {len(exc[n]['compare'])} paper-read fields; "
                  f"{exc[n]['reason']})")
        else:
            d = diff(ref_d, new_d)
            print(("OK       " if not d else "DIFFERS  ") + n)
        for line in d[:10]:
            print("    " + line)
        bad += bool(d)
    print(f"{len(names) - bad} of {len(names)} records match")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
