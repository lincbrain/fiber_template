"""A relocatable copy of a root of prepared brain folders, for a data release (the folders of record are not changed).

Three roots of record hold absolute symbolic links into a sibling root: canon_d3p_prod and canon_d3p_dev (the stand-in network's
training inputs) link <ID>/fine and <ID>/coarse/brain.nii.gz into canon_d3; canon_d3n_prod_lo0 (the 8 brains without MRI, as deployed)
links them into canon_d3n_prod. On another machine those links point nowhere. This writes a copy of one root in which every absolute
link is replaced:
  --mode copy       by a copy of what it points to (the root then stands alone; default)
  --mode relative   by a relative link to the same place among the copied siblings (ship the sibling root next to it, as in the
                    analysis project: e.g. canon_d3n_prod_lo0/<ID>/fine -> ../../canon_d3n_prod/<ID>/fine)
Regular files and relative links are copied unchanged, except in copy mode the JSON records (seam.json, canon_root.json, folds.json):
their analysis-machine paths are rewritten as placeholders ($LINC_PRODUCTS/..., shared.paths.placeholder_table()), which
shared.paths.rerooted resolves on any machine; nothing else in them changes (same keys, values and layout: indent 1). Every other file of
the copy, reached through its links, is checked to be byte-identical (md5) to the file of record it stands for, and every JSON record
to equal the record's with its paths scrubbed and no absolute path left (an empty or example table fails here, never silently). In relative mode the sibling root must already be
copied under the same destination parent, and the JSON records are copied unchanged.

    python -m blockface.synthesis.relocate <root of record> <new root> [--mode copy|relative]
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from shared import paths


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def placeholders():
    """(analysis prefix, placeholder) pairs, longest prefix first"""
    return sorted(paths.placeholder_table(), key=lambda ab: -len(ab[0]))


def scrub(o, pairs):
    if isinstance(o, str):
        for a, b in pairs:
            o = o.replace(a, b)
        return o
    if isinstance(o, list):
        return [scrub(x, pairs) for x in o]
    if isinstance(o, dict):
        return {scrub(k, pairs): scrub(v, pairs) for k, v in o.items()}
    return o


def _strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, list):
        for x in o:
            yield from _strings(x)
    elif isinstance(o, dict):
        for k, v in o.items():
            yield k
            yield from _strings(v)


def _dump(obj, like):
    """obj in the layout of the text `like` (the records of record are indent 1, no trailing newline)"""
    d = json.loads(like)
    for ind in (1, 2, None):
        s = json.dumps(d, indent=ind)
        if s == like.rstrip("\n"):
            return json.dumps(obj, indent=ind) + like[len(s):]
    return json.dumps(obj, indent=1)


def relocate(src, dst, mode="copy"):
    src, dst = Path(src).absolute(), Path(dst).absolute()
    if dst.exists():
        raise FileExistsError(f"{dst} exists: a relocated copy is never written over")
    if mode == "copy":                                                  # before writing anything: every path has a placeholder
        pairs = placeholders()
        for q in sorted(src.rglob("*.json")):
            left = [s for s in _strings(scrub(json.load(open(q)), pairs)) if s.startswith("/")]
            if left:
                raise ValueError(f"{q}: {left[0][:80]} has no placeholder (settings/internal/path_placeholders.local.yaml)")
    links = []
    for dirpath, dirnames, filenames in os.walk(src):                    # os.walk does not follow links to directories
        d = Path(dirpath); out = dst / d.relative_to(src)
        out.mkdir(parents=True, exist_ok=True)
        for name in sorted(dirnames + filenames):
            p, q = d / name, out / name
            if p.is_symlink():
                t = os.readlink(p)
                if not os.path.isabs(t):
                    os.symlink(t, q); continue
                links.append(str(p.relative_to(src)))
                if mode == "copy":
                    (shutil.copytree(t, q) if Path(t).is_dir() else shutil.copy2(t, q))
                else:
                    try:
                        rel_t = Path(t).relative_to(src.parent)
                    except ValueError:
                        raise ValueError(f"{p} -> {t}: not inside {src.parent}, so no relative link can reach it; use --mode copy")
                    os.symlink(os.path.relpath(dst.parent / rel_t, q.parent), q)
            elif p.is_file():
                shutil.copy2(p, q)
    pairs = placeholders()
    scrubbed = 0
    if mode == "copy":                                                  # the JSON records: analysis paths -> placeholders
        for q in sorted(dst.rglob("*.json")):
            txt = q.read_text(); new = scrub(json.loads(txt), pairs)
            if new != json.loads(txt):
                q.write_text(_dump(new, txt)); scrubbed += 1
    # every file of the copy (through its links) = the file of record it stands for (JSON records in copy mode: = scrubbed record)
    n, bad = 0, []
    for dirpath, _, filenames in os.walk(dst, followlinks=True):
        for name in filenames:
            q = Path(dirpath) / name; rel = q.relative_to(dst)
            if not q.exists():
                bad.append(f"{rel}: dangling"); continue
            if mode == "copy" and q.suffix == ".json":
                txt = q.read_text()
                if json.loads(txt) != scrub(json.load(open(src / rel)), pairs):
                    bad.append(f"{rel}: differs from the record with its paths scrubbed")
                left = [s for s in _strings(json.loads(txt)) if s.startswith("/")]
                if left:
                    bad.append(f"{rel}: still names an absolute path, {left[0][:80]} (no placeholder for it: "
                               f"settings/internal/path_placeholders.local.yaml)")
            elif md5(q) != md5(src / rel):
                bad.append(f"{rel}: differs")
            n += 1
    absolute = [str(q.relative_to(dst)) for q in dst.rglob("*") if q.is_symlink() and os.path.isabs(os.readlink(q))]
    return dict(files_checked=n, links_replaced=len(links), json_scrubbed=scrubbed, absolute_links_left=absolute, problems=bad)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path); ap.add_argument("dst", type=Path)
    ap.add_argument("--mode", choices=("copy", "relative"), default="copy")
    a = ap.parse_args(argv)
    r = relocate(a.src, a.dst, a.mode)
    print(f"{a.dst}: {r['links_replaced']} absolute links replaced ({a.mode}); {r['json_scrubbed']} JSON records with placeholders; "
          f"{r['files_checked']} files checked against the record; absolute links left {len(r['absolute_links_left'])}; "
          f"problems {len(r['problems'])}")
    for x in r["problems"][:20] + r["absolute_links_left"][:20]:
        print("  ", x)
    return 1 if r["problems"] or r["absolute_links_left"] else 0


if __name__ == "__main__":
    sys.exit(main())
