#!/usr/bin/env python3
"""Apply our changes to the FMRIB template-construction driver without shipping its text.

mm-template-construction (https://git.fmrib.ox.ac.uk/cart/mm-template-construction) carries no licence, so this repository holds only
our own lines: each edit file lists, per change, the range of the driver's lines it replaces (1-based, in the clean file), the sha256
of those lines (not the lines themselves) and our new lines. The clean file's md5 is checked first and the result's md5 afterwards.

    git clone https://git.fmrib.ox.ac.uk/cart/mm-template-construction.git && git -C mm-template-construction checkout f4907f4
    python linc/construction/patches/apply_edits.py linc/construction/patches/mm-template-construction_v1_linc.edits.json \
        mm-template-construction                  # the MRI template's driver; _v1_blockface.edits.json: the blockface template's

It rewrites <checkout>/<the edit file's "file"> in place (or writes --out FILE). Making an edit file from a clean and a changed copy:
    python apply_edits.py --make CLEAN CHANGED --file template_construction_v1/run_template_construction.py --commit f4907f4 -o EDITS
"""
import argparse
import difflib
import hashlib
import json
import sys
from pathlib import Path


def md5(b):
    return hashlib.md5(b).hexdigest()


def sha(lines):
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def make(clean, changed, file, commit):
    a, b = Path(clean).read_bytes(), Path(changed).read_bytes()
    A, B = a.decode().splitlines(keepends=True), b.decode().splitlines(keepends=True)
    edits = []
    for tag, i0, i1, j0, j1 in difflib.SequenceMatcher(None, A, B, autojunk=False).get_opcodes():
        if tag != "equal":
            edits.append(dict(start=i0 + 1, n_old=i1 - i0, old_sha256=sha(A[i0:i1]), new=B[j0:j1]))
    return dict(file=file, commit=commit, source_md5=md5(a), result_md5=md5(b), n_edits=len(edits), edits=edits)


def apply(spec, text):
    if md5(text) != spec["source_md5"]:
        raise SystemExit(f"{spec['file']}: md5 {md5(text)} is not the clean file at {spec['commit']} ({spec['source_md5']})")
    L = text.decode().splitlines(keepends=True)
    for e in sorted(spec["edits"], key=lambda e: -e["start"]):         # bottom up: earlier line numbers stay valid
        i0 = e["start"] - 1
        old = L[i0:i0 + e["n_old"]]
        if sha(old) != e["old_sha256"]:
            raise SystemExit(f"{spec['file']}: lines {e['start']}..{e['start'] + e['n_old'] - 1} are not those the edit replaces")
        L[i0:i0 + e["n_old"]] = e["new"]
    out = "".join(L).encode()
    if md5(out) != spec["result_md5"]:
        raise SystemExit(f"{spec['file']}: result md5 {md5(out)} != {spec['result_md5']}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("edits", nargs="?"); ap.add_argument("checkout", nargs="?")
    ap.add_argument("--out", help="write the result here instead of in place")
    ap.add_argument("--make", nargs=2, metavar=("CLEAN", "CHANGED")); ap.add_argument("--file"); ap.add_argument("--commit")
    ap.add_argument("-o", dest="o")
    a = ap.parse_args()
    if a.make:
        spec = make(*a.make, a.file, a.commit)
        Path(a.o).write_text(json.dumps(spec, indent=1) + "\n")
        print(f"{a.o}: {spec['n_edits']} edits, {sum(len(e['new']) for e in spec['edits'])} new lines, source {spec['source_md5']} -> {spec['result_md5']}")
        return
    if not (a.edits and a.checkout):
        ap.error("give an edit file and the driver checkout (or --make)")
    spec = json.loads(Path(a.edits).read_text())
    target = Path(a.checkout) / spec["file"]
    out = apply(spec, target.read_bytes())
    Path(a.out or target).write_bytes(out)
    print(f"{a.out or target}: {spec['n_edits']} edits applied, md5 {spec['result_md5']}")


if __name__ == "__main__":
    sys.exit(main())
