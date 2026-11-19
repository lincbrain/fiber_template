#!/usr/bin/env python3
"""Give the code its working names for the files of the data tree: one link per row of settings/files.tsv.

    python code/shared/link_data.py <data tree> <folder of links> [--table settings/files.tsv] [--built <folder> ...] [--refresh]

You need
    the data tree (LINC_DATA) and the table. Each row has new_path, the file's path in the data tree, and old_path, the path the code knows
    it by. A file the code knows by two names has two rows. With $LINC_000003 set to a copy of LINC dataset 000003 (the histology), raw/linc_000003
    is one link to that copy, so the code sees the whole dataset; without it, only the dataset's files that the tree holds are linked.
What it does
    Makes <folder of links>/<old_path>, a link to <data tree>/<new_path>, with real folders above it; a Zarr store is one link. A file the
    tree does not hold (a row with upload = local, in a tree downloaded from the archive) gets no link and is counted. A folder that holds
    none of the table's files is not a data tree: it stops there, says which folder looks like one, and changes nothing. Run again with the
    same table and tree it does nothing; with another table (a `git pull`) or --refresh on the same tree it updates the folder in place:
    a link that does not change is left as it is, a changed one is replaced in one step (a new link renamed over the old), a new one
    is added and one whose row went away is removed, so a step that is running meanwhile keeps reading its files. With another tree it
    starts over. It deletes only that folder's links, never the data.
    --refresh looks at the tree again although the table and the tree are the same: use it after files were added to the tree, which it
    cannot see by itself (`source settings/roots.sh refresh` does).
    --built names the folders made from these links (the working folders of settings/roots.sh): when one exists and the links were made
    from another tree, it stops instead of starting over, because those folders hold that tree's files and the results made from them.
You get
    the folder of links, and a line saying how many links it made and how many files were absent. settings/roots.sh runs this for you.
"""
import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path


def held(folder, rows, first=False):
    """The rows (new_path, old_path) whose file is in the folder; with first, only the first one found. A top-level folder that is not
    there is looked up once, not once per row."""
    tops = {t for t in {new.split("/", 1)[0] for new, _ in rows} if os.path.isdir(f"{folder}/{t}")}
    found = []
    for row in rows:
        if row[0].split("/", 1)[0] in tops and os.path.lexists(f"{folder}/{row[0]}"):
            found.append(row)
            if first:
                break
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tree", type=Path)
    ap.add_argument("links", type=Path)
    ap.add_argument("--table", type=Path, default=Path(__file__).resolve().parents[2] / "settings" / "files.tsv")
    ap.add_argument("--built", type=Path, nargs="*", default=[])
    ap.add_argument("--refresh", action="store_true", help="make the links again although the table and the tree are the same")
    a = ap.parse_args()
    tree = a.tree.resolve()
    if not tree.is_dir():
        sys.exit(f"{tree}: no such folder")
    linc = os.environ.get("LINC_000003", "").rstrip("/")
    state = [hashlib.md5(a.table.read_bytes()).hexdigest(), str(tree), linc]
    stamp = a.links / ".complete"
    if not a.refresh and stamp.exists() and stamp.read_text().split("\n")[:-1] == state:
        print(f"{a.links}: up to date")
        return
    with open(a.table) as f:
        head = f.readline().rstrip("\n").split("\t")
        i, j = head.index("new_path"), head.index("old_path")
        rows = [(r[i], r[j]) for r in (line.rstrip("\n").split("\t") for line in f)]
    if linc:
        rows = [r for r in rows if not r[1].startswith("raw/linc_000003/")]
    here = held(tree, rows)
    if not here:                                    # not a data tree: say so before anything is deleted or made
        tops = sorted({new.split("/", 1)[0] + "/" for new, _ in rows})
        near = [p for p in (tree.parent, tree.parent.parent) if held(p, rows, first=True)][:1]
        sys.exit(f"{tree} is not the data folder: none of the files that {a.table.parent.name}/{a.table.name} lists is in it. The data folder is the one "
                 f"that holds {', '.join(tops[:-1])} and {tops[-1]}" + (f"; {near[0]} looks like it." if near else "."))
    was = stamp.read_text().split("\n")[1:2] if stamp.exists() else []      # the tree the links were made from
    built = [b.name for b in a.built if b.exists()]
    if was and was[0] != str(tree) and os.path.isdir(was[0]) and built:
        sys.exit(f"{a.links.parent} was built from another data folder, {was[0]}: {', '.join(built)} hold its files and your results "
                 f"from it. Move {a.links.parent.name}/ aside or delete it, then source settings/roots.sh again.")
    want = {"raw/linc_000003": linc} if linc else {}                # old_path -> target
    for new, old in here:
        target = f"{tree}/{new}"
        if want.setdefault(old, target) != target:
            sys.exit(f"{old}: two rows give it different files ({want[old]}, {target})")
    in_place = a.links.is_dir() and was == [str(tree)]
    if in_place:                                # the same tree: update in place, so a running step keeps its files
        stamp.write_text("\n".join(["updating"] + state[1:]) + "\n")     # an interrupted update is redone, in place, next time
    else:
        if a.links.exists():
            shutil.rmtree(a.links)              # never follows a link: only this folder's own entries go
        a.links.mkdir(parents=True)
    count = dict(made=0, changed=0, removed=0, kept=0)
    if in_place:
        for d, dirs, files in os.walk(a.links, topdown=True):      # does not descend into linked folders (a Zarr store is one link)
            for name in dirs + files:
                p = os.path.join(d, name)
                if not os.path.islink(p):
                    continue
                rel = os.path.relpath(p, a.links)
                if rel not in want:
                    os.unlink(p); count["removed"] += 1
            dirs[:] = [x for x in dirs if not os.path.islink(os.path.join(d, x))]
    made_dirs = set()
    for old, target in want.items():
        link = f"{a.links}/{old}"
        d = os.path.dirname(link)
        if d not in made_dirs:
            os.makedirs(d, exist_ok=True)
            made_dirs.add(d)
        if os.path.islink(link):
            if os.readlink(link) == target:
                count["kept"] += 1
                continue
            tmp = f"{link}.relink{os.getpid()}"
            os.symlink(target, tmp)
            os.replace(tmp, link)                # one step: the name never disappears
            count["changed"] += 1
            continue
        if os.path.lexists(link):
            sys.exit(f"{link} is not a link: {a.links} holds a file of its own; move it aside")
        os.symlink(target, link)
        count["made"] += 1
    if in_place:                                 # folders a removed row left empty (never the links' root)
        for d, dirs, files in os.walk(a.links, topdown=False):
            if d != str(a.links) and not os.listdir(d):
                os.rmdir(d)
    stamp.write_text("\n".join(state) + "\n")
    n = len(want) - (1 if linc else 0)
    how = (f" (updated in place: {count['made']} made, {count['changed']} changed, {count['removed']} removed, {count['kept']} unchanged)"
           if in_place else "")
    print(f"{a.links}: {n} links to {tree}{how}; {len(rows) - len(here)} files of the table are not in this tree")


if __name__ == "__main__":
    main()
