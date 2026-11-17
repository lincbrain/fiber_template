#!/usr/bin/env python3
"""Give the code its working names for the files of the data tree: one link per row of settings/files.tsv.

    python code/shared/link_data.py <data tree> <folder of links> [--table settings/files.tsv] [--built <folder> ...]

You need
    the data tree (LINC_DATA) and the table. Each row has new_path, the file's path in the data tree, and old_path, the path the code knows
    it by. A file the code knows by two names has two rows. With $LINC_000003 set to a copy of LINC dataset 000003 (the histology), raw/linc_000003
    is one link to that copy, so the code sees the whole dataset; without it, only the dataset's files that the tree holds are linked.
What it does
    Makes <folder of links>/<old_path>, a link to <data tree>/<new_path>, with real folders above it; a Zarr store is one link. A file the
    tree does not hold (a row with upload = local, in a tree downloaded from the archive) gets no link and is counted. A folder that holds
    none of the table's files is not a data tree: it stops there, says which folder looks like one, and changes nothing. Run again with the
    same table and tree it does nothing; with another table or tree it starts over. It deletes only that folder's links, never the data.
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
    a = ap.parse_args()
    tree = a.tree.resolve()
    if not tree.is_dir():
        sys.exit(f"{tree}: no such folder")
    linc = os.environ.get("LINC_000003", "").rstrip("/")
    state = [hashlib.md5(a.table.read_bytes()).hexdigest(), str(tree), linc]
    stamp = a.links / ".complete"
    if stamp.exists() and stamp.read_text().split("\n")[:-1] == state:
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
    if a.links.exists():
        shutil.rmtree(a.links)                  # never follows a link: only this folder's own entries go
    a.links.mkdir(parents=True)
    made_dirs, n = set(), 0
    if linc:
        (a.links / "raw").mkdir()
        os.symlink(linc, a.links / "raw" / "linc_000003")
    for new, old in here:
        target, link = f"{tree}/{new}", f"{a.links}/{old}"
        d = os.path.dirname(link)
        if d not in made_dirs:
            os.makedirs(d, exist_ok=True)
            made_dirs.add(d)
        if os.path.lexists(link):
            if os.readlink(link) != target:
                sys.exit(f"{old}: two rows give it different files ({os.readlink(link)}, {target})")
            continue
        os.symlink(target, link)
        n += 1
    stamp.write_text("\n".join(state) + "\n")
    print(f"{a.links}: {n} links to {tree}; {len(rows) - len(here)} files of the table are not in this tree")


if __name__ == "__main__":
    main()
