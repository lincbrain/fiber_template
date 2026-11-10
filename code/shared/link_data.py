#!/usr/bin/env python3
"""Give the code its working names for the files of the data tree: one link per row of settings/files.tsv.

    python code/shared/link_data.py <data tree> <folder of links> [--table settings/files.tsv]

You need
    the data tree (LINC_DATA) and the table. Each row has new_path, the file's path in the data tree, and old_path, the path the code knows
    it by. A file the code knows by two names has two rows. With $LINC_000003 set to a copy of LINC dataset 000003 (the histology), raw/linc_000003
    is one link to that copy, so the code sees the whole dataset; without it, only the dataset's files that the tree holds are linked.
What it does
    Makes <folder of links>/<old_path>, a link to <data tree>/<new_path>, with real folders above it; a Zarr store is one link. A file the
    tree does not hold (a row with upload = local, in a tree downloaded from the archive) gets no link and is counted. Run again with the same
    table and tree it does nothing; with another table or tree it starts over. It deletes only that folder's links, never the data.
You get
    the folder of links, and a line saying how many links it made and how many files were absent. settings/roots.sh runs this for you.
"""
import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tree", type=Path)
    ap.add_argument("links", type=Path)
    ap.add_argument("--table", type=Path, default=Path(__file__).resolve().parents[2] / "settings" / "files.tsv")
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
    if a.links.exists():
        shutil.rmtree(a.links)                  # never follows a link: only this folder's own entries go
    a.links.mkdir(parents=True)
    made_dirs, n, absent = set(), 0, 0
    if linc:
        (a.links / "raw").mkdir()
        os.symlink(linc, a.links / "raw" / "linc_000003")
    with open(a.table) as f:
        head = f.readline().rstrip("\n").split("\t")
        i, j = head.index("new_path"), head.index("old_path")
        for line in f:
            row = line.rstrip("\n").split("\t")
            if linc and row[j].startswith("raw/linc_000003/"):
                continue
            target = f"{tree}/{row[i]}"
            if not os.path.lexists(target):
                absent += 1
                continue
            link = f"{a.links}/{row[j]}"
            d = os.path.dirname(link)
            if d not in made_dirs:
                os.makedirs(d, exist_ok=True)
                made_dirs.add(d)
            if os.path.lexists(link):
                if os.readlink(link) != target:
                    sys.exit(f"{row[j]}: two rows give it different files ({os.readlink(link)}, {target})")
                continue
            os.symlink(target, link)
            n += 1
    stamp.write_text("\n".join(state) + "\n")
    print(f"{a.links}: {n} links to {tree}; {absent} files of the table are not in this tree")


if __name__ == "__main__":
    main()
