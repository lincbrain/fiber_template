#!/usr/bin/env python3
"""Point a construction run at the files it starts from.

    python code/template/construction_inputs.py --work <the run's work folder> --paper "<folder, <ID> for the brain>" \\
        --own "<folder, <ID> for the brain>" MF283 MF284 MF285 MF286

You need
  settings/env/linc.yml (the standard library only).
What it does
  For each brain it makes <work>/prep/<ID> lead to the folder that holds the brain's construction inputs, the files step A7 (prepare)
  writes. The first of these that holds them is used:
    1. <work>/prep/<ID> itself, as it is. Point it at your own A7 outputs to build from those; it is then left alone.
    2. the paper's inputs in the data folder (--paper), so that the run can be compared with the paper's file by file.
    3. this build's own A7 outputs (--own), when the data folder does not hold the paper's.
  It prints which one each brain got. When none holds the inputs it stops, changes nothing for that brain and names the places it looked in.
You get
  the links and one line per brain, exit status 0; or one message and exit status 1.
"""
import argparse
import os
import sys
from pathlib import Path

NEEDS = ("t2_modelmask.nii.gz", "b0.nii.gz", "model_mask.nii.gz", "tensor_vr.nii.gz")   # written by prepare; the assembly reads them


def holds(folder, needs=NEEDS):
    """True when the folder (or what the link leads to) holds every file in needs"""
    return folder.is_dir() and all((folder / n).exists() for n in needs)


def choose(link, paper, own, needs=NEEDS):
    """(the folder to use, what to call it), or (None, None) when nothing holds the inputs"""
    if holds(link, needs):
        return Path(os.path.realpath(link)), "kept as it is"
    if holds(paper, needs):
        return paper, "the paper's inputs"
    if holds(own, needs):
        return own, "this build's own prepare outputs (A7): the data folder does not hold the paper's"
    return None, None


def point(work, paper, own, ids, needs=NEEDS, say=print):
    """make <work>/prep/<ID> lead to each brain's inputs; returns {ID: the folder}; SystemExit with one message when a brain has none"""
    prep = Path(work) / "prep"
    prep.mkdir(parents=True, exist_ok=True)
    used = {}
    for s in ids:
        link, p, o = prep / s, Path(paper.replace("<ID>", s)), Path(own.replace("<ID>", s))
        src, what = choose(link, p, o, needs)
        if src is None:
            raise SystemExit(f"no construction inputs for {s}. Looked in {p} (the paper's, from the data folder) and in {o} "
                             f"(this build's own: run step A7, prepare-{s}, first). Neither holds {', '.join(needs)}.")
        if what != "kept as it is":
            if link.is_symlink():
                link.unlink()
            elif link.exists():
                raise SystemExit(f"{link} is a folder of its own that does not hold the construction inputs: move it aside and run this again")
            link.symlink_to(src)
        used[s] = src
        say(f"{s}: {what} ({src})")
    return used


def main():
    ap = argparse.ArgumentParser(description="Point a construction run at the files it starts from (code/template/README.md)")
    ap.add_argument("ids", nargs="+", help="the brains, as in settings/brains.tsv")
    ap.add_argument("--work", required=True, help="the construction run's work folder; the links go in <work>/prep/")
    ap.add_argument("--paper", required=True, help="where the data folder keeps a brain's construction inputs, with <ID> for the brain")
    ap.add_argument("--own", required=True, help="where this build's prepare step (A7) writes them, with <ID> for the brain")
    a = ap.parse_args()
    point(a.work, a.paper, a.own, a.ids)
    return 0


if __name__ == "__main__":
    sys.exit(main())
