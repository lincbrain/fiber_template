#!/usr/bin/env python3
"""Check a local copy of the trained networks against a checksum list (sha256), before running the pipeline.

    python tests/verify_weights.py settings/models/segmentation.sha256 --root $LINC_MODELS/segment

Each line of the list is "<sha256>  <path relative to the root>"; lines starting with # are comments. Prints one line per file
(OK, MISSING or DIFFERS) and exits 0 only when every file is present and matches.
"""
import argparse
import hashlib
import sys
from pathlib import Path


def sha256(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checksums", type=Path)
    ap.add_argument("--root", type=Path, required=True, help="the folder the list's paths are relative to")
    a = ap.parse_args()
    root = a.root
    bad = 0
    for line in a.checksums.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        want, rel = line.split(None, 1)
        p = root / rel.strip()
        if not p.exists():
            print(f"MISSING  {rel}"); bad += 1
        elif sha256(p) != want:
            print(f"DIFFERS  {rel}"); bad += 1
        else:
            print(f"OK       {rel}")
    print("all files match" if not bad else f"{bad} file(s) missing or different")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
