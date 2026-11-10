"""Histology to blockface section matching: which blockface photograph each histology section was cut at (Methods, "Histology
registration and label propagation").

You need
  the brain's reconstruction store, the raw index of its first section (idx_range[0], settings/brains/<ID>.yaml), and the match
  table (one per brain and series): a histology section number -> the raw photograph index of each of its components (the whole
  section, the left and right hemispheres, the cerebellum). Rows are
      lower, upper, m_whole, a_whole, m_left, a_left, m_right, a_right, m_cere, a_cere
  and a section n with lower <= n <= upper ("inf" allowed as upper) matches photograph m * n + a for each component (-1: no match).
  Every step finds the table the same way (find_table): the path given by flag; else, in the brain's histology folder,
  match_table_edited.txt (a copy of the proposal changed by hand), match_table.txt (a table put there), match_table_proposed.txt
  (linc-histology propose).
What it does
  match_photos() reads, for every histology section, the reconstructed blockface section matched to each of its components,
  upright (rows and columns reversed from the store's (c, rows, sections, columns) order).
You get
  <section><suffix>_0000.png per matched component.

Ported from the analysis project's code/histo.py (_match_table, match_b2h).
"""
import csv
import os
import re
from functools import lru_cache
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
CONFIG = Path(__file__).resolve().parents[2] / "settings" / "histology.yaml"
EDITED, TABLE, PROPOSED, REVIEW = "match_table_edited.txt", "match_table.txt", "match_table_proposed.txt", "match_review.csv"


@lru_cache(maxsize=None)
def _load_config(path):
    import yaml
    return yaml.safe_load(Path(path).read_text())


def config(path=None):
    """The histology parameters (settings/histology.yaml, or `path`)."""
    return _load_config(str(path or CONFIG))


def flagged_count(review_csv):
    """How many components a match_review.csv flags."""
    with open(review_csv, newline="") as f:
        return sum(1 for r in csv.DictReader(f) if r["flag"])


def find_table(histology_dir, table=None):
    """The match table a step uses and a line saying which, (None, None) when there is none. The one order of every step: the path
    given by flag; match_table_edited.txt; match_table.txt; match_table_proposed.txt."""
    if table:
        return str(table), f"{table} (given)"
    h = Path(histology_dir)
    if (h / EDITED).exists():
        return str(h / EDITED), f"{h / EDITED} (edited by hand)"
    if (h / TABLE).exists():
        return str(h / TABLE), str(h / TABLE)
    if (h / PROPOSED).exists():
        n = flagged_count(h / REVIEW) if (h / REVIEW).exists() else "?"
        return str(h / PROPOSED), f"{h / PROPOSED} (an unreviewed proposal, {n} components flagged: see {h / REVIEW}; to change it, copy it to {EDITED})"
    return None, None


def piece_suffixes(cfg=None):
    """{suffix: (piece class, file tag)} of the hemisphere/cerebellum pieces (the whole section has suffix "")."""
    cfg = cfg or config()
    return {p["suffix"]: (p["cls"], p["tag"]) for p in cfg["pieces"].values() if p["suffix"]}


def piece_column(cfg=None):
    """{suffix: match-table column} (whole "" -> 0, _left -> 1, _right -> 2, _cere -> 3)."""
    cfg = cfg or config()
    return {p["suffix"]: p["column"] for p in cfg["pieces"].values()}


def column_tag(cfg=None):
    """{match-table column: piece file tag} (0 -> 0000, 1 -> 0001, 2 -> 0002, 3 -> 0003)."""
    cfg = cfg or config()
    return {p["column"]: p["tag"] for p in cfg["pieces"].values()}


def section_number(name, cfg=None):
    cfg = cfg or config()
    return int(re.search(cfg["section_regex"], name).group(1))


def sections(histo_dir, cfg=None, pattern=None):
    """The histology section files of one series, sorted (settings/histology.yaml section_glob)."""
    cfg = cfg or config()
    return sorted(glob(os.path.join(str(histo_dir), pattern or cfg["section_glob"])))


def match_table(path, cfg=None):
    """Parse a match table into lookup(section number) -> [whole, left, right, cere] raw photograph indices (-1 = none)."""
    cfg = cfg or config()
    rows = []
    with open(path) as f:
        for _ in range(cfg["match_table_skip_header"]):
            next(f)
        for line in f:
            v = re.split(cfg["match_table_separators"], line.strip())
            if len(v) < cfg["match_table_min_fields"]:
                continue
            lo = int(v[0])
            hi = np.inf if "inf" in v[1] else int(v[1])
            rows.append((lo, hi, [int(x) for x in v[2:10]]))

    def lookup(num):
        for lo, hi, c in rows:
            if lo <= num <= hi:
                return [num * c[2 * i] + c[2 * i + 1] for i in range(4)]
        return [-1, -1, -1, -1]

    return lookup


def match_photos(store, histo_dir, table, idx0, out_dir, cfg=None, pattern=None, log=print):
    """Save the blockface slice matched to each piece of each histology section -> out_dir/<section><suffix>_0000.png.

    store: the blockface OME-Zarr (<ID>_blockface.ome.zarr); idx0: the raw index of its first slice (idx_range[0] of the
    brain); a matched photograph outside the stack is reported and skipped. Returns the number of photos written."""
    import zarr
    cfg = cfg or config()
    lookup = match_table(table, cfg)
    bf = zarr.open_group(str(store), mode="r")[str(cfg["store_level"])]        # (c, rows, slices, cols)
    os.makedirs(out_dir, exist_ok=True)
    suffixes = ["", "_left", "_right", "_cere"]
    n = 0
    for jp2 in sections(histo_dir, cfg, pattern):
        base = os.path.basename(jp2)[:-4]
        num = section_number(base, cfg)
        for res, suffix in zip(lookup(num), suffixes):
            if res != -1 and not 0 <= res - idx0 < bf.shape[2]:
                log(f"  {base}{suffix}: matched blockface index {res - idx0} outside the stack "
                    f"(0..{bf.shape[2] - 1}) -> no blockface target, skipped")
                continue
            if res != -1:
                photo = np.asarray(bf[:, :, res - idx0, :]).transpose(1, 2, 0)[::-1, ::-1]   # upright
                Image.fromarray(photo).save(os.path.join(out_dir, base + suffix + "_0000.png"))
                n += 1
    log(f"matched {n} blockface slices to histology -> {out_dir}")
    return n
