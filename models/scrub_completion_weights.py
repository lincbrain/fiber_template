#!/usr/bin/env python3
"""Copies of the completion networks' files without analysis-machine paths, for a public data release.

The networks of record are not changed: each folder <src>/<name>/{last.pt, config.json, train_log.jsonl} is copied to <dst>/<name>/
with
  last.pt          cfg's strings re-rooted with linc.paths.placeholder_table() (cfg["root"], cfg["canon_root"]["source_root"]),
                   and its author_* fields (design requests quoted from the project's notes) dropped;
                   the weights G and the epoch are copied unchanged (every tensor checked equal, dtype and values)
  config.json      the same configuration, the same scrub (written as before: indent 1)
  train_log.jsonl  every epoch's row without its "host" field (the training machine's name; a release names no machine), the
                   other fields and their order unchanged (d3_methods reads epochs and wall time, and accepts a log without hosts)
Writes the old -> new checksum map (sha256 and md5 per file) and the checksum list of the copies (for models/verify_weights.py).
Nothing reads the scrubbed strings: linc.completion.network.load uses the weights and the architecture fields only, and
reproduce/numbers/d3_methods.py reads only the folder name of cfg["root"]. The copies are deterministic (a rerun is byte-identical).

    python models/scrub_completion_weights.py --src <runs of record> --dst <new folder> \
        [--map models/completion_checksum_map.tsv] [--sha models/completion_public.sha256]
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from linc import paths  # noqa: E402

NETS = ["d3_full_A_s0", "d3p_full_A_s0", "d3_lso_A_s0", "d3p_lso_A_s0"]
FILES = ["last.pt", "config.json", "train_log.jsonl"]
LEAK = re.compile(r'/autofs|/space/|/homes/|rg116|"/|ORCH')             # "/ = any absolute path left in a JSON string


def digest(p, algo="sha256"):
    h = hashlib.new(algo)
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def scrubber():
    pairs = sorted(paths.placeholder_table(), key=lambda ab: -len(ab[0]))

    def scrub(o):
        if isinstance(o, str):
            for a, b in pairs:
                o = o.replace(a, b)
            return o
        if isinstance(o, list):
            return [scrub(x) for x in o]
        if isinstance(o, dict):
            return {scrub(k): scrub(v) for k, v in o.items() if not str(k).startswith("author_")}
        return o
    return scrub


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path); ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--map", type=Path, default=None); ap.add_argument("--sha", type=Path, default=None)
    a = ap.parse_args()
    scrub = scrubber()
    for n in NETS:                                                      # before writing anything: every path has a placeholder
        if (a.dst / n).exists():
            sys.exit(f"{a.dst / n} exists: copies are never overwritten")
        ck = torch.load(a.src / n / "last.pt", map_location="cpu", weights_only=False)
        conf = json.loads((a.src / n / "config.json").read_text())
        for f, txt in (("last.pt", json.dumps(scrub(ck["cfg"]))), ("config.json", json.dumps(scrub(conf))),
                       ("train_log.jsonl", (a.src / n / "train_log.jsonl").read_text())):
            m = LEAK.search(txt)
            if m:
                sys.exit(f"{n}/{f}: would still hold {m.group(0)!r}: no placeholder for it (configs/path_placeholders.local.yaml)")
    rows, sums = [], []
    for n in NETS:
        s, d = a.src / n, a.dst / n
        d.mkdir(parents=True)
        ck = torch.load(s / "last.pt", map_location="cpu", weights_only=False)
        assert set(ck) == {"G", "cfg", "epoch"}, f"{n}: unexpected checkpoint keys {sorted(ck)}"
        new = {"G": ck["G"], "cfg": scrub(ck["cfg"]), "epoch": ck["epoch"]}
        torch.save(new, d / "last.pt")
        back = torch.load(d / "last.pt", map_location="cpu", weights_only=True)
        assert list(back["G"]) == list(ck["G"]) and back["epoch"] == ck["epoch"] and back["cfg"] == new["cfg"]
        for k, t in ck["G"].items():
            assert back["G"][k].dtype == t.dtype and torch.equal(back["G"][k], t), f"{n}: tensor {k} differs"
        cfg = scrub(json.loads((s / "config.json").read_text()))
        (d / "config.json").write_text(json.dumps(cfg, indent=1))
        (d / "train_log.jsonl").write_text("".join(json.dumps({k: v for k, v in json.loads(x).items() if k != "host"}) + "\n"
                                                    for x in (s / "train_log.jsonl").read_text().splitlines() if x.strip()))
        assert '"host"' not in (d / "train_log.jsonl").read_text()
        for f, txt in (("last.pt", json.dumps(new["cfg"])), ("config.json", (d / "config.json").read_text()),
                       ("train_log.jsonl", (d / "train_log.jsonl").read_text())):
            m = LEAK.search(txt)
            assert not m, f"{n}/{f}: still holds {m.group(0)!r}"
        changed = dict(zip(FILES, ["cfg paths re-rooted ($LINC_PRODUCTS), author_* notes dropped; weights and epoch unchanged",
                                   "paths re-rooted ($LINC_PRODUCTS), author_* notes dropped",
                                   "the host field of every row removed"]))
        for f in FILES:
            old, nw = digest(s / f), digest(d / f)
            rows.append(f"{n}/{f}\t{old}\t{nw}\t{digest(s / f, 'md5')}\t{digest(d / f, 'md5')}\t{changed[f]}")
            sums.append(f"{nw}  {n}/{f}")
        print(n, "OK:", len(ck["G"]), "tensors equal;", ", ".join(f"{f} {digest(d / f)[:12]}" for f in FILES))
    if a.map:
        a.map.write_text("# The completion networks' files: sha256 of the files of record (models/completion.sha256) -> sha256 of the copies\n"
                         "# without machine paths (models/scrub_completion_weights.py; weights identical). The data\n"
                         "# release ships the copies; models/completion_public.sha256 checks them. The md5 columns let records that name\n"
                         "# a network by its md5 (d3_methods, the completed volumes' summaries) recognize a copy.\n"
                         "file\tsha256_of_record\tsha256_public\tmd5_of_record\tmd5_public\tchange\n" + "\n".join(rows) + "\n")
    if a.sha:
        a.sha.write_text("# sha256 of the completion networks' files as released (copies without machine paths; the weights are\n"
                         "# those of record, see models/completion_checksum_map.tsv), relative to $LINC_MODELS/completion.\n"
                         "# Check a local copy with: python models/verify_weights.py models/completion_public.sha256 --root $LINC_MODELS/completion\n"
                         + "\n".join(sums) + "\n")


if __name__ == "__main__":
    main()
