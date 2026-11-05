#!/usr/bin/env python3
"""Test of the automated match table (linc-histology propose) against the records. Reads the records only; writes under --work.

    $PY_TIRL tests/match_propose_records.py --work <folder outside $LINC_RECORDS> --jobs 8

You need
  $LINC_RECORDS (labels/match_tables, labels/regions_t2_final, the brains' stores and masks); the LINC histology dataset (--linc-dataset,
  default $LINC_RAW/linc_000003); the folder of the five brains' hand-made tables (--hand-tables); env/tirl.yml.
What it does
  rule      the 12 tables written by the rule of record, recomputed from the record's saved scores: run lines, floor and table writer.
  full      the same 12 tables from the command, on the release's inputs, row by row against the record.
  outlines  the same, with the record's outline step (a 2D network's partition) in place of the command's: which differences come from
            that one step.
  noise     the same as full at two threads per process: which rows change through the registrations' thread count alone.
  repeat    one brain twice at one thread per process: identical or not.
  saved     the five brains with hand-made tables, from the record's saved fit of 20 Sep (an earlier variant, before the atlas regions
            were the classes): 17 of 318 kept components more than one section from the hand-made table. The record counts 17 of 23 such
            components flagged in one file and 22 in the other; its pieces give 17 of 22.
  five      the same comparison from the command, and with the record's outline step in its place. Those tables are another person's
            choice, not a known truth, and "kept" leaves out the components the floor dropped.
  edited    one row changed in an edited table: the match step uses it, and --table still wins.
You get
  tests/match_propose_records.json (or --json), and the proposals under --work.
"""
import argparse
import copy
import filecmp
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from linc import paths  # noqa: E402
from linc.histology import match as M, pieces as PC, propose as PR, sections as SC  # noqa: E402

RULE = {  # series: subject, spacing the record's rule ran at (um, section step); photographs per section = 1200 / that step
    "MF191": ("MF191", 600.0), "MF191_LY": ("MF191", 600.0), "MF196_FS": ("MF196", 870.0), "MF196_LY": ("MF196", 870.0),
    "MF204": ("MF204", 810.0), "MR244": ("MR244", 400.0), "MR253": ("MR253", 50.0), "MR254": ("MR254", 50.0),
    "MR257": ("MR257", 50.0), "MR263": ("MR263", 50.0), "MR273_FR": ("MR273", 50.0), "MR273_FS": ("MR273", 50.0)}
FIVE = {"MR243": 400.0, "MR252": 50.0, "MR256": 50.0, "MF278": 50.0, "MF275": 50.0}
HAND_COLUMN = {"whole": 0, "left": 1, "right": 2, "cere": 3}


def subject_cfg(S):
    return yaml.safe_load((REPO / "configs" / "subjects" / f"{S}.yaml").read_text())


def regions_of(T, S, vz, rec):
    """The regions volume the record's rule read and the spacing it was made at (MF191's first volume, made at 600 um, before its
    later version at the calibrated spacing)."""
    sc = subject_cfg(S)
    vx = float((sc.get("reconstruction_spacing_um") or sc)["vx"])
    if S == "MF191":
        return f"{rec}/labels/regions_t2_mf191_made_at_vz600/MF191_regions_t2.nii.gz", (vx, 600.0)
    return f"{rec}/labels/regions_t2_final/{S}_regions_t2.nii.gz", (vx, vz)


def source_args(T, S, dataset, rec):
    """--histology (and --section-map): the JPEG 2000 folder when the series has one, else the OME-Zarr stack with its section map."""
    st = T.split("_")[1] if "_" in T else None
    jp2 = f"{dataset}/sourcedata/sub-{S}/micr"
    if not st and glob.glob(jp2 + "/*nDF.jp2"):
        return dict(jp2_dir=jp2)
    z = SC.image_stacks(f"{dataset}/rawdata/sub-{S}/micr", S, st)
    return dict(stack=z[0], section_map=f"{rec}/labels/section_maps/{S}_{SC.stain_of(z[0])}.tsv")


def read_rows(path):
    rows = {}
    for line in Path(path).read_text().split("\n")[1:]:
        v = line.strip().split(",")
        if len(v) >= 10 and v[0] != "-1":
            rows[int(v[0])] = [int(x) for x in v[3:10:2]]
    return rows


def compare_tables(a, b, per_section):
    """Row by row: a, b = {section: [whole, left, right, cere]}. per_section: photographs per section."""
    secs = sorted(set(a) | set(b))
    none = [-1] * 4
    same = sum(a.get(n, none) == b.get(n, none) for n in secs)
    both, one, within = [], [], 0
    for n in secs:
        for c, (x, y) in enumerate(zip(a.get(n, none), b.get(n, none))):
            if x == y and x == -1:
                continue
            if x != -1 and y != -1:
                both.append(abs(x - y) / per_section)
                within += abs(x - y) <= per_section
            else:
                one.append((n, c))
    both = np.array(both) if both else np.zeros(0)
    return dict(sections=len(secs), rows_identical=same, components_in_both=int(both.size), components_identical=int((both == 0).sum()),
                within_one_section=int(within), median_sections=round(float(np.median(both)), 3) if both.size else None,
                max_sections=round(float(both.max()), 3) if both.size else None, in_one_table_only=len(one))


def run_command(T, S, a, rec, work, a0, vx, reg, made, dataset):
    """The command, timed, with its peak memory."""
    h = Path(work) / T / "histology"
    h.mkdir(parents=True, exist_ok=True)
    src = source_args(T, S, dataset, rec)
    cmd = ["/usr/bin/time", "-f", "%e %M", sys.executable, str(REPO / "cli/linc-histology"), "propose", "--subject", S,
           "--store", f"{rec}/data/sub-{S}/blockface/{S}_blockface.ome.zarr", "--brain-mask", f"{rec}/data/sub-{S}/blockface/{S}_mask.nii.gz",
           "--regions", reg, "--regions-spacing-um", str(made[0]), str(made[1]), "--photos-per-section", repr(a0),
           "--out", str(h / "registration"), "--jobs", str(a.jobs)]
    cmd += ["--histology", src["stack"], "--section-map", src["section_map"]] if "stack" in src else ["--histology", src["jp2_dir"]]
    r = subprocess.run(cmd, capture_output=True, text=True)
    last = r.stderr.strip().split("\n")[-1].split()
    if r.returncode:
        raise SystemExit(f"{T}: {r.stdout[-800:]}\n{r.stderr[-800:]}")
    return h, float(last[0]), round(float(last[1]) / 1024 ** 2, 2)


def inputs_of(T, S, a0, rec, reg, made, dataset, cfg):
    sc = subject_cfg(S)
    step = cfg["propose"]["sample_step_fine"] if cfg["propose"]["section_um"] / a0 <= cfg["propose"]["fine_step_um"] else 1
    scale = PC.regions_scale(sc, {"vx": made[0], "vz": made[1]})
    slices = PR.blockface_slices(f"{rec}/data/sub-{S}/blockface/{S}_blockface.ome.zarr", f"{rec}/data/sub-{S}/blockface/{S}_mask.nii.gz",
                                 reg, scale, step, cfg, lambda s: None)
    return slices, PR.section_components(SC.sections(**source_args(T, S, dataset, rec)), cfg), step


def with_record_outlines(P, comps, slices, a0, vx, rp, jobs, cfg):
    """The command's later steps on the record's outline step (its classes and places)."""
    Q = []
    for p in P:
        r = rp.get((p["section"], p["kind"]))
        Q.append(dict(p, cls=r["cls"], idx=r["idx"], offset=r["offset"], group=p["kind"] if p["kind"].startswith("pair") else r["cls"]) if r else dict(p))
    PR.add_scores(Q, comps, slices, a0, vx * cfg["propose"]["photo_px_factor"] / 1000.0, jobs, cfg, lambda s: None)
    return Q


def record_pieces(T, rec):
    return {(p["section"], p["kind"]): p for p in json.load(open(f"{rec}/labels/match_tables/proposed/{T}_pieces.json"))["pieces"]}


def part_rule(rec):
    out = {}
    cfg = M.config()
    for T, (S, vz) in RULE.items():
        f = json.load(open(f"{rec}/labels/match_tables/proposed/{T}_reg.json"))
        P = f["pieces"]
        for p in P:
            p["scores"] = {int(k): v for k, v in (p.get("scores") or {}).items()}
        PR.run_lines(P, abs(f["slope"]), f.get("sample_step") or 1, cfg)
        tmp = f"/dev/shm/match_propose_{os.getpid()}.txt"
        PR.write_table(tmp, PR.table_rows(P, subject_cfg(S)["idx_range"][0], cfg))
        out[T] = dict(bytes_identical=Path(tmp).read_text() == Path(f"{rec}/labels/match_tables/num_{T}.txt").read_text(),
                      components=len(P), flagged=sum(1 for p in P if p["flags"]))
        os.remove(tmp)
    return dict(tables_byte_identical=sum(v["bytes_identical"] for v in out.values()), tables=len(out), per_series=out)


def part_full(a, rec, dataset):
    out = {}
    cfg = M.config()
    for T, (S, vz) in RULE.items():
        a0 = 1200.0 / vz
        reg, made = regions_of(T, S, vz, rec)
        sc = subject_cfg(S)
        vx = float((sc.get("reconstruction_spacing_um") or sc)["vx"])
        i0 = sc["idx_range"][0]
        t0 = time.strftime("%H:%M:%S")
        h, wall, gb = run_command(T, S, a, rec, a.work, a0, vx, reg, made, dataset)
        print(f"{t0} {T}: command {wall:.0f} s, {gb} GB", flush=True)
        record = read_rows(f"{rec}/labels/match_tables/num_{T}.txt")
        full = read_rows(h / PR.PROPOSED)
        res = dict(photos_per_section=round(a0, 4), wall_s=wall, peak_gb_main_process=gb, jobs=a.jobs,
                   vs_record=compare_tables(full, record, a0), flagged=M.flagged_count(h / PR.REVIEW))
        slices, comps, step = inputs_of(T, S, a0, rec, reg, made, dataset, cfg)
        rp = record_pieces(T, rec)
        P = PR.outline_fit(comps, slices, a0, cfg)
        res["outline_step"] = dict(components=len(P), class_differs=sum(p["cls"] != rp[(p["section"], p["kind"])]["cls"] for p in P))
        Q = with_record_outlines(P, comps, slices, a0, vx, rp, a.jobs, cfg)
        PR.run_lines(Q, a0, step, cfg)
        res["with_record_outlines_vs_record"] = compare_tables(PR.table_rows(Q, i0, cfg), record, a0)
        # the registrations' thread count alone
        c2 = copy.deepcopy(cfg)
        c2["propose"]["registration"]["threads"] = 2
        N = [dict(p) for p in P]
        PR.add_scores(N, comps, slices, a0, vx * cfg["propose"]["photo_px_factor"] / 1000.0, max(a.jobs // 2, 1), c2, lambda s: None)
        PR.run_lines(N, a0, step, cfg)
        res["two_threads_vs_one"] = compare_tables(PR.table_rows(N, i0, cfg), full, a0)
        out[T] = res
        print(json.dumps({T: {k: res[k] for k in ("vs_record", "with_record_outlines_vs_record", "two_threads_vs_one")}}), flush=True)
    return out


def part_repeat(a, rec, dataset, T="MF204"):
    S, vz = RULE[T]
    a0 = 1200.0 / vz
    reg, made = regions_of(T, S, vz, rec)
    sc = subject_cfg(S)
    vx = float((sc.get("reconstruction_spacing_um") or sc)["vx"])
    w = Path(a.work)
    r1, _, _ = run_command(T, S, a, rec, w / "repeat1", a0, vx, reg, made, dataset)
    r2, _, _ = run_command(T, S, a, rec, w / "repeat2", a0, vx, reg, made, dataset)
    return dict(series=T, table_identical=filecmp.cmp(r1 / PR.PROPOSED, r2 / PR.PROPOSED, shallow=False),
                review_identical=filecmp.cmp(r1 / PR.REVIEW, r2 / PR.REVIEW, shallow=False))


def score_against_hand(rows, S, vz, hand, cfg):
    """rows: [(section, component, photograph on the run line, score, flagged)]. As on record: a component's hand-made photograph is its
    own column, else the section's first filled column; error in mm at the nominal step."""
    look = M.match_table(f"{hand}/num_{S}.txt", cfg)
    i0 = subject_cfg(S)["idx_range"][0]
    floor = cfg["propose"]["score_floor"]
    kept, dropped, big, big_flagged, n, flagged = [], [], 0, 0, 0, 0
    for sec, comp, line, score, flag in rows:
        n += 1
        flagged += bool(flag)
        ref = look(sec)
        c = ref[HAND_COLUMN[comp]]
        if c == -1:
            c = next((ref[HAND_COLUMN[k]] for k in ("left", "right", "whole", "cere") if ref[HAND_COLUMN[k]] != -1), -1)
        if c == -1 or line is None:
            continue
        err = abs(line - c) * vz / 1000.0
        if err > 1.2:
            big += 1
            big_flagged += bool(flag)
        (kept if (score or 0) >= floor else dropped).append(err)
    k = np.array(kept)
    return dict(components=n, flagged=flagged, kept=len(k), err_median_mm=round(float(np.median(k)), 3), wrong_kept=int((k > 1.2).sum()),
                dropped_by_floor=len(dropped), dropped_wrong=sum(x > 1.2 for x in dropped), large=big, large_flagged=big_flagged, i0=i0)


def pooled(per):
    keys = ("components", "flagged", "kept", "wrong_kept", "dropped_by_floor", "dropped_wrong", "large", "large_flagged")
    return {k: int(sum(v[k] for v in per.values())) for k in keys}


def part_saved(rec, hand):
    cfg = M.config()
    per = {}
    names = {"bottom": "cere"}
    for S, vz in FIVE.items():
        f = json.load(open(f"{rec}/labels/match_tables/proposed/{S}_regfit.json"))
        i0 = subject_cfg(S)["idx_range"][0]
        rows = [(p["section"], names.get(p["cls"], p["cls"]), None if p.get("idx") is None else p["idx"] + i0, p.get("score"), bool(p.get("flags")))
                for p in f["pieces"]]
        per[S] = score_against_hand(rows, S, vz, hand, cfg)
    return dict(pooled=pooled(per), per_brain=per, note="from the record's saved fit (20 Sep 2026): a 2D network's partition with whole "
                "components corrected by a 3D regions model, not yet the atlas regions as the classes")


def part_five(a, rec, dataset, hand):
    import csv
    cfg = M.config()
    per, per_rec = {}, {}
    for S, vz in FIVE.items():
        a0 = 1200.0 / vz
        sc = subject_cfg(S)
        vx = float((sc.get("reconstruction_spacing_um") or sc)["vx"])
        reg = f"{rec}/labels/regions_t2_final/{S}_regions_t2.nii.gz"
        made = (float(cfg["regions"]["made_at_spacing_um"][S]["vx"]), float(cfg["regions"]["made_at_spacing_um"][S]["vz"]))
        h, wall, gb = run_command(S, S, a, rec, Path(a.work) / "five", a0, vx, reg, made, dataset)
        rows = []
        for r in csv.DictReader(open(h / PR.REVIEW)):
            line = int(r["run_line_photograph"]) if r["run_line_photograph"] else None
            rows.append((int(r["section"]), r["component"], line, float(r["score"]) if r["score"] else None, bool(r["flag"])))
        per[S] = dict(score_against_hand(rows, S, vz, hand, cfg), wall_s=wall, peak_gb_main_process=gb)
        slices, comps, step = inputs_of(S, S, a0, rec, reg, made, dataset, cfg)
        Q = with_record_outlines(PR.outline_fit(comps, slices, a0, cfg), comps, slices, a0, vx, record_pieces(S, rec), a.jobs, cfg)
        PR.run_lines(Q, a0, step, cfg)
        per_rec[S] = score_against_hand([(p["section"], PR.COMPONENT[p["cls"]], None if p.get("idx") is None else p["idx"] + sc["idx_range"][0],
                                          p.get("score"), bool(p["flags"])) for p in Q], S, vz, hand, cfg)
        print(f"{S}: {json.dumps(per[S])}\n  with the record's outline step: {json.dumps(per_rec[S])}", flush=True)
    return dict(pooled=pooled(per), per_brain=per, with_record_outlines=dict(pooled=pooled(per_rec), per_brain=per_rec),
                note="compared with one person's hand-made tables, not with a known truth; kept excludes the components the floor dropped")


def part_edited(a, rec, dataset, T="MF191"):
    """match, three times on the same brain: the proposal; an edited copy with one row changed; --table over both."""
    S, vz = RULE[T]
    a0 = 1200.0 / vz
    reg, made = regions_of(T, S, vz, rec)
    sc = subject_cfg(S)
    vx = float((sc.get("reconstruction_spacing_um") or sc)["vx"])
    w = Path(a.work) / "edited"
    h, _, _ = run_command(T, S, a, rec, w, a0, vx, reg, made, dataset)
    prop = h / PR.PROPOSED
    lines = prop.read_text().split("\n")
    k = next(i for i, x in enumerate(lines) if i and x.split(",")[3] not in ("-1", "") and x.split(",")[0] != "-1")   # a whole-section row
    v = lines[k].split(",")
    section, old = int(v[0]), int(v[3])
    new = old + 2
    v[3] = str(new)
    lines[k] = ",".join(v)
    src = source_args(T, S, dataset, rec)

    def match_step(sub, table=None):
        out = h / "registration"
        matched = out / "matched"
        if matched.exists():
            shutil.rmtree(matched)
        cmd = [sys.executable, str(REPO / "cli/linc-histology"), "match", "--subject", S, "--store", f"{rec}/data/sub-{S}/blockface/{S}_blockface.ome.zarr",
               "--histology", src["jp2_dir"], "--out", str(out)] + (["--table", str(table)] if table else [])
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(r.stdout[-800:] + r.stderr[-800:])
        said = [x for x in r.stdout.split("\n") if x.startswith("match table:")]
        name = next(p for p in matched.glob("*_0000.png") if M.section_number(p.name) == section)
        return said[0] if said else "", name.read_bytes()

    said_prop, photo_prop = match_step("proposal")
    edited = h / M.EDITED
    edited.write_text("\n".join(lines))
    said_edit, photo_edit = match_step("edited")
    other = h / "other_table.txt"
    shutil.copy(prop, other)
    said_table, photo_table = match_step("table", other)
    # rerunning propose leaves an edited table alone
    before = edited.read_text()
    run_command(T, S, a, rec, w, a0, vx, reg, made, dataset)
    return dict(series=T, section=section, proposed_photograph=old, edited_photograph=new, uses_proposal_first="unreviewed proposal" in said_prop,
                edited_used="edited by hand" in said_edit, edited_photo_differs=photo_edit != photo_prop, table_flag_wins="(given)" in said_table,
                table_photo_equals_proposal=photo_table == photo_prop, edited_file_survives_rerun=edited.read_text() == before,
                said=[x.replace(str(w), "<work>") for x in (said_prop, said_edit, said_table)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--parts", default="rule,full,repeat,saved,five,edited")
    ap.add_argument("--linc-dataset", default=None)
    ap.add_argument("--hand-tables", help="the folder of the five brains' hand-made num_<ID>.txt (parts saved and five)")
    ap.add_argument("--json", default=str(REPO / "tests" / "match_propose_records.json"))
    a = ap.parse_args()
    rec = os.environ["LINC_RECORDS"]
    dataset = a.linc_dataset or os.path.join(os.environ["LINC_RAW"], "linc_000003")
    if {"saved", "five"} & set(a.parts.split(",")) and not a.hand_tables:
        sys.exit("parts saved and five need --hand-tables")
    paths.refuse_records(a.work)
    if (os.path.realpath(a.work) + os.sep).startswith(os.path.realpath(rec) + os.sep):
        sys.exit("--work must be outside $LINC_RECORDS")
    res = json.load(open(a.json)) if os.path.exists(a.json) else {}
    for part in a.parts.split(","):
        t0 = time.time()
        res[part] = (part_rule(rec) if part == "rule" else part_full(a, rec, dataset) if part == "full" else
                     part_repeat(a, rec, dataset) if part == "repeat" else part_saved(rec, a.hand_tables) if part == "saved" else
                     part_five(a, rec, dataset, a.hand_tables) if part == "five" else part_edited(a, rec, dataset))
        print(f"=== {part} done in {time.time() - t0:.0f} s", flush=True)
        Path(a.json).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k in ("repeat", "edited")}, indent=1))
    for k in ("saved", "five"):
        if k in res:
            print(k, json.dumps(res[k]["pooled"]))


if __name__ == "__main__":
    main()
