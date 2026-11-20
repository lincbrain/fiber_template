"""Tests of code/shared/link_data.py and of what settings/roots.sh checks before it builds anything (CPU, no data: toy folders).

    python -m pytest tests/test_link_data.py      or      python tests/run.py tests/test_link_data.py

You need
  settings/env/linc.yml (any Python 3 will do) and bash.
What it does
  It writes a toy table and toy data folders, and checks that the links are made once; that a folder holding none of the table's
  files is refused, with the folder that looks right named and nothing changed; that a second data folder is refused while a working
  folder built from the first exists; that files added to the data folder are linked on a refresh, and only then; and that
  settings/roots.sh stops on a wrong LINC_DATA before it makes anything, and with the word refresh makes the links again and fills
  the working folders again, and writes outputs/README.txt (from settings/outputs_README.txt) when it builds outputs/ and on a refresh
  only; that a changed table or a refresh updates the links in place (the folder and every unchanged link keep their inode, a changed
  row is replaced, a removed row's link goes), that an interrupted update is redone in place, and that a file read through the links
  all the while a refresh runs is never missing.
You get
  pytest's pass / fail per test.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "code" / "shared" / "link_data.py"
ROWS = [("brains/sub-A/inputs/mri/A_b0.nii.gz", "data/sub-A/mri/b0.nii.gz"),                       # (new_path, old_path)
        ("brains/sub-A/outputs/A_blockface.ome.zarr", "data/sub-A/blockface/A_blockface.ome.zarr"),
        ("template/outputs/template_T2w.nii.gz", "template/t2.nii.gz"),
        ("paper/numbers/x.json", "paper/numbers/x.json")]


def toy(tmp_path, name, keep=(0, 1, 2)):
    """a toy table, and a data folder holding the rows in keep (row 1 is a Zarr store, a folder)"""
    table = tmp_path / "files.tsv"
    table.write_text("new_path\told_path\tkind\n" + "".join(f"{new}\t{old}\tx\n" for new, old in ROWS))
    tree = tmp_path / name
    for k in keep:
        f = tree / ROWS[k][0]
        f.parent.mkdir(parents=True, exist_ok=True)
        if f.name.endswith(".zarr"):
            f.mkdir()
            (f / "c").write_text(name)
        else:
            f.write_text(name)
    return table, tree.resolve()


def link(tree, links, table, *more):
    return subprocess.run([sys.executable, str(TOOL), str(tree), str(links), "--table", str(table)] + [str(m) for m in more],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)


def test_links_made_once(tmp_path):
    table, tree = toy(tmp_path, "tree")
    links = tmp_path / "out" / "in"
    r = link(tree, links, table)
    assert r.returncode == 0, r.stderr
    assert "3 links" in r.stdout and "1 files of the table are not in this tree" in r.stdout
    for new, old in ROWS[:3]:
        assert os.readlink(str(links / old)) == f"{tree}/{new}"
    assert (links / ROWS[1][1]).is_symlink()                       # a Zarr store is one link
    assert not os.path.lexists(str(links / ROWS[3][1]))            # a file the folder does not hold gets no link
    assert "up to date" in link(tree, links, table).stdout


def test_a_folder_inside_the_data_folder_is_refused(tmp_path):
    table, tree = toy(tmp_path, "tree")
    links = tmp_path / "out" / "in"
    r = link(tree / "brains", links, table)                        # one level too deep
    assert r.returncode != 0 and "is not the data folder" in r.stderr and f"{tree} looks like it" in r.stderr
    assert not links.exists()                                      # nothing made
    assert link(tree, links, table).returncode == 0
    stamp = (links / ".complete").read_text()
    (tmp_path / "empty").mkdir()
    r = link(tmp_path / "empty", links, table)                     # no data folder anywhere near
    assert r.returncode != 0 and "is not the data folder" in r.stderr and "looks like it" not in r.stderr
    assert (links / ".complete").read_text() == stamp              # the links of the good folder are left alone
    assert os.readlink(str(links / ROWS[0][1])) == f"{tree}/{ROWS[0][0]}"


def test_another_data_folder_is_refused_while_built_folders_exist(tmp_path):
    table, a = toy(tmp_path, "a")
    _, b = toy(tmp_path, "b", keep=(0, 2))
    links, built = tmp_path / "out" / "in", tmp_path / "out" / "local_data"
    assert link(a, links, table, "--built", built).returncode == 0
    built.mkdir()                                                  # a working folder made from a's links
    r = link(b, links, table, "--built", built)
    assert r.returncode != 0 and f"was built from another data folder, {a}" in r.stderr and "local_data" in r.stderr
    assert os.readlink(str(links / ROWS[0][1])) == f"{a}/{ROWS[0][0]}"       # the links still lead to a
    assert "up to date" in link(a, links, table, "--built", built).stdout    # going back to a is fine
    shutil.rmtree(str(built))                                      # nothing built from a's links: it starts over with b
    r = link(b, links, table, "--built", built)
    assert r.returncode == 0 and "2 links" in r.stdout, r.stderr
    assert os.readlink(str(links / ROWS[0][1])) == f"{b}/{ROWS[0][0]}"
    built.mkdir()
    shutil.rmtree(str(b))                                          # b is gone (the folder moved): no reason to refuse a
    assert link(a, links, table, "--built", built).returncode == 0


def test_files_added_to_the_folder_are_linked_on_a_refresh(tmp_path):
    table, tree = toy(tmp_path, "tree", keep=(0, 2))
    links, built = tmp_path / "out" / "in", tmp_path / "out" / "local_data"
    assert "2 links" in link(tree, links, table, "--built", built).stdout
    built.mkdir()                                                  # a working folder made from these links
    for k in (1, 3):                                               # the folder gains a Zarr store and a file
        f = tree / ROWS[k][0]
        f.parent.mkdir(parents=True, exist_ok=True)
        f.mkdir() if f.name.endswith(".zarr") else f.write_text("later")
    assert "up to date" in link(tree, links, table, "--built", built).stdout        # the same table and folder: it cannot see them
    assert not os.path.lexists(str(links / ROWS[3][1]))
    r = link(tree, links, table, "--built", built, "--refresh")
    assert r.returncode == 0 and "4 links" in r.stdout and "0 files of the table are not in this tree" in r.stdout, r.stderr
    for new, old in ROWS:
        assert os.readlink(str(links / old)) == f"{tree}/{new}"
    _, other = toy(tmp_path, "other")                              # a refresh does not let another data folder in
    r = link(other, links, table, "--built", built, "--refresh")
    assert r.returncode != 0 and f"was built from another data folder, {tree}" in r.stderr
    assert os.readlink(str(links / ROWS[0][1])) == f"{tree}/{ROWS[0][0]}"


def source_roots(repo, word="", **env):
    """source <repo>/settings/roots.sh [word] in a bare shell: (stdout with rc= and tree=, stderr)"""
    r = subprocess.run(["bash", "-c", f'source "{repo}/settings/roots.sh" {word}; echo "rc=$? tree=${{LINC_DATA_TREE:-}}"'],
                       env=dict(PATH=os.environ.get("PATH", "/usr/bin:/bin"), PY=sys.executable, **{k: str(v) for k, v in env.items()}),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    return r.stdout, r.stderr


def test_roots_stops_on_a_wrong_data_folder(tmp_path):
    table, tree = toy(tmp_path, "tree")
    repo = tmp_path / "repo"                                       # the two files under test, with the toy table
    for rel in ("settings/roots.sh", "code/shared/link_data.py"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(REPO / rel), str(repo / rel))
    shutil.copy(str(table), str(repo / "settings" / "files.tsv"))

    out, err = source_roots(repo)                                  # LINC_DATA not set
    assert out.strip() == "rc=1 tree=" and "set LINC_DATA_TREE to the data folder first" in err
    assert not (repo / "outputs").exists()

    out, err = source_roots(repo, LINC_DATA=tree / "brains")       # one level too deep
    assert out.strip() == "rc=1 tree=" and "is not the data folder" in err and f"{tree} looks like it" in err

    # an earlier shell's LINC_DATA_TREE does not hide a wrong LINC_DATA: the folder you name is the one checked
    out, err = source_roots(repo, LINC_DATA=tree / "brains", LINC_DATA_TREE=tree)
    assert out.strip() == f"rc=1 tree={tree}" and f"{tree}/brains is not the data folder" in err

    # in a shell that sourced another clone, LINC_DATA is that clone's working folder, and the folder kept in LINC_DATA_TREE is checked
    out, err = source_roots(repo, LINC_DATA="/another/clone/outputs/local_data/data", LINC_DATA_TREE=tree / "brains")
    assert out.strip() == f"rc=1 tree={tree}/brains" and f"{tree}/brains is not the data folder" in err

    for built in ("in", "local_data", "local_rebuild", "local_mri", "models"):
        assert not (repo / "outputs" / built).exists()             # it stopped before making anything


def test_roots_refresh_links_again_and_fills_the_working_folders_again(tmp_path):
    table, tree = toy(tmp_path, "tree", keep=(0, 2))
    repo = tmp_path / "repo"                                       # the files under test, the toy table, and stand-ins for the two helpers
    for rel in ("settings/roots.sh", "code/shared/link_data.py", "settings/outputs_README.txt"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(REPO / rel), str(repo / rel))
    shutil.copy(str(table), str(repo / "settings" / "files.tsv"))
    calls = repo / "calls.txt"
    (repo / "code" / "shared" / "join_weights.sh").write_text("exit 0\n")
    helper = repo / "code" / "shared" / "make_local_data.sh"       # what the real one does that matters here: it writes <dest>/.complete last
    args = repo / "args.txt"                                        # every argument of each call, one call per line
    helper.write_text(f'#!/bin/bash\nmkdir -p "$1"\necho "$1" >> "{calls}"\necho "$@" >> "{args}"\necho done > "$1/.complete"\n')
    helper.chmod(0o755)
    out = repo / "outputs"

    stdout, err = source_roots(repo, LINC_DATA=tree)
    assert stdout.strip() == f"rc=0 tree={tree}", err
    assert calls.read_text().split() == [str(out / "local_data"), str(out / "local_rebuild")]   # the toy holds no template brains:
    assert "building" in err and "(once; " in err and "FAILED" not in err                       # local_mri is not made, and it says so once
    assert "outputs/local_mri is not made: the data folder does not hold the brain masks of MF283 MF284 MF285 MF286" in err
    first = args.read_text().splitlines()                          # built with the brains each folder starts with
    assert first[0].startswith(f"{out / 'local_data'} MR252 MF191 MF283 --raw ")
    assert first[1].startswith(f"{out / 'local_rebuild'} --rebuild MR252 --rebuild MF191 --rebuild MR263 --raw ")
    readme = (REPO / "settings" / "outputs_README.txt").read_text()
    assert (out / "README.txt").read_text() == readme              # outputs/README.txt, written when outputs/ is built
    (out / "README.txt").write_text("an older one")
    mine = out / "local_data" / "my_result.txt"                    # something the user made in a working folder
    mine.write_text("mine")
    for d in ("local_data/data/sub-MR252", "local_data/data/sub-MR243", "local_rebuild/data/sub-MR252", "local_rebuild/data/sub-MR256"):
        (out / d).mkdir(parents=True)                              # brains in the working folders: two they start with, two added since

    later = tree / ROWS[3][0]                                      # the data folder gains a file
    later.parent.mkdir(parents=True, exist_ok=True)
    later.write_text("later")
    stdout, err = source_roots(repo, LINC_DATA=tree)               # sourced again as usual: nothing changes
    assert stdout.strip() == f"rc=0 tree={tree}" and "building" not in err and "adding" not in err and "not made" not in err
    assert not os.path.lexists(str(out / "in" / ROWS[3][1])) and len(calls.read_text().split()) == 2
    assert (out / "README.txt").read_text() == "an older one"

    stdout, err = source_roots(repo, "refresh", LINC_DATA=tree)    # with the word: the new file is linked, the working folders filled again
    assert stdout.strip() == f"rc=0 tree={tree}", err
    assert os.readlink(str(out / "in" / ROWS[3][1])) == f"{tree}/{ROWS[3][0]}"
    assert "the links are made again (3 links" in err and err.count("adding what is missing to") == 2 and "once" not in err
    assert len(calls.read_text().split()) == 4 and "not made" not in err
    again = args.read_text().splitlines()                          # a brain added to a working folder since is filled again too, once
    assert again[2].startswith(f"{out / 'local_data'} MR252 MF191 MF283 MR243 --raw ")
    assert again[3].startswith(f"{out / 'local_rebuild'} --rebuild MR252 --rebuild MF191 --rebuild MR263 --rebuild MR256 --raw ")
    assert all((out / d / ".complete").exists() for d in ("local_data", "local_rebuild"))
    assert mine.read_text() == "mine"                              # nothing of the user's is touched
    assert (out / "README.txt").read_text() == readme              # the README is written again on a refresh

    stdout, err = source_roots(repo, LINC_DATA=tree)               # and afterwards it is quiet again
    assert "adding" not in err and len(calls.read_text().split()) == 4

    stdout, err = source_roots(repo, LINC_DATA_TREE=tree)          # the documented way: the data folder in LINC_DATA_TREE, the same
    assert stdout.strip() == f"rc=0 tree={tree}" and "building" not in err and len(calls.read_text().split()) == 4, err
    stdout, err = source_roots(repo, LINC_DATA=out / "local_data" / "data", LINC_DATA_TREE=tree)   # a shell that sourced it before
    assert stdout.strip() == f"rc=0 tree={tree}" and "building" not in err, err
    assert "data folder" not in err                                # not interactive: no line naming the folders
    r = subprocess.run(["bash", "--norc", "-i", "-c", f'source "{repo}/settings/roots.sh"; echo "data=$LINC_DATA"'],   # interactive: one line
                       env=dict(PATH=os.environ.get("PATH", "/usr/bin:/bin"), PY=sys.executable, LINC_DATA_TREE=str(tree)),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    assert f"settings/roots.sh: data folder {tree}; working folder {out / 'local_data' / 'data'}" in r.stderr, r.stderr
    assert r.stdout.strip() == f"data={out / 'local_data' / 'data'}"


def test_a_changed_table_updates_the_links_in_place(tmp_path):
    """after a `git pull` that changes the table, the links folder and every unchanged link stay as they are (same inode), a changed
    row is replaced, a removed row's link goes, a new row's link comes"""
    table, tree = toy(tmp_path, "tree", keep=(0, 1, 2, 3))
    links = tmp_path / "out" / "in"
    assert "4 links" in link(tree, links, table).stdout
    keep_ino = os.lstat(str(links / ROWS[0][1])).st_ino; dir_ino = os.stat(str(links)).st_ino
    (tree / "template/outputs/template_T2w_v2.nii.gz").write_text("v2")
    (tree / "brains/sub-A/inputs/mri/A_t2.nii.gz").write_text("t2")
    rows = [ROWS[0], ROWS[1], ("template/outputs/template_T2w_v2.nii.gz", ROWS[2][1]),          # row 2 now gives another file
            ("brains/sub-A/inputs/mri/A_t2.nii.gz", "data/sub-A/mri/t2.nii.gz")]               # row 3 went, a new row came
    table.write_text("new_path\told_path\tkind\n" + "".join(f"{n}\t{o}\tx\n" for n, o in rows))
    r = link(tree, links, table)
    assert r.returncode == 0 and "updated in place: 1 made, 1 changed, 1 removed, 2 unchanged" in r.stdout, r.stdout + r.stderr
    assert os.stat(str(links)).st_ino == dir_ino and os.lstat(str(links / ROWS[0][1])).st_ino == keep_ino
    assert os.readlink(str(links / ROWS[2][1])) == f"{tree}/template/outputs/template_T2w_v2.nii.gz"
    assert not os.path.lexists(str(links / ROWS[3][1])) and not os.path.exists(str(links / "paper"))   # its empty folders go too
    assert os.readlink(str(links / "data/sub-A/mri/t2.nii.gz")) == f"{tree}/brains/sub-A/inputs/mri/A_t2.nii.gz"
    assert "up to date" in link(tree, links, table).stdout


def test_an_interrupted_update_is_redone_in_place(tmp_path):
    table, tree = toy(tmp_path, "tree")
    links = tmp_path / "out" / "in"
    link(tree, links, table)
    dir_ino = os.stat(str(links)).st_ino
    stamp = links / ".complete"
    lines = stamp.read_text().split("\n")
    stamp.write_text("\n".join(["updating"] + lines[1:]))           # what an update leaves when it is stopped halfway
    r = link(tree, links, table)
    assert r.returncode == 0 and "updated in place" in r.stdout and os.stat(str(links)).st_ino == dir_ino
    assert stamp.read_text().split("\n")[0] != "updating"


def test_a_running_step_keeps_its_files_during_a_refresh(tmp_path):
    """a reader opening a file through the links all the while `--refresh` runs (several times, on a table of 3,000 rows) never
    finds it missing; the code before 19 Nov 2026 removed the whole folder and made every link again"""
    import threading
    table = tmp_path / "files.tsv"; tree = (tmp_path / "tree").resolve()
    rows = [(f"brains/sub-A/inputs/f{i:04d}.txt", f"data/sub-A/f{i:04d}.txt") for i in range(3000)]
    for new, _ in rows:
        f = tree / new; f.parent.mkdir(parents=True, exist_ok=True); f.write_text("x")
    table.write_text("new_path\told_path\tkind\n" + "".join(f"{n}\t{o}\tx\n" for n, o in rows))
    links = tmp_path / "out" / "in"
    link(tree, links, table)
    missing, stop = [], threading.Event()

    def read():
        p = str(links / rows[1500][1])
        while not stop.is_set():
            try:
                open(p).read()
            except FileNotFoundError:
                missing.append(1)
    t = threading.Thread(target=read); t.start()
    try:
        for _ in range(4):
            assert link(tree, links, table, "--refresh").returncode == 0
    finally:
        stop.set(); t.join()
    assert not missing, f"the file was missing {len(missing)} times during the refreshes"


def test_roots_mri_folder_is_made_from_the_template_brains_and_its_failure_is_loud(tmp_path):
    """with the four template brains' masks in the data folder, local_mri is built; a failure of that build is said as one"""
    rows = list(ROWS[:3])
    for S in ("MF283", "MF284", "MF285", "MF286"):
        for K in ("dwi", "t2", "micro"):
            rows.append((f"brains/sub-{S}/inputs/mri/{K}_brain_mask.nii.gz", f"mri/data/sub-{S}/masks/{K}_brain_mask.nii.gz"))
        rows.append((f"brains/sub-{S}/inputs/mri/final_mask.nii.gz", f"mri/data/sub-{S}/canonical_rebuild_manual_annotations_frac/model_mask_source_uint8.nii.gz"))
    tree = tmp_path / "tree"; table = tmp_path / "files.tsv"
    table.write_text("new_path\told_path\tkind\n" + "".join(f"{new}\t{old}\tx\n" for new, old in rows))
    for new, _ in rows:
        f = tree / new; f.parent.mkdir(parents=True, exist_ok=True)
        (f.mkdir() if new.endswith(".zarr") else f.write_text("x"))
    repo = tmp_path / "repo"
    for rel in ("settings/roots.sh", "code/shared/link_data.py"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(REPO / rel), str(repo / rel))
    shutil.copy(str(table), str(repo / "settings" / "files.tsv"))
    (repo / "code" / "shared" / "join_weights.sh").write_text("exit 0\n")
    helper = repo / "code" / "shared" / "make_local_data.sh"       # the MRI folder's build fails; the others succeed
    helper.write_text('#!/bin/bash\ncase "$1" in *local_mri) echo "boom: a copy failed"; exit 1 ;; esac\nmkdir -p "$1"\necho done > "$1/.complete"\n')
    helper.chmod(0o755)
    stdout, err = source_roots(repo, LINC_DATA_TREE=tree)
    assert stdout.strip() == f"rc=0 tree={tree}", err
    assert "building" in err and "local_mri" in err and "not made" not in err
    assert "settings/roots.sh: FAILED: boom: a copy failed" in err
