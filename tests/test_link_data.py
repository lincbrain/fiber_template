"""Tests of code/shared/link_data.py and of what settings/roots.sh checks before it builds anything (CPU, no data: toy folders).

    python -m pytest tests/test_link_data.py      or      python tests/run.py tests/test_link_data.py

You need
  settings/env/linc.yml (any Python 3 will do) and bash.
What it does
  It writes a toy table and toy data folders, and checks that the links are made once; that a folder holding none of the table's
  files is refused, with the folder that looks right named and nothing changed; that a second data folder is refused while a working
  folder built from the first exists; that files added to the data folder are linked on a refresh, and only then; and that
  settings/roots.sh stops on a wrong LINC_DATA before it makes anything, and with the word refresh makes the links again and fills
  the working folders again.
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
    assert out.strip() == "rc=1 tree=" and "set LINC_DATA to the data folder first" in err
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
    repo = tmp_path / "repo"                                       # the two files under test, the toy table, and stand-ins for the two helpers
    for rel in ("settings/roots.sh", "code/shared/link_data.py"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(REPO / rel), str(repo / rel))
    shutil.copy(str(table), str(repo / "settings" / "files.tsv"))
    calls = repo / "calls.txt"
    (repo / "code" / "shared" / "join_weights.sh").write_text("exit 0\n")
    helper = repo / "code" / "shared" / "make_local_data.sh"       # what the real one does that matters here: it writes <dest>/.complete last
    helper.write_text(f'#!/bin/bash\nmkdir -p "$1"\necho "$1" >> "{calls}"\necho done > "$1/.complete"\n')
    helper.chmod(0o755)
    out = repo / "outputs"

    stdout, err = source_roots(repo, LINC_DATA=tree)
    assert stdout.strip() == f"rc=0 tree={tree}", err
    assert calls.read_text().split() == [str(out / "local_data"), str(out / "local_rebuild"), str(out / "local_mri")]
    assert "building" in err and "(once; " in err
    mine = out / "local_data" / "my_result.txt"                    # something the user made in a working folder
    mine.write_text("mine")

    later = tree / ROWS[3][0]                                      # the data folder gains a file
    later.parent.mkdir(parents=True, exist_ok=True)
    later.write_text("later")
    stdout, err = source_roots(repo, LINC_DATA=tree)               # sourced again as usual: nothing changes
    assert stdout.strip() == f"rc=0 tree={tree}" and "building" not in err and "adding" not in err
    assert not os.path.lexists(str(out / "in" / ROWS[3][1])) and len(calls.read_text().split()) == 3

    stdout, err = source_roots(repo, "refresh", LINC_DATA=tree)    # with the word: the new file is linked, the working folders filled again
    assert stdout.strip() == f"rc=0 tree={tree}", err
    assert os.readlink(str(out / "in" / ROWS[3][1])) == f"{tree}/{ROWS[3][0]}"
    assert "the links are made again (3 links" in err and err.count("adding what is missing to") == 3 and "once" not in err
    assert len(calls.read_text().split()) == 6
    assert all((out / d / ".complete").exists() for d in ("local_data", "local_rebuild", "local_mri"))
    assert mine.read_text() == "mine"                              # nothing of the user's is touched

    stdout, err = source_roots(repo, LINC_DATA=tree)               # and afterwards it is quiet again
    assert "adding" not in err and len(calls.read_text().split()) == 6
