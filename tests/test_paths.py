"""Tests of shared.paths.rerooted and the placeholder table (CPU, no data; synthetic prefixes only).

    python -m pytest tests/test_paths.py      or      python tests/test_paths.py

You need
  settings/env/linc.yml.
What it does
  It checks both recorded forms (a prefix of the machine that made the products, a $LINC_<ROOT> placeholder), the pass-through,
  that the local table is preferred over the committed example, and the clear failure without one.
You get
  pytest's pass / fail per test.
"""
import contextlib
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from shared import paths  # noqa: E402

ANALYSIS, RAW = "/example/analysis", "/example/raw"
TABLE = [(ANALYSIS, "$LINC_PRODUCTS"), (f"{RAW}/blockface_curated", "$LINC_RAW/blockface_curated")]


@contextlib.contextmanager
def roots(**kw):
    """set exactly these LINC_* roots for the block (every other root unset), then restore the environment"""
    old = {k: os.environ.get(k) for k in paths.ROOTS}
    try:
        for k in paths.ROOTS:
            os.environ.pop(k, None)
        os.environ.update(kw)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def local_table(lines):
    """point shared.paths at a temporary local table (None: no local table) for the block"""
    old = paths.PLACEHOLDERS_LOCAL
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "path_placeholders.local.yaml"
        if lines is not None:
            f.write_text("".join(f'- ["{a}", "{b}"]\n' for a, b in lines))
        paths.PLACEHOLDERS_LOCAL = f
        try:
            yield
        finally:
            paths.PLACEHOLDERS_LOCAL = old


def test_recorded_prefix():
    with roots(LINC_PRODUCTS="/p", LINC_RAW="/r"):
        assert paths.rerooted(f"{ANALYSIS}/data/sub-MR243/seam.json", table=TABLE) == "/p/data/sub-MR243/seam.json"
        assert paths.rerooted(ANALYSIS, table=TABLE) == "/p"
        assert paths.rerooted(f"{RAW}/blockface_curated/MR243/x.png", table=TABLE) == "/r/blockface_curated/MR243/x.png"
        assert paths.rerooted(Path(f"{ANALYSIS}/a"), table=TABLE) == "/p/a"            # a Path in, a str out


def test_placeholder():
    with roots(LINC_PRODUCTS="/p", LINC_DATA="/d", LINC_TEMPLATE="/t", LINC_RAW="/r", LINC_MODELS="/m", LINC_NUMBERS="/n", LINC_WORK="/w"):
        for var, base in (("PRODUCTS", "/p"), ("DATA", "/d"), ("TEMPLATE", "/t"), ("RAW", "/r"), ("MODELS", "/m"), ("NUMBERS", "/n"),
                          ("WORK", "/w")):
            assert paths.rerooted(f"$LINC_{var}/a/b.nii.gz", table=TABLE) == f"{base}/a/b.nii.gz"
            assert paths.rerooted(f"$LINC_{var}", table=TABLE) == base
    with roots():                                                                   # roots with a default: numbers, work
        assert paths.rerooted("$LINC_NUMBERS/x.json", table=TABLE) == "numbers/x.json"
        assert paths.rerooted("$LINC_WORK/y", table=TABLE) == "work/y"
        try:
            paths.rerooted("$LINC_DATA/sub-MR243", table=TABLE)                   # no default: the configured-root error
            raise AssertionError("an unset root must raise")
        except RuntimeError as e:
            assert "LINC_DATA" in str(e)


def test_neither_form_unchanged():
    with roots(LINC_PRODUCTS="/p"):
        for p in ("/some/other/place/file.json", "data/sub-MR243/seam.json", "$LINC_FOO/x", "$LINC_PRODUCTSX/a", "$HOME/a", "LINC_PRODUCTS/a",
                  f"{ANALYSIS}X/a", ""):
            assert paths.rerooted(p, table=TABLE) == p, p


def test_local_table_preferred():
    with roots(LINC_PRODUCTS="/p"), local_table(TABLE):
        assert paths.placeholder_table() == TABLE
        assert paths.rerooted(f"{ANALYSIS}/a/b") == "/p/a/b"
        assert paths.rerooted("/some/other/place/file.json") == "/some/other/place/file.json"     # as before: passed through
    with local_table(None):                                                         # the committed example: matches no path
        assert paths.placeholder_table() and all(a.startswith("<") for a, _ in paths.placeholder_table())


def test_no_local_table_fails_clearly():
    with roots(LINC_PRODUCTS="/p", LINC_RAW="/r"), local_table(None):
        for p in (f"{ANALYSIS}/data/sub-MR243/seam.json", "/some/other/place/file.json"):
            try:
                paths.rerooted(p)
                raise AssertionError(f"{p}: a recorded path that cannot be mapped must raise")
            except RuntimeError as e:
                assert p in str(e) and "path_placeholders.local.yaml" in str(e)
        assert paths.rerooted("/p/data/x") == "/p/data/x"                           # inside a configured root: fine
        assert paths.rerooted("$LINC_RAW/blockface_curated/x") == "/r/blockface_curated/x"
        assert paths.rerooted("data/x") == "data/x"                                 # relative: unchanged


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(); print("ok", name)
