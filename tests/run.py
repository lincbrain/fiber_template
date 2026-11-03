#!/usr/bin/env python3
"""Run test files without pytest (the release's environments do not install it): every `test_*` function of each file, in order.

    python tests/run.py tests/test_segment.py [tests/test_completion.py ...]

A test that takes `tmp_path` gets a fresh temporary folder. When pytest is not installed, a stand-in provides the two pieces the
tests use (`pytest.raises`, `pytest.skip`). Prints PASS / SKIP / FAIL per test and a summary; the exit status is the number of
failures (0 = all passed). Runs the same tests as `python -m pytest <file>` where pytest exists.
"""
import contextlib
import importlib.util
import inspect
import sys
import tempfile
import time
import traceback
import types
from pathlib import Path


class Skipped(Exception):
    pass


def _stand_in_pytest():
    m = types.ModuleType("pytest")

    @contextlib.contextmanager
    def raises(exc):
        try:
            yield
        except exc:
            return
        raise AssertionError(f"{exc.__name__} not raised")

    def skip(reason=""):
        raise Skipped(reason)

    m.raises, m.skip = raises, skip
    m.skip.Exception = Skipped
    return m


def run_file(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(path).resolve().parent))
    spec.loader.exec_module(mod)
    counts = dict(passed=0, skipped=0, failed=0)
    for name, fn in list(vars(mod).items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        t0 = time.time()
        try:
            fn(*([Path(tempfile.mkdtemp(prefix=f"{name}_"))] if "tmp_path" in inspect.signature(fn).parameters else []))
            counts["passed"] += 1; print(f"PASS {Path(path).name}::{name} ({time.time() - t0:.1f} s)", flush=True)
        except Exception as e:                                                   # noqa: BLE001
            if type(e).__name__ in ("Skipped", "Skip") or "skip" in type(e).__module__.lower():
                counts["skipped"] += 1; print(f"SKIP {Path(path).name}::{name}: {e}", flush=True)
            else:
                counts["failed"] += 1; print(f"FAIL {Path(path).name}::{name}", flush=True); traceback.print_exc()
    return counts


def main(paths):
    try:
        import pytest  # noqa: F401
    except ImportError:
        sys.modules["pytest"] = _stand_in_pytest()
    total = dict(passed=0, skipped=0, failed=0)
    for p in paths:
        for k, v in run_file(p).items():
            total[k] += v
    print(f"{total['passed']} passed, {total['skipped']} skipped, {total['failed']} failed")
    return total["failed"]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or sorted(str(p) for p in Path(__file__).resolve().parent.glob("test_*.py"))))
