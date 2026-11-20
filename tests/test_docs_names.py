"""Every variable the documentation uses is explained where a reader looks for it (CPU, no data, standard library only).

    python -m pytest tests/test_docs_names.py      or      python tests/run.py tests/test_docs_names.py      or      python tests/test_docs_names.py

You need
  any Python 3.
What it does
  It reads README.md and every page of docs/ and collects each $NAME (capital letters, digits and _) used in a code block or between
  backticks. A name passes when the first column of a table row of docs/tutorials/00_start_here.md or docs/overview.md names it, when it
  is set in the same code block or in the same section, a # or ## heading with the ### headings under it (NAME=..., export NAME=...,
  for NAME in ..., a command that starts with NAME=...), or when the shell or the system sets it (HOME, PATH, ...).
You get
  pytest's pass / fail; on a failure, and when run as a script, one line per name that is not explained: file:line  $NAME.
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TABLES = ("docs/tutorials/00_start_here.md", "docs/overview.md")
SYSTEM = {"HOME", "PATH", "PWD", "USER", "SHELL", "TMPDIR", "HOSTNAME", "LD_LIBRARY_PATH", "PYTHONPATH", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS"}
USE = re.compile(r"\$\{?([A-Z][A-Z0-9_]*)")
SET = re.compile(r"(?:^|[\s;&|(`])(?:export\s+|for\s+|read\s+)?([A-Z][A-Z0-9_]*)(?:=|\s+in\s)")
FENCE = re.compile(r"^\s*(```|~~~)")
TICKS = re.compile(r"(`+)(.+?)\1")


def pages():
    return [REPO / "README.md"] + sorted((REPO / "docs").rglob("*.md"))


def parts(path):
    """(line number, text, block id, section id) for each piece of code in a page: a code block's lines and the backtick spans"""
    out, block, section, fence = [], None, 0, None
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        m = FENCE.match(line)
        if m and fence is None:
            fence, block = m.group(1), f"block{i}"
            continue
        if fence is not None:
            if line.strip().startswith(fence):
                fence, block = None, None
            else:
                out.append((i, line, block, section))
            continue
        if re.match(r"##? ", line):                                               # a # or ## heading starts a section; ### stays in it
            section = i
        out += [(i, t.group(2), None, section) for t in TICKS.finditer(line)]
    return out


def explained():
    """the names the first column of the two pages' tables gives"""
    names = set()
    for rel in TABLES:
        for line in (REPO / rel).read_text().splitlines():
            cells = line.strip().split("|")
            if line.strip().startswith("|") and len(cells) > 2 and not set(cells[1].strip()) <= set("-: "):
                names |= set(USE.findall(cells[1]))
    return names


def unexplained():
    known = explained() | SYSTEM
    miss = []
    for path in pages():
        P = parts(path)
        in_block, in_section = {}, {}
        for _, text, block, section in P:
            for n in SET.findall(text):
                in_section.setdefault(section, set()).add(n)
                if block:
                    in_block.setdefault(block, set()).add(n)
        for i, text, block, section in P:
            for n in USE.findall(text):
                if n not in known and n not in in_block.get(block, ()) and n not in in_section.get(section, ()):
                    miss.append(f"{path.relative_to(REPO)}:{i}  ${n}")
    return sorted(set(miss), key=lambda s: (s.split(":")[0], int(s.split(":")[1].split()[0]), s))


def test_every_name_is_explained():
    miss = unexplained()
    assert not miss, (f"{len(miss)} uses of a name that no table of {' or '.join(TABLES)} gives and that its code block or section does not set:\n"
                      + "\n".join(miss))


if __name__ == "__main__":
    miss = unexplained()
    print("\n".join(miss) if miss else "every name is explained")
    sys.exit(1 if miss else 0)
