#!/usr/bin/env python3
"""Write every README.txt of the repository from the README.md beside it, so that the two never drift.

The README.md is the source: GitHub shows it. The README.txt is the same text, plain.

    python3 code/docs/readme_txt.py            checks: lists every README.txt that differs from its README.md, exits 1 if any
    python3 code/docs/readme_txt.py --write    writes them

The Markdown of a README is kept to what a plain text file can say, one rule each:

    # Title, ## Heading, ### Heading    the heading in capitals (text in parentheses as written), without the #
    a fenced block (```)                its lines, indented 4 spaces
    a table                             its columns, indented 4 spaces, each as wide as written in the .md (the
                                        last one wrapped at 100 characters); a header row of empty cells is left out
    - item, 1. item                     indented 4 spaces (- dropped); a continuation line, indented 2 or 3, 8 spaces
    `code`, a backslash escape          the backticks and the backslash dropped
    [text](link)                        the text

Every other line, and every blank line, is copied as it is. No other Markdown is used in the READMEs.
"""
import argparse
import re
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WIDTH = 100
ESC = re.compile(r"\\([\\`*_{}\[\]<>()#+\-.!|])")


LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")


def plain(s):
    """Inline Markdown to plain text: a link becomes its text; backticks and backslash escapes are dropped."""
    return ESC.sub(r"\1", LINK.sub(r"\1", s).replace("`", ""))


def heading(s):
    """A heading's text in capitals, but for the text in parentheses."""
    out, depth = [], 0
    for ch in plain(s):
        depth += ch == "("
        out.append(ch if depth else ch.upper())
        depth -= ch == ")" and depth > 0
    return "".join(out)


def cells(line):
    """The cells of a table row, each as written between '| ' and ' |' (so its padding is kept)."""
    body = line.rstrip("\n")[1:]
    parts, cur, i = [], "", 0
    while i < len(body):
        if body[i] == "\\" and i + 1 < len(body):
            cur += body[i:i + 2]
            i += 2
            continue
        if body[i] == "|":
            parts.append(cur)
            cur = ""
        else:
            cur += body[i]
        i += 1
    return [p[1:-1] if p.startswith(" ") and p.endswith(" ") else p.strip() for p in parts]


def table(rows):
    """A Markdown table as plain columns."""
    head, body = rows[0], rows[2:]
    out = []
    for r in ([head] if any(c.strip() for c in head) else []) + body:
        lead = "    " + "".join(plain(c) for c in r[:-1])
        last = plain(r[-1]).strip()
        if not last:
            out.append(lead.rstrip())
            continue
        width = max(WIDTH - len(lead), 20)
        parts = textwrap.wrap(last, width, break_long_words=False, break_on_hyphens=False)
        out.append(lead + parts[0])
        out += [" " * len(lead) + p for p in parts[1:]]
    return out


def txt_of(md):
    """The README.txt of a README.md's text."""
    lines, out, i = md.split("\n"), [], 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                out.append(("    " + lines[i]) if lines[i].strip() else "")
                i += 1
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(cells(lines[i]))
                i += 1
            out += table(rows)
            continue
        m = re.match(r"#{1,6} (.*)$", line)
        if m:
            out.append(heading(m.group(1)))
        elif re.match(r"- ", line):
            out.append("    " + plain(line[2:]))
        elif re.match(r"\d+\. ", line):
            out.append("    " + plain(line))
        elif re.match(r" {2,3}\S", line) and out and out[-1].startswith("    "):
            out.append("        " + plain(line.strip()))
        else:
            out.append(plain(line))
        i += 1
    return "\n".join(out)


def readmes():
    """Every README.md the repository tracks, and any new one beside it."""
    tracked = subprocess.run(["git", "-C", str(REPO), "ls-files", "--cached", "--others", "--exclude-standard", "*README.md"],
                             stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout.split()
    return [REPO / p for p in sorted(tracked)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--write", action="store_true", help="write every README.txt from its README.md")
    a = ap.parse_args()
    differ = []
    for md in readmes():
        txt = md.with_name("README.txt")
        new = txt_of(md.read_text(encoding="utf-8"))
        if not txt.exists() or txt.read_text(encoding="utf-8") != new:
            differ.append(txt)
            if a.write:
                txt.write_text(new, encoding="utf-8")
    for t in differ:
        print(("written: " if a.write else "differs from its README.md: ") + str(t.relative_to(REPO)))
    print(f"{len(readmes())} README.md, {len(differ)} README.txt {'written' if a.write else 'differ'}")
    return 0 if a.write or not differ else 1


if __name__ == "__main__":
    sys.exit(main())
