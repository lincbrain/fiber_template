"""Tests of the documentation site's settings (mkdocs.yml) against the pages of docs/ (CPU, no data, MkDocs not needed).

    python -m pytest tests/test_docs_site.py      or      python tests/run.py tests/test_docs_site.py

You need
  settings/env/linc.yml.
What it does
  It checks that every page the site's menu names exists, that every page of docs/ is in the menu (the short README.md notes of the
  subfolders apart), and that every relative link of every page leads to a file of the repository: inside docs/ to a page or an image,
  outside docs/ to the file that code/docs/site_links.py sends the link to on GitHub.
You get
  pytest's pass / fail per test.
"""
import importlib.util
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def site():
    return yaml.safe_load((REPO / "mkdocs.yml").read_text())


def menu(nav):
    """every page a menu names, in order"""
    out = []
    for item in nav:
        for value in (item.values() if isinstance(item, dict) else [item]):
            out += menu(value) if isinstance(value, list) else [value]
    return out


def hook():
    spec = importlib.util.spec_from_file_location("site_links", REPO / "code" / "docs" / "site_links.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_menu_names_every_page():
    cfg = site()
    assert cfg["docs_dir"] == "docs" and all((REPO / h).is_file() for h in cfg["hooks"])
    pages = menu(cfg["nav"])
    assert pages and all((DOCS / p).is_file() for p in pages), [p for p in pages if not (DOCS / p).is_file()]
    notes = {p.relative_to(DOCS).as_posix() for p in DOCS.rglob("README.md")} - {"README.md"}     # a subfolder's own note
    every = {p.relative_to(DOCS).as_posix() for p in DOCS.rglob("*.md")}
    assert every - notes == set(pages), sorted((every - notes) ^ set(pages))


def test_every_relative_link_leads_to_a_file():
    outside, broken, sent = hook().outside, [], 0
    for page in sorted(DOCS.rglob("*.md")):
        rel = page.relative_to(DOCS).as_posix()
        text = re.sub(r"```.*?```", "", page.read_text(encoding="utf-8"), flags=re.S)
        for url in LINK.findall(text):
            if re.match(r"^(https?:|mailto:|#)", url):
                continue
            target = outside(rel, url)
            if target is None:                                     # inside docs/: a page or an image beside this one
                ok = (page.parent / url.split("#")[0]).exists()
            else:                                                  # outside: the file the link is sent to on GitHub
                ok = (REPO / target.split("#")[0]).exists()
                sent += 1
            if not ok:
                broken.append(f"{rel}: {url}")
    assert not broken, broken
    assert sent > 0                                                # the pages do link to the code, so the rule is exercised
    assert outside("tutorials/01_mri.md", "../overview.md#the-data-folder") is None
    assert outside("tutorials/01_mri.md", "../../tests/README.md") == "tests/README.md"
    assert outside("README.md", "https://dandiarchive.org/dandiset/001372") is None
