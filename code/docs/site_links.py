"""For the documentation site (mkdocs.yml at the top of the repository): a link that leaves docs/, to the code, the settings or the
tests, goes to that file in the repository on GitHub.

You need
    nothing: MkDocs loads this file itself (mkdocs.yml, `hooks`).
What it does
    The Markdown files keep their relative links, which GitHub follows as they are. A site built from docs/ holds only docs/, so
    on each page a link whose target is outside docs/ is rewritten to https://github.com/lincbrain/fiber_template/blob/main/<path>.
    Links inside docs/, links to other sites and images are left alone.
You get
    pages whose every link leads somewhere.
"""
import posixpath
import re

REPO = "https://github.com/lincbrain/fiber_template/blob/main/"
LINK = re.compile(r"(?<!\!)\[([^\]]*)\]\(([^)\s]+)\)")


def outside(page_path, url):
    """the repository path a relative link on docs/<page_path> leads to when that is outside docs/, else None"""
    if re.match(r"^(https?:|mailto:|#)", url):
        return None
    path, sep, anchor = url.partition("#")
    target = posixpath.normpath(posixpath.join(posixpath.dirname("docs/" + page_path), path))
    return None if target == "docs" or target.startswith("docs/") else target + sep + anchor


def on_page_markdown(markdown, page, **kwargs):
    def fix(m):
        target = outside(page.file.src_uri, m.group(2))
        return m.group(0) if target is None else f"[{m.group(1)}]({REPO}{target})"

    return LINK.sub(fix, markdown)
