"""Where the data, the template and the numbers live. Nothing in `linc` hard-codes a path; everything is resolved here.

Set once per shell (or pass the same values on the command line of each entry point):

    LINC_DATA      per-brain data, one folder per brain: $LINC_DATA/sub-<ID>/...            (see README.txt)
    LINC_TEMPLATE  the released template (channels, masks, transforms): $LINC_TEMPLATE/...
    LINC_NUMBERS   where code/figures/paper_numbers writes its JSON records (default: ./numbers)
    LINC_WORK      scratch for intermediate files (default: ./work)
    LINC_RAW       the raw acquisitions as delivered (blockface photographs, MRI, histology); read-only inputs
    LINC_MODELS    the trained networks' weights (see models/: cards, checksums, download)
    LINC_PRODUCTS  the intermediate products the scripts in code/figures and code/template read (per-piece tables, carried labels, warped inputs, ...);
                   logical names map to paths relative to this root in settings/internal/products_<lane>.yaml (see products())
"""
import os
from pathlib import Path


def _env(name, default=None):
    v = os.environ.get(name)
    if v:
        return Path(v).expanduser()
    if default is None:
        raise RuntimeError(f"set {name} (see code/shared/paths.py and README.txt)")
    return Path(default)


def data_root():
    return _env("LINC_DATA")


def refuse_records(*folders):
    """Stop before writing into `folders` when that could write into the products of record: with LINC_RECORDS set (a tree made by
    code/shared/make_local_data.sh, whose files link to the records), a folder that is, or holds an entry that is, a link resolving under
    $LINC_RECORDS. Opening an existing linked file for writing would change the record itself. No effect without LINC_RECORDS."""
    rec = os.environ.get("LINC_DATA_TREE") or os.environ.get("LINC_RECORDS")      # the data tree, when settings/roots.sh has set it
    if not rec:
        return
    R = os.path.realpath(rec).rstrip(os.sep) + os.sep
    inside = lambda q: (os.path.realpath(q) + os.sep).startswith(R)
    for f in folders:
        f = str(f)
        if not os.path.lexists(f):
            continue
        hits = [f] if inside(f) else [os.path.join(f, e) for e in os.listdir(f) if inside(os.path.join(f, e))] if os.path.isdir(f) else []
        if hits:
            raise SystemExit(f"{f}: {len(hits)} entr{'y' if len(hits) == 1 else 'ies'} link to the records under {rec} (e.g. {hits[0]}); "
                             "writing here could change them. Write to a new folder (--out), or make a rebuild folder for this brain: "
                             "code/shared/make_local_data.sh <dest> --rebuild <ID>")


def template_root():
    return _env("LINC_TEMPLATE")


def numbers_root():
    return _env("LINC_NUMBERS", "numbers")


def work_root():
    return _env("LINC_WORK", "work")


def subject_dir(subject):
    """$LINC_DATA/sub-<ID>; <ID> is the LINC identifier used in file names (e.g. MR252), not the paper name."""
    return data_root() / f"sub-{subject}"


def products_root():
    return _env("LINC_PRODUCTS")


def products(config_name):
    """{logical name: absolute path} from settings/<config_name>.yaml (paths there are relative to $LINC_PRODUCTS).
    Validation sets LINC_PRODUCTS to the analysis project; a data release only edits the YAML."""
    import yaml
    cfg = Path(__file__).resolve().parents[2] / "settings" / "internal" / f"{config_name}.yaml"
    m = yaml.safe_load(cfg.read_text()) or {}
    root = products_root()
    return {k: (root / v if isinstance(v, str) else v) for k, v in m.items()}


ROOTS = ("LINC_DATA", "LINC_TEMPLATE", "LINC_NUMBERS", "LINC_WORK", "LINC_RAW", "LINC_MODELS", "LINC_PRODUCTS")
PLACEHOLDERS = Path(__file__).resolve().parents[2] / "settings" / "internal" / "path_placeholders.yaml"            # committed: a generic example
PLACEHOLDERS_LOCAL = PLACEHOLDERS.with_name("path_placeholders.local.yaml")                          # this installation's real table (not in git)


def placeholder_table():
    """[(recorded prefix, placeholder), ...] in file order: settings/internal/path_placeholders.local.yaml when present (the real prefixes of the
    machine that made the products; gitignored), else settings/internal/path_placeholders.yaml (the committed example, whose generic "<...>"
    prefixes match no path). Every reader of the table uses this function."""
    import yaml
    f = PLACEHOLDERS_LOCAL if PLACEHOLDERS_LOCAL.exists() else PLACEHOLDERS
    return [tuple(ab) for ab in (yaml.safe_load(f.read_text()) or [])]


def _under_a_root(p):
    for var in ROOTS:
        v = os.environ.get(var, "").rstrip("/")
        if v and (p == v or p.startswith(v + "/")):
            return True
    return False


def rerooted(p, table=None):
    """A path recorded inside a product's own metadata, re-rooted on this machine. Two recorded forms:
      * an analysis-machine path: settings/internal/path_placeholders.yaml maps each recorded prefix to a placeholder ($LINC_PRODUCTS,
        $LINC_DATA, $LINC_TEMPLATE or $LINC_RAW/...), which resolves through the environment (LINC_RAW for raw-data folders);
      * a placeholder string, $<ROOT> or $<ROOT>/..., for the seven roots of this module (ROOTS), as relocated products write them:
        expanded from the configured root (data_root(), ..., products_root(); their defaults apply).
    Either way the result is a str. A path in neither form is returned unchanged, with one exception: when there is no local table
    (settings/internal/path_placeholders.local.yaml, see placeholder_table), an absolute path outside every configured root raises, because it
    was most likely recorded on the machine that made the products and cannot be mapped. `table` ([(prefix, placeholder), ...])
    replaces the table files (tests). (Generalized from blockface.synthesis.data.recorded_path.)"""
    p = str(p)
    local = table is not None or PLACEHOLDERS_LOCAL.exists()
    for a, b in (placeholder_table() if table is None else table):
        if p == a or p.startswith(a.rstrip("/") + "/"):
            var, _, rest = b.lstrip("$").partition("/")
            base = os.environ.get(var)
            if not base:
                raise RuntimeError(f"{p} was recorded under {a}; set ${var} to re-root it")
            return str(Path(base) / rest / p[len(a):].lstrip("/")) if rest else str(Path(base) / p[len(a):].lstrip("/"))
    var, _, rest = p[1:].partition("/") if p.startswith("$") else ("", "", "")
    if var in ROOTS:
        root = {"LINC_DATA": data_root, "LINC_TEMPLATE": template_root, "LINC_NUMBERS": numbers_root, "LINC_WORK": work_root,
                "LINC_RAW": raw_root, "LINC_MODELS": models_root, "LINC_PRODUCTS": products_root}[var]()
        return str(root / rest)
    if not local and p.startswith("/") and not _under_a_root(p):
        raise RuntimeError(f"{p}: an absolute path outside every configured root ({', '.join('$' + v for v in ROOTS)}), and no "
                           f"{PLACEHOLDERS_LOCAL.name} to map it (a path recorded on the machine that made the products?). Copy "
                           f"settings/{PLACEHOLDERS.name} to settings/{PLACEHOLDERS_LOCAL.name} and fill in that machine's prefixes.")
    return p


def raw_root():
    return _env("LINC_RAW")


def models_root():
    return _env("LINC_MODELS")
