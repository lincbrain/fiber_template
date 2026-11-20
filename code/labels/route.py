"""The route of a brain's labels into the MRI template, per brain (settings/labels.yaml template_routes): blockface, through the
blockface template of record (step 9), or mri, through the brain's own MRI (its R1 of record and its b0's registration to the MRI
template). One implementation for the figures and the records (tracer.py, s_tracer_fidelity.py, paper_numbers/tracer_sections_v2.py).

The blockface route, MRI-template world points y -> the brain's blockface world points:
    y -> the link of the blockface template to the MRI template (products_figures bft_pull_map) -> x, in the blockface template of
    record (step 9 keeps step 8's frame) -> x + U(x), the brain's step-9 field (bft_step9_fields: a member's last-round field, or its
    registration to the template when it is not a member) -> its member warp of step 8 (bft_member_warps, fsl_rel, its canon grid)
    -> its native map: the inverse of its S2 rigid placement for a brain without MRI ($LINC_DATA_TREE/brains/sub-<ID>/intermediate/registration/<ID>_s2lin.json),
    else its R1 of record after its placement (r1_record: components 2 and 3 of bf-native__to__b0-native), the map the template's
    build carried the brain with.
A brain outside the paper (no step-9 field in the settings) takes, after the link, its own registration to the blockface template
refined by step 9: $LINC_DATA/sub-<ID>/xfm_step9/bf-native__to__bf-template (code/template/blockface_step9.py refine: the refine
field, then the brain's entry from `linc-register blockface-template`). A brain not in template_routes takes the blockface route, the
paper's route for histology labels (15 Nov 2026).
"""
import json
from pathlib import Path

import numpy as np

from shared import paths, xfm
from blockface.registration import xfmtools as X


def routes():
    """{brain: "blockface" | "mri"} (settings/labels.yaml template_routes)."""
    import yaml
    return yaml.safe_load((Path(__file__).resolve().parents[2] / "settings" / "labels.yaml").read_text())["template_routes"]


def route_of(S):
    """S's route: settings/labels.yaml template_routes, else the blockface route."""
    return routes().get(S, "blockface")


def paper_brain(S, P):
    """S is one of the paper's brains on the blockface route: its step-9 field is in the settings (products_figures bft_step9_fields)."""
    return S in (P.get("bft_step9_fields") or {})


def new_brain_entry(S):
    """A brain outside the paper: its registration to the blockface template refined by step 9 (blockface_step9.py refine)."""
    d = paths.data_root() / f"sub-{S}" / "xfm_step9" / "bf-native__to__bf-template"
    if not (d / "meta.json").exists():
        raise SystemExit(f"{S}: no refined entry {d}. On the blockface route a brain outside the paper needs its registration to the "
                         f"blockface template (`linc-register blockface-template`, 02 Step 4), refined by step 9: "
                         f"$ANATOMIX_PYTHON code/template/blockface_step9.py refine --subject {S} --gpu 0")
    return xfm.Entry(d)


def uses_pull(S, P):
    """Whether S's labels are carried with a pull map (carry's pull=) rather than through shared.xfm.apply (chain=), as the paper
    carried them: a brain without MRI (its S2 rigid placement), and a brain outside the paper."""
    return not paper_brain(S, P) or (Path(P["data"]) / f"sub-{S}" / "mri" / "s2lin_prod" / f"{S}_s2lin.json").exists()


def blockface_native_map(S, P):
    """The brain's canon (run-113 placed) world -> its blockface world, and its description."""
    s2 = Path(P["data"]) / f"sub-{S}" / "mri" / "s2lin_prod" / f"{S}_s2lin.json"
    if s2.exists():
        Ti = np.linalg.inv(np.array(json.load(open(s2))["T_world_bf_to_run113"], float))
        return (lambda p: p @ Ti[:3, :3].T + Ti[:3, 3]), "s2lin"
    B = json.loads(Path(P["r1_record"]).read_text()); B = (B.get("data") or B)["brains"]
    assert B[S]["qc"] == "pass", (S, B[S]["qc"])
    c = xfm.Entry(paths.rerooted(B[S]["registry"])).components()
    assert "placed" in c[0].label and type(c[1]).__name__ == "Warp", f"{S}: unexpected components of its R1 of record"
    return (lambda p: c[2].pull(c[1].pull(p))), "R1 of record"


def blockface_pull(S, P):
    """MRI-template world points -> the brain's blockface world points on the blockface route (P: products_figures), and the route's
    description (one line per link)."""
    link = X.load_map(str(P["bft_pull_map"]), "full")
    if not paper_brain(S, P):
        e = new_brain_entry(S)

        def g(y):
            return np.asarray(e.pull(np.asarray(link(y))))
        return g, ["mri-template->bf-template (d2bs8_s5sym)", f"bf-template->{S} bf-native (its step-9 refined entry, {e.dir})"]
    row = [r for r in json.load(open(P["bft_member_warps"]))["data"] if r["member"] == S][0]
    W = xfm.Warp(paths.rerooted(row["file"]), "fsl_rel", src_grid=paths.rerooted(row["src_grid"]))
    U = X.GridField(str(paths.products_root() / P["bft_step9_fields"][S]), str(P["bft_step9_grid"]))
    native, how = blockface_native_map(S, P)

    def f(y):
        return native(np.asarray(W.pull(U(link(y)))))
    return f, ["mri-template->bf-template (d2bs8_s5sym)", f"bf-template step 9->step 8 ({S} step-9 field)", f"bf-template->{S} canon (d2bs8 member warp)",
               f"{S} canon->bf-native ({how})"]


# --------------------------------------------------------------------------- chains for shared.xfm.apply (the LGN and STN)

LGNSTN_BRAINS = ("MR243", "MR252", "MR256")                      # the brains with the LGN and STN drawn (MM1-MM3)
MRI_LINKS = [("bf-native", "b0-native"), ("b0-native", "mri-template")]
BLOCKFACE_LINKS = [("bf-template", "mri-template"), ("bf-template-step8", "bf-template"), ("canon", "bf-template-step8"),
                   ("bf-native", "canon")]                      # in pull order, as blockface_pull's description
NEW_BRAIN_LINKS = [("bf-template", "mri-template"), ("bf-native", "bf-template")]   # a brain outside the paper


def link_names(S, route):
    """The links of S's route into the MRI template, as the label records name them ("<src>->(dst) (<ID>)")."""
    return [f"{s}->{d} ({S})" for s, d in (BLOCKFACE_LINKS if route == "blockface" else MRI_LINKS)]


def r1_entry(S):
    """S's blockface -> b0 registration of record (settings/brains/<ID>.yaml variant_of_record); its QC must pass."""
    import yaml
    c = yaml.safe_load((Path(__file__).resolve().parents[2] / "settings" / "brains" / f"{S}.yaml").read_text())
    v = c["registration_to_own_mri"]["variant_of_record"]
    e = xfm.Entry(paths.products("products_blockface")["data"] / f"sub-{S}" / f"xfm_{v}" / "bf-native__to__b0-native")
    assert e.qc_status() in ("pass", "pass-visual"), (S, e.qc_status())
    return e


class RouteChain:
    """S's blockface route (blockface_pull) as a chain for shared.xfm.apply: pull(points), links (no registry entry) and describe()."""

    def __init__(self, S, P=None):
        P = P or paths.products("products_figures")
        self.S, self.paper = S, paper_brain(S, P)
        self._f, self._how = blockface_pull(S, P)
        self.links = [] if self.paper else [new_brain_entry(S)]        # registry entries (xfm.apply records their meta.json)

    def pull(self, Y):
        return np.asarray(self._f(np.asarray(Y, np.float64)), np.float64)

    def describe(self):
        return [dict(src=s, dst=d, subject=self.S, map=h) for (s, d), h in zip(BLOCKFACE_LINKS if self.paper else NEW_BRAIN_LINKS, self._how)]


def template_chain(S, route):
    """bf-native -> mri-template for S: the blockface route (RouteChain), or the MRI route (its R1 of record, then its b0 -> MRI template
    entry; strict)."""
    if route == "blockface":
        return RouteChain(S)
    return xfm.Chain([r1_entry(S), xfm.Entry(paths.products("products_blockface")["data"] / f"sub-{S}" / "xfm" / "b0-native__to__mri-template")],
                     strict=True)


def lgnstn_sets():
    """The LGN and STN label sets the records cover, as [(tag, folder, {brain: route})]: the MRI route of record always (products_mri
    push_current, tag from its folder name), and, when a brain of LGNSTN_BRAINS takes the blockface route (template_routes), that set
    (products_mri lgnstn_template_labels, whose folder must be push_current<template_chain_tag>). The paper reports both."""
    P = paths.products("products_mri")
    d0 = P["push_current"]
    assert d0.name.startswith("push_current"), d0
    out = [(d0.name[len("push_current"):], d0, {S: "mri" for S in LGNSTN_BRAINS})]
    R = routes()
    if any(R.get(S) == "blockface" for S in LGNSTN_BRAINS):
        tag, d = paths.template_chain_tag(), P["lgnstn_template_labels"]
        assert d.name == "push_current" + tag, f"products_mri lgnstn_template_labels ({d}) is not push_current{tag} (template_chain_tag)"
        out.append((tag, d, {S: R.get(S, "mri") for S in LGNSTN_BRAINS}))
    return out
