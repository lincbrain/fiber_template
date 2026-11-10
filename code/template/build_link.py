#!/usr/bin/env python3
"""Build the registration of the blockface template to the MRI template, its consistency check, and the registrations of the MRI
template to published templates; every parameter comes from settings/link.yaml.

You need
  settings/env/linc.yml; settings/env/anatomix.yml ($ANATOMIX_PYTHON) and one GPU for the [anatomix] steps (about 5 GB at 0.35 mm, more for the
  0.25 mm registrations to published templates); ANTs 2.5.4 on PATH or $ANTSPATH for their affine start;
  under $LINC_PRODUCTS (settings/link.yaml inputs): the blockface template's b0 and mask, the MRI template's b0, mask and T2, the
  bridge between the two frames, the template's member warps, the one-directional registration of record
  (settings/internal/products_blockface.yaml link_greedy); under $LINC_DATA the per-brain registrations; your own copies of the published
  templates (settings/targets.yaml says where to get them).
What it does (--plan lists every step with its command, inputs, outputs and time)
  1. the blockface template's b0 registered to the MRI template's b0 (an affine start, then the symmetric feature-based
     registration)
  2. a consistency check: the registrations composed through each member brain (blockface template -> brain -> MRI template),
     averaged, and compared with the registration of step 1
  3. the MRI template registered to each published template in settings/link.yaml atlas.atlases, and scored
You get
  under --work (default $LINC_WORK/build_link): link/sym (the registration of step 1), link/maps (its two transforms, as pull
  maps), link/blockface_in_mri (the template's blockface channel in the MRI template's frame: linc_bf_*), the consistency check (link/s5_*), mri/tc/atlas_v2/<template>/ (the registrations to published templates), xfm/ (the template-level
  entries of the transform registry, both directions of each registration); the numbers records in $LINC_NUMBERS.

    python code/template/build_link.py --plan                    every step, its environment, time, inputs, outputs
    python code/template/build_link.py --step link-prepare       one step (or several, in order)
    python code/template/build_link.py --from compose --to compare
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _driver as D  # noqa: E402

ATLASES = list(__import__("yaml").safe_load((D.REPO / "settings" / "link.yaml").read_text())["atlas"]["atlases"])   # the mapped templates, in order


def make_steps(W, paths):
    P1 = paths.products("products_blockface")
    greedy = P1["link_greedy"]
    L = W / "link"
    sym, maps = L / "sym", L / "maps"
    comp, tpl = L / "s5_compose_d2bs8_r05_guard", L / "s5_tpl_d2bs8_r05_guard"
    AT = W / "mri" / "tc" / "atlas_v2"
    R, E, A = ["-m", "template.link.register"], ["-m", "template.link.estimate"], ["-m", "template.link.atlas"]

    def greedy_link():
        """The one-way link of record beside the symmetric one (compare reads <link root>/greedy); not rebuilt by this recipe."""
        L.mkdir(parents=True, exist_ok=True)
        os.symlink(greedy, L / "greedy")

    S = D.Step
    steps = [
        S("link-prepare", "1. blockface template -> MRI template", "linc", [*R, "prepare", "--out", sym], [sym / "prepare.json"], [], "2 min",
          "linear start, working grid, engine inputs"),
        S("link-solve", "1. blockface template -> MRI template", "anatomix", [*R, "solve", "--out", sym, "--device", "cuda:0"], [sym / "solve.json"],
          [sym / "prepare.json"], "2 min", "the symmetric engine -> link/sym/sym/", gpu=True),
        S("link-finish", "1. blockface template -> MRI template", "linc", [*R, "finish", "--out", sym], [sym / "s5_link.json"], [sym / "solve.json"], "2 min",
          "link_warp, moved b0 / mask / grey, s5_link.json"),
        S("link-export", "1. blockface template -> MRI template", "linc", [*R, "export", sym, maps], [maps / "export.json"], [sym / "s5_link.json"], "2 min",
          "the two full pull maps + round trips"),
        S("blockface-in-mri", "1. blockface template -> MRI template", "linc", [*R, "blockface-in-mri", maps, L / "blockface_in_mri"],
          [L / "blockface_in_mri" / "blockface_in_mri.json"], [maps / "export.json"], "3 min",
          "the template's blockface channel in the MRI template's frame, under the released names (linc_bf_*)"),
        S("greedy-link", "2. consistency check", "linc", [], [L / "greedy"], [greedy], "1 s",
          "links the one-way link of record (compare's second direct estimate; not rebuilt here)", fn=greedy_link),
        S("compose", "2. consistency check", "linc", [*E, "compose", "--out", comp, "-j", "8"],
          [comp / "s5_compose.json"], [], "3 min", "all brain members (TPL excluded); add --order-from <record>/s5_compose.json for the "
          "record's member order (bit-exact; otherwise ~1e-7 mm)"),
        S("compose-tpl", "2. consistency check", "linc", [*E, "compose", "--out", tpl, "--only", "TPL"],
          [tpl / "s5_compose.json"], [], "1 min", "the MRI template member alone"),
        S("compare", "2. consistency check", "linc", [*E, "compare", comp, L, L / "s5_compare_d2bs8.json"],
          [L / "s5_compare_d2bs8.json"], [comp / "s5_compose.json", tpl / "s5_compose.json", L / "greedy", sym / "s5_link.json"], "1 min"),
    ]
    for a in ATLASES:
        steps += [
            S(f"atlas-affine-{a}", "3. mappings to published templates", "linc", [*A, "affine", a, "--out", AT], [AT / a / "prep.json"], [],
              "5 min", "ANTs affines in both directions, averaged in the log domain (ANTs 2.5.4; not bit-reproducible: code/template/README.txt)"),
            S(f"atlas-solve-{a}", "3. mappings to published templates", "anatomix", [*A, "solve", a, "--out", AT],
              [AT / a / "run" / "metrics.json"], [AT / a / "prep.json"], "10 min", "the symmetric engine", gpu=True),
            S(f"atlas-score-{a}", "3. mappings to published templates", "linc", [*A, "score", a, "--out", AT], [AT / a / "score.json"],
              [AT / a / "run" / "metrics.json"], "2 min", "score.json + $LINC_NUMBERS/atlas_final_v2_<atlas>.json"),
        ]
    steps.append(S("registry", "3. mappings to published templates", "linc", [*R, "registry", maps, sym, AT, W / "xfm"],
                   [W / "xfm" / "bf-template__to__mri-template" / "meta.json"], [maps / "export.json", sym / "s5_link.json"] + [AT / a / "score.json" for a in ATLASES], "1 min",
                   "the template-level entries of the transform registry (code/shared/xfm.py), both directions of each registration: $LINC_TEMPLATE/xfm/ in the released template"))
    steps.append(S("atlas-summary", "3. mappings to published templates", "linc", ["code/figures/paper_numbers/atlas_final_v2_summary.py"],
                   [Path(os.environ.get("LINC_NUMBERS", "numbers")) / "atlas_final_v2_summary.json"],
                   [AT / a / "score.json" for a in ATLASES], "1 s", "reads each mapping's score record under <work>/mri/tc/atlas_v2",
                   setenv={"LINC_PRODUCTS": str(W)}))
    return steps


if __name__ == "__main__":
    D.main("Link and template mappings build (code/template/README.txt)", make_steps, "build_link", doc=__doc__)
