#!/usr/bin/env python3
"""Build the link (the blockface template's b0 -> the MRI template's b0), its construction estimate and comparison, and the mappings
to NMT v2, INIA19 and CIVM-R, from the released products: the recipe of record (reproduce/templates/link.md), one step per command,
with configs/link.yaml's parameters (the commands read it; nothing is restated here).

    python reproduce/templates/build_link.py --plan                    every step, its environment, time, inputs, outputs
    python reproduce/templates/build_link.py --step link-prepare       one step (or several, in recipe order)
    python reproduce/templates/build_link.py --from compose --to compare

Run in env/linc.yml. The [anatomix] steps are the feature-based engine (`solve`), run with $ANATOMIX_PYTHON (env/anatomix.yml) on one
GPU (about 5 GB for the link at 0.35 mm; more for the 0.25 mm atlas mappings); the atlas affines need ANTs 2.5.4 on PATH (or
$ANTSPATH). Inputs: configs/link.yaml `inputs` under $LINC_PRODUCTS (the two templates, the frame bridge, the member warps; your own
copies of CIVM-R and ONPRC18), the per-brain registrations under $LINC_DATA. Outputs under --work (default $LINC_WORK/build_link):
link/sym (the link), link/maps (its two pull maps), link/s5_compose_d2bs8_r05_guard and link/s5_tpl_d2bs8_r05_guard (the
construction estimate), link/s5_compare_d2bs8.json, mri/tc/atlas_v2/<atlas>/ (the mappings), and the numbers records written to
$LINC_NUMBERS by `score` and the summary.
One input the recipe does not rebuild: the one-way (greedy) link that `compare` reads beside the symmetric one; the greedy-link step
links the product of record (configs/products_blockface.yaml link_greedy). Times: link.md's (CPU, 8 threads) where it gives them.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _driver as D  # noqa: E402

ATLASES = ["NMT_v2.0_sym", "INIA19", "CIVM-R"]                      # link.md's order (atlas_final_v2_summary.py's record order)


def make_steps(W, paths):
    P1 = paths.products("products_blockface")
    greedy = P1["link_greedy"]
    L = W / "link"
    sym, maps = L / "sym", L / "maps"
    comp, tpl = L / "s5_compose_d2bs8_r05_guard", L / "s5_tpl_d2bs8_r05_guard"
    AT = W / "mri" / "tc" / "atlas_v2"
    R, E, A = ["-m", "linc.link.register"], ["-m", "linc.link.estimate"], ["-m", "linc.link.atlas"]

    def greedy_link():
        """The one-way link of record beside the symmetric one (compare reads <link root>/greedy); not rebuilt by this recipe."""
        L.mkdir(parents=True, exist_ok=True)
        os.symlink(greedy, L / "greedy")

    S = D.Step
    steps = [
        S("link-prepare", "1. the link", "linc", [*R, "prepare", "--out", sym], [sym / "prepare.json"], [], "2 min",
          "linear start, working grid, engine inputs"),
        S("link-solve", "1. the link", "anatomix", [*R, "solve", "--out", sym, "--device", "cuda:0"], [sym / "solve.json"],
          [sym / "prepare.json"], "2 min", "the symmetric engine -> link/sym/sym/", gpu=True),
        S("link-finish", "1. the link", "linc", [*R, "finish", "--out", sym], [sym / "s5_link.json"], [sym / "solve.json"], "2 min",
          "link_warp, moved b0 / mask / grey, s5_link.json"),
        S("link-export", "1. the link", "linc", [*R, "export", sym, maps], [maps / "export.json"], [sym / "s5_link.json"], "2 min",
          "the two full pull maps + round trips"),
        S("greedy-link", "2. construction estimate and comparison", "linc", [], [L / "greedy"], [greedy], "1 s",
          "links the one-way link of record (compare's second direct estimate; not rebuilt here)", fn=greedy_link),
        S("compose", "2. construction estimate and comparison", "linc", [*E, "compose", "--out", comp, "-j", "8"],
          [comp / "s5_compose.json"], [], "3 min", "all brain members (TPL excluded); add --order-from <record>/s5_compose.json for the "
          "record's member order (bit-exact; otherwise ~1e-7 mm)"),
        S("compose-tpl", "2. construction estimate and comparison", "linc", [*E, "compose", "--out", tpl, "--only", "TPL"],
          [tpl / "s5_compose.json"], [], "1 min", "the MRI template member alone"),
        S("compare", "2. construction estimate and comparison", "linc", [*E, "compare", comp, L, L / "s5_compare_d2bs8.json"],
          [L / "s5_compare_d2bs8.json"], [comp / "s5_compose.json", tpl / "s5_compose.json", L / "greedy", sym / "s5_link.json"], "1 min"),
    ]
    for a in ATLASES:
        steps += [
            S(f"atlas-affine-{a}", "3. mappings to published templates", "linc", [*A, "affine", a, "--out", AT], [AT / a / "prep.json"], [],
              "5 min", "ANTs affines in both directions, averaged in the log domain (ANTs 2.5.4; not bit-reproducible: link.md)"),
            S(f"atlas-solve-{a}", "3. mappings to published templates", "anatomix", [*A, "solve", a, "--out", AT],
              [AT / a / "run" / "metrics.json"], [AT / a / "prep.json"], "10 min", "the symmetric engine", gpu=True),
            S(f"atlas-score-{a}", "3. mappings to published templates", "linc", [*A, "score", a, "--out", AT], [AT / a / "score.json"],
              [AT / a / "run" / "metrics.json"], "2 min", "score.json + $LINC_NUMBERS/atlas_final_v2_<atlas>.json"),
        ]
    steps.append(S("atlas-summary", "3. mappings to published templates", "linc", ["reproduce/numbers/atlas_final_v2_summary.py"],
                   [Path(os.environ.get("LINC_NUMBERS", "numbers")) / "atlas_final_v2_summary.json"],
                   [AT / a / "score.json" for a in ATLASES], "1 s", "reads the three score records under <work>/mri/tc/atlas_v2",
                   setenv={"LINC_PRODUCTS": str(W)}))
    return steps


if __name__ == "__main__":
    D.main("Link and template mappings build (reproduce/templates/link.md)", make_steps, "build_link")
