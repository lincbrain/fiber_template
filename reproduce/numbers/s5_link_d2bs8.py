#!/usr/bin/env python3
"""The link between the blockface template and the MRI template (paper: Methods, "Link to the MRI channels and placement of
photographed brains"; Results, "The link"). The link is a direct registration of the two b0 channels (anatomix + MIND-SSC features,
warp smoothing sigma 1.25), run with a greedy and a symmetric engine; the construction's own warps give an independent estimate
(the composition of the member brains' warps, and the template's build warp, each with an inverse-residual guard of 0.1 mm).
Per link (products: bftemplate_cv/s5_direct_d2bs8/<engine>/s5_link.json, s5_compose_d2bs8_r05_guard/ and s5_tpl_d2bs8_r05_guard/
s5_compose.json): the outline gate of the moved brain mask against the fixed one, the b0-b0 normalized cross-correlation, folds and
the Jacobian range, and for the symmetric engine the inverse round trip of its two maps. Consistency (s5_compare_d2bs8.json) = the
per-voxel distance between the composition and each link over MRI-template brain voxels (the paper's "median ... from its
construction estimate" and "inside this spread"). The envelope Dice is a support envelope, not brain overlap (kept, labeled so).
-> $LINC_NUMBERS/s5_link_d2bs8.json
    python reproduce/numbers/s5_link_d2bs8.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from linc import paths, provenance  # noqa: E402

# the record's own labels, verbatim (they are part of the record's data)
TEMPLATE = "bftemplate/d2b/final_s8 (2b step 8)"
FIXED = "template/mri/linc_mri_b0 (MRI template of record)"
TPL_NOTE = "the TPL max (23 mm) comes from MRI-mask points whose TPL inverse did not converge (left at identity by the guard); use p50/p95"
ENGINES = ("greedy", "sym")                                         # direct links, s5_direct_d2bs8/<engine>/
CHECKS = (("composition_of_member_warps", "s5_compose_d2bs8_r05_guard"), ("TPL_build_warp", "s5_tpl_d2bs8_r05_guard"))


def r(x, n=3):
    return round(float(x), n)


def main():
    C = paths.products("products_mri")["bftemplate_cv"]
    ins, out = [], {"template": TEMPLATE, "fixed": FIXED, "links": {}}
    for arm in ENGINES:
        f = C / f"s5_direct_d2bs8/{arm}/s5_link.json"; ins.append(f); d = json.loads(f.read_text()); j = d["jacobian"]
        row = dict(outline=d["outline_status"], outline_reasons=d["outline_reasons"], ncc_b0=r(d["ncc"]), folds_in_mask=j["folds_n_in_mask"],
                   detJ_min=r(j["jac_min"]), detJ_max=r(j["jac_max"]), sd_log_detJ=r(j["sd_log_detJ"]),
                   regqc_envelope_dice_support_not_brain_overlap=r(d["envelope_dice"]))
        if arm == "sym":
            m = C / "s5_direct_d2bs8/sym/sym/metrics.json"; ins.append(m); mt = json.loads(m.read_text())
            row["inverse_roundtrip_mm"] = {k: {q: r(v, 4) for q, v in mt[k].items()} for k in ("roundtrip_GF_mm_in_Amask", "roundtrip_FG_mm_in_Bmask")}
            row["folds_fraction_both_maps"] = mt["folds_fraction"]
        out["links"][f"direct_{arm}"] = row
    for name, d in CHECKS:
        f = C / d / "s5_compose.json"; ins.append(f); s = json.loads(f.read_text())
        out["links"][name] = dict(outline=s["outline_status"], outline_reasons=s["outline_reasons"],
                                  member_spread_mm={k: r(v) for k, v in s["spread_mm_in_mri_mask"].items()},
                                  link_symmetry_mm={k: r(v) for k, v in s["link_symmetry_mm_in_mask"].items()},
                                  mri_mask_points_without_converged_inverse=s.get("inverse_guard", {}).get("mri_mask_points_without_any_member"))
    f = C / "s5_compare_d2bs8.json"; ins.append(f); cmp_ = json.loads(f.read_text())
    out["consistency_vs_composition_mm"] = {k.replace("composition_vs_", ""): {q: r(v) for q, v in cmp_[k].items()}
                                            for k in cmp_ if k.startswith("composition_vs_")}
    # As in the record of the paper: the note is looked up under "tpl", but the key is "tpl_mm", so the note stays empty.
    out["note"] = TPL_NOTE if out["consistency_vs_composition_mm"].get("tpl", {}).get("max", 0) > 10 else ""
    provenance.write_numbers("s5_link_d2bs8", out, inputs=ins, md5=False, cmd="python reproduce/numbers/s5_link_d2bs8.py")


if __name__ == "__main__":
    main()
