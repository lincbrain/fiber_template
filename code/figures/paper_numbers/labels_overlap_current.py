#!/usr/bin/env python3
"""Overlap of the LGN and STN labels carried from histology into the MRI template (paper: Results, "Labels in the template"; Table 7).

The labels of three histology brains (MM1-MM3) were drawn on histology, carried to the blockface, interpolated between sections, and
carried to the MRI template through each brain's own MRI (blockface -> b0 -> template; products: push_current/<ID>/). The reference is
a template-space annotation drawn on an earlier MRI template and carried onto the current one (products: template_gt/). Per
hemisphere-structure (lhLGN, lhSTN, rhLGN, rhSTN), every brain against the reference (subject_vs_GT) and every pair of brains
(pairwise_subjects, the rows the paper quotes): Dice, ASSD (mean symmetric surface distance, mm), centroid distance (mm), volume ratio,
voxel counts, and each label's one-voxel reference (the Dice it scores against itself moved one 0.25 mm voxel along its thinnest axis).
Each row lists the transform links on each side and which are shared.
Three variants of the interpolated labels:
  sheet_sdf             sections rebuilt by plain signed-distance interpolation
  demons                demons-interpolated labels, whole volume (the primary variant)
  demons_real_sections  the same, scored only on template voxels whose provenance is a real section, and only where both sides have
                        >= 50 voxels
Top level: the one-voxel reference per structure (and LGN / STN means over the primary variant's brain labels) and each brain's
blockface A-P spacing.
-> $LINC_NUMBERS/labels_overlap_current.json
    python code/figures/paper_numbers/labels_overlap_current.py"""
import json
import sys
from itertools import combinations
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
from shared import paths, provenance  # noqa: E402
from figures.evaluation.labels_agreement import AA_RULE, ONE_VOXEL_RULE, bf_ap_spacing, one_voxel_ref, overlap_metrics

SUBJECTS = ["MR243", "MR252", "MR256"]                             # LINC IDs of MM1, MM2, MM3
STRUCTS = ["lhLGN", "lhSTN", "rhLGN", "rhSTN"]
PRIMARY = "demons"                                                 # primary variant (decided before comparing variants)
MIN_VOX = 50                                                       # demons_real_sections: scored only where both sides have >= this
VARIANTS = [("sheet_sdf", ""), ("demons", "_demons"), ("demons_real_sections", "_demons")]   # (variant, label file suffix)
R1_MANIFEST = "r1_current_2026-10-03.json"                         # record naming each brain's blockface -> b0 registration of record

# the record's own text, verbatim (it is part of the record's data)
GT_NAME = "GT(run113, Chiara)"
GT_NOTE = "GT is single-rater, drawn on the old run-113 template (in-sample for neither route)"
LINKS_SUBJECT = [("bf-native", "b0-native"), ("b0-native", "mri-template")]    # each brain's chain, both links per brain
LINKS_GT = [("mri-run113", "mri-template")]                                      # the reference's bridge (not per brain)
LINK_HISTO = "histology piece->bf-native (TIRL)"
LINK_GT_DRAWN = "GT drawn on run-113"
SHARED_PAIR = "mri-template (common target)"
CHAIN_OVERRIDE_PREFIX = "R1 of record per paper/numbers/r1_current_2026-10-03.json: "
PRIMARY_RATIONALE = ("demons (whole volume) is primary: statistical adequacy, decided before comparing variants "
                     "(orchestrator 2026-09-27); real-section scoring kept as a secondary row where both sides have >= "
                     f"{MIN_VOX} voxels (it scored higher, so the choice is conservative)")
VARIANT_TEXT = {"sheet_sdf": "sections rebuilt by plain SDF interpolation, ends extended half a spacing",
                "demons": "demons-interpolated labels (labels/histo_interp.py), whole volume",
                "demons_real_sections": "demons labels scored on real-section voxels only (provenance 1)"}


def load(p):
    im = nib.load(p)
    return np.asarray(im.dataobj) > 0, im


def links(S):
    return [f"{s}->{d} ({S})" for s, d in LINKS_SUBJECT]


def adequate(variant, m):
    """demons_real_sections: n/a unless both sides have >= MIN_VOX voxels (voxel counts kept)."""
    if variant == "demons_real_sections" and min(m.get("vox_a") or 0, m.get("vox_b") or 0) < MIN_VOX:
        return dict(dice=None, assd_mm=None, centroid_mm=None, vol_ratio=None, vox_a=m.get("vox_a"), vox_b=m.get("vox_b"),
                    na=f"< {MIN_VOX} voxels")
    return m


def main():
    P = paths.products("products_mri")
    pushdir = P["push_current"]
    REF = P["template_mri"] / "linc_mri_b0.nii.gz"
    ref = nib.load(REF)
    vox = ref.header.get_zooms()[:3]
    man = json.loads((P["numbers_of_record"] / R1_MANIFEST).read_text()); man = (man.get("data") or man)["brains"]
    assert all(man[S]["qc"] == "pass" for S in SUBJECTS), "every brain's registration of record must pass QC"
    gt_links = [f"{s}->{d}" for s, d in LINKS_GT]
    rows, inputs = [], []

    ovr = {}                                                                  # (variant, label, struct) -> one-voxel Dice
    def ov(variant, key, st, m):                                              # noqa: E306
        if (variant, key, st) not in ovr:
            ovr[variant, key, st] = one_voxel_ref(m, ref.affine)
        return ovr[variant, key, st]

    def extra(variant, st, ma, mb, ka, kb):
        oa, ob = ov(variant, ka, st, ma), ov(variant, kb, st, mb)
        return dict(one_voxel_ref=dict(a=oa, b=ob, mean=round((oa + ob) / 2, 4) if None not in (oa, ob) else None),
                    previous_carry=None)

    for variant, sfx in VARIANTS:
        real = {}
        for S in SUBJECTS:
            pv = pushdir / S / f"{S}_labels_demons_prov_mri-template.nii.gz"
            real[S] = (np.asarray(nib.load(pv).dataobj) == 1) if pv.exists() else None
        for st in STRUCTS:
            g, _ = load(P["template_gt"] / f"GT_{st}_mri-template.nii.gz")
            files = {S: pushdir / S / f"{S}_{st}_mri-template_mask{sfx}.nii.gz" for S in SUBJECTS}
            have = [S for S in SUBJECTS if files[S].exists() and (variant != "demons_real_sections" or real[S] is not None)]
            inputs += [files[S] for S in have]
            lab = {S: load(files[S])[0] for S in have}
            R = lambda S: real[S] if variant == "demons_real_sections" else np.ones(g.shape, bool)       # noqa: E731
            for S in have:
                m = adequate(variant, overlap_metrics(lab[S] & R(S), g & R(S), vox, ref.affine))
                ex = extra(variant, st, lab[S] & R(S), g & R(S), S, "GT" if variant != "demons_real_sections" else f"GT&real({S})")
                rows.append(dict(variant=variant, primary=variant == PRIMARY, kind="subject_vs_GT", struct=st, a=S, b=GT_NAME, **m, **ex,
                                 links_a=[LINK_HISTO] + links(S), links_b=[LINK_GT_DRAWN] + gt_links, shared_links=[], note=GT_NOTE))
            for S1, S2 in combinations(have, 2):
                both = R(S1) & R(S2)
                m = adequate(variant, overlap_metrics(lab[S1] & both, lab[S2] & both, vox, ref.affine))
                rk = "" if variant != "demons_real_sections" else f"&real({S1},{S2})"
                ex = extra(variant, st, lab[S1] & both, lab[S2] & both, S1 + rk, S2 + rk)
                rows.append(dict(variant=variant, primary=variant == PRIMARY, kind="pairwise_subjects", struct=st, a=S1, b=S2, **m, **ex,
                                 links_a=links(S1), links_b=links(S2), shared_links=[SHARED_PAIR]))

    ovp = {st: {k: v for (var, k, s_), v in ovr.items() if var == PRIMARY and s_ == st} for st in STRUCTS}
    mean = lambda v: round(float(np.mean(v)), 4) if v else None                                     # noqa: E731
    data = dict(subjects=SUBJECTS, reference=str(REF), primary_variant=PRIMARY, min_voxels_real_sections=MIN_VOX,
                anti_aliasing=AA_RULE, bf_ap_spacing_mm={S: bf_ap_spacing(S, str(pushdir)) for S in SUBJECTS},
                one_voxel_reference=dict(rule=ONE_VOXEL_RULE, variant=PRIMARY,
                                         LGN_dice=mean([v for st in STRUCTS if "LGN" in st for k, v in ovp[st].items() if v is not None and k != "GT"]),
                                         STN_dice=mean([v for st in STRUCTS if "STN" in st for k, v in ovp[st].items() if v is not None and k != "GT"]),
                                         per_struct=ovp),
                previous_carry=None,
                primary_rationale=PRIMARY_RATIONALE,
                chain_override=CHAIN_OVERRIDE_PREFIX + ", ".join(f"{S} {man[S]['variant']}" for S in SUBJECTS),
                variants=VARIANT_TEXT,
                rows=rows)
    provenance.write_numbers("labels_overlap_current", data, inputs=inputs, md5=False,
                             cmd="python code/figures/paper_numbers/labels_overlap_current.py")


if __name__ == "__main__":
    main()
