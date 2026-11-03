#!/usr/bin/env python3
"""LGN/STN label geometry beyond Dice and ASSD (paper: Results, "Labels in the template"; Table 7 and its supplementary detail).

Inputs: exactly the labels scored by $LINC_NUMBERS/labels_overlap_current.json (run reproduce/numbers/labels_overlap_current.py first),
primary variant "demons" (whole volume): products push_current/<ID>/<ID>_<struct>_mri-template_mask_demons.nii.gz for MM1-MM3 and the
template-space reference products template_gt/GT_<struct>_mri-template.nii.gz, all on the MRI template grid (0.25 mm isotropic,
axis-aligned RAS: x = left->right, y = posterior->anterior (the blockface/histology section axis), z = inferior->superior).

Comparison design = the overlap record's: per hemisphere-structure (lhLGN, lhSTN, rhLGN, rhSTN), every pair of brains
(pairwise_subjects, the rows the paper quotes) and every brain against the single-rater template reference (subject_vs_GT).

Reproduction (asserted): Dice, ASSD, centroid distance and voxel counts of all 24 demons rows of the overlap record, and its
thinnest-axis one-voxel references, recomputed from the same files.

Per row: signed and absolute per-axis centroid offsets (b - a, mm), HD95, label volumes. Per label: volume (mm^3), the thinnest-axis
one-voxel reference, a direction-averaged one-voxel reference (Dice against itself shifted one voxel along +x, -x, +y, -y, +z, -z; mean
of the six), the same for ASSD and HD95, and the direction-averaged Dice at 1-4 voxel shifts, from which each row's "equivalent shift"
(the direction-averaged shift that reproduces the observed Dice, by linear interpolation) is read.
-> $LINC_NUMBERS/lgnstn_geometry_current.json
    python reproduce/numbers/lgnstn_geometry_current.py"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "4"                                           # CPU only, at most 4 threads

import json  # noqa: E402
import sys  # noqa: E402
from itertools import combinations  # noqa: E402
from pathlib import Path  # noqa: E402

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import distance_transform_edt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from linc import paths, provenance  # noqa: E402
from linc.evaluation.labels_agreement import one_voxel_ref, surface  # noqa: E402

OVERLAP = "labels_overlap_current.json"                            # the overlap record these labels were scored in
SUBS = ["MR243", "MR252", "MR256"]                                 # LINC IDs of MM1, MM2, MM3
STRUCTS = ["lhLGN", "lhSTN", "rhLGN", "rhSTN"]
GTNAME = "GT(run113, Chiara)"                                      # the reference's name in the overlap record's rows
AX = "xyz"
SHIFTS = (1, 2, 3, 4)                                              # integer voxel shifts of the Dice-vs-shift curve
TEMPLATE_VOX_MM = 0.25                                             # MRI template grid (asserted)
TOL_DICE, TOL_MM = 6e-5, 6e-4                                      # reproduction tolerance against the overlap record (its rounding)

# the record's own text, verbatim (it is part of the record's data)
REVIEW_ITEM = "manuscript/docs/reviews/ay_referee_2026-10-01.md M2"
VARIANT = "demons (primary of record)"
COMPARISON_DESIGN = ("as the record: per hemisphere-structure, each pair of brains (pairwise_subjects; the rows the paper "
                     "quotes) and each brain vs the single-rater template GT (subject_vs_GT); no consensus label")
DEFINITIONS = dict(
    axes="template grid is axis-aligned RAS at 0.25 mm: x = left->right, y = posterior->anterior (blockface/histology section "
         "axis, labels drawn every 1.2 mm), z = inferior->superior",
    dice="2|A&B|/(|A|+|B|) on the binary masks (as overlap_h7.py)",
    surface="mask voxels with a background 6-neighbor (m & ~binary_erosion(m)), as overlap_h7.py",
    assd_mm="mean of the pooled directed surface-to-surface EDT distances (both directions), as overlap_h7.py",
    hd95_mm="95th percentile of the same pooled symmetric surface distances (primary HD95)",
    hd95_max_directed_mm="max of the two directed 95th percentiles (the MONAI-style HD95), for comparison",
    hd100_mm="maximum surface distance (Hausdorff)",
    centroid_offset_mm_signed="centroid(b) - centroid(a) in world mm per axis (b = second brain, or GT)",
    centroid_offset_voxels_abs="|offset| / 0.25 mm",
    volume_mm3="voxel count x 0.015625 mm^3",
    extent_mm="bounding-box extent per axis",
    one_voxel_dice_thinnest_axis_of_record="label_refs.one_voxel_ref (the rule of record: integer one-voxel shift along the grid "
                                            "axis closest to the PCA-thinnest axis, mean of +/-)",
    one_voxel_dice_per_axis="Dice of the label vs itself shifted one voxel along that axis, mean of +/-",
    one_voxel_dice_direction_averaged="mean over the six shifts +-x, +-y, +-z",
    one_voxel_assd_mm_direction_averaged="ASSD of the label vs itself shifted one voxel, mean over the six shifts (HD95 likewise)",
    pair_reference="each row's references are the mean of its two labels' references (as the record's one_voxel_ref.mean)",
    dice_vs_shift_direction_averaged="self-Dice at an integer shift of k voxels, mean over the six axis directions",
    equivalent_shift_mm_direction_averaged="shift (mm) at which the row's mean direction-averaged self-Dice curve equals the "
                                           "observed Dice (linear interpolation between integer shifts; None beyond 4 voxels)",
    reproduction="all 24 demons rows' dice/assd_mm/centroid_mm/voxel counts and all 16 thinnest-axis references of the record "
                 "were recomputed from the same files and asserted equal to rounding")
PAPER_NOTE = ("results_histology.tex quotes Dice from labels_overlap_s125 but the one-voxel reference from labels_overlap.json "
              "(the earlier labels/push carry), not the s125 file's own reference")


def load(p):
    return np.asarray(nib.load(p).dataobj) > 0


def surf_dists(a, b, vox):
    """Directed surface distances (same surface/EDT definition as the overlap record)."""
    sa, sb = surface(a), surface(b)
    return distance_transform_edt(~sb, sampling=vox)[sa], distance_transform_edt(~sa, sampling=vox)[sb]


def shifted(m, ax, k):
    """Exact integer shift by k voxels along grid axis ax (zero fill; labels are far from the volume edge)."""
    out = np.zeros_like(m)
    src = [slice(None)] * 3; dst = [slice(None)] * 3
    if k > 0:
        src[ax] = slice(0, -k); dst[ax] = slice(k, None)
    else:
        src[ax] = slice(-k, None); dst[ax] = slice(0, k)
    out[tuple(dst)] = m[tuple(src)]
    assert out.sum() == m.sum(), "label touches the volume edge"
    return out


def dice(a, b):
    return float(2 * (a & b).sum() / (a.sum() + b.sum()))


def r(x, n=4):
    return None if x is None else round(float(x), n)


def label_refs(m, vox, aff):
    """One-voxel references of one label (Dice, ASSD, HD95 vs itself shifted) + direction-averaged Dice-vs-shift curve."""
    per = {}
    for ax in range(3):
        for sgn in (1, -1):
            s = shifted(m, ax, sgn)
            da, db = surf_dists(m, s, vox)
            d = np.concatenate([da, db])
            per[f"{'+' if sgn > 0 else '-'}{AX[ax]}"] = dict(dice=dice(m, s), assd_mm=float(d.mean()), hd95_mm=float(np.percentile(d, 95)))
    axis_mean = {AX[ax]: r(np.mean([per[f"{g}{AX[ax]}"]["dice"] for g in "+-"])) for ax in range(3)}
    curve = {0: 1.0}
    for k in SHIFTS:
        curve[k] = float(np.mean([dice(m, shifted(m, ax, sgn * k)) for ax in range(3) for sgn in (1, -1)]))
    thin = one_voxel_ref(m, aff)                                         # the thinnest-axis rule of the overlap record
    thin_axes = [a for a, v in axis_mean.items() if abs(v - thin) < TOL_DICE]
    return dict(
        volume_mm3=r(m.sum() * float(np.prod(vox)), 2), voxels=int(m.sum()),
        extent_mm={AX[ax]: r((np.ptp(np.nonzero(m.any(axis=tuple(i for i in range(3) if i != ax)))[0]) + 1) * vox[ax], 2) for ax in range(3)},
        one_voxel_dice_thinnest_axis_of_record=thin, thinnest_axis_of_record=thin_axes,
        one_voxel_dice_per_axis=axis_mean,
        one_voxel_dice_direction_averaged=r(np.mean([v["dice"] for v in per.values()])),
        one_voxel_assd_mm_direction_averaged=r(np.mean([v["assd_mm"] for v in per.values()]), 3),
        one_voxel_hd95_mm_direction_averaged=r(np.mean([v["hd95_mm"] for v in per.values()]), 3),
        one_voxel_per_direction={k: {kk: r(vv, 4 if kk == "dice" else 3) for kk, vv in v.items()} for k, v in per.items()},
        dice_vs_shift_direction_averaged={str(k): r(v) for k, v in curve.items()})


def equiv_shift(obs, curve, vox):
    """Direction-averaged shift (mm) whose self-Dice equals obs (linear interpolation of the curve; None beyond its range)."""
    ks = sorted(curve); ds = [curve[k] for k in ks]
    for k0, k1, d0, d1 in zip(ks, ks[1:], ds, ds[1:]):
        if d1 <= obs <= d0:
            return r((k0 + (d0 - obs) / (d0 - d1) * (k1 - k0)) * vox, 3)
    return None


def row_metrics(a, b, vox, aff):
    da, db = surf_dists(a, b, vox)
    d = np.concatenate([da, db])
    ca = nib.affines.apply_affine(aff, np.argwhere(a).mean(0)); cb = nib.affines.apply_affine(aff, np.argwhere(b).mean(0))
    off = cb - ca
    return dict(dice=r(dice(a, b)), assd_mm=r(d.mean(), 3), hd95_mm=r(np.percentile(d, 95), 3),
                hd95_max_directed_mm=r(max(np.percentile(da, 95), np.percentile(db, 95)), 3), hd100_mm=r(d.max(), 3),
                centroid_mm=r(np.linalg.norm(off), 3),
                centroid_offset_mm_signed={AX[i]: r(off[i], 3) for i in range(3)},
                centroid_offset_mm_abs={AX[i]: r(abs(off[i]), 3) for i in range(3)},
                centroid_offset_voxels_abs={AX[i]: r(abs(off[i]) / vox[i], 2) for i in range(3)},
                vox_a=int(a.sum()), vox_b=int(b.sum()))


def stats(v):
    v = np.asarray([x for x in v if x is not None], float)
    return dict(n=len(v), mean=r(v.mean()), median=r(np.median(v)), min=r(v.min()), max=r(v.max()),
                sd=r(v.std(ddof=1)) if len(v) > 1 else None)


def main():
    P = paths.products("products_mri")
    REF = P["template_mri"] / "linc_mri_b0.nii.gz"
    src = paths.numbers_root() / OVERLAP                                  # the regenerated overlap record
    ref = nib.load(REF)
    vox = tuple(float(z) for z in ref.header.get_zooms()[:3]); aff = ref.affine
    assert np.allclose(vox, TEMPLATE_VOX_MM) and np.allclose(aff[:3, :3], np.diag(vox)), "template grid is not axis-aligned 0.25 mm RAS"
    vv = vox[0]
    rec = json.loads(src.read_text())["data"]
    assert rec["primary_variant"] == "demons" and rec["subjects"] == SUBS
    recrows = {(x["kind"], x["struct"], x["a"], x["b"]): x for x in rec["rows"] if x["variant"] == "demons"}
    paper_ref = rec["one_voxel_reference"]                                # the overlap record is also the paper's one-voxel reference

    inputs, labels, rows, checks = [], {}, [], []
    for st in STRUCTS:
        for S in SUBS + ["GT"]:
            p = P["template_gt"] / f"GT_{st}_mri-template.nii.gz" if S == "GT" else P["push_current"] / S / f"{S}_{st}_mri-template_mask_demons.nii.gz"
            im = nib.load(p); assert im.shape == ref.shape and np.allclose(im.affine, aff), p
            labels[st, S] = load(p); inputs.append(p)

    per_label = {st: {S: label_refs(labels[st, S], vox, aff) for S in SUBS + ["GT"]} for st in STRUCTS}
    for st in STRUCTS:                                                    # reproduction: thinnest-axis references
        for S in SUBS + ["GT"]:
            got, want = per_label[st][S]["one_voxel_dice_thinnest_axis_of_record"], rec["one_voxel_reference"]["per_struct"][st][S]
            assert abs(got - want) < TOL_DICE, (st, S, got, want)
            assert per_label[st][S]["thinnest_axis_of_record"], (st, S, "record value is not one of the per-axis values")
            checks.append(dict(what=f"one_voxel_ref {st} {S}", recorded=want, recomputed=got))

    pairs = [("pairwise_subjects", S1, S2) for S1, S2 in combinations(SUBS, 2)] + [("subject_vs_GT", S, "GT") for S in SUBS]
    for st in STRUCTS:
        for kind, A, B in pairs:
            m = row_metrics(labels[st, A], labels[st, B], vox, aff)
            want = recrows[kind, st, A, GTNAME if B == "GT" else B]
            for k, tol in (("dice", TOL_DICE), ("assd_mm", TOL_MM), ("centroid_mm", TOL_MM), ("vox_a", 0), ("vox_b", 0)):
                assert abs(m[k] - want[k]) <= tol, (kind, st, A, B, k, m[k], want[k])
            checks.append(dict(what=f"{kind} {st} {A}-{B}", recorded={k: want[k] for k in ("dice", "assd_mm", "centroid_mm")},
                               recomputed={k: m[k] for k in ("dice", "assd_mm", "centroid_mm")}))
            la, lb = per_label[st][A], per_label[st][B]
            refs = dict(
                one_voxel_dice_thinnest_axis_of_record_mean=r((la["one_voxel_dice_thinnest_axis_of_record"] + lb["one_voxel_dice_thinnest_axis_of_record"]) / 2),
                one_voxel_dice_direction_averaged_mean=r((la["one_voxel_dice_direction_averaged"] + lb["one_voxel_dice_direction_averaged"]) / 2),
                one_voxel_assd_mm_direction_averaged_mean=r((la["one_voxel_assd_mm_direction_averaged"] + lb["one_voxel_assd_mm_direction_averaged"]) / 2, 3),
                one_voxel_hd95_mm_direction_averaged_mean=r((la["one_voxel_hd95_mm_direction_averaged"] + lb["one_voxel_hd95_mm_direction_averaged"]) / 2, 3))
            curve = {k: (float(la["dice_vs_shift_direction_averaged"][str(k)]) + float(lb["dice_vs_shift_direction_averaged"][str(k)])) / 2
                     for k in (0,) + SHIFTS}
            es = equiv_shift(m["dice"], curve, vv)
            rows.append(dict(kind=kind, struct=st, a=A, b=B, **m,
                             volume_mm3_a=la["volume_mm3"], volume_mm3_b=lb["volume_mm3"], **refs,
                             equivalent_shift_mm_direction_averaged=es,
                             equivalent_shift_voxels_direction_averaged=r(es / vv, 2) if es is not None else None))

    summary = {}
    for name, sts in (("LGN", ["lhLGN", "rhLGN"]), ("STN", ["lhSTN", "rhSTN"]), *[(s, [s]) for s in STRUCTS]):
        out = {}
        for kind in ("pairwise_subjects", "subject_vs_GT"):
            rr = [x for x in rows if x["kind"] == kind and x["struct"] in sts]
            out[kind] = dict(
                n_rows=len(rr),
                **{k: stats([x[k] for x in rr]) for k in ("dice", "assd_mm", "hd95_mm", "hd95_max_directed_mm", "centroid_mm",
                                                         "equivalent_shift_mm_direction_averaged")},
                centroid_offset_mm_abs={a: stats([x["centroid_offset_mm_abs"][a] for x in rr]) for a in AX},
                centroid_offset_mm_signed={a: stats([x["centroid_offset_mm_signed"][a] for x in rr]) for a in AX},
                rows_dice_below_direction_averaged_ref=sum(x["dice"] < x["one_voxel_dice_direction_averaged_mean"] for x in rr),
                rows_dice_below_thinnest_axis_ref=sum(x["dice"] < x["one_voxel_dice_thinnest_axis_of_record_mean"] for x in rr),
                rows_assd_above_direction_averaged_ref=sum(x["assd_mm"] > x["one_voxel_assd_mm_direction_averaged_mean"] for x in rr),
                rows_hd95_above_direction_averaged_ref=sum(x["hd95_mm"] > x["one_voxel_hd95_mm_direction_averaged_mean"] for x in rr))
        brains = [per_label[s][S] for s in sts for S in SUBS]
        out["labels_3_brains"] = dict(
            volume_mm3=stats([b["volume_mm3"] for b in brains]),
            volume_cv=r(np.std([b["volume_mm3"] for b in brains], ddof=1) / np.mean([b["volume_mm3"] for b in brains])),
            volume_max_over_min=r(max(b["volume_mm3"] for b in brains) / min(b["volume_mm3"] for b in brains), 3),
            one_voxel_dice_thinnest_axis_of_record=stats([b["one_voxel_dice_thinnest_axis_of_record"] for b in brains]),
            one_voxel_dice_direction_averaged=stats([b["one_voxel_dice_direction_averaged"] for b in brains]),
            one_voxel_dice_per_axis={a: stats([b["one_voxel_dice_per_axis"][a] for b in brains]) for a in AX},
            one_voxel_assd_mm_direction_averaged=stats([b["one_voxel_assd_mm_direction_averaged"] for b in brains]),
            one_voxel_hd95_mm_direction_averaged=stats([b["one_voxel_hd95_mm_direction_averaged"] for b in brains]),
            extent_mm={a: stats([b["extent_mm"][a] for b in brains]) for a in AX})
        out["GT_volume_mm3"] = [per_label[s]["GT"]["volume_mm3"] for s in sts]
        summary[name] = out

    data = dict(
        review_item=REVIEW_ITEM,
        record=str(P["numbers_of_record"] / OVERLAP),                     # names the overlap record of the analysis (read: src)
        variant=VARIANT, subjects=SUBS, structures=STRUCTS, reference_grid=str(REF), voxel_mm=vox,
        comparison_design=COMPARISON_DESIGN, definitions=DEFINITIONS,
        paper_text_reference_note=dict(
            note=PAPER_NOTE,
            labels_overlap_json=dict(LGN_dice=paper_ref["LGN_dice"], STN_dice=paper_ref["STN_dice"]),
            labels_overlap_s125_json=dict(LGN_dice=rec["one_voxel_reference"]["LGN_dice"], STN_dice=rec["one_voxel_reference"]["STN_dice"])),
        summary=summary, per_label=per_label, rows=rows, reproduction_checks=checks)
    provenance.write_numbers("lgnstn_geometry_current", data, inputs=inputs + [src], md5=False,
                             cmd="python reproduce/numbers/lgnstn_geometry_current.py")


if __name__ == "__main__":
    main()
