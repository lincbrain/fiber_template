#!/usr/bin/env python3
"""An error budget of the between-brain LGN/STN agreement (paper: Supplementary S3, the error-budget table above its network rows,
and its figure; the paper reads summary.<LGN|STN>.*, sarm.nuclei.*, per_brain.<ID>.* and nominal_section_step_mm).

Brains MM1, MM2, MM3 (MR243, MR252, MR256): the observer's outlines on sections ~1.1 mm apart, carried to the MRI template by the
chain of record (the R1 registry entry named by numbers_of_record/r1_current_2026-10-03.json, then the brain's b0 -> MRI template
entry). Template grid: 0.25 mm, axis-aligned RAS, symmetric about x = 0 (x left->right, y posterior->anterior = the section axis,
z inferior->superior).

Labels, per brain and hemisphere-nucleus (lhLGN, lhSTN, rhLGN, rhSTN), all on the template grid (products lgnstn_labels/<ID>/):
  demons   the labels of record (<ID>_<struct>_mri-template_mask_demons.nii.gz: demons-interpolated between sections; the labels
           lgnstn_geometry_current and labels_overlap_current read); reproduced here first (asserted against lgnstn_geometry_current)
  sdf      the same chain with plain signed-distance interpolation between sections (<ID>_<struct>_mri-template_mask.nii.gz)
  slab     NOT interpolated: each labeled section stands for its own slab, half-way to its neighbors along the section axis (the
           outer sections: half the adjacent gap). A template voxel is labeled when its point, pulled through the chain of record
           into the blockface, falls inside a labeled section's slab and inside that section's outline (nearest in-plane sample of
           the un-interpolated sheet, <ID>_<struct>_bfnative_L2.nii.gz).
Between brains: per brain pair, hemisphere and nucleus, the centroid difference per axis (signed and absolute, mm; |dy| also against the
nominal 1.2 mm and the calibrated section spacing) for demons, sdf and slab; per brain, what the interpolation moves (centroid(demons)
- centroid(slab)); the labeled sections per nucleus, their spacing, and each nucleus's y extent.
Same brain: the left label against the mirrored right label (x -> -x; the grid is symmetric).
Atlas: SARM level 6 in our template (sarm_in_template, carried from NMT v2 by the NMT v2 mapping of record, nearest neighbor; DLG = 151
for the LGN, STh = 91 for the STN, split by hemisphere at x = 0): each brain's label against the SARM nucleus of its side; the
mapping's own quality from numbers_of_record/atlas_final_v2_NMT_v2.0_sym.json.
Metrics as lgnstn_geometry_current: Dice, ASSD and HD95 over the pooled symmetric surface distances (surface = m & ~erosion),
centroid offsets b - a in world mm.
-> $LINC_NUMBERS/lgnstn_budget_r059.json; the slab labels and the SARM nuclei on the template grid in $LINC_WORK/lgnstn_budget_r059/slab/
    python code/figures/paper_numbers/lgnstn_budget_r059.py
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import json  # noqa: E402
import sys  # noqa: E402
from itertools import combinations  # noqa: E402
from pathlib import Path  # noqa: E402

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import binary_erosion, distance_transform_edt  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
from shared import paths, provenance, xfm  # noqa: E402

LABELS = REPO / "settings" / "specimen_labels.json"
SARM_ID = {"LGN": 151, "STN": 91}                                   # DLG dorsal_lateral_geniculate; STh subthalamic_nucleus
SUBS = ["MR243", "MR252", "MR256"]
STRUCTS = ["lhLGN", "lhSTN", "rhLGN", "rhSTN"]
AX = "xyz"
NOMINAL_STEP_MM = 1.2                                                # the histology sections' nominal spacing (Methods)
MARGIN_MM = 4.0

# the record's own text, verbatim (it is part of the record's data)
REQUEST = "requested in the manuscript's revision (items 1, 2 and 5 of its analysis list)"


def load(p):
    return np.asarray(nib.load(p).dataobj) > 0


def r(x, n=4):
    return None if x is None else round(float(x), n)


def surface(m):
    return m & ~binary_erosion(m)


def metrics(a, b, vox, aff):
    sa, sb = surface(a), surface(b)
    da = distance_transform_edt(~sb, sampling=vox)[sa]; db = distance_transform_edt(~sa, sampling=vox)[sb]
    d = np.concatenate([da, db])
    ca = nib.affines.apply_affine(aff, np.argwhere(a).mean(0)); cb = nib.affines.apply_affine(aff, np.argwhere(b).mean(0))
    off = cb - ca
    return dict(dice=r(2 * (a & b).sum() / (a.sum() + b.sum())), assd_mm=r(d.mean(), 3), hd95_mm=r(np.percentile(d, 95), 3),
                centroid_mm=r(np.linalg.norm(off), 3), centroid_offset_mm_signed={AX[i]: r(off[i], 3) for i in range(3)},
                centroid_offset_mm_abs={AX[i]: r(abs(off[i]), 3) for i in range(3)}, vox_a=int(a.sum()), vox_b=int(b.sum()))


def centroid(m, aff):
    return nib.affines.apply_affine(aff, np.argwhere(m).mean(0))


def stats(v):
    v = np.asarray([x for x in v if x is not None], float)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return dict(n=len(v), median=r(med), iqr=[r(q1), r(q3)], mean=r(v.mean()), min=r(v.min()), max=r(v.max()))


def chain(S, B, data):
    assert B[S]["qc"] == "pass", S
    return xfm.Chain([xfm.Entry(paths.rerooted(B[S]["registry"])), xfm.Entry(data / f"sub-{S}" / "xfm" / "b0-native__to__mri-template")],
                     strict=True)


def slab_label(S, st, ch, ref_shape, aff, bbox, push):
    """the un-interpolated slab stack of one hemisphere-nucleus on the template grid (module docstring), and the sections"""
    sp = push / S / f"{S}_{st}_bfnative_L2.nii.gz"
    im = nib.load(sp); sheet = np.asanyarray(im.dataobj) > 0; Abf = im.affine
    dy = abs(Abf[1, 1]); assert np.allclose(Abf[:3, :3], np.diag(np.diag(Abf)[:3])), "blockface grid not axis-aligned"
    planes = np.nonzero(sheet.any(axis=(0, 2)))[0]
    mids = (planes[:-1] + planes[1:]) / 2.0
    gaps = np.diff(planes)
    lo = np.r_[planes[0] - gaps[0] / 2.0, mids]; hi = np.r_[mids, planes[-1] + gaps[-1] / 2.0]
    (i0, j0, k0), (i1, j1, k1) = bbox
    ii, jj, kk = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), np.arange(k0, k1), indexing="ij")
    vox = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], 1).astype(np.float64)
    P = vox @ aff[:3, :3].T + aff[:3, 3]
    X = np.concatenate([np.asarray(ch.pull(P[s:s + 200000])) for s in range(0, len(P), 200000)])
    v = X @ np.linalg.inv(Abf)[:3, :3].T + np.linalg.inv(Abf)[:3, 3]
    lab = np.zeros(len(v), bool)
    for p, a_, b_ in zip(planes, lo, hi):
        sel = (v[:, 1] >= a_) & (v[:, 1] < b_)
        ix = np.rint(v[sel, 0]).astype(int); iz = np.rint(v[sel, 2]).astype(int)
        ok = (ix >= 0) & (ix < sheet.shape[0]) & (iz >= 0) & (iz < sheet.shape[2])
        hit = np.zeros(sel.sum(), bool); hit[ok] = sheet[ix[ok], p, iz[ok]]
        lab[np.flatnonzero(sel)] = hit
    out = np.zeros(ref_shape, bool)
    out[i0:i1, j0:j1, k0:k1] = lab.reshape(ii.shape)
    sec = dict(n_sections=int(len(planes)), section_spacing_mm=[r(g * dy, 3) for g in gaps],
               section_spacing_median_mm=r(np.median(gaps) * dy, 3), span_first_to_last_mm=r((planes[-1] - planes[0]) * dy, 3),
               slab_extent_mm=r((hi[-1] - lo[0]) * dy, 3), blockface_step_mm=r(dy, 4),
               voxels_per_section=[int(sheet[:, p, :].sum()) for p in planes])
    return out, sec, sp


def main():
    P = paths.products("products_completion")
    PUSH = P["lgnstn_labels"]
    NUM = P["numbers_of_record"]
    REF, SARM, SARM_KEY = P["template_mri"] / "linc_mri_b0.nii.gz", P["sarm_in_template"], P["sarm_key"]
    R1MAN, GEOM, ATLAS = NUM / "r1_current_2026-10-03.json", NUM / "lgnstn_geometry_current.json", NUM / "atlas_final_v2_NMT_v2.0_sym.json"
    SLABDIR = paths.work_root() / "lgnstn_budget_r059" / "slab"
    ref = nib.load(REF); aff = ref.affine; shape = ref.shape[:3]
    vox = tuple(float(z) for z in ref.header.get_zooms()[:3])
    assert np.allclose(vox, 0.25) and np.allclose(aff[:3, :3], np.diag(vox)), "template grid not axis-aligned 0.25 mm"
    xc = aff[0, 3] + aff[0, 0] * np.arange(shape[0]); assert np.allclose(xc, -xc[::-1], atol=1e-6), "template grid not symmetric"
    B = json.load(open(R1MAN)); B = (B.get("data") or B)["brains"]
    lab = json.load(open(LABELS))["labels"]
    inputs = [REF, R1MAN, GEOM, ATLAS, SARM, SARM_KEY, LABELS]
    SLABDIR.mkdir(parents=True, exist_ok=True)
    L = {}
    for v_, suf in (("demons", "_demons"), ("sdf", "")):
        for S in SUBS:
            for st in STRUCTS:
                p = PUSH / S / f"{S}_{st}_mri-template_mask{suf}.nii.gz"; im = nib.load(p)
                assert im.shape[:3] == shape and np.allclose(im.affine, aff), p
                L[v_, S, st] = load(p); inputs.append(p)
    # reproduction: the record's demons pair rows (lgnstn_geometry_current) from the same files
    geom = json.load(open(GEOM))["data"]
    rec = {(x["struct"], x["a"], x["b"]): x for x in geom["rows"] if x["kind"] == "pairwise_subjects"}
    checks = []
    for st in STRUCTS:
        for A, Bb in combinations(SUBS, 2):
            m = metrics(L["demons", A, st], L["demons", Bb, st], vox, aff); w = rec[st, A, Bb]
            for k in ("dice", "assd_mm", "centroid_mm"):
                assert abs(m[k] - w[k]) <= 1e-3, (st, A, Bb, k, m[k], w[k])
            for a_ in AX:
                assert abs(m["centroid_offset_mm_signed"][a_] - w["centroid_offset_mm_signed"][a_]) <= 1e-3
            checks.append(dict(row=f"{st} {A}-{Bb}", recorded=w["centroid_mm"], recomputed=m["centroid_mm"]))
    # the slab stacks (un-interpolated)
    sections, prov = {}, {}
    for S in SUBS:
        ch = chain(S, B, P["brain_data"])
        for st in STRUCTS:
            m = L["demons", S, st] | L["sdf", S, st]
            idx = np.argwhere(m); pad = int(np.ceil(MARGIN_MM / vox[0]))
            bbox = (tuple(np.maximum(idx.min(0) - pad, 0)), tuple(np.minimum(idx.max(0) + pad + 1, shape)))
            sl, sec, sp = slab_label(S, st, ch, shape, aff, bbox, PUSH)
            assert sl.any(), (S, st, "empty slab stack")
            L["slab", S, st] = sl; sections[S, st] = sec; inputs.append(sp)
            nib.save(nib.Nifti1Image(sl.astype(np.uint8), aff), SLABDIR / f"{S}_{st}_mri-template_slab.nii.gz")
            print(S, lab[S], st, "sections", sec["n_sections"], "spacing", sec["section_spacing_median_mm"], "slab vox", int(sl.sum()),
                  "demons vox", int(L["demons", S, st].sum()), flush=True)
        prov[S] = dict(r1=B[S]["registry"], r1_variant=B[S]["variant"], r5=f"data/sub-{S}/xfm/b0-native__to__mri-template")
    VARS = ("demons", "sdf", "slab")
    # item 1: pairs
    rows = []
    for st in STRUCTS:
        for A, Bb in combinations(SUBS, 2):
            row = dict(struct=st, nucleus=st[2:], hemisphere=st[:2], a=A, b=Bb, a_name=lab[A], b_name=lab[Bb])
            for v_ in VARS:
                m = metrics(L[v_, A, st], L[v_, Bb, st], vox, aff)
                m["abs_dy_over_nominal_step"] = r(m["centroid_offset_mm_abs"]["y"] / NOMINAL_STEP_MM, 3)
                step = np.median([sections[A, st]["section_spacing_median_mm"], sections[Bb, st]["section_spacing_median_mm"]])
                m["abs_dy_over_section_spacing"] = r(m["centroid_offset_mm_abs"]["y"] / step, 3)
                row[v_] = m
            rows.append(row)
    # item 1: per brain (what the interpolation moves; extents)
    per_brain = {}
    for S in SUBS:
        for st in STRUCTS:
            cs = {v_: centroid(L[v_, S, st], aff) for v_ in VARS}
            ext = {v_: r((np.ptp(np.nonzero(L[v_, S, st].any(axis=(0, 2)))[0]) + 1) * vox[1], 3) for v_ in VARS}
            per_brain.setdefault(S, {})[st] = dict(
                name=lab[S], sections=sections[S, st],
                centroid_mm={v_: [r(c, 3) for c in cs[v_]] for v_ in VARS},
                interpolation_shift_demons_minus_slab_mm={AX[i]: r(cs["demons"][i] - cs["slab"][i], 3) for i in range(3)},
                interpolation_shift_demons_minus_slab_norm_mm=r(np.linalg.norm(cs["demons"] - cs["slab"]), 3),
                interpolation_shift_sdf_minus_slab_mm={AX[i]: r(cs["sdf"][i] - cs["slab"][i], 3) for i in range(3)},
                y_extent_template_mm=ext, volume_mm3={v_: r(L[v_, S, st].sum() * np.prod(vox), 2) for v_ in VARS})
    # item 2: within-brain mirror floor (left vs mirrored right)
    mirror = []
    for S in SUBS:
        for nuc in ("LGN", "STN"):
            row = dict(brain=S, name=lab[S], nucleus=nuc)
            for v_ in VARS:
                row[v_] = metrics(L[v_, S, "lh" + nuc], L[v_, S, "rh" + nuc][::-1], vox, aff)
            mirror.append(row)
    # item 5: SARM reference
    sarm = np.asarray(nib.load(SARM).dataobj).astype(np.int32)
    key = {int(l.split("\t")[0]): l.split("\t")[1:3] for l in open(SARM_KEY).read().splitlines()[1:] if l.strip()}
    assert key[151][0] == "DLG" and key[91][0] == "STh", (key[151], key[91])
    xw = (aff[0, 3] + aff[0, 0] * np.arange(shape[0]))[:, None, None]
    ref_lab, sarm_info = {}, {}
    for nuc, sid in SARM_ID.items():
        m = sarm == sid
        ref_lab["lh" + nuc] = m & (xw < 0); ref_lab["rh" + nuc] = m & (xw > 0)
        sym = metrics(ref_lab["lh" + nuc], ref_lab["rh" + nuc][::-1], vox, aff)
        sarm_info[nuc] = dict(sarm_id=sid, abbreviation=key[sid][0], name=key[sid][1],
                              volume_mm3={h: r(ref_lab[h + nuc].sum() * np.prod(vox), 2) for h in ("lh", "rh")},
                              centroid_mm={h: [r(c, 3) for c in centroid(ref_lab[h + nuc], aff)] for h in ("lh", "rh")},
                              left_vs_mirrored_right=dict(dice=sym["dice"], centroid_mm=sym["centroid_mm"]))
        for h in ("lh", "rh"):
            nib.save(nib.Nifti1Image(ref_lab[h + nuc].astype(np.uint8), aff), SLABDIR / f"SARM_{h}{nuc}_mri-template.nii.gz")
    vs_sarm = []
    for S in SUBS:
        for st in STRUCTS:
            row = dict(brain=S, name=lab[S], struct=st, nucleus=st[2:], hemisphere=st[:2])
            for v_ in ("demons", "slab"):
                row[v_] = metrics(ref_lab[st], L[v_, S, st], vox, aff)              # offset = brain label - SARM
            vs_sarm.append(row)
    A_ = json.load(open(ATLAS))["data"]
    mapping = dict(atlas="NMT_v2.0_sym", engine=A_["engine"], template_channel=A_["template_channel"], mask_dice=A_["mask_dice"],
                   inverse_consistency_mm=A_["roundtrip_mm"], outline=A_["outline"], folds=A_["folds_fraction"],
                   label_carry="nearest neighbor (mri/code/atlas_v2.py: SARM level 6 into the template, labels for figures)",
                   local_accuracy="not measured: the mapping's quality measures are global (brain-mask Dice, inverse consistency, "
                                  "outline at the brain surface); its accuracy at the LGN and STN has no measurement of its own")
    # summaries per nucleus
    def summ_pairs(nuc, v_):
        rr = [x[v_] for x in rows if x["nucleus"] == nuc]
        return dict(n=len(rr), centroid_mm=stats([x["centroid_mm"] for x in rr]),
                    abs_offset_mm={a: stats([x["centroid_offset_mm_abs"][a] for x in rr]) for a in AX},
                    abs_dy_over_nominal_step=stats([x["abs_dy_over_nominal_step"] for x in rr]),
                    dice=stats([x["dice"] for x in rr]), assd_mm=stats([x["assd_mm"] for x in rr]), hd95_mm=stats([x["hd95_mm"] for x in rr]))
    summary = {}
    for nuc in ("LGN", "STN"):
        sts = ["lh" + nuc, "rh" + nuc]
        summary[nuc] = dict(
            between_brains={v_: summ_pairs(nuc, v_) for v_ in VARS},
            within_brain_mirror={v_: dict(n=3, centroid_mm=stats([x[v_]["centroid_mm"] for x in mirror if x["nucleus"] == nuc]),
                                          abs_offset_mm={a: stats([x[v_]["centroid_offset_mm_abs"][a] for x in mirror if x["nucleus"] == nuc]) for a in AX},
                                          dice=stats([x[v_]["dice"] for x in mirror if x["nucleus"] == nuc]))
                                 for v_ in VARS},
            to_sarm={v_: dict(n=6, centroid_mm=stats([x[v_]["centroid_mm"] for x in vs_sarm if x["nucleus"] == nuc]),
                              abs_offset_mm={a: stats([x[v_]["centroid_offset_mm_abs"][a] for x in vs_sarm if x["nucleus"] == nuc]) for a in AX},
                              signed_offset_mm={a: stats([x[v_]["centroid_offset_mm_signed"][a] for x in vs_sarm if x["nucleus"] == nuc]) for a in AX},
                              assd_mm=stats([x[v_]["assd_mm"] for x in vs_sarm if x["nucleus"] == nuc]),
                              dice=stats([x[v_]["dice"] for x in vs_sarm if x["nucleus"] == nuc]))
                     for v_ in ("demons", "slab")},
            interpolation_shift_norm_mm=stats([per_brain[S][s]["interpolation_shift_demons_minus_slab_norm_mm"] for S in SUBS for s in sts]),
            interpolation_shift_abs_mm={a: stats([abs(per_brain[S][s]["interpolation_shift_demons_minus_slab_mm"][a]) for S in SUBS for s in sts]) for a in AX},
            sections=dict(n_sections=stats([sections[S, s]["n_sections"] for S in SUBS for s in sts]),
                          spacing_median_mm=stats([sections[S, s]["section_spacing_median_mm"] for S in SUBS for s in sts]),
                          y_extent_demons_mm=stats([per_brain[S][s]["y_extent_template_mm"]["demons"] for S in SUBS for s in sts])))
    data = dict(
        request=REQUEST,
        brains={S: lab[S] for S in SUBS}, structures=STRUCTS, chain=prov, nominal_section_step_mm=NOMINAL_STEP_MM,
        definitions=dict(
            axes="template RAS at 0.25 mm: x left->right, y posterior->anterior (the section axis), z inferior->superior",
            variants="demons = labels of record; sdf = signed-distance interpolation, same chain; slab = un-interpolated: each labeled "
                     "section stands for its slab half-way to its neighbors (outer sections: half the adjacent gap), labeled where the "
                     "pulled template point lies in the slab and inside the section's outline (nearest in-plane sample)",
            offsets="centroid(b) - centroid(a), world mm (pairs); brain label - SARM (item 5); left - mirrored right (item 2)",
            metrics="Dice; ASSD and HD95 over the pooled symmetric surface distances (surface = m & ~erosion), as lgnstn_geometry.py",
            abs_dy_over_nominal_step="|dy| / 1.2 mm (the sections' nominal spacing)",
            abs_dy_over_section_spacing="|dy| / the median calibrated spacing of the two brains' labeled sections (blockface frame)",
            interpolation_shift="centroid(demons) - centroid(slab): what the interpolation between sections moves",
            mirror="left label vs right label mirrored x -> -x on the symmetric grid",
            sarm="SARM level 6 in our template (nearest-neighbor carry through the NMT v2 mapping of record), DLG for the LGN, STh "
                 "for the STN, split at x = 0"),
        reproduction=dict(note="all 12 demons pair rows of lgnstn_geometry_current (Dice, ASSD, centroid, per-axis offsets) "
                               "recomputed from the same files and asserted equal to 1e-3", checks=checks),
        summary=summary, pairs=rows, per_brain=per_brain, mirror=mirror, sarm=dict(nuclei=sarm_info, mapping=mapping, rows=vs_sarm))
    provenance.write_numbers("lgnstn_budget_r059", data, inputs=inputs, md5=False, cmd="python code/figures/paper_numbers/lgnstn_budget_r059.py")
    for nuc in ("LGN", "STN"):
        s = summary[nuc]
        print(nuc, "between", {v_: (s["between_brains"][v_]["centroid_mm"]["median"],
                                    {a: s["between_brains"][v_]["abs_offset_mm"][a]["median"] for a in AX}) for v_ in VARS})
        print(nuc, "mirror", {v_: s["within_brain_mirror"][v_]["centroid_mm"]["median"] for v_ in VARS},
              "| to SARM", {v_: s["to_sarm"][v_]["centroid_mm"]["median"] for v_ in ("demons", "slab")},
              "| interp shift", s["interpolation_shift_norm_mm"]["median"])


if __name__ == "__main__":
    main()
