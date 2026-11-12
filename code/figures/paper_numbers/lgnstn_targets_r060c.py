#!/usr/bin/env python3
"""How well the LGN and STN of MM1-MM3 agree between brains in each of the three registration targets of the target comparison (NMT
v2, CIVM-R, our template; code/figures/evaluation/targets.py), with the labels transformed into each target (Results, "The template as a
registration target", Table 3). Reference row: the same labels transformed in the same run through the paper's own MRI-based
alignment onto our template.

    python code/figures/paper_numbers/lgnstn_targets_r060c.py [--carry-only S ...] [--score-only]

You need
  the observer's labels of record on the blockface (products_mri.yaml `push_current`: <S>/<S>_<st>_bfnative_L2_demons.nii.gz); each
  brain's blockface-to-b0 registration of record (registry bf-native -> b0-native, settings/brains/<ID>.yaml variant_of_record;
  its QC must pass); the target comparison's entries (products_blockface.yaml `r060_runs`/xfm/sub-<S>/b0-native__to__<target>) and,
  for the reference row, the b0-to-template registration of record (registry b0-native -> mri-template); each target's prepared
  image in $LINC_WORK/r060/targets (run code/figures/paper_numbers/r060_target_comparison.py, or `python -m figures.evaluation.targets prep`,
  first).
What it does
  Transform (the dense rule of the labels of record): each label cropped to its box + 2 mm (at least 1 mm along the section axis),
  Gaussian anti-aliasing per source axis (FWHM = max(source spacing, target voxel)), one linear resample through the chain,
  probability > 0.5. Agreement per target: the three pairs of brains x four hemisphere-structures; the Dice coefficient, the
  average symmetric surface distance (mm) and the centroid distance (mm) between the two brains' labels; also in "ours-scale" mm
  (x (V_ours / V_target)^(1/3), V = the target's brain-mask volume).
You get
  $LINC_WORK/r060c/<S>/<S>_<st>_<target>_{prob,mask}_demons.nii.gz and carry_<S>.json; $LINC_NUMBERS/lgnstn_targets_r060c_h9.json,
  which also holds, unchanged, the superseded r060b summaries (products: numbers_of_record) and the r060b run's blockface volumes
  (`lgnstn_r060b_carries`).

Ported from the analysis project's labels/r060/lgnstn_targets_r060.py (1 Nov 2026, the r060c version). The record's free-text
`method` is kept as the analysis wrote it (METHOD below; its internal request names removed on 3 Nov 2026), as for every record.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
from shared import paths, provenance, xfm  # noqa: E402
from figures.evaluation import targets  # noqa: E402
from figures.evaluation.labels_agreement import AA_RULE  # noqa: E402
from labels.carry import aa_sigma  # noqa: E402

CASES = {"MR243": "MM1", "MR252": "MM2", "MR256": "MM3"}
STRUCTS = ["lhLGN", "lhSTN", "rhLGN", "rhSTN"]
TGT = {"nmt": "atlas-NMT_v2.0_sym", "civmr": "atlas-CIVM-R", "ours": "mri-template"}
REF = "ours_R5_of_record"
# the record's method text (the analysis script's docstring; its paths are the analysis project's)
METHOD = """The target comparison (30 Oct 2026), labels part: the LGN and STN of MM1-MM3 (the observer's labels of record) carried into each of the three
registration targets through the target-comparison registrations, and their agreement between brains in each target space.

Carry (the rule of the labels of record, labels/labels_to_template.py --dense, unchanged): the demons-interpolated blockface label
(labels/push_current/<S>/<S>_<st>_bfnative_L2_demons.nii.gz) cropped to its box + 2 mm, Gaussian anti-aliasing per source axis
(label_refs.aa_sigma, FWHM = max(source spacing, target voxel)), ONE linear resample through the chain, probability > 0.5.
Chain: the R1 of record (bf-native -> b0-native; paper/numbers/r1_current_2026-10-03.json, QC must pass), then the target-comparison entry
(mri/tc/r060_up/xfm/sub-<S>/b0-native__to__<target>; the same engine and settings for every target). Those entries carry no registry
QC status (the comparison's own QC: mid-space mask Dice >= 0.968, no folds), so the chain is built non-strict and the record says so.
Targets: NMT v2.0 sym (in vivo T1w average), CIVM-R (ex vivo; its intensities are NOT_FOR_SHARING, only its grid is used here), ours
(our template reached through the comparison's engine; NOT the R5 of record). Reference row ours_R5_of_record: the same labels carried by this
script in the same run through the R1 of record and the R5 of record (data/sub-<S>/xfm/b0-native__to__mri-template, strict QC) onto
template/mri/linc_mri_b0.nii.gz; checked against the labels of record (labels/push_current/<S>/<S>_<st>_mri-template_mask_demons.nii.gz,
the paper's Table 3; reference_check: voxel counts and Dice).

Agreement per target: the three pairs of brains x four hemisphere-structures; Dice, ASSD (mm) and centroid distance (mm), the metrics
of labels/r059/model_agreement_r059.py. The targets differ in size (an in vivo average vs ex vivo brains), and distances scale with
the brain, so each is also given in "ours-scale" mm: x s = (V_ours / V_target)^(1/3), V = the target's brain-mask volume.
Per row, ours is "better" when its centroid distance is smaller (raw and ours-scale both reported)."""


def crop(img, margin_mm=2.0):
    """the label's box + margin_mm (at least 1 mm along axis 1, the section axis), as float32 (the dense rule: no fill)"""
    a = np.asarray(img.dataobj) > 0
    ijk = np.argwhere(a)
    if not len(ijk):
        return None
    z = np.array(img.header.get_zooms()[:3])
    m = np.ceil(margin_mm / z).astype(int)
    m[1] = max(m[1], int(np.ceil(1.0 / z[1])))
    lo = np.maximum(ijk.min(0) - m, 0); hi = np.minimum(ijk.max(0) + m + 1, a.shape)
    sub = a[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
    aff = img.affine.copy(); aff[:3, 3] = nib.affines.apply_affine(img.affine, lo)
    return sub, aff, z


def metrics(a, b, A):
    """Dice, ASSD (mm), centroid distance (mm) of two masks on one isotropic grid (the analysis project's labels/r059/model_agreement_r059.py)"""
    vx = float(abs(A[0, 0]))
    d = 2 * (a & b).sum() / (a.sum() + b.sum())
    sa, sb = a & ~ndi.binary_erosion(a), b & ~ndi.binary_erosion(b)
    lo = np.maximum(np.minimum(np.argwhere(a).min(0), np.argwhere(b).min(0)) - 5, 0)
    hi = np.minimum(np.maximum(np.argwhere(a).max(0), np.argwhere(b).max(0)) + 6, a.shape)
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    da = ndi.distance_transform_edt(~sb[sl])[sa[sl]]; db = ndi.distance_transform_edt(~sa[sl])[sb[sl]]
    assd = float(np.concatenate([da, db]).mean() * vx)
    ca = nib.affines.apply_affine(A, np.argwhere(a).mean(0)); cb = nib.affines.apply_affine(A, np.argwhere(b).mean(0))
    dv = cb - ca
    return dict(dice=round(float(d), 4), assd_mm=round(assd, 3), centroid_mm=round(float(np.linalg.norm(dv)), 3),
                dx=round(float(dv[0]), 3), dy=round(float(dv[1]), 3), dz=round(float(dv[2]), 3), vol_a_mm3=round(float(a.sum() * vx ** 3), 2),
                vol_b_mm3=round(float(b.sum() * vx ** 3), 2))


def r1_entry(S):
    import yaml
    c = yaml.safe_load((Path(__file__).resolve().parents[3] / "settings" / "brains" / f"{S}.yaml").read_text())
    v = c["registration_to_own_mri"]["variant_of_record"]
    e = xfm.Entry(paths.products("products_blockface")["data"] / f"sub-{S}" / f"xfm_{v}" / "bf-native__to__b0-native")
    assert e.qc_status() in ("pass", "pass-visual"), (S, e.qc_status())
    return e


def carry(S, W, out):
    P1, P2 = paths.products("products_blockface"), paths.products("products_mri")
    d = out / S; d.mkdir(parents=True, exist_ok=True)
    r1 = r1_entry(S); log = dict(r1=str(r1.dir), targets={})
    ents = {t: xfm.Entry(P1["r060_runs"] / "xfm" / f"sub-{S}" / f"b0-native__to__{tname}") for t, tname in TGT.items()}
    ents[REF] = xfm.Entry(P1["data"] / f"sub-{S}" / "xfm" / "b0-native__to__mri-template")
    for t, ent in ents.items():
        chain = xfm.Chain([r1, ent], strict=(t == REF))
        ref = P1["template_mri"] / "linc_mri_b0.nii.gz" if t == REF else W.tgt_ch(t); vox = float(np.mean(nib.load(str(ref)).header.get_zooms()[:3]))
        log["targets"][t] = dict(entry=str(ent.dir), r060_entry_qc=ent.qc_status(), ref=str(ref), chain=chain.describe(), structs={})
        for st in STRUCTS:
            t0 = time.time()
            img = nib.load(str(P2["push_current"] / S / f"{S}_{st}_bfnative_L2_demons.nii.gz"))
            sub, aff, zooms = crop(img)
            sig = aa_sigma(zooms, vox)
            tmp = d / f"_tmp_{st}_{t}.nii.gz"
            nib.save(nib.Nifti1Image(gaussian_filter(sub, sig), aff), str(tmp))
            prob = d / f"{S}_{st}_{t}_prob_demons.nii.gz"
            xfm.apply(chain, str(tmp), str(ref), str(prob), order=1); tmp.unlink()
            p = nib.load(str(prob)); pa = np.asarray(p.dataobj); m = (pa > 0.5).astype(np.uint8)
            nib.save(nib.Nifti1Image(m, p.affine), str(d / f"{S}_{st}_{t}_mask_demons.nii.gz"))
            vz = float(np.prod(p.header.get_zooms()[:3]))
            log["targets"][t]["structs"][st] = dict(vol_bf_mm3=round(float(sub.sum() * np.prod(zooms)), 3), voxels=int(m.sum()),
                                                    vol_mm3=round(float(m.sum() * vz), 3), aa_sigma_src_vox=[round(float(x), 3) for x in sig],
                                                    seconds=round(time.time() - t0, 1))
            print(S, t, st, log["targets"][t]["structs"][st], flush=True)
    log["anti_aliasing"] = AA_RULE; log["dense_rule"] = True
    json.dump(log, open(d / f"carry_{S}.json", "w"), indent=1)


def score(W, out):
    P1, P2 = paths.products("products_blockface"), paths.products("products_mri")
    SRC = P2["push_current"]
    inputs = []

    def load(path):
        inputs.append(Path(path)); im = nib.load(str(path)); m = np.asanyarray(im.dataobj) > 0
        return (m if m.any() else None), im.affine
    V = {}
    for t in TGT:
        mk = nib.load(str(W.tgt_mask(t))); inputs.append(W.tgt_mask(t))
        V[t] = float((np.asanyarray(mk.dataobj) > 0).sum() * np.prod(mk.header.get_zooms()[:3]))
    scale = {t: round((V["ours"] / V[t]) ** (1 / 3), 4) for t in TGT}
    pth = {t: (lambda S, st, t=t: out / S / f"{S}_{st}_{t}_mask_demons.nii.gz") for t in [*TGT, REF]}
    check = {}                                                        # the reference row against the labels of record (Table 3)
    for S in CASES:
        for st in STRUCTS:
            m, _ = load(pth[REF](S, st)); r, _ = load(SRC / S / f"{S}_{st}_mri-template_mask_demons.nii.gz")
            check.setdefault(CASES[S], {})[st] = dict(voxels=int(m.sum()), voxels_record=int(r.sum()), identical=bool((m == r).all()),
                                                      dice=round(float(2 * (m & r).sum() / (m.sum() + r.sum())), 4))
    sets = {}
    for t, pf in pth.items():
        rows = []
        for a, b in itertools.combinations(CASES, 2):
            for st in STRUCTS:
                ma, A = load(pf(a, st)); mb, _ = load(pf(b, st))
                r = dict(a=CASES[a], b=CASES[b], struct=st)
                if ma is None or mb is None:
                    r["missing"] = True
                else:
                    r.update(metrics(ma, mb, A))
                    s = scale.get(t, 1.0); r["centroid_mm_ours_scale"] = round(r["centroid_mm"] * s, 3); r["assd_mm_ours_scale"] = round(r["assd_mm"] * s, 3)
                rows.append(r)
        summ = {}
        for nuc in ("LGN", "STN"):
            rr = [r for r in rows if r["struct"].endswith(nuc) and "centroid_mm" in r]
            q = lambda k: [round(float(x), 3) for x in np.percentile([r[k] for r in rr], [25, 50, 75])]  # noqa: E731
            summ[nuc] = dict(n=len(rr), **{f"{k}_q25_50_75": q(k) for k in ("centroid_mm", "centroid_mm_ours_scale", "assd_mm", "assd_mm_ours_scale", "dice")})
        sets[t] = dict(rows=rows, summary=summ)
        print(t, json.dumps(summ), flush=True)
    better = {}
    for t in ("nmt", "civmr"):
        for k in ("centroid_mm", "centroid_mm_ours_scale"):
            for nuc in ("LGN", "STN"):
                pr = [(o, x) for o, x in zip(sets["ours"]["rows"], sets[t]["rows"]) if o["struct"].endswith(nuc) and k in o and k in x]
                better.setdefault(f"ours_vs_{t}", {}).setdefault(k, {})[nuc] = dict(ours_better=sum(o[k] < x[k] for o, x in pr), n=len(pr))
    carries = {S: json.load(open(out / S / f"carry_{S}.json")) for S in CASES}
    old_carries = {S: json.load(open(P1["lgnstn_r060b_carries"] / S / f"carry_{S}.json")) for S in CASES}
    old = P1["numbers_of_record"] / "lgnstn_targets_r060b.json"; inputs += [old] + [out / S / f"carry_{S}.json" for S in CASES]
    O = json.loads(old.read_text()); O = O.get("data") or O
    assert all(c["dense_rule"] for c in carries.values())
    rec = dict(method=METHOD, target_brain_volume_mm3={t: round(v, 1) for t, v in V.items()}, ours_scale_factor=scale,
               sets=sets, ours_better=better, reference_check=check,
               r060_entry_qc={S: {t: c["targets"][t]["r060_entry_qc"] for t in [*TGT, REF]} for S, c in carries.items()},
               carry_volumes_mm3={CASES[S]: {t: {st: v["vol_mm3"] for st, v in c["targets"][t]["structs"].items()} for t in [*TGT, REF]} for S, c in carries.items()},
               bf_volume_mm3={CASES[S]: {st: dict(dense=v["vol_bf_mm3"], r060b_filled=old_carries[S]["targets"]["nmt"]["structs"][st]["vol_bf_mm3"])
                                         for st, v in c["targets"]["nmt"]["structs"].items()} for S, c in carries.items()},
               superseded_r060b=dict(summary={t: v["summary"] for t, v in O["sets"].items()}, ours_better=O["ours_better"]))
    provenance.write_numbers("lgnstn_targets_r060c_h9", rec, inputs=sorted(set(inputs)), md5=False,
                             cmd=" ".join(["python code/figures/paper_numbers/lgnstn_targets_r060c.py"] + sys.argv[1:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--carry-only", nargs="*", default=None, metavar="S", help="carry these brains only (default: all three, then score)")
    ap.add_argument("--score-only", action="store_true", help="score the carried labels in $LINC_WORK/r060c")
    a = ap.parse_args()
    W = targets.Work(paths.work_root() / "r060"); out = paths.work_root() / "r060c"
    assert all(W.tgt_ch(t).exists() for t in TGT), "prepare the targets first: python -m figures.evaluation.targets prep"
    if not a.score_only:
        for S in (a.carry_only or CASES):
            carry(S, W, out)
    if a.carry_only is None:
        score(W, out)


if __name__ == "__main__":
    main()
