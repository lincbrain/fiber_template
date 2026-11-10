#!/usr/bin/env python3
"""How many marker tubes each sectioned brain has (paper: Methods, "Sectioning and blockface photography").

    python code/figures/paper_numbers/bf_tubes_per_brain_r073.py

You need
  each brain's tube mask in the products (products_blockface `data`: sub-<ID>/blockface/<ID>_holes.nii.gz, level 4 in plane, native
  sections) and settings/specimen_labels.json.
What it does
  The masks do not label tubes, so it gives three counts per brain:
    components   3D connected parts of the mask (26-connectivity); a tube that is not visible over some sections is split in parts
    tracks       the parts joined into tubes: part b continues part a when b starts after a ends (along the sections) and b's first
                 in-plane center lies within JOIN_MM of a's last (union over all such pairs)
    per section  the number of separate tube profiles in a section, over the sections with at least one (median, 90th percentile,
                 maximum)
  and their median, quartiles and range over the 21 brains.
You get
  $LINC_NUMBERS/bf_tubes_per_brain_r073.json.

Ported from the analysis project's paper/r073_l1/bf_tubes_per_brain_r073.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
from shared import paths, provenance  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
LABELS = REPO / "settings" / "specimen_labels.json"
JOIN_MM = 1.5


def one(path):
    h = nib.load(str(path)); H = np.asanyarray(h.dataobj) > 0; vx = abs(h.affine[0, 0])
    per = [n2 for n2 in (ndi.label(H[:, k, :])[1] for k in range(H.shape[1])) if n2]
    L, n = ndi.label(H, np.ones((3, 3, 3)))
    parts = []
    for i, o in enumerate(ndi.find_objects(L), 1):
        m = L[o] == i
        ks = np.where(m.any(axis=(0, 2)))[0]

        def cen(kk):
            yy, xx = np.where(m[:, kk, :])
            return np.array([yy.mean() + o[0].start, xx.mean() + o[2].start]) * vx
        parts.append(dict(z0=int(ks[0] + o[1].start), z1=int(ks[-1] + o[1].start), c0=cen(ks[0]), c1=cen(ks[-1])))
    par = list(range(n))

    def root(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for a in range(n):
        for b in range(n):
            if a != b and parts[b]["z0"] > parts[a]["z1"] and np.linalg.norm(parts[a]["c1"] - parts[b]["c0"]) <= JOIN_MM:
                par[root(b)] = root(a)
    return dict(components=n, tracks=len({root(a) for a in range(n)}), per_section_median=float(np.median(per)),
                per_section_p90=float(np.percentile(per, 90)), per_section_max=int(max(per)))


def summ(v):
    v = np.asarray(v, float)
    return dict(median=float(np.median(v)), q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75)), min=float(v.min()), max=float(v.max()))


def main():
    lab = json.loads(LABELS.read_text())["labels"]
    data = paths.products("products_blockface")["data"]
    subs = sorted((s for s in lab if (data / f"sub-{s}" / "blockface" / f"{s}_holes.nii.gz").exists()), key=lambda s: (lab[s][:2], int(lab[s][2:])))
    assert len(subs) == 21, subs
    per, ins = {}, []
    for S in subs:
        p = data / f"sub-{S}" / "blockface" / f"{S}_holes.nii.gz"; ins.append(p)
        per[S] = dict(name=lab[S], **one(p)); print(S, per[S], flush=True)
    out = dict(per_brain=per, join_mm=JOIN_MM,
               summary=dict(tracks=summ([v["tracks"] for v in per.values()]), components=summ([v["components"] for v in per.values()]),
                            per_section_median=summ([v["per_section_median"] for v in per.values()]),
                            per_section_max=summ([v["per_section_max"] for v in per.values()])),
               note="tube masks of record; tracks = parts joined along the sections when the next starts within join_mm in plane of the last's end")
    provenance.write_numbers("bf_tubes_per_brain_r073", out, inputs=ins, md5=False, cmd="python code/figures/paper_numbers/bf_tubes_per_brain_r073.py")
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
