#!/usr/bin/env python3
"""Join each tracer case's bundle label into one connected pathway, marking the joins as inferred (Fig. 10; Supplementary S2).

You need (configs/products_mri.yaml)
  tracer_template_v2: each case's interpolated bundle density and its connected solid in the template; template_mri: the brain mask
  and the white matter (FA > 0.25).

What it does, per case
  1. splits the connected solid into its connected components and keeps the largest as the pathway;
  2. joins every other component along the cheapest path inside the brain (cheap where the density is high, cheaper in white
     matter), the cheapest component first, the path widened to a 1 mm tube; drops only specks smaller than 8 voxels;
  3. marks the join voxels that were not in the solid as inferred.

You get
  $LINC_WORK/tracer_v2c/<ID>/<ID>_Bundles_mri-template_connected.nii.gz (uint8: 1 = charted, 2 = inferred), _solid_v2c.nii.gz
  (the pathway, 0/1), _bridges.nii.gz (each join's component number), <ID>_connect_v2c.json; and
  $LINC_NUMBERS/tracer_connected_v2c.json (per case: the components, every join with its length, the inferred fraction, the longest join)

    python reproduce/numbers/tracer_connected_v2c.py [--cases MR243 ...]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from linc import paths, provenance  # noqa: E402

NAME = "tracer_connected_v2c"
CASES = {"MR243": "MM1", "MR252": "MM2", "MR256": "MM3", "MF191": "MF8"}
GROUP = "Bundles"
VOX, EPS, LONG_MM, MIN_VOX, TUBE_R, PAD = 0.25, 0.05, 10.0, 8, 2, 40
S26 = np.ones((3, 3, 3), bool)
# the record's method text, fixed (it was this script's docstring when the record was made; the docstring may change)
METHOD = [
    'One connected pathway per tracer case (paper: Results, tracer labels; Fig. 10; Supplementary S2): the interpolated bundle label of each',
    'case joined into a single component, the joins marked as inferred.',
    '',
    'Inputs (products_mri `tracer_template_v2`): <S>/<S>_Bundles_mri-template_density.nii.gz (the transformed density of the interpolated bundle',
    'label, 0-1 on the 0.25 mm template grid) and <S>_Bundles_mri-template_solid2.nii.gz (its connected solid: density > 0.5 grown through',
    'connected voxels of density >= 0.25); the template brain mask and white matter (template_mri: linc_mri_brainmask, linc_mri_wm_fa025).',
    'Rule: the 26-connected components of the connected solid, largest first; every other component is joined to the pathway along the cheapest',
    'path inside the template brain mask (voxel cost 1 / (density + EPS), halved in white matter and inside the footprint, density >= 0.1;',
    '6-connected graph; multi-source Dijkstra from the current pathway, the cheapest component first, so chains of components connect through each',
    'other), the path dilated to a tube of TUBE_R voxels; components smaller than MIN_VOX voxels are dropped (listed), no component is dropped for',
    'distance; every join is recorded with its length and voxel count and flagged when longer than LONG_MM. Join voxels that were not in',
    'the solid are inferred: label 2 in the written label (1 = the solid).',
]


def load(p):
    im = nib.load(str(p)); return np.asanyarray(im.dataobj), im.affine


def ball(r):
    z, y, x = np.mgrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return (x * x + y * y + z * z) <= r * r


def build_graph(domain, cost):
    """6-connected graph over the domain voxels (node = voxel index), edge weight = mean voxel cost x VOX."""
    idx = -np.ones(domain.shape, np.int64); n = int(domain.sum()); idx[domain] = np.arange(n)
    rows, cols, w = [], [], []
    for ax in range(3):
        a, c = np.moveaxis(idx, ax, 0), np.moveaxis(cost, ax, 0)
        i1, i2 = a[:-1], a[1:]; ok = (i1 >= 0) & (i2 >= 0)
        rows.append(i1[ok]); cols.append(i2[ok]); w.append(0.5 * (c[:-1][ok] + c[1:][ok]) * VOX)
    r, c_, ww = np.concatenate(rows), np.concatenate(cols), np.concatenate(w)
    G = coo_matrix((np.concatenate([ww, ww]), (np.concatenate([r, c_]), np.concatenate([c_, r]))), shape=(n + 1, n + 1)).tocsr()
    return idx, n, G


def connect(S, lab_dir, tpl, out_dir, log=print):
    t0 = time.time()
    D, A = load(lab_dir / S / f"{S}_{GROUP}_mri-template_density.nii.gz"); D = np.clip(D.astype(np.float32), 0, 1)
    S0 = load(lab_dir / S / f"{S}_{GROUP}_mri-template_solid2.nii.gz")[0] > 0
    brain = load(tpl / "linc_mri_brainmask.nii.gz")[0] > 0; wm = load(tpl / "linc_mri_wm_fa025.nii.gz")[0] > 0
    F = D >= 0.1
    lab, n = ndi.label(S0, S26); sizes = np.bincount(lab.ravel())[1:]
    order = np.argsort(-sizes) + 1
    main_id = int(order[0]); main = lab == main_id
    rec = dict(case=CASES[S], subject=S, params=dict(vox_mm=VOX, cost="1/(density + eps), x0.5 in white matter (FA > 0.25) or the footprint (density >= 0.1)",
                                                     eps=EPS, long_bridge_flag_mm=LONG_MM, min_piece_vox=MIN_VOX, tube_radius_vox=TUBE_R, graph="6-connected, inside the template brain mask",
                                                     drop="specks only (no distance cap: every charted component is joined)",
                                                     pieces="26-connected components of v2's connected solid (density > 0.5 grown through >= 0.25)"),
               v2_solid=dict(voxels=int(S0.sum()), pieces=int(n), largest_share=round(float(sizes.max() / S0.sum()), 4), piece_voxels=sorted(sizes.tolist(), reverse=True)),
               attached=[], dropped=[])
    remaining = [int(c) for c in order[1:]]
    for cid in list(remaining):
        if sizes[cid - 1] < MIN_VOX:
            rec["dropped"].append(dict(piece=cid, voxels=int(sizes[cid - 1]), reason=f"smaller than {MIN_VOX} voxels")); remaining.remove(cid)
    nz = np.argwhere(F | S0); lo = np.maximum(nz.min(0) - PAD, 0); hi = np.minimum(nz.max(0) + PAD + 1, np.array(D.shape))
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    dom = brain[sl]; Dc = D[sl]; cost = 1.0 / (Dc + EPS); cost[(wm[sl] | F[sl])] *= 0.5
    idx, nn, G = build_graph(dom, cost)
    log(f"{S}: pieces {n} (largest {rec['v2_solid']['largest_share']}), domain {dom.shape} nodes {nn}, {time.time() - t0:.0f} s")
    labc = lab[sl]; final = main.copy(); inferred = np.zeros_like(S0); bridges = np.zeros(S0.shape, np.int32)
    while remaining:
        src = idx[final[sl]]; src = src[src >= 0]
        dist, pred, _ = dijkstra(G, directed=True, indices=src, min_only=True, return_predecessors=True)
        cands = []
        for cid in remaining:
            nodes = idx[labc == cid]; nodes = nodes[nodes >= 0]
            if len(nodes) == 0:
                cands.append((np.inf, cid, -1, 0.0)); continue
            j = int(np.argmin(dist[nodes])); d = float(dist[nodes[j]])
            if not np.isfinite(d):
                cands.append((np.inf, cid, -1, 0.0)); continue
            path = []; k = nodes[j]
            while k >= 0:
                path.append(k); k = pred[k]
            cands.append((d, cid, nodes[j], (len(path) - 1) * VOX, path))
        ok = [c for c in cands if np.isfinite(c[0])]
        if not ok:
            for c in cands:
                rec["dropped"].append(dict(piece=c[1], voxels=int(sizes[c[1] - 1]), reason="no path inside the brain mask"))
            break
        d, cid, node, length, path = min(ok, key=lambda c: c[0])
        pm = np.zeros(dom.shape, bool); flat = np.zeros(nn, bool); flat[path] = True; pm[dom] = flat
        tube = ndi.binary_dilation(pm, ball(TUBE_R)) & dom
        piece = labc == cid
        newvox = (tube | piece) & ~final[sl]
        inf = tube & ~S0[sl] & ~final[sl]
        final[sl] |= tube | piece; inferred[sl] |= inf; bridges[sl][inf] = cid
        rec["attached"].append(dict(piece=cid, voxels=int(sizes[cid - 1]), path_mm=round(length, 2), long=bool(length > LONG_MM), cost=round(d, 3), bridge_voxels=int(inf.sum()), added_voxels=int(newvox.sum())))
        remaining.remove(cid)
    labf, nf = ndi.label(final, S26)
    assert nf == 1, f"{S}: the final solid has {nf} pieces"
    tot = int(final.sum()); ninf = int(inferred.sum())
    rec.update(final=dict(voxels=tot, mm3=round(tot * VOX ** 3, 3), pieces=int(nf), inferred_voxels=ninf, inferred_mm3=round(ninf * VOX ** 3, 3),
                          inferred_share=round(ninf / tot, 4), v2_solid_voxels_kept=int((final & S0).sum()), v2_solid_voxels_dropped=int((S0 & ~final).sum()),
                          dropped_share_of_v2_solid=round(float((S0 & ~final).sum() / S0.sum()), 4), pieces_attached=len(rec["attached"]), pieces_dropped=len(rec["dropped"]),
                          bridges_longer_than_flag=sum(a["long"] for a in rec["attached"]), longest_bridge_mm=max([a["path_mm"] for a in rec["attached"]] or [0.0]),
                          bridge_total_mm=round(sum(a["path_mm"] for a in rec["attached"]), 2)))
    log_wall = round(time.time() - t0, 1)
    od = out_dir / S; od.mkdir(parents=True, exist_ok=True)
    con = np.where(final & S0, 1, 0).astype(np.uint8); con[inferred] = 2
    nib.save(nib.Nifti1Image(con, A), str(od / f"{S}_{GROUP}_mri-template_connected.nii.gz"))
    nib.save(nib.Nifti1Image(final.astype(np.uint8), A), str(od / f"{S}_{GROUP}_mri-template_solid_v2c.nii.gz"))
    nib.save(nib.Nifti1Image(bridges, A), str(od / f"{S}_{GROUP}_mri-template_bridges.nii.gz"))
    (od / f"{S}_connect_v2c.json").write_text(json.dumps(rec, indent=1))
    log(f"{S}: final {tot} vox in 1 piece; inferred {ninf} ({100 * ninf / tot:.2f} %); attached {len(rec['attached'])}, dropped {len(rec['dropped'])}; {log_wall} s")
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="+", default=list(CASES)); ap.add_argument("--labels", default="tracer_template_v2", help="products_mri key of the interpolated labels")
    a = ap.parse_args()
    P = paths.products("products_mri"); lab_dir, tpl = P[a.labels], P["template_mri"]; out_dir = paths.work_root() / "tracer_v2c"
    rec = dict(method=METHOD, cases={}); inputs = []
    for S in a.cases:
        rec["cases"][S] = connect(S, lab_dir, tpl, out_dir)
        inputs += [lab_dir / S / f"{S}_{GROUP}_mri-template_density.nii.gz", lab_dir / S / f"{S}_{GROUP}_mri-template_solid2.nii.gz"]
    inputs += [tpl / "linc_mri_brainmask.nii.gz", tpl / "linc_mri_wm_fa025.nii.gz"]
    print(provenance.write_numbers(NAME, rec, inputs=inputs, md5=False, cmd=" ".join(["python reproduce/numbers/tracer_connected_v2c.py"] + sys.argv[1:])))


if __name__ == "__main__":
    main()
