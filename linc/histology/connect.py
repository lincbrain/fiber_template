"""Join a tracer case's bundle label into one connected pathway in the template, marking the joins as inferred (paper: Supplementary S2).

You need
  the case's carried densities in one folder, <ID>_<group>_mri-template_density.nii.gz (linc-labels carry), and the template's brain
  mask and white matter (FA > 0.25).

What it does
  1. solid: per group, the voxels of density > 0.5 grown through 26-connected voxels of density >= 0.25;
  2. keeps the largest component of the bundles' solid as the pathway;
  3. joins every other component along the cheapest path inside the brain (cheap where the density is high, cheaper in white matter),
     the cheapest component first, the path widened to a 1 mm tube; drops only specks smaller than 8 voxels;
  4. marks the join voxels that were not in the solid as inferred.

You get, in the folder
  <ID>_<group>_mri-template_solid2.nii.gz (uint8, 0/1), <ID>_Bundles_mri-template_connected.nii.gz (uint8: 1 = charted, 2 = inferred),
  _solid_v2c.nii.gz (the pathway, 0/1), _bridges.nii.gz (each join's component number), and <ID>_connect_v2c.json (every join with its
  length, the components dropped, the inferred fraction).
"""
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

CASES = {"MR243": "MM1", "MR252": "MM2", "MR256": "MM3", "MF191": "MF8"}
GROUP, GROUPS = "Bundles", ("Bundles", "Terminals", "Injection_site")
VOX, EPS, LONG_MM, MIN_VOX, TUBE_R, PAD = 0.25, 0.05, 10.0, 8, 2, 40
S26 = np.ones((3, 3, 3), bool)


def solid(density, hi=0.5, lo=0.25):
    """Voxels of density > hi grown through 26-connected voxels of density >= lo."""
    lab, n = ndi.label(density >= lo, S26)
    keep = np.zeros(n + 1, bool); keep[np.unique(lab[density > hi])] = True; keep[0] = False
    return keep[lab]


def write_solids(S, case_dir):
    """<ID>_<group>_mri-template_solid2.nii.gz beside each carried group density."""
    for g in GROUPS:
        f = Path(case_dir) / f"{S}_{g}_mri-template_density.nii.gz"
        if f.exists():
            im = nib.load(str(f)); d = np.clip(np.asanyarray(im.dataobj).astype(np.float32), 0, 1)
            nib.save(nib.Nifti1Image(solid(d).astype(np.uint8), im.affine), str(f).replace("_density.nii.gz", "_solid2.nii.gz"))


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


def connect(S, case_dir, tpl, out_dir, log=print):
    t0 = time.time()
    D, A = load(Path(case_dir) / f"{S}_{GROUP}_mri-template_density.nii.gz"); D = np.clip(D.astype(np.float32), 0, 1)
    S0 = load(Path(case_dir) / f"{S}_{GROUP}_mri-template_solid2.nii.gz")[0] > 0
    brain = load(tpl / "linc_mri_brainmask.nii.gz")[0] > 0; wm = load(tpl / "linc_mri_wm_fa025.nii.gz")[0] > 0
    F = D >= 0.1
    lab, n = ndi.label(S0, S26); sizes = np.bincount(lab.ravel())[1:]
    order = np.argsort(-sizes) + 1
    main_id = int(order[0]); main = lab == main_id
    rec = dict(case=CASES.get(S, S), subject=S, params=dict(vox_mm=VOX, cost="1/(density + eps), x0.5 in white matter (FA > 0.25) or the footprint (density >= 0.1)",
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
    od = Path(out_dir); od.mkdir(parents=True, exist_ok=True)
    con = np.where(final & S0, 1, 0).astype(np.uint8); con[inferred] = 2
    nib.save(nib.Nifti1Image(con, A), str(od / f"{S}_{GROUP}_mri-template_connected.nii.gz"))
    nib.save(nib.Nifti1Image(final.astype(np.uint8), A), str(od / f"{S}_{GROUP}_mri-template_solid_v2c.nii.gz"))
    nib.save(nib.Nifti1Image(bridges, A), str(od / f"{S}_{GROUP}_mri-template_bridges.nii.gz"))
    (od / f"{S}_connect_v2c.json").write_text(json.dumps(rec, indent=1))
    log(f"{S}: final {tot} vox in 1 piece; inferred {ninf} ({100 * ninf / tot:.2f} %); attached {len(rec['attached'])}, dropped {len(rec['dropped'])}; {log_wall} s")
    return rec
