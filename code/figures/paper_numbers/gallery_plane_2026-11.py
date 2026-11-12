#!/usr/bin/env python3
"""The axial plane of Supplementary S2 (the gallery of the 21 completed blockface volumes): the rule the author chose on 11 Nov 2026,
and the plane it gives. Every brain on its canonical fine grid (the run-113 world, its rigid placement), one world axial plane for
all 21, shown from the caudal end to 22 mm rostral of the junction line's median (code/figures/s_synthesis_gallery.py).

The score of a brain S on an axial plane z, in pixels of 0.25 x 0.25 mm in the tile's crop (equal weights; lower is cleaner):
  s(S, z) = G + M + C
  G  gaps: holes of the brain mask in the crop
  M  missing tissue: pixels of the mask's mirror image about the world midline x = 0 that the mask lacks, in the crop
  C  cerebellum at the junction: pixels of the brain's own cerebellum within 1 mm of the junction (|seam_depth| <= 1 mm), in the brain
The cerebellum per brain: SARM's cerebellum (level 2, index 241; NMT v2 sym) in the MRI template (atlas-NMT_v2.0_sym__to__mri-template),
carried forward by the brain's own registration of record (the 13 with MRI: R5, then the rigid b0-native__to__mri-run113 inverted;
the 8 without: S2 xfm_d2s2_prod_s125, then T_world_bf_to_run113 of its S2 placement), each template voxel as 8 half-voxel sub-points
splatted onto the fine grid and closed by 1 voxel (registration-level, used only to choose the plane).
The plane: among the lattice planes whose cohort median synthesized area (brain voxels with seam_depth < 0) is at least 0.8 of the
largest median, the one with the smallest sum of s over the 21 brains (a tie goes to the larger median synthesized area).
Reads the published NMT v2 SARM under settings/link.yaml inputs.atlas_reference (not redistributed; the Makefile skips this record
without it).
-> $LINC_NUMBERS/gallery_plane_2026-11.json
    python code/figures/paper_numbers/gallery_plane_2026-11.py
"""
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
from shared import paths, provenance, xfm  # noqa: E402

VOX, ROSTRAL_MM, FRAC, JUNCTION_MM, CB = 0.25, 22.0, 0.8, 1.0, 241
RULE = ("one world axial plane for all 21 brains (canonical fine grids): among the planes whose cohort median synthesized area is at "
        "least 0.8 of the largest median, the smallest sum over the brains of G + M + C in each tile's crop (G mask gaps, M pixels of "
        "the mask's mirror image about x = 0 that the mask lacks, C the brain's own cerebellum within 1 mm of the junction), in "
        "0.25 mm pixels with equal weights")


def key(z):
    return round(float(z), 2)


def cerebellum_points(R, inputs):
    """the template's cerebellum (SARM level 2 = 241 through the NMT mapping of record) as 8 half-voxel sub-points per voxel"""
    tm = nib.load(str(R / "template/mri/linc_mri_brainmask.nii.gz")); At = tm.affine; Bt = np.asanyarray(tm.dataobj) > 0
    ref = R / yaml.safe_load((REPO / "settings" / "link.yaml").read_text())["inputs"]["atlas_reference"]
    sf = ref / "NMT_v2.0_sym/NMT_v2.0_sym/SARM_in_NMT_v2.0_sym.nii.gz"; sarm = nib.load(str(sf))
    L2 = np.asanyarray(sarm.dataobj[..., 0, 1]).astype(np.int32)                       # level 2
    ent = R / "template/xfm/atlas-NMT_v2.0_sym__to__mri-template"; nmt = xfm.Entry(str(ent))
    inputs += [R / "template/mri/linc_mri_brainmask.nii.gz", sf, ent]
    W = nib.affines.apply_affine(At, np.argwhere(Bt)); lab = np.zeros(len(W), np.int32)
    for k0 in range(0, len(W), 1_000_000):
        P = np.asarray(nmt.pull(W[k0:k0 + 1_000_000])); q = np.rint(nib.affines.apply_affine(np.linalg.inv(sarm.affine), P)).astype(int)
        ok = np.all((q >= 0) & (q < np.array(L2.shape)), 1); lab[k0:k0 + 1_000_000][ok] = L2[tuple(q[ok].T)]
    cbW = W[lab == CB]
    sub = np.array([[dx, dy, dz] for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)], float) * VOX / 4
    return (cbW[:, None, :] + sub[None]).reshape(-1, 3), int(len(cbW))


def carried(S, kind, pts, A, shape, R, inputs):
    """the template cerebellum carried into brain S's canonical fine grid"""
    D = R / "data" / f"sub-{S}"
    if kind == "mri":
        ent, rig = D / "xfm/b0-native__to__mri-template", D / "xfm/b0-native__to__mri-run113/run113_to_b0native_pull.txt"
        e = xfm.Entry(str(ent)); y = np.concatenate([np.asarray(e.pull(pts[k:k + 2_000_000])) for k in range(0, len(pts), 2_000_000)])
        x = nib.affines.apply_affine(np.linalg.inv(np.loadtxt(rig)), y); inputs += [ent, rig]
    else:
        ent, lin = D / "xfm_d2s2_prod_s125/bf-native__to__mri-template", D / f"mri/s2lin_prod/{S}_s2lin.json"
        e = xfm.Entry(str(ent)); y = np.concatenate([np.asarray(e.pull(pts[k:k + 2_000_000])) for k in range(0, len(pts), 2_000_000)])
        x = nib.affines.apply_affine(np.asarray(json.load(open(lin))["T_world_bf_to_run113"]), y); inputs += [ent, lin]
    q = np.rint(nib.affines.apply_affine(np.linalg.inv(A), x[np.isfinite(x).all(1)])).astype(int)
    ok = np.all((q >= 0) & (q < np.array(shape)), 1)
    cb = np.zeros(shape, bool); cb[tuple(q[ok].T)] = True
    return ndi.binary_closing(cb, iterations=1) | cb


def score(v, z):
    A = v["A"]; k = int(round((z - A[2, 3]) / A[2, 2]))
    if not (0 <= k < v["B"].shape[2]):
        return None
    B, dep, cb = v["B"][:, :, k], v["dep"][:, :, k], v["cb"][:, :, k]
    if not B.any():
        return None
    a0 = np.where(B.any(1))[0]; a1 = np.where(B.any(0))[0]
    jy = np.nonzero(B & (np.abs(dep) <= VOX / 2))[1]
    y_j = int(np.round(np.median(jy))) if len(jy) else int(a1.min())
    sl = (slice(a0.min(), a0.max() + 1), slice(a1.min(), y_j + int(round(ROSTRAL_MM / VOX))))
    G = int((ndi.binary_fill_holes(B) & ~B)[sl].sum())
    i0 = int(round(-2 * A[0, 3] / A[0, 0])); ii = i0 - np.arange(B.shape[0]); okm = (ii >= 0) & (ii < B.shape[0])
    Bm = np.zeros_like(B); Bm[okm] = B[ii[okm]]
    M = int((Bm & ~B)[sl].sum())
    C = int((cb & B & (np.abs(dep) <= JUNCTION_MM))[sl].sum())
    return dict(G=G, M=M, C=C, s=G + M + C)


def main():
    P = paths.products("products_completion"); R = paths.products_root(); OUT = P["completed"]
    lab = json.load(open(REPO / "settings" / "specimen_labels.json"))["labels"]
    brains = {}
    for r, kind in (("canon_d3", "mri"), ("canon_d3n_prod", "nomri")):
        root = P["completion_inputs"] / r
        for S in json.load(open(root / "canon_root.json"))["brains"]:
            brains[S] = (root, kind)
    assert len(brains) == 21
    inputs = []
    pts, n_tpl = cerebellum_points(R, inputs)
    V, cbrec = {}, {}
    for S, (root, kind) in sorted(brains.items()):
        f = root / S / "fine/brain.nii.gz"; im = nib.load(str(f)); A = im.affine; B = np.asanyarray(im.dataobj) > 0
        fd = OUT / S / f"{S}_seam_depth.nii.gz"; dep = np.asanyarray(nib.load(str(fd)).dataobj).astype(np.float32)
        cb = carried(S, kind, pts, A, B.shape, R, inputs); inputs += [f, fd]
        V[S] = dict(A=A, B=B, dep=dep, cb=cb, syn={key(A[2, 3] + A[2, 2] * k): int(n) for k, n in enumerate((B & (dep < 0)).sum((0, 1)))})
        cbrec[lab[S]] = dict(cerebellum_ml=round(cb.sum() * VOX ** 3 / 1000, 3), share_inside_brain=round(float((cb & B).sum() / max(cb.sum(), 1)), 3))
        print(lab[S], S, cbrec[lab[S]], flush=True)
    zs = sorted({z for v in V.values() for z in v["syn"]})
    med = {z: float(np.median([v["syn"].get(z, 0) for v in V.values()])) for z in zs}
    Z = [z for z in zs if med[z] >= FRAC * max(med.values())]
    tot = {}
    for z in Z:
        sc = [score(v, z) for v in V.values()]
        if all(sc):
            tot[z] = sum(d["s"] for d in sc)
    z_a = min(tot, key=lambda z: (tot[z], -med[z]))
    per = {lab[S]: score(V[S], z_a) for S in V}
    data = dict(rule=RULE, plane_z_mm=z_a, total=tot[z_a], candidates=dict(z_range_mm=[Z[0], Z[-1]], n=len(Z)),
                totals_per_plane={f"{z:.2f}": t for z, t in tot.items()}, cohort_median_synthesized_vox={f"{z:.2f}": med[z] for z in Z},
                per_brain_at_plane=dict(sorted(per.items(), key=lambda t: (t[0][:2], int(t[0][2:])))),
                cerebellum=dict(template_vox=n_tpl, per_brain=dict(sorted(cbrec.items(), key=lambda t: (t[0][:2], int(t[0][2:]))))))
    provenance.write_numbers("gallery_plane_2026-11", data, inputs=sorted({str(p) for p in inputs}), md5=False,
                             cmd="python code/figures/paper_numbers/gallery_plane_2026-11.py")
    print("plane", z_a, "total", tot[z_a], "candidates", Z[0], "..", Z[-1])


if __name__ == "__main__":
    main()
