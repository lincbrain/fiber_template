"""A sanity check of reconstructed blockface volumes: is each one oriented, consistent and sane?

    python -m linc.blockface.sanity --data $LINC_DATA MR252 [...]

You need
  the reconstructions under <data>/sub-<ID>/blockface (<ID>_blockface.ome.zarr, <ID>_mask, <ID>_holes).
What it does
  ORIENT   the store's section axis must run anterior to posterior (negative step, section 0 frontal).
  MIRROR   the grayscale volume must agree with its own brain mask in world space (an A-P flip of one but not the other cannot be
           seen in either volume alone, and ruins every registration that uses the grayscale).
  CHIRAL   the marker tubes are in the LEFT hemisphere, so their centroid must sit at negative world x relative to the brain's.
  SIZE     the L-R and D-V extent against the cohort's median (more than 18 % off is a voxel-size error, not a brain); A-P is not
           checked, since the brains were cut caudally by different amounts.
  ALIGN    the brain-centroid step from section to section: what the camera-jump correction is there to deliver.
You get
  JSON on stdout: one row per brain with each check's numbers and its findings (empty when all pass), and the cohort medians.

Ported from the analysis project's code/qc_bf_sanity.py (the checks unchanged; the data folder is an argument).
"""
import argparse
import json
import os
import sys

import numpy as np

LR_TOL, DV_TOL = 0.18, 0.18          # fraction off the cohort median that counts as a finding
MIRROR_MIN = 0.90                    # grey-vs-mask world agreement below this means they disagree


def _world(idx, A):
    return (A[:3, :3] @ idx.T).T + A[:3, 3]


def _extent(mask, A):
    """World bounding box, from the three axis projections - argwhere on a full volume is needlessly slow."""
    idx = []
    for ax in range(3):
        p = mask.any(axis=tuple(k for k in range(3) if k != ax))
        nz = np.flatnonzero(p)
        if not nz.size:
            return None
        idx.append((int(nz.min()), int(nz.max())))
    corners = np.array([[idx[0][i], idx[1][j], idx[2][k]]
                        for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)
    w = _world(corners, A)
    return w.min(0), w.max(0), w.max(0) - w.min(0)


def _store_slice_step(path):
    """The store's declared slice-axis world step (mm), from its OME metadata - no array is read."""
    import json as _j
    for name in ("zarr.json", ".zattrs"):
        f = os.path.join(path, name)
        if os.path.exists(f):
            d = _j.load(open(f))
            at = d.get("attributes", d)
            ms = (at.get("ome") or at).get("multiscales") or at["multiscales"]
            ct = ms[0]["datasets"][0]["coordinateTransformations"]
            sc = [t for t in ct if t["type"] == "scale"][0]["scale"]
            return float(sc[2]) / 1000.0        # index 2 is the SLICE axis (same as fix_ap_orientation), um
    raise SystemExit(f"no OME metadata in {path}")


def _dice(a, b):
    return float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))


def check(S, data_root):
    """The checks of one reconstruction under <data_root>/sub-<ID>/blockface (the module docstring)."""
    import nibabel as nib
    import nibabel.processing as nip
    from scipy import ndimage as ndi
    bd = os.path.join(data_root, f"sub-{S}", "blockface")
    final = lambda n: os.path.join(bd, f"{S}_{n}.nii.gz")      # noqa: E731
    mask_store = os.path.join(bd, "02_blockface_mask.ome.zarr")
    r = dict(subject=S, findings=[])
    if not os.path.isdir(mask_store):
        r["findings"].append("no mask store (not reconstructed)")
        return r

    # The STORE is consulted for its declared slice axis only - that is metadata, and loading a 1400-slice
    # mask pyramid to read a sign costs minutes. Everything measurable is measured on the deliverable masks,
    # which are small and are what the rest of the pipeline actually consumes.
    r["slice_step_mm"] = round(_store_slice_step(mask_store), 4)
    if r["slice_step_mm"] >= 0:
        r["findings"].append(f"ORIENT: store slice-axis step {r['slice_step_mm']:+.3f} is not negative "
                             f"(A-P not migrated)")
    mp = final("mask")
    if not (mp and os.path.exists(mp)):
        r["findings"].append("no deliverable brain mask")
        return r
    mi = nib.load(mp)
    m = np.asanyarray(mi.dataobj) > 0
    A = mi.affine
    ax = int(np.argmax(np.abs(A[:3, :3][1])))
    r["n_slices"] = int(m.shape[ax])

    ext = _extent(m, A)
    if ext is None:
        r["findings"].append("empty brain mask")
        return r
    lo, hi, size = ext
    r["LR_mm"], r["DV_mm"], r["AP_mm"] = [round(float(v), 1) for v in (size[0], size[2], size[1])]
    r["y_slice0_mm"], r["y_slice_last_mm"] = round(float(hi[1]), 1), round(float(lo[1]), 1)

    # MIRROR: the iso grey against this mask, both in world space
    g = final("blockface_iso_grey")
    if g and os.path.exists(g):
        gi = nib.load(g)
        st = mi
        G = (np.asanyarray(gi.dataobj).astype(np.float32) > 0).astype(np.uint8)
        gr = np.asanyarray(nip.resample_from_to(nib.Nifti1Image(G, gi.affine), st, order=0).dataobj) > 0
        r["grey_vs_mask_dice"] = round(_dice(m, gr), 3)
        if r["grey_vs_mask_dice"] < MIRROR_MIN:
            n = gi.shape
            Af = gi.affine.copy(); Af[:, 1] = -gi.affine[:, 1]
            Af[:3, 3] = gi.affine[:3, :3] @ np.array([0, n[1] - 1, 0]) + gi.affine[:3, 3]
            fr = np.asanyarray(nip.resample_from_to(nib.Nifti1Image(G, Af), st, order=0).dataobj) > 0
            d2 = _dice(m, fr)
            r["grey_vs_mask_dice_mirrored"] = round(d2, 3)
            r["findings"].append(
                f"MIRROR: grey disagrees with its mask (dice {r['grey_vs_mask_dice']:.3f}"
                + (f", but {d2:.3f} A-P MIRRORED -> re-run prep_bf" if d2 > r["grey_vs_mask_dice"] + 0.05
                   else ") -> investigate"))

    # CHIRAL: drill holes must be LEFT of the brain centre
    h = final("holes")
    if h and os.path.exists(h):
        hi_ = nib.load(h)
        H = np.asanyarray(hi_.dataobj) > 0
        r["holes_voxels"] = int(H.sum())
        if H.sum() >= 20:
            hx = float(_world(np.argwhere(H), hi_.affine)[:, 0].mean())
            bx = float(_world(np.argwhere(m), A)[:, 0].mean())
            r["holes_dx_mm"] = round(hx - bx, 1)
            if hx - bx > 0:
                r["findings"].append(f"CHIRAL: drill holes sit {hx - bx:+.1f} mm RIGHT of brain centre "
                                     f"(they mark the LEFT hemisphere)")
        else:
            r["findings"].append(f"holes mask has only {int(H.sum())} voxels (under-segmented?)")

    # ALIGN: frame-to-frame centroid step
    nz = m.shape[ax]
    stride = max(1, nz // 400)
    ks = range(0, nz, stride)
    c = np.full((len(list(ks)), 2), np.nan)
    for i, k in enumerate(range(0, nz, stride)):
        s = np.take(m, k, axis=ax)
        if s.sum() > 200:
            c[i] = ndi.center_of_mass(s)
    vox = float(np.abs(A[0, 0]))
    d = np.linalg.norm(np.diff(c, axis=0), axis=1) * vox
    d = d[np.isfinite(d)]
    if d.size:
        r["step_median_mm"] = round(float(np.median(d)), 3)
        r["step_p95_mm"] = round(float(np.percentile(d, 95)), 3)
        r["step_stride"] = stride
    return r


def judge_size(rows):
    done = [r for r in rows if "LR_mm" in r]
    lr = np.median([r["LR_mm"] for r in done]); dv = np.median([r["DV_mm"] for r in done])
    for r in done:
        r["findings"] = [f for f in r["findings"] if not f.startswith("SIZE:")]
        for k, med, tol in (("LR_mm", lr, LR_TOL), ("DV_mm", dv, DV_TOL)):
            if abs(r[k] - med) / med > tol:
                r["findings"].append(f"SIZE: {k} {r[k]} vs cohort median {med:.1f} "
                                     f"({100 * (r[k] - med) / med:+.0f} %)")
    return float(lr), float(dv)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subjects", nargs="+")
    ap.add_argument("--data", default=os.environ.get("LINC_DATA"), help="per-brain data root ($LINC_DATA)")
    a = ap.parse_args()
    rows = [check(S, a.data) for S in a.subjects]
    lr, dv = judge_size(rows)
    json.dump(dict(rows=rows, cohort_median_LR_mm=lr, cohort_median_DV_mm=dv), sys.stdout, indent=1)


if __name__ == "__main__":
    main()
