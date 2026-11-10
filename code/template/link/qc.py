"""Scores of the link and of the atlas mappings: the outline check, the b0-b0 agreement, the Jacobian, the working-grid edge checks.

Ported from the analysis project:
  outline_regions / outline_gate   code/regqc_outline.py (unchanged; the declared-box list is passed in, not read from a fixed path)
  load / ncc / dice / jacobian_stats / global_qc
                                   code/regqc.py: the metrics that enter the link's record (ncc at a matched FWHM, envelope Dice,
                                   regeval edge_mean, Jacobian). The registry bookkeeping of regqc (QC store, PNG montage, linear-part
                                   decomposition and gate) is not part of the link's record and is not ported.
  grid_from / edge / detj          mri/code/r5_canon_test.py
"""
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

REGIONS = ("left", "right", "posterior", "anterior", "ventral", "dorsal")       # -x, +x, -y, +y, -z, +z in RAS world
LPS_SIGN = np.array([-1.0, -1.0, 1.0], np.float32)                               # saved FireANTs/ANTs fields: x/y LPS-signed


def _gate_cfg():
    from template.link.common import config
    return config()["outline"]


# ------------------------------------------------------------------------------------ outline check
def _fill(m, affine):
    """Per-coronal-slice hole fill along the grid axis closest to world y."""
    ax = int(np.argmax(np.abs(affine[1, :3])))
    out = np.zeros_like(m)
    for j in range(m.shape[ax]):
        sl = [slice(None)] * 3; sl[ax] = j
        out[tuple(sl)] = ndi.binary_fill_holes(m[tuple(sl)])
    return out


def outline_regions(moved, target, affine, fill=True, ignore_below_target=True, declared=()):
    """Moved brain outline vs the target's, region by region (six poles), in mm: overshoot (moved outside target, by distance to the
    target) and undershoot (target not covered, by distance to the moved mask); max, p99 and volumes per region. Masks hole-filled per
    coronal slice first. Overshoot below the target's lowest axial slice is reported, not gated; `declared` boxes (approved specimen
    differences, world RAS) are reported, not gated. Both masks on one grid."""
    moved = np.asarray(moved) > 0.5 if moved.dtype != bool else moved
    target = np.asarray(target) > 0.5 if target.dtype != bool else target
    assert moved.shape == target.shape, "masks must share one grid"
    if fill:
        moved, target = _fill(moved, affine), _fill(target, affine)
    vox = np.sqrt((affine[:3, :3] ** 2).sum(0))
    d_to_target = ndi.distance_transform_edt(~target, sampling=vox)
    d_to_moved = ndi.distance_transform_edt(~moved, sampling=vox)
    over, under = moved & ~target, target & ~moved
    below = np.zeros_like(over)
    if ignore_below_target:
        zax = int(np.argmax(np.abs(affine[2, :3])))
        zs = np.where(target.any(tuple(i for i in range(3) if i != zax)))[0]
        lowest = zs[0] if affine[2, zax] > 0 else zs[-1]
        sl = [slice(None)] * 3
        sl[zax] = slice(0, lowest) if affine[2, zax] > 0 else slice(lowest + 1, None)
        below[tuple(sl)] = over[tuple(sl)]
        over = over & ~below
    dec_rep = []
    if declared:
        ijk = np.indices(over.shape).reshape(3, -1).T
        W = (ijk @ affine[:3, :3].T + affine[:3, 3]).reshape(over.shape + (3,))
        for dcl in declared:
            (x0, x1), (y0, y1), (z0, z1) = dcl["box_world"]
            inb = (W[..., 0] >= x0) & (W[..., 0] <= x1) & (W[..., 1] >= y0) & (W[..., 1] <= y1) & (W[..., 2] >= z0) & (W[..., 2] <= z1)
            o_d, u_d = over & inb, under & inb
            dec_rep.append(dict(name=dcl.get("name", ""), evidence=dcl.get("evidence", ""), over_mm3=round(float(o_d.sum() * np.prod(vox)), 1),
                                under_mm3=round(float(u_d.sum() * np.prod(vox)), 1),
                                over_max_mm=round(float(d_to_target[o_d].max()), 3) if o_d.any() else 0.0,
                                under_max_mm=round(float(d_to_moved[u_d].max()), 3) if u_d.any() else 0.0))
            over, under = over & ~inb, under & ~inb
        del W, ijk
    tij = np.argwhere(target); tw = tij @ affine[:3, :3].T + affine[:3, 3]
    c = tw.mean(0); lo, hi = np.percentile(tw, 1, 0), np.percentile(tw, 99, 0); half = np.maximum((hi - lo) / 2, 1e-3)

    def region_of(ij):
        w = ij @ affine[:3, :3].T + affine[:3, 3]
        n = (w - c) / half
        k = np.argmax(np.abs(n), 1)
        return 2 * k + (np.take_along_axis(n, k[:, None], 1)[:, 0] > 0)

    vv = float(np.prod(vox)); out = {}
    oij, uij = np.argwhere(over), np.argwhere(under)
    od, ud = d_to_target[over], d_to_moved[under]
    oreg, ureg = (region_of(oij) if len(oij) else np.zeros(0, int)), (region_of(uij) if len(uij) else np.zeros(0, int))
    for i, name in enumerate(REGIONS):
        a, b = od[oreg == i], ud[ureg == i]
        out[name] = dict(over_max_mm=round(float(a.max()), 3) if a.size else 0.0, over_p99_mm=round(float(np.percentile(a, 99)), 3) if a.size else 0.0,
                         under_max_mm=round(float(b.max()), 3) if b.size else 0.0, under_p99_mm=round(float(np.percentile(b, 99)), 3) if b.size else 0.0,
                         over_mm3=round(a.size * vv, 1), under_mm3=round(b.size * vv, 1))
    out["_all"] = dict(declared=dec_rep, over_below_target_mm3=round(float(below.sum() * vv), 1),
                       over_below_target_max_mm=round(float(d_to_target[below].max()), 3) if below.any() else 0.0,
                       dice_filled=round(float(2 * (moved & target).sum() / (moved.sum() + target.sum())), 4),
                       over_max_mm=round(float(od.max()), 3) if od.size else 0.0, under_max_mm=round(float(ud.max()), 3) if ud.size else 0.0)
    return out


def outline_gate(r, fail_p99=None, fail_max=None, warn_p99=None):
    """-> ("PASS" | "WARN" | "FAIL", reasons). FAIL when any region's over- or undershoot has p99 > fail_p99 or max > fail_max; WARN when
    p99 > warn_p99 (settings/link.yaml `outline`: 3.5 / 5.0 / 2.5 mm)."""
    g = _gate_cfg()
    fail_p99 = g["fail_p99_mm"] if fail_p99 is None else fail_p99
    fail_max = g["fail_max_mm"] if fail_max is None else fail_max
    warn_p99 = g["warn_p99_mm"] if warn_p99 is None else warn_p99
    fail, warn = [], []
    for k in REGIONS:
        for kind in ("over", "under"):
            p99, mx = r[k][kind + "_p99_mm"], r[k][kind + "_max_mm"]
            if p99 > fail_p99 or mx > fail_max:
                fail.append(f"{k} {kind}shoot p99 {p99:.2f} / max {mx:.2f} mm")
            elif p99 > warn_p99:
                warn.append(f"{k} {kind}shoot p99 {p99:.2f} mm (warn)")
    return ("FAIL" if fail else "WARN" if warn else "PASS"), fail + warn


# ------------------------------------------------------------------------------------ global agreement (regqc subset)
def load(p, ref=None, order=1):
    img = nib.load(str(p))
    a = np.asanyarray(img.dataobj).astype(np.float32)
    if a.ndim == 5:
        a = a[..., 0, :]
    if a.ndim == 4 and a.shape[-1] in (3, 4):                         # RGB -> luminance
        a = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
    elif a.ndim == 4:
        a = a[..., 0]
    if ref is not None and (a.shape != ref.shape[:3] or not np.allclose(img.affine, ref.affine, atol=1e-4)):
        import nibabel.processing as nip
        a = np.asanyarray(nip.resample_from_to(nib.Nifti1Image(a, img.affine), (ref.shape[:3], ref.affine),
                                               order=order).dataobj).astype(np.float32)
        return a, True
    return a, False


def ncc(a, b, m):
    x, y = a[m].astype(np.float64), b[m].astype(np.float64)
    if x.size < 100:
        return float("nan")
    x -= x.mean(); y -= y.mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum() + 1e-12))


def dice(a, b):
    return float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))


def jacobian_stats(warp_path, mask=None, mask_affine=None):
    """Signed det J of x -> x + u(x) with u a saved displacement (x/y LPS-signed, mm), on the field's own grid (the mask is resampled
    onto it, nearest)."""
    wimg = nib.load(str(warp_path))
    W = np.squeeze(np.asanyarray(wimg.dataobj)).astype(np.float32)
    vox = np.abs(np.diag(wimg.affine)[:3])
    if mask is not None and mask.shape != W.shape[:3]:
        if mask_affine is None:
            mask = None
        else:
            import nibabel.processing as nip
            mask = np.asanyarray(nip.resample_from_to(nib.Nifti1Image(mask.astype(np.uint8), mask_affine),
                                                      (W.shape[:3], wimg.affine), order=0).dataobj) > 0
    if W.ndim != 4 or W.shape[-1] != 3:
        return dict(jac_error=f"unexpected warp shape {W.shape}")
    step = 2 if np.prod(W.shape[:3]) > 60e6 else 1
    W = W[::step, ::step, ::step] * LPS_SIGN
    v = np.asarray(vox, float) * step
    J = np.empty(W.shape[:3] + (3, 3), np.float32)
    for i in range(3):
        g = np.gradient(W[..., i], *v)
        for j in range(3):
            J[..., i, j] = g[j]
        J[..., i, i] += 1.0
    d = np.linalg.det(J)
    pos = d[d > 0]
    mag = np.sqrt((W ** 2).sum(-1))
    out = dict(jac_min=float(d.min()), jac_max=float(d.max()), jac_std=float(d.std()),
               sd_log_detJ=float(np.log(pos).std()) if pos.size else float("nan"),
               folds_n=int((d <= 0).sum()), folds_pct=float(100.0 * (d <= 0).mean()),
               jac_out_0p5_2_pct=float(100.0 * ((d < 0.5) | (d > 2.0)).mean()), jac_step=step)
    if mask is not None:
        mm = mask[::step, ::step, ::step]
        out["folds_n_in_mask"] = int(((d <= 0) & mm).sum())
        out["folds_n_in_mask_dil2"] = int(((d <= 0) & ndi.binary_dilation(mm, iterations=2)).sum())
        if mm.any():
            dm = mag[mm]
            out.update(drift_p95_mm=float(np.percentile(dm, 95)), drift_median_mm=float(np.median(dm)),
                       drift_max_mm=float(dm.max()))
    return out


def global_qc(fixed, moved, *, fixed_mask, moved_mask, warp=None, ncc_fwhm_mm=None, edge_eval=True):
    """ncc (inside the 1 mm-eroded intersection of the two masks; at a matched FWHM if given), envelope Dice of the two masks, the
    interior edge distance (figures.evaluation.regeval) and the Jacobian of `warp` inside the fixed mask. Same values as the analysis
    project's regqc.run_qc for these fields."""
    fimg = nib.load(str(fixed))
    F, _ = load(fixed)
    M, resampled = load(moved, ref=fimg)
    vox = np.abs(np.diag(fimg.affine)[:3])
    fm = load(fixed_mask, ref=fimg, order=0)[0] > 0.5
    mm = load(moved_mask, ref=fimg, order=0)[0] > 0.5
    em = ndi.binary_erosion(fm & mm, iterations=max(1, int(round(1.0 / float(vox.min())))))
    row = dict(eval_mask_vox=int(em.sum()))
    if ncc_fwhm_mm:
        sg = ncc_fwhm_mm / 2.3548 / vox
        row["ncc_raw"] = ncc(F, M, em)
        F, M = ndi.gaussian_filter(F, sg), ndi.gaussian_filter(M, sg)
        row["ncc_fwhm_mm"] = float(ncc_fwhm_mm)
    row["ncc"] = ncc(F, M, em)
    row["envelope_dice"] = dice(fm, mm)
    if warp:
        row["jacobian"] = jacobian_stats(warp, mask=fm, mask_affine=fimg.affine)
    if edge_eval:
        if resampled:
            raise ValueError("moved image must be on the fixed grid for the edge evaluation")
        try:
            from figures.evaluation import regeval
            r = regeval.evaluate(str(fixed), str(moved), None, tag="link")
            row["regeval"] = {k: r[k] for k in ("edge_mean", "edge_sym", "edge_frac1mm", "nmi_interior",
                                                "n_bf_edges", "n_mri_edges") if k in r}
        except Exception as e:                                        # as regqc: an edge failure is recorded, not raised
            row["regeval"] = {"error": f"{type(e).__name__}: {e}"}
    return row


# ------------------------------------------------------------------------------------ working grid
def grid_from(lo, hi, v, mult=16):
    """Grid of spacing v covering [lo, hi] (world RAS), sizes rounded up to a multiple of `mult`, centered on the box."""
    shp = np.ceil((hi - lo) / v).astype(int) + 1; shp = shp + (-shp) % mult
    ctr = (lo + hi) / 2; lo2 = ctr - v * (shp - 1) / 2
    A = np.diag([v, v, v, 1.0]); A[:3, 3] = lo2
    return tuple(int(x) for x in shp), A


def edge(mask_grid, gaff, jac=None):
    """min distance (mm) of the mask to the grid boundary, fraction of the mask within 2 mm of it, det J in the 4 mm boundary band."""
    v = np.sqrt((gaff[:3, :3] ** 2).sum(0)); sh = np.array(mask_grid.shape)
    ij = np.argwhere(mask_grid)
    if not len(ij):
        return {}
    dlo = ij * v; dhi = (sh - 1 - ij) * v
    dmin = np.minimum(dlo, dhi).min(1)
    out = dict(min_dist_to_boundary_mm=round(float(dmin.min()), 2), frac_within_2mm=round(float((dmin <= 2.0).mean()), 5))
    if jac is not None:
        I = np.indices(sh).reshape(3, -1).T
        band = (np.minimum(I * v, (sh - 1 - I) * v).min(1) <= 4.0).reshape(tuple(sh))
        jb = jac[band]
        out.update(band_detJ_min=round(float(jb.min()), 3), band_detJ_lt05_pct=round(float(100 * (jb < 0.5).mean()), 3))
    return out


def detj(warp_path):
    """det J of a saved FireANTs/ITK displacement field (x/y LPS-signed) on its own grid."""
    w = nib.load(str(warp_path)); u = np.squeeze(np.asanyarray(w.dataobj)).astype(np.float32)
    u[..., :2] *= -1
    vox = np.sqrt((w.affine[:3, :3] ** 2).sum(0))
    g = np.stack([np.stack(np.gradient(u[..., c], *vox), -1) for c in range(3)], -2)
    return np.linalg.det(np.eye(3) + g), w.affine, u.shape[:3]
