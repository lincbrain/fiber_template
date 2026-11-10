"""Agreement of a predicted mask with its reference, per foreground class: Dice, and the symmetric surface distances in mm (the image's
voxel size): mean (ASSD) and 95th percentile (HD95) of the pooled distances from each surface to the other. Surface = mask voxels with a
background face-neighbor (m & ~erosion). For 2D images also the connected components of each mask and the false-positive and
false-negative areas (mm^2). numpy + scipy."""
import numpy as np
from scipy import ndimage as ndi


def _surface(m):
    return m & ~ndi.binary_erosion(m)


def score(pred, ref, spacing, classes=(1,)):
    out = {}
    for c in classes:
        a, b = pred == c, ref == c
        na, nb = int(a.sum()), int(b.sum())
        d = dict(dice=float(2 * (a & b).sum() / (na + nb)) if na + nb else 1.0, vox_pred=na, vox_ref=nb)
        if na and nb:
            sa, sb = _surface(a), _surface(b)
            da = ndi.distance_transform_edt(~sb, sampling=spacing)[sa]; db = ndi.distance_transform_edt(~sa, sampling=spacing)[sb]
            dd = np.concatenate([da, db])
            d.update(assd_mm=float(dd.mean()), hd95_mm=float(np.percentile(dd, 95)))
        else:
            d.update(assd_mm=None, hd95_mm=None)
        if a.ndim == 2:                                                 # a photograph: pieces and false-positive area
            px = float(np.prod(spacing))
            d.update(components_pred=int(ndi.label(a)[1]), components_ref=int(ndi.label(b)[1]),
                     fp_area_mm2=float((a & ~b).sum() * px), fn_area_mm2=float((b & ~a).sum() * px))
        out[str(c)] = d
    return out


def summarize(rows, n_classes, counted_only=True):
    """mean / median / min / max of each metric over the cases (rows: {id: {"counted": bool, "<class>": {metric: value}}})"""
    out = {}
    use = [r for r in rows.values() if r.get("counted", True) or not counted_only]
    for k in range(1, n_classes):
        for m in ("dice", "assd_mm", "hd95_mm", "components_pred", "fp_area_mm2"):
            v = [r[str(k)][m] for r in use if r[str(k)].get(m) is not None]
            out[f"{k}_{m}"] = dict(n=len(v), mean=float(np.mean(v)), median=float(np.median(v)), min=float(np.min(v)),
                                   max=float(np.max(v))) if v else None
    out["n_cases"] = len(use)
    return out
