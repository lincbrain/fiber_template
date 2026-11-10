"""Marker-tube offsets: how far a tube hole in the histology lies from where the registration puts the blockface tube.

Paper: Methods "Histology registration and label propagation" (the marker tubes, tubes hidden from the registration); Results
"Histology registered to the blockface"; Table 3; Fig. 6.

For every registered piece whose blockface slice carries a tube (the tube mask of the reconstruction, `<ID>_holes.nii.gz`):
  blockface center  the centroid of the hole's DARK CORE inside the tube mask, on the piece's photograph with the hole visible (the
                    pre-inpaint blockface): Otsu threshold over the mask dilated by RING_PX, darker mask pixels, holes filled, largest
                    component. If the core is < MIN_KEEP of the mask (no dark core), the mask centroid is used (counted).
  carried center    the blockface center pushed through the piece's TIRL chain (registered with the tubes hidden: inpainted target)
                    into the histology section (JP2 level-0 pixels).
  histology hole    in a WIN_MM window at JP2 level LEVEL: dark components (max RGB < DARK x the window's tissue median, holes filled)
                    of 0.03-1.5 mm^2, not touching the window border, axis ratio >= MIN_AXIS_RATIO; the nearest within SEARCH_MM.
  offset            the distance (mm) between the carried center and that hole; no hole within SEARCH_MM = not found.
Both the hole-only center ("hole") and the mask center ("mask") are scored, and a DIAG_WIN_MM window records the hole-like and
tube-sized (>= TUBE_MM2) components near the carried center.

Each piece runs in its own process (TIRL 3.6.2 can corrupt the heap after loading transforms; the worker exits hard).

    python -m figures.evaluation.tubes brain --subject MR252 --tirl <hidden-tubes tirl dir> --tirl-visible <pre-inpaint tirl dir>
        --holes <ID>_holes.nii.gz --blockface-l2 <ID>_blockface_L2.nii.gz --match-table <num_ID.txt> --first-index <i0>
        --histology <JP2 dir> --histology-masks <tissue masks dir> --pieces <blockface pieces dir> --out <dir> [-j 12]
    -> <out>/<ID>_native/<piece>.json and <out>/<ID>_native.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import numpy as np

# detector parameters of record (paper: Methods, marker tubes)
WIN_MM, LEVEL, SEARCH_MM, DARK, PX0_MM, MIN_AXIS_RATIO = 2.0, 3, 1.0, 0.35, 0.4397e-3, 0.4   # PX0_MM: JP2 level-0 pixel (mm)
RING_PX, MIN_KEEP, DIAG_WIN_MM, TUBE_MM2 = 6, 0.10, 4.0, 0.10                                   # hole-only center; diagnostics


def match_table(num_file):
    """The histology <-> blockface match table: histology section number -> [whole, left, right, cerebellum] blockface photo
    numbers (-1 = no match). Same parser as the histology registration (code/histology)."""
    rows = []
    with open(num_file) as f:
        next(f)
        for line in f:
            v = re.split(r"[,;.:]+", line.strip())
            if len(v) < 10:
                continue
            lo = int(v[0]); hi = np.inf if "inf" in v[1] else int(v[1])
            rows.append((lo, hi, [int(x) for x in v[2:10]]))

    def lookup(num):
        for lo, hi, c in rows:
            if lo <= num <= hi:
                return [num * c[2 * i] + c[2 * i + 1] for i in range(4)]
        return [-1, -1, -1, -1]
    return lookup


def _borders(img):
    """Distances (top, bottom, left, right) from each edge to the tissue, less a 2-pixel margin (as the histology registration)."""
    def first(a):
        for i in range(a.shape[0]):
            if np.sum(a[i]) > 0:
                return i
        return a.shape[0]
    return [max(0, k - 2) for k in (first(img), first(img[::-1]), first(np.rot90(img, 3)), first(np.rot90(img, 1)))]


def geometry(blue_masked, photo):
    """Where the masked histology was placed on the blockface piece grid (the registration's moving-image geometry)."""
    H, W = photo.shape[:2]
    tt, tb, ll, lr = _borders(photo)
    box_h, box_w = H - tt - tb, W - ll - lr
    t, b, l, r = _borders(blue_masked)
    h0, w0 = blue_masked.shape[0] - t - b, blue_masked.shape[1] - l - r
    h, w = float(h0), float(w0)
    while h < box_h and w < box_w:
        h, w = 1.2 * h, 1.2 * w
    while h > box_h or w > box_w:
        h, w = 0.9 * h, 0.9 * w
    return dict(tt=tt, ll=ll, h=max(1, int(h)), w=max(1, int(w)), t=t, l=l, h0=h0, w0=w0)


def otsu(v):
    h, e = np.histogram(v, 64); c = 0.5 * (e[1:] + e[:-1]); w = np.cumsum(h); mu = np.cumsum(h * c)
    return float(c[np.argmax((mu[-1] / w[-1] * w - mu) ** 2 / np.maximum(w * (w[-1] - w), 1e-12))])


def holes_in(j2k, y0, x0, win_mm, lvl):
    """Hole-like components around level-0 point (y0, x0): [(distance mm, area mm^2, center level-0)]."""
    from scipy.ndimage import label, center_of_mass, binary_fill_holes
    half = int(win_mm / 2 / PX0_MM)
    r0, c0 = int(max(0, y0 - half)), int(max(0, x0 - half))
    win = j2k[r0:int(y0 + half):2 ** lvl, c0:int(x0 + half):2 ** lvl][:, :, :3].astype(np.float32)
    mx_ = win.max(-1)
    tissue = mx_ > np.percentile(mx_, 20)
    dark = mx_ < DARK * np.percentile(mx_[tissue], 50) if tissue.any() else np.zeros_like(tissue)
    lb, nb = label(binary_fill_holes(dark))
    px_mm = PX0_MM * 2 ** lvl
    cen = np.array([(y0 - r0) / 2 ** lvl, (x0 - c0) / 2 ** lvl])
    out = []
    for i in range(1, nb + 1):
        comp = lb == i
        a = comp.sum() * px_mm ** 2
        if not 0.03 <= a <= 1.5:
            continue
        if comp[0].any() or comp[-1].any() or comp[:, 0].any() or comp[:, -1].any():
            continue
        yy, xx = np.nonzero(comp)
        ev = np.linalg.eigvalsh(np.cov(np.vstack([yy, xx]).astype(float)))
        if (float(np.sqrt(max(ev[0], 0) / ev[1])) if ev[1] > 0 else 0.0) < MIN_AXIS_RATIO:
            continue
        c = np.array(center_of_mass(comp))
        out.append((float(np.linalg.norm(c - cen) * px_mm), float(a), [float(r0 + c[0] * 2 ** lvl), float(c0 + c[1] * 2 ** lvl)]))
    return out


def one_piece(a, piece):
    """Score one piece (run in its own process; writes <out>/<ID>_native/<piece>.json and exits hard)."""
    import glymur
    import nibabel as nib
    from PIL import Image
    from scipy.ndimage import label, center_of_mass, binary_fill_holes, binary_dilation
    from tirl import load
    Image.MAX_IMAGE_PIXELS = None
    d = os.path.join(a.tirl, piece)
    rec = dict(piece=piece)
    look = match_table(a.match_table)
    idx = look(int(re.search(r"_s(\d+)", piece).group(1)))[int(piece[-4:])] - a.first_index
    A_d = nib.load(a.blockface_l2).affine
    him = nib.load(a.holes)
    Hv = np.asarray(him.dataobj) > 0
    fx = np.asarray(Image.open(os.path.join(d, "fixed.png")).convert("L"))
    Hd, Wd = fx.shape
    U, V = np.meshgrid(np.arange(Hd), np.arange(Wd), indexing="ij")
    R, C = Hd - 1 - U, Wd - 1 - V
    vox = np.stack([(C - 0.5) / 2, np.full(C.shape, float(idx)), (R - 0.5) / 2, np.ones(C.shape)], -1)
    v = np.round((vox @ A_d.T) @ np.linalg.inv(him.affine).T)[..., :3].astype(int)
    ok = np.all((v >= 0) & (v < np.array(Hv.shape)), -1)
    tb = np.zeros((Hd, Wd), bool); tb[ok] = Hv[v[ok, 0], v[ok, 1], v[ok, 2]]
    tb &= binary_fill_holes(fx > 0)
    lab, n = label(tb)
    if not n:
        rec.update(bf_tube=False); _save(a, rec); os._exit(0)
    k = int(np.argmax(np.bincount(lab.ravel())[1:])) + 1
    comp = lab == k
    cm_mask = center_of_mass(comp)
    g = np.asarray(Image.open(os.path.join(a.tirl_visible, piece, "fixed.png")).convert("L")).astype(np.float32)
    assert g.shape == fx.shape, (piece, g.shape, fx.shape)
    zone = binary_dilation(comp, iterations=RING_PX) & (g > 0)
    th = otsu(g[zone]) if zone.sum() > 10 else None
    hole = binary_fill_holes(comp & (g < th)) if th is not None else np.zeros_like(comp)
    lh, nh = label(hole)
    if nh:
        hole = lh == (int(np.argmax(np.bincount(lh.ravel())[1:])) + 1)
    refined = bool(nh) and hole.sum() >= MIN_KEEP * comp.sum()
    cm_hole = center_of_mass(hole) if refined else cm_mask
    fix_px = float(np.linalg.norm(np.subtract(cm_hole, cm_mask)))
    m, f = load(os.path.join(d, "moving.timg")), load(os.path.join(d, "fixed4_nonlinear.timg"))
    sid = piece.rsplit("_", 1)[0]
    j2k = glymur.Jp2k(os.path.join(a.histology, sid + ".jp2"))
    nres = j2k.codestream.segment[2].num_res
    lv = 6 if nres >= 6 else nres - 1
    rgb6 = j2k.read_bands(rlevel=lv)[:, :, :3]
    pmask = np.asarray(Image.open(os.path.join(a.histology_masks, piece + ".png"))) > 0
    photo = np.asarray(Image.open(os.path.join(a.pieces, piece + ".png")))
    geo = geometry(np.where(pmask, rgb6[:, :, 2], 0).astype(np.uint8), photo)
    lvl = LEVEL
    res = {}
    for name, (cy, cx) in (("mask", cm_mask), ("hole", cm_hole)):
        my, mx = m.domain.map_physical_coordinates(f.domain.map_voxel_coordinates(np.array([[cy, cx]])))[0]
        y0 = (geo["t"] + (my - geo["tt"]) * geo["h0"] / geo["h"]) * 2 ** lv
        x0 = (geo["l"] + (mx - geo["ll"]) * geo["w0"] / geo["w"]) * 2 ** lv
        cand = [c for c in holes_in(j2k, y0, x0, WIN_MM, lvl) if c[0] <= SEARCH_MM]
        best = min(cand) if cand else None
        diag = holes_in(j2k, y0, x0, DIAG_WIN_MM, lvl)
        tube = [c for c in diag if c[1] >= TUBE_MM2]
        res[name] = dict(found=best is not None, error_mm=round(best[0], 4) if best else None, hole_area_mm2=round(best[1], 3) if best else None,
                         bf_tube_l0=[round(float(y0), 1), round(float(x0), 1)], hole_l0=[round(q, 1) for q in best[2]] if best else None,
                         diag_n_holes=len(diag), diag_n_tube_sized=len(tube),
                         diag_nearest_mm=round(min(diag)[0], 3) if diag else None,
                         diag_nearest_tube_sized_mm=round(min(tube)[0], 3) if tube else None)
    rec.update(bf_tube=True, bf_slice=int(idx), sid=sid, mask_px=int(comp.sum()), hole_px=int(hole.sum()) if refined else None,
               refined=refined, center_shift_fixed_px=round(fix_px, 2), detect_level=lvl, **res)
    _save(a, rec); os._exit(0)


def _save(a, rec):
    d = os.path.join(a.out, f"{a.subject}_native")
    os.makedirs(d, exist_ok=True)
    json.dump(rec, open(os.path.join(d, rec["piece"] + ".json"), "w"), default=lambda o: o.item())


def brain(a, argv):
    pieces = sorted(os.path.basename(os.path.dirname(p)) for p in glob(os.path.join(a.tirl, "*", "fixed4_nonlinear.timg")))

    def run(p):
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))      # the repository
        r = subprocess.run([sys.executable, "-m", "figures.evaluation.tubes", "piece", "--piece", p] + argv, capture_output=True, text=True,
                           cwd=root, env=dict(os.environ, PYTHONPATH=os.path.join(root, "code") + os.pathsep + os.environ.get("PYTHONPATH", "")))
        return p, r.returncode, r.stderr[-300:]
    with ThreadPoolExecutor(a.jobs) as ex:
        for p, rc, err in ex.map(run, pieces):
            if rc != 0:
                print(f"! {p} rc={rc} {err}", flush=True)
    rows = [json.load(open(q)) for q in sorted(glob(os.path.join(a.out, f"{a.subject}_native", "*.json")))]
    summ = dict(subject=a.subject, pieces=len(rows), with_bf_tube=sum(1 for r in rows if r.get("bf_tube")),
                found_mask=sum(1 for r in rows if r.get("mask", {}).get("found")), found_hole=sum(1 for r in rows if r.get("hole", {}).get("found")))
    json.dump(dict(summary=summ, pieces=rows), open(os.path.join(a.out, f"{a.subject}_native.json"), "w"), indent=1)
    print(summ, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["brain", "piece"])
    ap.add_argument("--piece")
    for k in ("subject", "tirl", "tirl-visible", "holes", "blockface-l2", "match-table", "histology", "histology-masks", "pieces", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--first-index", type=int, required=True, help="raw index of the first blockface photograph of the stack")
    ap.add_argument("-j", "--jobs", type=int, default=12)
    a = ap.parse_args()
    if a.mode == "piece":
        one_piece(a, a.piece)
    else:
        argv = [x for x in sys.argv[2:]]
        brain(a, argv)


if __name__ == "__main__":
    main()
