"""Propose a brain's match table automatically, for a person to check and edit: which blockface photograph each histology section
was cut at.

You need
  the brain's reconstruction store (<ID>_blockface.ome.zarr) and brain mask (<ID>_mask.nii.gz); its histology sections (JPEG 2000
  files, or an OME-Zarr stack with its section map); the regions volume on the blockface (linc/histology/README.md); the raw index of
  the store's first photograph; the number of photographs per histology section (configs/histology.yaml propose.section_um / the
  section step the volume was built at, or --photos-per-section).
What it does
  components  each section (coarse level, dark background) splits into its tissue components: two side by side are its left and
              right, one is whole, left, right or cerebellum.
  outlines    each component's outline is compared with the outline of every sampled photograph's whole brain, hemispheres and
              cerebellum (the regions volume sampled on the brain mask, each class's largest connected part). In section order, each
              class follows a line of fixed spacing whose offset may jump where a block was cut apart. This fixes the class of a single
              component and a window of photographs around its position.
  scores      each component is registered to its class on every photograph in the window (SimpleITK: moments start, similarity,
              then affine, Mattes mutual information); score = half the Dice of the masks + half the correlation inside the overlap.
  run lines   the same line fit, now on the scores: within a cutting run the sections keep the fixed spacing and order, so a
              component follows its run's offset, never its own best photograph (neighboring photographs look alike).
  floor       a component scoring under propose.score_floor on its run line stays unmatched instead of being guessed.
  flags       a component is flagged when a photograph more than one section away scores higher, or when its score is low. The flags
              only set the order of the review, which goes row by row: they miss some large disagreements (the test below).
You get
  in the brain's histology folder: match_table_proposed.txt (one row per section, the format of match.py: multiplier 0, adder = the
  photograph, -1 = no match, so a person can read and change a number directly); match_review.csv (one row per component: section,
  component, photograph, run-line photograph, score, flag, next-best photographs); match_review/, one image per flagged component
  (the section beside its proposed photograph, whose number is in the file name).

To correct it, copy match_table_proposed.txt to match_table_edited.txt and change numbers: every step then uses the edited file
(match.find_table), and nothing overwrites it.

Tested in tests/match_propose_records.py. Ported from the analysis project's labels/propose_pieces.py, labels/propose_reg.py (regfit,
atlas regions as the classes) and labels/review_page.py write_table, with one change: the record took its first outlines from a
2D network's partition of the photographs, and here they come from the regions volume, as the later steps' classes already did.
"""
import csv
import os
from multiprocessing import Pool

import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes, distance_transform_edt, gaussian_filter, label, zoom

from . import match as M
from . import pieces as PC

CLASSES = ("whole", "left", "right", "bottom")                 # bottom = cerebellum + brainstem
COLUMN = {"whole": 0, "left": 1, "right": 2, "bottom": 3}      # match-table column
COMPONENT = {"whole": "whole", "left": "left", "right": "right", "bottom": "cere"}
TABLE_HEADER = ("lower_val,higher_val,multiplier,adder,left_multiplier,left_adder,right_multiplier,right_adder,"
                "cere_multiplier,cere_adder\n")
PROPOSED, REVIEW = M.PROPOSED, M.REVIEW


def photos_per_section(subject_cfg, cfg=None):
    """Photographs per histology section: the section spacing over the section step the volume was built at."""
    cfg = cfg or M.config()
    s = subject_cfg.get("reconstruction_spacing_um") or subject_cfg
    return cfg["propose"]["section_um"] / float(s["vz"])


def largest_part(m):
    """A mask's largest connected part: stray pixels of a class would stretch the box its outline is cropped to."""
    lb, n = label(m)
    return m if n < 2 else lb == 1 + int(np.bincount(lb.ravel())[1:].argmax())


def outline(m, size):
    """A mask cropped to its box and resized to size x size (a scale-free outline); all False if nearly empty."""
    r, c = np.nonzero(m)
    if r.size < 20:
        return np.zeros((size, size), bool)
    m = m[r.min():r.max() + 1, c.min():c.max() + 1]
    return np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((size, size), Image.BILINEAR)) > 127


def _dice_matrix(h, b):
    h = h.reshape(len(h), -1).astype(np.float32)
    b = b.reshape(len(b), -1).astype(np.float32)
    return 2 * (h @ b.T) / (h.sum(1)[:, None] + b.sum(1)[None, :] + 1e-6)


def _crop(img, m, pad=6):
    rr, cc = np.nonzero(m)
    r0, r1 = max(rr.min() - pad, 0), min(rr.max() + pad + 1, m.shape[0])
    c0, c1 = max(cc.min() - pad, 0), min(cc.max() + pad + 1, m.shape[1])
    return img[r0:r1, c0:c1], m[r0:r1, c0:c1]


# ------------------------------------------------------------------------------------------------------ inputs
def blockface_slices(store, mask, regions, scale_xyz, step, cfg=None, log=print):
    """{photograph index in the store: (gray upright photograph, {class: mask})} for every step-th photograph that has tissue. Classes
    from the regions volume sampled at each pixel's world position, restricted to the filled brain mask."""
    import nibabel as nib
    import zarr
    cfg = cfg or M.config()
    pc, rc = cfg["propose"], cfg["regions"]
    level = pc["photo_level"]
    st = zarr.open_group(str(store), mode="r")[str(level)]             # (c, rows, photographs, cols)
    A = PC.store_affine(store, level)
    rim = nib.load(str(regions))
    R3 = np.asarray(rim.dataobj).astype(np.uint8)
    R3[R3 == rc["punch_hole_label"]] = rc["punch_hole_as"]
    A_r = rim.affine if scale_xyz is None else np.diag(list(scale_xyz) + [1.0]) @ rim.affine
    Rinv = np.linalg.inv(A_r)
    mk = nib.load(str(mask))
    brain = np.asarray(mk.dataobj) > 0
    Minv = np.linalg.inv(mk.affine)
    # the anatomical label on the image left (class 1), once per brain from the geometry: the column axis maps to world x with the sign
    # of A[0, 0], and the upright photograph is turned 180 degrees from the store
    xs = {v: nib.affines.apply_affine(A_r, np.argwhere(R3 == v))[:, 0].mean() for v in (1, 2) if (R3 == v).any()}
    trans = np.array([0, 0, 0, 3], np.uint8)
    if len(xs) == 2:
        img_left = max(xs, key=xs.get) if A[0, 0] > 0 else min(xs, key=xs.get)
        trans[img_left], trans[3 - img_left] = 1, 2
    rows, cols = st.shape[1], st.shape[3]
    U, V = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    R, C = rows - 1 - U, cols - 1 - V                                  # upright -> store
    out = {}
    for i in range(0, st.shape[2], step):
        grey = np.asarray(st[:, :, i, :]).astype(np.float32).mean(0)[::-1, ::-1]
        if grey.max() == 0:
            continue
        w = nib.affines.apply_affine(A, np.stack([C.ravel(), np.full(C.size, i), R.ravel()], 1))
        v = np.round(nib.affines.apply_affine(Minv, w)).astype(int)
        ok = np.all((v >= 0) & (v < brain.shape), 1)
        m = np.zeros(v.shape[0], bool)
        m[ok] = brain[tuple(v[ok].T)]
        m = binary_fill_holes(m.reshape(U.shape))
        v = np.round(nib.affines.apply_affine(Rinv, w)).astype(int)
        ok = np.all((v >= 0) & (v < R3.shape), 1)
        lab = np.zeros(v.shape[0], np.uint8)
        lab[ok] = R3[tuple(v[ok].T)]
        lab = trans[np.clip(lab.reshape(U.shape), 0, 3)]
        lab[~m] = 0
        cls = {"whole": m}
        if lab.any():
            lab = PC.split_separate_hemispheres(lab, m, cfg)
            _, (ii, jj) = distance_transform_edt(lab == 0, return_indices=True)     # tissue outside the regions: the nearest region
            near = lab[ii, jj]
            for c, cid in (("left", 1), ("right", 2), ("bottom", 3)):
                cls[c] = m & (near == cid)
        out[i] = (np.where(m, grey, 0), cls)
    log(f"  {len(out)} photographs sampled every {step} (store level {level})")
    return out


def section_components(secs, cfg=None):
    """[{section, sid, kind, gray, mask, rlevel}] from sections.sections(): kind pair-left / pair-right / single, image-left first."""
    from .register import _rlevel
    pc = (cfg or M.config())["propose"]
    out = []
    for sid, num, mk in secs:
        o = mk()
        rl = _rlevel(o, pc["section_level"])
        a = o.read(rlevel=rl)
        m = binary_fill_holes(gaussian_filter(a.max(-1).astype(float), pc["tissue_sigma_px"]) > pc["tissue_threshold"])
        lb, n = label(m)
        if n == 0:
            continue
        sizes = np.bincount(lb.ravel())[1:]
        keep = [i + 1 for i in np.argsort(sizes)[::-1] if sizes[i] >= pc["second_component_min_fraction"] * sizes.max()][:2]
        comps = [binary_fill_holes(lb == k) for k in keep]
        comps.sort(key=lambda c: np.nonzero(c)[1].mean())
        kinds = ["pair-left", "pair-right"] if len(comps) == 2 else ["single"]
        grey = a.astype(np.float32).mean(-1)
        for kind, c in zip(kinds, comps):
            out.append(dict(section=int(num), sid=sid, kind=kind, gray=np.where(c, grey, 0), mask=c, rlevel=rl))
    return out


# ------------------------------------------------------------------------------------------------------ outlines
def outline_fit(comps, slices, a0, cfg=None):
    """Each component's class and place from outlines: [{section, sid, kind, cls, idx, ...}], idx None where no line fits. a0:
    photographs per section. Whole sections take one offset and anchor the rest; other classes follow a line whose offset may jump."""
    pc = (cfg or M.config())["propose"]
    size, jump, none_score = pc["outline_px"], pc["outline_jump"], pc["outline_none_score"]
    max_off = pc["outline_max_offset_mm"] * 1000.0 / (pc["section_um"] / a0)
    bidx = np.array(sorted(slices))
    D = {}
    H = np.array([outline(c["mask"], size) for c in comps])
    for c in CLASSES:
        B = np.array([outline(largest_part(slices[i][1].get(c, np.zeros((1, 1), bool))), size) for i in bidx])
        D[c] = _dice_matrix(H, B)
    kinds = [c["kind"] for c in comps]
    allowed = [("left",) if k == "pair-left" else ("right",) if k == "pair-right" else CLASSES for k in kinds]
    nums = np.array([c["section"] for c in comps])
    step = np.median(np.diff(bidx)) if len(bidx) > 1 else 1
    offs = np.arange(-bidx.max() - a0 * nums.max(), 2 * bidx.max() + a0 * nums.max() + 1, step)

    def viterbi(units, sign, single_offset=False, center=None):
        """Offset along units in section order; a unit is the (component, class) pairs scored together (both halves of a pair)."""
        units = sorted(units, key=lambda u: nums[u[0][0]])
        E = np.zeros((len(units), len(offs)))
        for k, u in enumerate(units):
            pos = sign * a0 * nums[u[0][0]] + offs
            j = np.searchsorted(bidx, pos).clip(0, len(bidx) - 1)
            on = (pos >= bidx.min() - step) & (pos <= bidx.max() + step)
            E[k] = np.where(on, np.mean([D[c][i, j] for i, c in u], 0), -np.inf)
        if center is not None:
            E[:, np.abs(offs - center) > max_off] = -np.inf
        if single_offset:
            q = int(np.argmax(np.maximum(E, none_score).sum(0)))
            o = {i: (float(offs[q]) if E[k, q] >= none_score else None) for k, u in enumerate(units) for i, _ in u}
            return float(np.maximum(E[:, q], none_score).sum()), o
        E = np.concatenate([E, np.full((len(units), 1), none_score)], 1)     # the last column: no match
        O = len(offs)
        V = E[0].copy()
        back = []
        for k in range(1, len(units)):
            ao = int(V[:O].argmax())
            cand = np.stack([V, np.r_[np.full(O, V[ao] - jump), V[ao]], np.r_[np.full(O, V[O]), -np.inf]])
            src = np.stack([np.arange(O + 1), np.full(O + 1, ao), np.full(O + 1, O)])
            w = cand.argmax(0)
            back.append(src[w, np.arange(O + 1)])
            V = cand[w, np.arange(O + 1)] + E[k]
        path = [int(V.argmax())]
        for bk in reversed(back):
            path.append(int(bk[path[-1]]))
        path = path[::-1]
        return float(V.max()), {i: (None if q == O else float(offs[q])) for u, q in zip(units, path) for i, _ in u}

    def groups(cls):
        by_sec = {}
        for i in range(len(comps)):
            if kinds[i].startswith("pair"):
                by_sec.setdefault(nums[i], []).append((i, cls[i]))
        g = {"pair": list(by_sec.values())}
        for c in CLASSES:
            g[c] = [[(i, c)] for i in range(len(comps)) if kinds[i] == "single" and cls[i] == c]
        return g

    def solve(cls):
        best = None
        for sign in (1, -1):
            tot, off, center = 0.0, {}, None
            g = groups(cls)
            if g["whole"]:
                sc, o = viterbi(g["whole"], sign, single_offset=True)
                tot += sc
                off.update(o)
                vals = [v for v in o.values() if v is not None]
                center = vals[0] if vals else None
            for name, units in g.items():
                if units and name != "whole":
                    sc, o = viterbi(units, sign, center=center)
                    tot += sc
                    off.update(o)
            if best is None or tot > best[0]:
                best = (tot, sign, off)
        return best

    def path_dice(i, c, sign, off, cls):
        pc_ = [k for k in np.where(cls == c)[0] if off.get(int(k)) is not None and k != i]
        if not pc_:
            return -1.0
        k = min(pc_, key=lambda k: abs(nums[k] - nums[i]))
        pos = sign * a0 * nums[i] + off[int(k)]
        if pos < bidx.min() - step or pos > bidx.max() + step:
            return -1.0
        return float(D[c][i, int(np.clip(np.searchsorted(bidx, pos), 0, len(bidx) - 1))])

    cls = np.array([max(al, key=lambda c: D[c][i].max()) for i, al in enumerate(allowed)])
    _, sign, off = solve(cls)
    for _ in range(pc["outline_rounds"]):                               # a single component's class, against each class's fitted line
        new = cls.copy()
        for i, al in enumerate(allowed):
            if len(al) > 1:
                new[i] = max(al, key=lambda c: path_dice(i, c, sign, off, cls))
        if (new == cls).all():
            break
        cls = new
        _, sign, off = solve(cls)
    res = []
    for i, c in enumerate(comps):
        idx = None
        if off[i] is not None:
            idx = int(bidx[int(np.clip(np.searchsorted(bidx, sign * a0 * c["section"] + off[i]), 0, len(bidx) - 1))])
        k = c["kind"]
        res.append(dict(section=c["section"], sid=c["sid"], kind=k, cls=str(cls[i]), idx=idx, offset=off[i],
                        group=k if k.startswith("pair") else str(cls[i])))
    return res


# ------------------------------------------------------------------------------------------------------ registration scores
def _set_threads(n):
    import SimpleITK as sitk
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(n)


def registration_score(job):
    """(key, fixed gray, fixed mask, moving gray, moving mask, settings) -> (key, score)."""
    import SimpleITK as sitk
    key, fg, fm, mg, mm, s = job
    if fm.sum() < 30 or mm.sum() < 30:
        return key, 0.0
    F = sitk.GetImageFromArray(fg.astype(np.float32))
    Mv = sitk.GetImageFromArray(mg.astype(np.float32))
    Fm = sitk.GetImageFromArray(fm.astype(np.uint8))
    Mm = sitk.GetImageFromArray(mm.astype(np.uint8))

    def run(tf):
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsMattesMutualInformation(s["bins"])
        R.SetMetricSamplingStrategy(R.RANDOM)
        R.SetMetricSamplingPercentage(s["sampling"], seed=s["seed"])
        R.SetInterpolator(sitk.sitkLinear)
        R.SetOptimizerAsRegularStepGradientDescent(s["learning_rate"], s["min_step"], s["iterations"], relaxationFactor=s["relaxation"])
        R.SetOptimizerScalesFromPhysicalShift()
        R.SetShrinkFactorsPerLevel(s["shrink"])
        R.SetSmoothingSigmasPerLevel(s["sigma"])
        R.SetInitialTransform(tf, inPlace=True)
        R.Execute(F, Mv)
        return tf

    try:
        sim = run(sitk.Similarity2DTransform(sitk.CenteredTransformInitializer(
            Fm, Mm, sitk.Similarity2DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)))
        aff = sitk.AffineTransform(2)
        aff.SetCenter(sim.GetCenter())
        aff.SetMatrix(sim.GetMatrix())
        aff.SetTranslation(sim.GetTranslation())
        final = run(aff)
        wm = sitk.GetArrayFromImage(sitk.Resample(Mm, F, final, sitk.sitkNearestNeighbor, 0)) > 0
        wg = sitk.GetArrayFromImage(sitk.Resample(Mv, F, final, sitk.sitkLinear, 0.0))
    except Exception:                                                   # a pair that does not converge scores nothing
        return key, 0.0
    dice = 2 * (wm & fm).sum() / max(wm.sum() + fm.sum(), 1)
    ov = wm & fm
    ncc = float(np.corrcoef(fg[ov], wg[ov])[0, 1]) if ov.sum() > 30 and fg[ov].std() > 0 and wg[ov].std() > 0 else 0.0
    return key, round(float(0.5 * dice + 0.5 * max(ncc, 0.0)), 4)


def add_scores(P, comps, slices, a0, photo_px_mm, jobs=1, cfg=None, log=print):
    """P[k]["scores"] = {photograph index: score} for the photographs inside the window around the component's outline place."""
    pc = (cfg or M.config())["propose"]
    bidx = np.array(sorted(slices))
    step = int(np.median(np.diff(bidx))) if len(bidx) > 1 else 1
    win = int(round(pc["window_mm"] * 1000.0 / (pc["section_um"] / a0) / step))
    work = []
    for k, p in enumerate(P):
        if p["idx"] is None:
            continue
        c = comps[k]
        s_h = pc["section_pixel_um"] * 2 ** c["rlevel"] / 1000.0 / photo_px_mm
        g, m = _crop(c["gray"], c["mask"])
        mg, mm = zoom(g, s_h, order=1), zoom(m.astype(np.uint8), s_h, order=0) > 0
        j0 = int(np.searchsorted(bidx, p["idx"]))
        for i in bidx[max(0, j0 - win):min(len(bidx), j0 + win + 1)]:
            grey, cl = slices[int(i)]
            fm = cl.get(p["cls"], cl["whole"])
            if fm.sum() < 30:
                continue
            fg, fm2 = _crop(np.where(fm, grey, 0), fm)
            work.append(((k, int(i)), fg, fm2, mg, mm, pc["registration"]))
    log(f"  {len(P)} components, {len(work)} registrations (window +-{pc['window_mm']} mm)")
    with Pool(jobs, initializer=_set_threads, initargs=(pc["registration"]["threads"],)) as pool:
        res = pool.map(registration_score, work, chunksize=4)
    for p in P:
        p["scores"] = {}
    for (k, i), s in res:
        P[k]["scores"][i] = s
    return P


# ------------------------------------------------------------------------------------------------------ run lines + the rule
def run_lines(P, a0, step, cfg=None):
    """Re-derive each class's line from the scores and write idx (the photograph on the run line), score, flags into P. Per group
    (whole, single left, right, cerebellum, pair-left, pair-right), in section order: the state is the line's offset, a component scores
    its registration at the photograph nearest the line (else outline_floor), staying on an offset is free, changing it costs
    run_jump. Returns the number of runs."""
    pc = (cfg or M.config())["propose"]
    jump, floor = pc["run_jump"], pc["run_floor"]
    groups = {}
    for p in P:
        p["flags"] = []
        if p.get("scores"):
            groups.setdefault(p["group"], []).append(p)
        else:
            p.update(idx=None, score=None, flags=["no-score"])
    runs = 0
    for ps in groups.values():
        ps.sort(key=lambda p: p["section"])
        cand = [{int(k): v for k, v in p["scores"].items()} for p in ps]
        lo = min(min(c) - a0 * p["section"] for c, p in zip(cand, ps))
        hi = max(max(c) - a0 * p["section"] for c, p in zip(cand, ps))
        offs = np.arange(np.floor(lo / step) * step, hi + step, step)
        E = np.full((len(ps), len(offs)), floor)
        for k, (c, p) in enumerate(zip(cand, ps)):
            keys = np.array(sorted(c))
            vals = np.array([c[i] for i in keys])
            pos = a0 * p["section"] + offs
            j = np.abs(keys[None, :] - pos[:, None]).argmin(1)
            E[k] = np.where(np.abs(keys[j] - pos) <= step / 2 + 1e-6, vals[j], floor)
        V = E[0].copy()
        back = []
        for k in range(1, len(ps)):
            ao = int(V.argmax())
            move = np.full_like(V, V[ao] - jump)
            back.append(np.where(V >= move, np.arange(len(offs)), ao))
            V = np.maximum(V, move) + E[k]
        path = [int(V.argmax())]
        for bk in reversed(back):
            path.append(int(bk[path[-1]]))
        path = path[::-1]
        prev = None
        for p, q, c in zip(ps, path, cand):
            if q != prev:
                runs += 1
                prev = q
            keys = np.array(sorted(c))
            i = int(keys[np.abs(keys - (a0 * p["section"] + offs[q])).argmin()])
            here = c[i]
            own = max(c, key=c.get)
            far = [x for x in c if abs(x - i) > a0]
            alt = max(far, key=c.get) if far else None
            p.update(idx=i, score=here)
            if abs(own - i) > a0 and c[own] - here > pc["own_best_margin"]:
                p["flags"].append("own-best-elsewhere")
            if alt is not None and c[alt] > here:
                p["flags"].append("better-photograph-elsewhere")
            if here < pc["low_score"]:
                p["flags"].append("low-score")
    return runs


def table_rows(P, idx0, cfg=None):
    """{section: [whole, left, right, cere] raw photograph indices}, -1 = no match. A component under the floor stays unmatched."""
    floor = (cfg or M.config())["propose"]["score_floor"]
    rows = {}
    for p in P:
        if p.get("idx") is None or (p.get("score") or 0) < floor:
            continue
        rows.setdefault(p["section"], [-1, -1, -1, -1])[COLUMN[p["cls"]]] = p["idx"] + idx0
    return rows


def write_table(path, rows):
    """One row per section (multiplier 0, adder = the photograph) that match.match_table reads; the last row covers every other section."""
    with open(path, "w") as f:
        f.write(TABLE_HEADER)
        for n in sorted(rows):
            f.write(f"{n},{n}," + ",".join(f"0,{v}" for v in rows[n]) + "\n")
        f.write("-1,inf,0,-1,0,-1,0,-1,0,-1")


def write_review(path, P, idx0, cfg=None):
    floor = (cfg or M.config())["propose"]["score_floor"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "component", "photograph", "run_line_photograph", "score", "flag", "next_best"])
        for p in sorted(P, key=lambda p: (p["section"], p["kind"], p["cls"])):
            line = p.get("idx")
            kept = line is not None and (p.get("score") or 0) >= floor
            others = sorted(((s, i) for i, s in (p.get("scores") or {}).items() if i != line), reverse=True)[:3]
            w.writerow([p["section"], COMPONENT[p["cls"]], line + idx0 if kept else -1, "" if line is None else line + idx0,
                        "" if p.get("score") is None else f"{p['score']:.4f}", ";".join(p["flags"]),
                        " ".join(f"{i + idx0}:{s:.4f}" for s, i in others)])


def review_images(folder, P, comps, slices, idx0):
    """One image per flagged component: the section beside its proposed photograph, both scaled to one height."""
    def show(g):
        lo, hi = np.percentile(g[g > 0], [2, 99.5])
        im = Image.fromarray(np.uint8(255 * np.clip((g - lo) / max(hi - lo, 1e-3), 0, 1) * (g > 0)))
        return im.resize((max(1, round(im.width * 280 / im.height)), 280), Image.BILINEAR)

    os.makedirs(folder, exist_ok=True)
    for k, p in (x for x in enumerate(P) if x[1]["flags"] and x[1].get("idx") is not None):
        grey, cl = slices[p["idx"]]
        fm = cl.get(p["cls"], cl["whole"])
        ims = [show(_crop(comps[k]["gray"], comps[k]["mask"])[0]), show(_crop(np.where(fm, grey, 0), fm)[0])]
        out = Image.new("L", (ims[0].width + ims[1].width + 8, 280), 40)
        out.paste(ims[0], (0, 0))
        out.paste(ims[1], (ims[0].width + 8, 0))
        out.save(os.path.join(folder, f"s{p['section']:03d}_{p['kind'].split('-')[-1]}_{COMPONENT[p['cls']]}_photo{p['idx'] + idx0}.png"))


def propose(store, mask, regions, secs, idx0, out_dir, a0, vx_um, scale_xyz=None, jobs=1, cfg=None, log=print):
    """The whole proposal -> out_dir/{match_table_proposed.txt, match_review.csv, match_review/}. a0: photographs per section; vx_um:
    the photograph pixel the volume was built at. Returns the summary {components, matched, flagged, runs, ...} and the components."""
    cfg = cfg or M.config()
    pc = cfg["propose"]
    os.makedirs(out_dir, exist_ok=True)
    photo_um = pc["section_um"] / a0
    step = pc["sample_step_fine"] if photo_um <= pc["fine_step_um"] else 1
    slices = blockface_slices(store, mask, regions, scale_xyz, step, cfg, log)
    comps = section_components(secs, cfg)
    log(f"  {len(comps)} tissue components in {len({c['section'] for c in comps})} sections")
    P = outline_fit(comps, slices, a0, cfg)
    add_scores(P, comps, slices, a0, vx_um * pc["photo_px_factor"] / 1000.0, jobs, cfg, log)
    runs = run_lines(P, a0, step, cfg)
    rows = table_rows(P, idx0, cfg)
    write_table(os.path.join(out_dir, PROPOSED), rows)
    write_review(os.path.join(out_dir, REVIEW), P, idx0, cfg)
    review_images(os.path.join(out_dir, "match_review"), P, comps, slices, idx0)
    summ = dict(components=len(P), matched=sum(v != -1 for r in rows.values() for v in r), flagged=sum(1 for p in P if p["flags"]),
                runs=runs, photos_per_section=a0, sample_step=step)
    log(f"  proposed {summ['matched']} of {summ['components']} components matched; {summ['flagged']} flagged for review -> "
        f"{os.path.join(out_dir, PROPOSED)}")
    return summ, P
