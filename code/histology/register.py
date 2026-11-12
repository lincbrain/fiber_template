"""Histology to blockface registration, one component at a time (Methods, "Histology registration and label propagation").

You need
  the histology sections (JPEG 2000), the blockface components (pieces.cut_pieces) and TIRL 3.6.3 or 3.6.2 (the registration
  library; settings/env/tirl.yml); parameters: settings/histology.yaml.
What it does
    tissue_masks    each section (green band, low resolution) -> an Otsu tissue mask per component
    moving_images   the blue band masked by the component's tissue mask, scaled into the blockface component's box
    fixed_images    the blockface component, contrast x1.6, grayscale
    run_tirl        `linc.h2b` (run by the script's path) per component: rotation, rigid, affine, nonlinear; the nonlinear stage's
                    stiffness per component from its affine-stage tissue Dice (settings/histology.yaml tirl.stiff_nonlinear)
    registered_rgb  the fitted TIRL transforms applied to all three JPEG 2000 bands
  The moving and fixed images share the blockface component's pixel grid, so TIRL registers them at one pixel size
  (settings/histology.yaml tirl.resolution_mm). run_tirl checks TIRL's version first; registered_rgb imports tirl, so it runs under
  the TIRL environment's Python, one process per component (TIRL's evaluate can corrupt the heap).
You get
  <masks>/<section>_<tag>.png, <moving>/<component>.png, <fixed>/<component>.png, <tirl>/<component>/ (the transforms and the
  registered images, stiff_nonlinear.json: the affine Dice and the weight it chose) and <tirl>/<component>/moving4_nonlinear_rgb.png.
  Tags: 0000 whole, 0001 left, 0002 right, 0003 cerebellum.

Ported from the analysis project's code/register.py (histo_masks, histo_moving, bf_fixed, tirl_h2b), labels/histo_masks_par.py,
labels/tirl_worker.py and labels/histo_rgb.py, the code of the products of record.
"""
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageEnhance

from . import match as M

Image.MAX_IMAGE_PIXELS = None


# --------------------------------------------------------------------------- tissue extent

def _first_nonzero_row(img):
    for i in range(img.shape[0]):
        if np.sum(img[i]) > 0:
            return i
    return img.shape[0]


def borders(img, margin):
    """Distances (top, bottom, left, right) from each edge to the tissue, less a `margin` px margin."""
    t = _first_nonzero_row(img)
    b = _first_nonzero_row(img[::-1])
    l = _first_nonzero_row(np.rot90(img, 3))
    r = _first_nonzero_row(np.rot90(img, 1))
    return [max(0, k - margin) for k in (t, b, l, r)]


def _rlevel(j2k, want):
    levels = j2k.codestream.segment[2].num_res
    return want if levels >= want else levels - 1


# --------------------------------------------------------------------------- tissue masks

def _enhance(a, mc):
    """The contrast-stretched band with enhance_offset added to every nonzero pixel, saturated at 255. (Until 12 Nov 2026 the sum
    wrapped in uint8 and was then multiplied by 255 in uint8: moderately bright tissue, e.g. a darkfield injection core, fell near 0,
    below the Otsu threshold, and where it reached the border the outer contour detoured around it. The paper's registrations use the
    saturated form; analysis: labels/fix_r058/masks_r058.py gen(fixed=True).)"""
    param = mc["enhance_low_contrast"] if np.std(a) < mc["enhance_std_threshold"] else mc["enhance_default"]
    out = np.array(ImageEnhance.Contrast(Image.fromarray(a)).enhance(param)).astype(np.int32)
    return np.where(out > 0, np.minimum(out + mc["enhance_offset"], 255), 0).astype(np.uint8)


def generate_mask(jp2_path, cfg=None):
    """Two piece masks of one section (mask2 is zeros for a single-piece section), mask1 the image-left one."""
    import cv2
    import glymur
    from skimage import measure
    from skimage.draw import polygon2mask
    cfg = cfg or M.config()
    mc = cfg["masks"]
    j2k = glymur.Jp2k(jp2_path)
    band = j2k.read(rlevel=_rlevel(j2k, mc["jp2_rlevel"]))[:, :, mc["band"]]
    bs = mc["pad_px"]
    pad = np.zeros((band.shape[0] + 2 * bs, band.shape[1] + 2 * bs), np.uint8)
    pad[bs:-bs, bs:-bs] = band
    enh = _enhance(pad, mc)
    _, binary = cv2.threshold(enh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours = sorted(measure.find_contours(binary), key=len, reverse=True)
    two = len(contours) > 1 and len(contours[1]) / len(contours[0]) > mc["two_pieces_min_ratio"]
    h, w = binary.shape
    m1 = np.asarray(polygon2mask((h, w), contours[0]), np.uint8)
    m2 = np.asarray(polygon2mask((h, w), contours[1]), np.uint8) if two else np.zeros((h, w), np.uint8)
    if two and _first_nonzero_row(np.rot90(m1, 3)) > _first_nonzero_row(np.rot90(m2, 3)):
        m1, m2 = m2, m1
    return m1[bs:-bs, bs:-bs], m2[bs:-bs, bs:-bs]


def _dilate(m, mc):
    import cv2
    k = mc["dilate_kernel"]
    return cv2.dilate(m, np.ones((k, k), np.uint8), iterations=mc["dilate_iterations"])


def _mask_one(args):
    jp2, table, out, cfg_path = args
    cfg = M.config(cfg_path)
    mc = cfg["masks"]
    tag = M.column_tag(cfg)
    sid = os.path.basename(jp2)[:-4]
    whole, left, right, cere = M.match_table(table, cfg)(M.section_number(sid, cfg))
    m1, m2 = generate_mask(jp2, cfg)

    def save(m, col):
        Image.fromarray(_dilate(m, mc)).save(os.path.join(out, f"{sid}_{tag[col]}.png"))

    if whole != -1:
        save(m1, 0)
    elif cere != -1:
        save(m1, 3)
    elif left != -1 and right != -1:
        save(m1, 1); save(m2, 2)
    elif right != -1:
        save(m1, 2)
    elif left != -1:
        save(m1, 1)
    return sid


def tissue_masks(histo_dir, table, out_dir, jobs=1, cfg_path=None, pattern=None, log=print):
    """Per-piece tissue masks of every section -> out_dir/<section>_<tag>.png (one process per section when jobs > 1).
    A whole or cerebellum row gets one mask; a left + right row gets the image-left mask as left and the other as right."""
    cfg = M.config(cfg_path)
    os.makedirs(out_dir, exist_ok=True)
    jp2s = M.sections(histo_dir, cfg, pattern)
    args = [(j, str(table), str(out_dir), cfg_path) for j in jp2s]
    if jobs > 1:
        with Pool(jobs) as p:
            done = p.map(_mask_one, args, chunksize=1)
    else:
        done = [_mask_one(a) for a in args]
    assert len(done) == len(jp2s)
    log(f"generated histology masks for {len(done)} sections -> {out_dir}")
    return len(done)


# --------------------------------------------------------------------------- moving and fixed images

def resize_to_grid(histo_obj, photo, cfg=None):
    """Scale the masked histology to fit inside the blockface piece's tissue box and paste it onto a canvas the size of the
    piece, so moving and fixed share one pixel grid. Blank if either is empty."""
    cfg = cfg or M.config()
    mv = cfg["moving"]
    H, W = photo.shape[:2]
    canvas = np.zeros((H, W), np.uint8)
    tt, tb, ll, lr = borders(photo, mv["border_margin_px"])
    box_h, box_w = H - tt - tb, W - ll - lr
    t, b, l, r = borders(histo_obj, mv["border_margin_px"])
    h, w = max(0, histo_obj.shape[0] - t - b), max(0, histo_obj.shape[1] - l - r)
    if box_h <= 0 or box_w <= 0 or h <= 0 or w <= 0:
        return canvas
    obj = histo_obj[t:histo_obj.shape[0] - b, l:histo_obj.shape[1] - r]
    while h < box_h and w < box_w:
        h, w = mv["grow"] * h, mv["grow"] * w
    while h > box_h or w > box_w:
        h, w = mv["shrink"] * h, mv["shrink"] * w
    h, w = max(1, int(h)), max(1, int(w))
    canvas[tt:tt + h, ll:ll + w] = np.asarray(Image.fromarray(obj).resize((w, h)))
    return canvas


def moving_images(histo_dir, masks_dir, pieces_dir, out_dir, cfg=None, pattern=None, log=print):
    """Blue band of each section masked by each piece's tissue mask, on its blockface piece grid -> out_dir/<piece>.png."""
    import glymur
    cfg = cfg or M.config()
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for jp2 in M.sections(histo_dir, cfg, pattern):
        sid = os.path.basename(jp2)[:-4]
        j2k = glymur.Jp2k(jp2)
        gray = j2k.read_bands(rlevel=_rlevel(j2k, cfg["moving"]["jp2_rlevel"]))[:, :, cfg["moving"]["band"]]
        for mpath in sorted(glob(os.path.join(str(masks_dir), sid + "_*.png"))):
            piece = os.path.basename(mpath)[:-4]
            photo_path = os.path.join(str(pieces_dir), piece + ".png")
            if not os.path.exists(photo_path):
                continue
            photo = np.asarray(Image.open(photo_path))
            if not (photo > 0).any():
                continue
            mask = np.asarray(Image.open(mpath))
            masked = np.where(mask, gray, 0)
            Image.fromarray(resize_to_grid(masked, photo, cfg)).save(os.path.join(out_dir, piece + ".png"))
            n += 1
    log(f"built {n} moving (histology) images -> {out_dir}")
    return n


def fixed_images(pieces_dir, out_dir, cfg=None, log=print):
    """Grayscale, contrast-boosted blockface pieces -> out_dir/<piece>.png."""
    cfg = cfg or M.config()
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for p in sorted(glob(os.path.join(str(pieces_dir), "*.png"))):
        gray = np.array(ImageEnhance.Contrast(Image.open(p)).enhance(cfg["fixed"]["contrast"]).convert("L"))
        Image.fromarray(gray).save(os.path.join(out_dir, os.path.basename(p)))
        n += 1
    log(f"built {n} fixed (blockface) images -> {out_dir}")
    return n


# --------------------------------------------------------------------------- TIRL

def tirl_python(executable=None, cfg=None):
    """The Python interpreter of the `tirl` executable: its shebang, or, for a shell wrapper (FSL's `tirl` runs
    `<fsl>/bin/python -I <fsl>/bin/tirl "$@"`), the first python it calls. None if neither is found."""
    import re
    cfg = cfg or M.config()
    here = os.path.join(os.path.dirname(sys.executable), cfg["tirl"]["executable"])
    exe = here if not executable and os.access(here, os.X_OK) else shutil.which(executable or cfg["tirl"]["executable"])
    if not exe:                                      # (beside the running interpreter first, as check_tirl: FSL's tirl is on PATH)
        return None
    with open(exe, "rb") as f:
        head = f.read(4096).decode(errors="replace").splitlines()
    is_py = re.compile(r"python[0-9.]*$")
    if head and head[0].startswith("#!"):
        parts = head[0][2:].split()
        if parts and os.path.basename(parts[0]) == "env" and len(parts) > 1:
            parts = [shutil.which(parts[1]) or parts[1]]
        if parts and is_py.search(os.path.basename(parts[0])):
            return parts[0]
    for line in head[1:]:
        for tok in line.split():
            if is_py.search(os.path.basename(tok)) and os.access(tok, os.X_OK):
                return tok
    return None


def check_tirl(executable=None, cfg=None):
    """Return (path to tirl, version); raise if it is missing or not the version of record."""
    cfg = cfg or M.config()
    here = os.path.join(os.path.dirname(sys.executable), cfg["tirl"]["executable"])
    if not executable and os.access(here, os.X_OK):  # beside the running interpreter first (`$PY_TIRL code/histology/linc-histology ...`):
        exe = here                                   # FSL's setup puts its own, older `tirl` (2.2.1) on PATH
    else:
        exe = shutil.which(executable or cfg["tirl"]["executable"])
    if not exe:
        raise RuntimeError(f"`{executable or cfg['tirl']['executable']}` not found on PATH (TIRL {cfg['tirl']['version']} needed)")
    py = tirl_python(exe, cfg)
    if not py:
        raise RuntimeError(f"cannot find the Python interpreter of {exe}")
    r = subprocess.run([py, "-c", "import tirl; print(tirl.__version__)"], capture_output=True, text=True)
    ver = r.stdout.strip()
    if r.returncode != 0:
        raise RuntimeError(f"cannot import tirl with {py} ({exe}): {r.stderr.strip()[-200:]}")
    if ver != str(cfg["tirl"]["version"]) and ver not in [str(v) for v in cfg["tirl"].get("also_accepted", [])]:
        raise RuntimeError(f"TIRL {cfg['tirl']['version']} required, found {ver} ({exe})")
    return exe, ver


def tirl_script(executable=None, cfg=None):
    """The path of the TIRL script (settings/histology.yaml tirl.script_module) in the environment of `tirl`. _tirl_one runs it as
    the `tirl` launcher does (`<its python> <script> <args>`, the script's folder appended to PATH), so neither `tirl register`
    (whose registry, ~/.config/tirlscripts.yml, every TIRL environment of the account shares) nor the registry file is needed."""
    cfg = cfg or M.config()
    py = tirl_python(executable, cfg)
    r = subprocess.run([py, "-c", "import importlib.util, sys; print(importlib.util.find_spec(sys.argv[1]).origin)",
                        cfg["tirl"]["script_module"]], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(r.stdout.strip()):
        raise RuntimeError(f"{cfg['tirl']['script_module']} not found by {py}: install tirlscripts-mgh-linc (settings/env/tirl.yml)")
    return r.stdout.strip()


def tirl_params(moving, fixed, piece_dir, cfg=None):
    """The TIRL configuration of one piece (the configuration of record with this piece's files)."""
    cfg = cfg or M.config()
    p = copy.deepcopy(cfg["tirl"]["config"])
    p["histology"]["file"] = str(moving)
    p["histology"]["resolution"] = cfg["tirl"]["resolution_mm"]
    p["block"]["file"] = str(fixed)
    p["block"]["resolution"] = cfg["tirl"]["resolution_mm"]
    p["general"]["outputdir"] = str(piece_dir)
    p["general"]["logfile"] = os.path.join(str(piece_dir), "logfile.log")
    p["general"]["paramlogfile"] = os.path.join(str(piece_dir), "paramlogfile.log")
    p["general"]["system"] = "linux"
    return p


def affine_dice(piece_dir, cfg=None):
    """The registration QC's tissue Dice after the affine stage (the analysis's labels/qc_histo2bf.piece_qc): the blockface piece
    (fixed.png above 0) against the histology after the affine stage (moving3_affine.png; its 255 = outside the domain = background),
    both block-averaged dice_block_px x dice_block_px; rounded to 4 decimals."""
    cfg = cfg or M.config()
    k = int(cfg["tirl"]["stiff_nonlinear"]["dice_block_px"])

    def load(name):
        a = np.asarray(Image.open(os.path.join(piece_dir, name)).convert("L"))
        a = a[: a.shape[0] // k * k, : a.shape[1] // k * k]
        return a.reshape(a.shape[0] // k, k, a.shape[1] // k, k).mean((1, 3))
    f = load("fixed.png") > 0
    m = load("moving3_affine.png")
    m = np.where(m >= 254.5, 0, m) > 0
    return round(float(2 * (f & m).sum() / max(f.sum() + m.sum(), 1)), 4)


def _h2b(piece, moving_dir, fixed_dir, d, exe, script, threads, cfg, regweight):
    import yaml
    os.makedirs(d, exist_ok=True)
    fname = os.path.join(d, "tirlparam.yml")
    p = tirl_params(os.path.join(moving_dir, piece + ".png"), os.path.join(fixed_dir, piece + ".png"), d, cfg)
    p["regparams"]["nonlinear"]["regweight"] = regweight
    with open(fname, "w") as f:
        yaml.dump(p, f)
    env = dict(os.environ, PATH=os.environ.get("PATH", "") + ":" + os.path.dirname(script))
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[v] = str(threads)
    with open(os.path.join(d, "worker_stdout.log"), "w") as log:
        rc = subprocess.run([tirl_python(exe, cfg), script, "--config", fname, "--verbose"], stdout=log,
                            stderr=subprocess.STDOUT, env=env).returncode
    return rc, rc == 0 and os.path.exists(os.path.join(d, cfg["tirl"]["done_file"]))


def _tirl_one(piece, moving_dir, fixed_dir, out_dir, exe, script, threads, cfg):
    """One component: TIRL at the stiff weight, then the rule on its affine stage (which does not depend on the nonlinear weight):
    below the gate, rerun from scratch at the configuration's weight. A component registered before (done file, no decision) takes
    the same rule from its own affine stage and is rerun only if it ran at the other weight."""
    import yaml
    rule = cfg["tirl"]["stiff_nonlinear"]
    stiff, base = float(rule["regweight"]), float(cfg["tirl"]["config"]["regparams"]["nonlinear"]["regweight"])
    d = os.path.join(out_dir, piece)
    t0 = time.time()
    rc, ok = 0, os.path.exists(os.path.join(d, cfg["tirl"]["done_file"]))
    if not ok:
        rc, ok = _h2b(piece, moving_dir, fixed_dir, d, exe, script, threads, cfg, stiff)
    dec = None
    if ok:
        used = float(yaml.safe_load(open(os.path.join(d, "tirlparam.yml")))["regparams"]["nonlinear"]["regweight"])
        dice = affine_dice(d, cfg)
        want = stiff if dice >= float(rule["affine_dice_min"]) else base
        dec = dict(dice_affine=dice, affine_dice_min=float(rule["affine_dice_min"]), regweight=want, first_run_regweight=used)
        if used != want:
            shutil.rmtree(d)
            rc, ok = _h2b(piece, moving_dir, fixed_dir, d, exe, script, threads, cfg, want)
            if ok:                                   # the affine stage is the same at either weight: the decision stands
                dec["dice_affine_rerun"] = affine_dice(d, cfg)
                assert dec["dice_affine_rerun"] == dice, f"{piece}: affine Dice {dice} -> {dec['dice_affine_rerun']} on the rerun"
        if ok:
            with open(os.path.join(d, "stiff_nonlinear.json"), "w") as f:
                json.dump(dec, f)
    return piece, ok, rc, time.time() - t0, dec


def tirl_clip_applied(executable=None, cfg=None):
    """True / False: the TIRL that runs carries our start-point clip (code/histology/patches/README.txt); None if not decidable."""
    import hashlib
    cfg = cfg or M.config()
    spec = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "patches", "tirl-3.6.3_optnl_clip.edits.json")))
    r = subprocess.run([tirl_python(executable, cfg), "-c", "import tirl.optimisation.optnl as o; print(o.__file__)"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    h = hashlib.md5(open(r.stdout.strip(), "rb").read()).hexdigest()
    return True if h == spec["result_md5"] else False if h == spec["source_md5"] else None


def run_tirl(moving_dir, fixed_dir, out_dir, jobs=1, threads=None, executable=None, pieces=None, cfg=None, log=print):
    """TIRL per piece -> out_dir/<piece>/ (a piece with fixed4_nonlinear.timg and stiff_nonlinear.json is skipped; one with the first
    only takes the stiffness rule, _tirl_one). Returns {piece: {"ok", "rc", "wall_s", "dice_affine", "regweight"}} for the pieces run
    now; per-piece wall-clock and the weight also go to out_dir/timing.tsv."""
    cfg = cfg or M.config()
    exe, ver = check_tirl(executable, cfg)
    script = tirl_script(exe, cfg)
    threads = threads or cfg["tirl"]["threads"]
    os.makedirs(out_dir, exist_ok=True)
    todo = []
    for mv in sorted(glob(os.path.join(str(moving_dir), "*.png"))):
        piece = os.path.basename(mv)[:-4]
        if pieces and piece not in pieces:
            continue
        if all(os.path.exists(os.path.join(out_dir, piece, f)) for f in (cfg["tirl"]["done_file"], "stiff_nonlinear.json")):
            continue
        if not os.path.exists(os.path.join(str(fixed_dir), piece + ".png")):
            log(f"! no fixed image for {piece}, skipped")
            continue
        todo.append(piece)
    clip = tirl_clip_applied(exe, cfg)
    log(f"TIRL {ver} ({exe} {script}): {len(todo)} pieces, {jobs} at a time, {threads} threads each; start-point clip "
        f"{'applied' if clip else 'NOT applied (code/histology/patches/README.txt)' if clip is False else 'unknown (optnl.py not the 3.6.3 file)'}")
    res = {}
    with ThreadPoolExecutor(max(1, jobs)) as ex:
        futs = [ex.submit(_tirl_one, p, str(moving_dir), str(fixed_dir), str(out_dir), exe, script, threads, cfg) for p in todo]
        for fu in futs:
            piece, ok, rc, dt, dec = fu.result()
            res[piece] = dict(ok=ok, rc=rc, wall_s=round(dt, 2), dice_affine=dec and dec["dice_affine"], regweight=dec and dec["regweight"])
            with open(os.path.join(out_dir, "timing.tsv"), "a") as f:
                f.write(f"{piece}\t{dt:.2f}\t{'ok' if ok else f'rc={rc}'}\t{dec and dec['regweight']}\n")
            log(f"  {piece} {'ok' if ok else f'FAILED rc={rc}'} {dt / 60:.1f} min" + (f", affine Dice {dec['dice_affine']}, nonlinear weight "
                f"{dec['regweight']}" if dec else ""))
    log(f"TIRL histology -> blockface: {sum(r['ok'] for r in res.values())} ok, "
        f"{sum(not r['ok'] for r in res.values())} failed -> {out_dir}")
    return res


# --------------------------------------------------------------------------- registered RGB

def _geometry_resize_rgb(obj_rgb, blue_masked, photo, cfg):
    """resize_to_grid with the box taken from the masked blue band (what TIRL registered), applied to all channels."""
    mv = cfg["moving"]
    H, W = photo.shape[:2]
    canvas = np.zeros((H, W, 3), np.uint8)
    tt, tb, ll, lr = borders(photo, mv["border_margin_px"])
    box_h, box_w = H - tt - tb, W - ll - lr
    t, b, l, r = borders(blue_masked, mv["border_margin_px"])
    h, w = max(0, blue_masked.shape[0] - t - b), max(0, blue_masked.shape[1] - l - r)
    if box_h <= 0 or box_w <= 0 or h <= 0 or w <= 0:
        return canvas
    obj = obj_rgb[t:blue_masked.shape[0] - b, l:blue_masked.shape[1] - r]
    while h < box_h and w < box_w:
        h, w = mv["grow"] * h, mv["grow"] * w
    while h > box_h or w > box_w:
        h, w = mv["shrink"] * h, mv["shrink"] * w
    h, w = max(1, int(h)), max(1, int(w))
    for c in range(3):
        canvas[tt:tt + h, ll:ll + w, c] = np.asarray(Image.fromarray(np.ascontiguousarray(obj[..., c])).resize((w, h)))
    return canvas


def rgb_piece(jp2, mask_png, piece_png, moving_png, piece_dir, cfg_path=None):
    """One piece's registered RGB (run in its own process under the TIRL environment) -> piece_dir/moving4_nonlinear_rgb.png
    + rgb_check.json (the blue channel must reproduce the moving image: max |difference| recorded)."""
    import glymur
    from tirl import load
    from tirl.timage import TImage
    from tirl.interpolation.scipyinterpolator import ScipyInterpolator
    cfg = M.config(cfg_path)
    j2k = glymur.Jp2k(jp2)
    rgb = j2k.read_bands(rlevel=_rlevel(j2k, cfg["moving"]["jp2_rlevel"]))[:, :, :3]
    mask = np.asarray(Image.open(mask_png)) > 0
    photo = np.asarray(Image.open(piece_png))
    rgbm = np.where(mask[..., None], rgb, 0).astype(np.uint8)
    canvas = _geometry_resize_rgb(rgbm, rgbm[..., cfg["moving"]["band"]], photo, cfg)
    mv_ref = np.asarray(Image.open(moving_png))
    blue_diff = int(np.abs(canvas[..., 2].astype(int) - mv_ref.astype(int)).max()) if mv_ref.shape == canvas.shape[:2] else -1
    m, f = load(os.path.join(piece_dir, "moving.timg")), load(os.path.join(piece_dir, cfg["tirl"]["done_file"]))
    out = []
    for c in range(3):
        T = TImage(canvas[..., c].astype(np.float32)); T.domain = m.domain.copy(); T.interpolator = ScipyInterpolator(order=1)
        out.append(np.array(T.evaluate(f.domain)).squeeze())
    k = cfg["rgb"]["domain_marker"]
    dom = TImage(np.full(canvas.shape[:2], k, np.float32)); dom.domain = m.domain.copy(); dom.interpolator = ScipyInterpolator(order=0)
    inside = np.isclose(np.array(dom.evaluate(f.domain)).squeeze(), k)
    res = np.stack([np.where(inside, np.clip(o, 0, 255), cfg["rgb"]["outside_value"]) for o in out], -1).astype(np.uint8)
    Image.fromarray(res).save(os.path.join(piece_dir, "moving4_nonlinear_rgb.png"))
    piece = os.path.basename(os.path.normpath(piece_dir))
    with open(os.path.join(piece_dir, "rgb_check.json"), "w") as fh:
        json.dump(dict(piece=piece, blue_vs_moving_png_maxabs=blue_diff, inside_px=int(inside.sum())), fh)
    print(f"{piece}: blue vs moving.png max|diff| {blue_diff}, inside {int(inside.sum())} px", flush=True)


def registered_rgb(histo_dir, masks_dir, pieces_dir, moving_dir, tirl_dir, jobs=1, python=None, cfg_path=None, log=print):
    """Registered RGB for every registered piece that lacks it or is older than its TIRL result. Each piece runs in its own
    process under `python` (default: the tirl executable's interpreter). Returns the list of failed pieces."""
    cfg = M.config(cfg_path)
    done = cfg["tirl"]["done_file"]
    python = python or tirl_python(None, cfg) or sys.executable
    pieces = sorted(os.path.basename(os.path.dirname(p)) for p in glob(os.path.join(str(tirl_dir), "*", done)))

    def stale(p):
        o = os.path.join(str(tirl_dir), p, "moving4_nonlinear_rgb.png")
        return not os.path.exists(o) or os.path.getmtime(o) < os.path.getmtime(os.path.join(str(tirl_dir), p, done))
    todo = [p for p in pieces if stale(p)]
    repo = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "code")

    def run(p):
        sid = p.rsplit("_", 1)[0]
        args = [os.path.join(str(histo_dir), sid + ".jp2"), os.path.join(str(masks_dir), p + ".png"),
                os.path.join(str(pieces_dir), p + ".png"), os.path.join(str(moving_dir), p + ".png"),
                os.path.join(str(tirl_dir), p), cfg_path]
        code = ("import sys, os; sys.path.insert(0, sys.argv[1]); from histology.register import rgb_piece; "
                "a = sys.argv[2:]; rgb_piece(*a[:5], cfg_path=a[5] or None); os._exit(0)")
        r = subprocess.run([python, "-c", code, repo, *args[:5], args[5] or ""], capture_output=True, text=True)
        return p, r.returncode, r.stderr[-300:]
    bad = []
    with ThreadPoolExecutor(max(1, jobs)) as ex:
        for p, rc, err in ex.map(run, todo):
            if rc != 0 or not os.path.exists(os.path.join(str(tirl_dir), p, "moving4_nonlinear_rgb.png")):
                bad.append(p); log(f"! {p} rc={rc} {err}")
    checks = [json.load(open(os.path.join(str(tirl_dir), p, "rgb_check.json"))) for p in pieces
              if os.path.exists(os.path.join(str(tirl_dir), p, "rgb_check.json"))]
    bd = [c["blue_vs_moving_png_maxabs"] for c in checks]
    log(f"{len(pieces)} pieces, {len(todo)} rendered now, {len(bad)} failed; blue-vs-moving max|diff| max {max(bd) if bd else None}")
    return bad
