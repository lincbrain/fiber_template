#!/usr/bin/env python3
"""Step 9 of the blockface template: the published template, one step finer than MMORF's eighth.

MMORF's eighth step ends with the members aligned at 0.25 mm. Step 9 aligns 11 of the 21 brains (each also mirrored: 22 inputs)
further, whole brain, with the feature-based engine (anatomix + FireANTs), and averages them again from their own photographs.
The brains are those photographed at every section with at most 10 % of the sections interpolated and with even lighting
(settings/blockface_template.yaml, step9). The template keeps step 8's frame, so everything registered to step 8's template (the
template link, the brains' chains) applies to it unchanged.

You need
  settings/env/anatomix.yml's Python ($ANATOMIX_PYTHON) for every step: the averaging uses the same torch as the registration, and
  the template of record was made with it. The registrations need one GPU with 48 GB (each job peaks near 36 GB); everything
  else runs on the CPU.
  Under $LINC_PRODUCTS (settings/internal/products_blockface.yaml, products_completion.yaml): step 8's template and member warps,
  its box and precision rule, each brain's photographs (the L2 level), its registration to its own b0 (or, for a brain without
  MRI, its placement), its completed volume with the synthesized tissue, and its seam record.
What it does
  start      every input carried once from its photographs onto the rounds' grid (0.125 mm): template point -> step 8's member
             warp -> mirror (for a mirrored input) -> the brain's photographs; then averaged into T0<tag>. Each input's tissue
             is a probability (the photograph's own mask; the brain mask where the tissue was synthesized), every value is
             normalized by its mask, and the weights are step 8's precision rule.
  round      round k = register, unbias and carry below, in one command on one card (about 3 h on an RTX 6000 Ada).
  register   round k: each input registered whole brain to T(k-1) on a 0.15 mm grid; the field is resampled once onto the
             0.125 mm grid (one job per input, about 7 minutes on an RTX 6000 Ada). Run it on more cards, on any machine that
             sees the work folder, and they share the round's jobs.
  unbias     round k: a member whose field folds is left unmoved (more than 2 stops the build); the mean field is inverted
             and composed with every field, so the template does not drift; no field may fold afterwards.
  carry      round k: every input carried again from its photographs through its field -> Tk<tag> (color in the last round).
  final      the last round's fields resampled once onto the 0.15 mm grid, every input carried again from its photographs, and
             the start and the template averaged there; the checks (sharpness against the start; r and shift against step 8;
             exact symmetry) -> final/checks.json.
  in-mri     the template in the MRI template's frame through the template link, as code/template/link/register.py
             blockface_in_mri does, and multiplied by its carried brain mask.
  refine     optional, one 48 GB card: a brain registered whole brain to the finished template with a member's job. A brain of
             the paper starts from its step-8 warp; a new brain from its registration to the blockface template (photographs only),
             and gets a refined transform entry (the field, then its own entry) in $LINC_DATA/sub-<ID>/xfm_step9/.
You get
  under the work folder: nln_step_09/iteration_00..03 (the rounds, as MMORF's run folders are laid out: inputs/, fields_raw/,
  fields/, the template T<k><tag>_*, with the member set's tag from the settings), final/ (the 0.15 mm template
  T3<tag>_{grey,rgb,brainmask,real_mask,photo_share,P}, the start T0<tag>, the fields, checks.json) and in_mri/ (linc_bf_*).

    $ANATOMIX_PYTHON code/template/blockface_step9.py start [-j 8]
    $ANATOMIX_PYTHON code/template/blockface_step9.py round --it 1 --gpu 0         # rounds 2 and 3 the same way
    $ANATOMIX_PYTHON code/template/blockface_step9.py register --it 1 --gpu 1      # optional: another card joins round 1
    $ANATOMIX_PYTHON code/template/blockface_step9.py final [-j 8]
    $ANATOMIX_PYTHON code/template/blockface_step9.py in-mri

Ported from the analysis project's bftemplate/hr_2026-11-12/hr.py (the carry, its polish of 14 Nov 2026, the combination, the
unbias by composition), step9/step9.py, step9/s9reg15.py and step9/s9_015.py (the 0.15 mm grid), and step9/s9_015_mri.py;
function bodies unchanged where the arithmetic is concerned. The template of record was built with them on 14-15 Nov 2026.
"""
import argparse
import json
import os
import resource
import shutil
import socket
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import nibabel as nib

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))
from shared import paths, xfm  # noqa: E402
from blockface.registration import xfmtools as X  # noqa: E402
from template import blockface_finish as F  # noqa: E402

TCFG = __import__("yaml").safe_load((REPO / "settings" / "blockface_template.yaml").read_text())
C9 = TCFG["step9"]
W_LUM = np.array([0.299, 0.587, 0.114], np.float32)               # the photographs' luminance (the iso grey's rule)
PAIR_BELOW_MM = float(C9["pair_below_mm"])
KZ = int(C9["slab_vox"])
POLISH_NEAR_VOX = int(C9["polish"]["near_vox"])
POLISH_DEN_MIN = float(C9["polish"]["den_min"])
CROPS = dict(a_occipital=(85, 24, 55), b_frontal=(85, 219, 80))   # two 20 mm cubes of step 8's box (voxels), reported beside the whole

# the grid being worked on: "box" = step 8's box refined corner to corner (the rounds, 0.125 mm); "mirror" = the published grid
# (0.15 mm), whose count per axis is the box's refined, y and z corner-anchored, x centered on the mirror plane x = 0. Set by
# use_grid() before any worker starts (the workers inherit it).
G = dict(vox=float(C9["grid_mm"]), kind="box", root=None)


def use_grid(kind, root):
    G.update(kind=kind, vox=float(C9["grid_mm"] if kind == "box" else C9["template_mm"]), root=Path(root))


def P():
    return paths.products("products_blockface")


def PC():
    return paths.products("products_completion")


# ------------------------------------------------------------------------------------------------------------------- grids
def box():
    b = nib.load(P()["bf_step8_box"])
    return tuple(int(x) for x in b.shape[:3]), b.affine.astype(np.float64)


def grid():
    """The current grid (shape, affine)."""
    VOX = G["vox"]
    (n, A) = box()
    v25 = float(A[0, 0]); f = v25 / VOX
    shp = tuple(int(round(x * f)) for x in n)
    corner = A[:3, 3] - v25 / 2
    Ag = np.diag([VOX, VOX, VOX, 1.0]); Ag[:3, 3] = corner + VOX / 2
    if G["kind"] == "mirror":
        Ag[0, 3] = -(shp[0] - 1) / 2 * VOX
    return shp, Ag


def centers(ax):
    shp, A = grid()
    return A[ax, 3] + G["vox"] * np.arange(shp[ax])


def idx25(ax):
    """The 0.25 mm box voxel of each grid voxel: by its index ratio on the box's own refinement, by its center otherwise."""
    shp, _ = grid(); (n, A25) = box(); VOX = G["vox"]
    if G["kind"] == "box":
        return np.minimum((np.arange(shp[ax]) * VOX / 0.25 + VOX / 0.25 / 2).astype(int), n[ax] - 1)
    c0 = A25[ax, 3] - 0.125
    return np.clip(np.floor((centers(ax) - c0) / 0.25).astype(int), 0, n[ax] - 1)


def record_crop(name):
    """A file of step 8's template, cropped to the box (it is embedded on the run-113 grid's 2x refinement at an integer offset)."""
    (bs, bA), (fs, fA) = box(), F.run113_grid(2)
    off = (bA[:3, 3] - fA[:3, 3]) / np.diag(fA)[:3]; o = np.rint(off).astype(int)
    assert np.allclose(off, o, atol=1e-4)
    v = np.asanyarray(nib.load(P()["bftemplate_step8"] / name).dataobj)
    assert tuple(v.shape[:3]) == tuple(fs), f"{name}: not on the run-113 2x grid"
    return v[o[0]:o[0] + bs[0], o[1]:o[1] + bs[1], o[2]:o[2] + bs[2]]


def support():
    """Where any member of step 8 has brain, dilated 2 mm, on the current grid (the containing 0.25 mm voxel)."""
    from scipy import ndimage as ndi
    cov = ndi.binary_dilation(record_crop("d2bs8_coverage.nii.gz") > 0, iterations=int(C9["support_dilate_vox"]))
    return cov[np.ix_(*[idx25(a) for a in range(3)])]


def down25(v):
    """Exact area averaging of the current grid onto the box's 0.25 mm voxels."""
    (n, A25) = box(); out = v.astype(np.float32); VOX = G["vox"]
    for ax in range(3):
        nf = out.shape[ax]; M = np.zeros((n[ax], nf), np.float32)
        if G["kind"] == "box":
            for i in range(n[ax]):
                lo, hi = i * 0.25, (i + 1) * 0.25
                j0, j1 = int(np.floor(lo / VOX)), int(np.ceil(hi / VOX))
                for j in range(j0, min(j1, nf)):
                    M[i, j] = max(0.0, min(hi, (j + 1) * VOX) - max(lo, j * VOX)) / 0.25
        else:
            c0 = A25[ax, 3] - 0.125; lo15 = centers(ax) - VOX / 2 - c0
            for i in range(n[ax]):
                lo, hi = i * 0.25, (i + 1) * 0.25
                ov = np.clip(np.minimum(hi, lo15 + VOX) - np.maximum(lo, lo15), 0, None)
                M[i] = ov / 0.25
        out = np.moveaxis(np.tensordot(M, np.moveaxis(out, ax, 0), axes=(1, 0)), 0, ax)
    return out


def regions():
    """Step 8's photographed tissue (its real mask at the containing 0.25 mm voxel), eroded 1 mm; and the same in the two cubes."""
    from scipy import ndimage as ndi
    ix = [idx25(a) for a in range(3)]; VOX = G["vox"]
    real = (record_crop("d2bs8_real_mask.nii.gz") > 0)[np.ix_(*ix)]
    er = ndi.binary_erosion(real, iterations=int(round(1.0 / VOX)))
    regs = dict(whole=er)
    for k, c in CROPS.items():
        if G["kind"] == "box":
            f = 0.25 / VOX; m = np.zeros_like(er)
            m[int(c[0] * f):int((c[0] + 80) * f), int(c[1] * f):int((c[1] + 80) * f), int(c[2] * f):int((c[2] + 80) * f)] = True
            regs[k] = er & m
        else:
            sel = [(ix[a] >= c[a]) & (ix[a] < c[a] + 80) for a in range(3)]
            regs[k] = er & (sel[0][:, None, None] & sel[1][None, :, None] & sel[2][None, None, :])
    return regs


# ------------------------------------------------------------------------------------------------------- folders and inputs
def itdir(it):
    return G["root"] / "nln_step_09" / f"iteration_{it:02d}"


def inputs_dir(it):
    return itdir(it) / "inputs"


def fields_dir(it):
    return itdir(it) / "fields"


def tpl_name(it):
    return f"T{it}{C9['tag']}"


def tpl(it):
    return itdir(it) / tpl_name(it)


def subjects():
    """Each brain's completed volume and synthesis cache, and whether it has MRI (step 8's members)."""
    pc, mb = PC(), TCFG["members"]
    return {S: dict(kind="mri" if S in mb["with_mri"] else "nomri", root=str(pc["completed"]), cache=str(pc["completed_masks"]))
            for S in mb["with_mri"] + mb["without_mri"]}


def input_ids(brains=None):
    """The inputs, in step 8's run order (the averages add them in this order): each brain, then its mirror."""
    return [s for S in (brains or C9["members"]) for s in (S, S + "f")]


def l2_path(S):
    return P()["data"] / f"sub-{S}" / "blockface" / f"{S}_blockface_L2.nii.gz"


def member_warp(sid):
    """Step 8's warp of input sid (FSL relative, on its 0.25 mm canonical grid)."""
    S = sid.rstrip("f")
    return xfm.Warp(str(P()["bftemplate_step8"] / "warps" / f"{sid}_d2bs8_warp.nii.gz"), "fsl_rel",
                    src_grid=str(P()["bf_canon2_tpl"] / S / f"t1_grey{'_flip' if sid.endswith('f') else ''}.nii.gz"))


# ------------------------------------------------------------------------------------------------------- native and record
def seam_json(S):
    for d in ("canon_d3", "canon_d3n_prod_lo0"):
        p = PC()["completion_inputs"] / d / S / "seam.json"
        if p.exists():
            return json.load(open(p))
    raise FileNotFoundError(f"{S}: no seam.json in {PC()['completion_inputs']}")


def native_map(S):
    """Placed (run-113) world -> the brain's native blockface world: its registration to its own b0, or, for a brain without
    MRI, its placement (FSL dof 9 to its placed reference)."""
    sj = seam_json(S); tag = sj["nonlin"].replace("nonlinear_", "")
    D = P()["data"] / f"sub-{S}"
    iso = D / "blockface" / f"{S}_blockface_iso_grey.nii.gz"
    if tag.startswith("s2lin"):
        ref = D / "mri" / sj["nonlin"] / f"{S}_mri_full.nii.gz"
        T = xfm.read_fsl_affine(paths.rerooted(sj["matrix"]), str(iso), str(ref))
        return (lambda Q: Q @ T[:3, :3].T + T[:3, 3]), float(sj["blo"]), float(sj["bhi"]), dict(kind="placement (dof 9)")
    e = xfm.Entry(str(D / f"xfm_{tag}" / "bf-native__to__b0-native")); c = e.components()
    assert "placed" in c[0].label and c[1].__class__.__name__ == "Warp", f"{S}: unexpected components"
    return (lambda Q: c[2].pull(c[1].pull(Q))), float(sj["blo"]), float(sj["bhi"]), dict(kind="registration to its own b0")


class Native:
    """The brain's photographs from the L2 level, mask-normalized: n01 = clip((g - blo)/(bhi - blo), 0, 1) * valid, valid = g > 0,
    g = the luminance; sections closer than PAIR_BELOW_MM are averaged in pairs (anti-aliasing)."""

    def __init__(self, S, rgb=False, nmap=None, l2=None):
        self.X = X; self.rgb = rgb
        self.f, blo, bhi, self.how = nmap or native_map(S)
        im = nib.load(l2 or l2_path(S)); A = im.affine.astype(np.float64)
        assert np.count_nonzero(A[:3, 1]) == 1 and A[1, 1] != 0, f"{S}: L2 section axis is not axis 1"
        step = abs(float(A[1, 1])); pair = step < PAIR_BELOW_MM
        rgb = np.asanyarray(im.dataobj)
        nx, ny, nz = rgb.shape[:3]; ny2 = ny // 2 if pair else ny
        self.n01 = np.empty((nx, ny2, nz), np.float32); self.valid = np.empty((nx, ny2, nz), np.float32)
        self.nrgb = [np.empty((nx, ny2, nz), np.float32) for _ in range(3)] if self.rgb else None
        ch = 64
        for j0 in range(0, ny2, ch):
            j1 = min(ny2, j0 + ch)
            src = rgb[:, 2 * j0:2 * j1] if pair else rgb[:, j0:j1]
            g = src.astype(np.float32) @ W_LUM
            val = (g > 0).astype(np.float32)
            n = np.clip((g - blo) / max(bhi - blo, 1e-6), 0, 1) * val
            if self.rgb:
                for c in range(3):
                    x = src[..., c].astype(np.float32) / 255.0 * (g > 0)
                    self.nrgb[c][:, j0:j1] = 0.5 * (x[:, 0::2] + x[:, 1::2]) if pair else x
            if pair:
                n = 0.5 * (n[:, 0::2] + n[:, 1::2]); val = 0.5 * (val[:, 0::2] + val[:, 1::2])
            self.n01[:, j0:j1] = n; self.valid[:, j0:j1] = val
        del rgb
        if pair:
            A = A.copy(); A[:3, 3] = A[:3, 3] + 0.5 * A[:3, 1]; A[:3, 1] = 2 * A[:3, 1]
        self.A = A
        self.how.update(l2_shape=[nx, ny, nz], section_step_mm=step, paired=bool(pair), used_step_mm=abs(float(A[1, 1])))

    def rgb_at(self, Q, den):
        """The mask-normalized native RGB in [0, 1] (raw /255), then the brain's per-channel fit to the record's RGB scale."""
        out = np.empty((Q.shape[0], 3), np.float32)
        for c in range(3):
            x = self.X.sample(self.nrgb[c], self.A, Q, order=1) / np.maximum(den, 1e-6)
            out[:, c] = np.clip(self.fit[c][0] * x + self.fit[c][1], 0, 1)
        return out

    def fit_rgb(self, S, q):
        """Per channel, record RGB = g * native + o over the member's photographed 0.25 mm voxels (seam ramp >= 0.99,
        photographed fraction >= 0.9, brain; record values strictly inside (0, 1)): least squares, 1 M voxels at most."""
        src = f"{q['root']}/{S}"
        ci = nib.load(f"{src}/{S}_bf_completed_rgb.nii.gz"); rec = np.asanyarray(ci.dataobj).astype(np.float32)
        ramp = np.asanyarray(nib.load(f"{src}/{S}_seam_ramp_real.nii.gz" if os.path.exists(f"{src}/{S}_seam_ramp_real.nii.gz")
                                      else f"{src}/{S}_seam_ramp.nii.gz").dataobj)
        phi = np.asanyarray(nib.load(f"{src}/{S}_photo_fraction.nii.gz").dataobj)
        br = np.asanyarray(nib.load(f"{q['cache']}/{S}/deploy/brain.nii.gz").dataobj)
        ii = np.argwhere((ramp >= 0.99) & (phi >= 0.9) & (br > 0.5))
        rs = np.random.default_rng(0)
        if len(ii) > 10 ** 6:
            ii = ii[rs.choice(len(ii), 10 ** 6, replace=False)]
        W = ii @ ci.affine[:3, :3].T + ci.affine[:3, 3]
        Q = self.f(W)
        den = self.X.sample(self.valid, self.A, Q, order=1); ok = den >= 0.5
        self.fit = []
        rows = []
        for c in range(3):
            x = self.X.sample(self.nrgb[c], self.A, Q[ok], order=1) / den[ok]
            y = rec[ii[ok, 0], ii[ok, 1], ii[ok, 2], c]
            m = (y > 1e-3) & (y < 1 - 1e-3)
            (g, o), *_ = np.linalg.lstsq(np.c_[x[m], np.ones(m.sum())], y[m], rcond=None)
            self.fit.append((float(g), float(o)))
            rows.append(dict(gain=round(float(g), 4), offset=round(float(o), 4), r=round(float(np.corrcoef(x[m], y[m])[0, 1]), 4),
                             n=int(m.sum())))
        self.how["rgb_fit"] = rows


def record_source(S, q, rgb=False):
    """The member's completed volume and its companions on its 0.25 mm grid, sampled linearly, with:
    kappa = the caudal synthesized part (seam depth < 0 in the brain) and its POLISH_NEAR_VOX-voxel rim outside the brain;
    ramp_nr / phi_nr = the seam ramp / photo fraction at the nearest photographed (real-mask) voxel; near = the brain dilated
    POLISH_NEAR_VOX voxels; rm = the real mask (normalizer of the record's real image, which is 0 off it)."""
    from scipy import ndimage as ndi
    src = f"{q['root']}/{S}"
    ci = nib.load(f"{src}/{S}_bf_completed.nii.gz")
    flat, _ = F.unflatten_map(S, q["cache"], ci.affine, ci.shape)
    brain = np.asanyarray(nib.load(f"{q['cache']}/{S}/deploy/brain.nii.gz").dataobj).astype(np.float32)
    rm = np.asanyarray(nib.load(f"{q['cache']}/{S}/deploy/real_mask.nii.gz").dataobj) > 0
    vols = dict(brain=brain, rm=rm.astype(np.float32))
    ramp_nm = "seam_ramp_real" if os.path.exists(f"{src}/{S}_seam_ramp_real.nii.gz") else "seam_ramp"
    for key, nm in (("real", "bf_real"), ("synth", "bf_synth"), ("ramp", ramp_nm), ("depth", "seam_depth"), ("phi", "photo_fraction")):
        vols[key] = np.asanyarray(nib.load(f"{src}/{S}_{nm}.nii.gz").dataobj).astype(np.float32)
    b = brain > 0.5
    d = vols["depth"]; fin = d > -998.0
    caudal = b & fin & (d < 0)
    vols["kappa"] = (caudal | (ndi.binary_dilation(caudal, iterations=POLISH_NEAR_VOX) & ~b)).astype(np.float32)
    vols["near"] = ndi.binary_dilation(b, iterations=POLISH_NEAR_VOX).astype(np.float32)
    idx = ndi.distance_transform_edt(~rm, return_distances=False, return_indices=True)
    vols["ramp_nr"] = vols["ramp"][tuple(idx)]; vols["phi_nr"] = vols["phi"][tuple(idx)]; del idx
    vols["shell"] = F.shell_mask(S, q, ci.affine, brain).astype(np.float32)
    if rgb:
        r = np.asanyarray(nib.load(f"{src}/{S}_bf_completed_rgb.nii.gz").dataobj).astype(np.float32)
        for c in range(3):
            vols[f"rgb{c}"] = r[..., c]
    vols["depth"] = np.where(fin, d, float(d[fin].min()) if fin.any() else 0.0).astype(np.float32)
    aff = ci.affine

    def sample(pts):
        qq = flat(pts)
        return {k: X.sample(v, aff, qq, order=1) for k, v in vols.items()}
    return sample


# ---------------------------------------------------------------------------------------------------------------- the carry
def member_job(job):
    """One brain: the input and its mirror carried from the photographs through (y + u(y)) -> step 8's warp -> mirror -> native:
      t   = (1 - kappa) * den + kappa * brain        den = the photograph's own mask (valid) sampled linearly
      r'  = (1 - kappa) * ramp_nr,   phi' = phi_nr   the photograph's reliability, read at the nearest photographed voxel
      a'  = t r' (phi'/tau2 + (1 - phi')/(tau2 + s2_interp));  b' = t (1 - r') m_i / (tau2 + s2_syn(past))
      v'  = (a' real + b' synth) / (a' + b');  w' = (a' + b') tau2;  p' = t r' phi'
    real = native n01 / den where den > POLISH_DEN_MIN, else the record's real / its real mask; synth = the record's synth / its
    brain. Writes <sid>_{t,w2,v2,p2}[,_rgb2].npy (float16, Fortran order) under the round's inputs/."""
    S, sids, it, subj, rgb, od = job
    t0 = time.time()
    shp, A01 = grid(); sup = support()
    pj = json.load(open(P()["bf_precision_rule"])); f_s2 = F.s2_syn_fn(pj); t2, s2i = float(pj["tau2"]), float(pj["s2_interp"])
    m_i = 1.0 if subj[S]["kind"] == "mri" else 0.0
    nat = Native(S, rgb=rgb)
    if rgb:
        nat.fit_rgb(S, subj[S])
        chj = json.load(open(f"{subj[S]['root']}/{S}/{S}_chroma.json"))
        chroma = np.array(chj["chroma"], np.float32); sl_, ic_ = chj["grey_to_luma"]
    smp = record_source(S, subj[S], rgb)
    os.makedirs(od, exist_ok=True)
    rows = {}
    for sid in sids:
        t1 = time.time(); flip = sid.endswith("f")
        W = member_warp(sid)
        U = np.load(f"{fields_dir(it)}/{sid}_u.npy", mmap_mode="r") if it > 0 else None
        keys = [("t", np.float16), ("w2", np.float16), ("v2", np.float16), ("p2", np.float16)]
        mm = {k: np.lib.format.open_memmap(f"{od}/{sid}_{k}.tmp.npy", "w+", dt, shp, fortran_order=True) for k, dt in keys}
        if rgb:
            mm["rgb2"] = np.lib.format.open_memmap(f"{od}/{sid}_rgb2.tmp.npy", "w+", np.float16, shp + (3,), fortran_order=True)
        st = dict(t_out_of_record_brain=0.0, t=0.0, fallback_a=0.0, a=0.0)
        for k0 in range(0, shp[2], KZ):
            k1 = min(shp[2], k0 + KZ)
            s = sup[:, :, k0:k1]
            ii, jj, kk = np.nonzero(s)
            out = {k: np.zeros(s.shape, np.float32) for k, _ in keys}
            orgb = np.zeros(s.shape + (3,), np.float32) if rgb else None
            if ii.size:
                Pw = np.c_[ii, jj, kk + k0].astype(np.float64) @ A01[:3, :3].T + A01[:3, 3]
                if U is not None:
                    Us = np.asarray(U[:, :, k0:k1, :]).astype(np.float32)
                    Pw = Pw + Us[ii, jj, kk].astype(np.float64)
                Q = W.pull(Pw)
                if flip:
                    Q = Q @ F.MIRROR[:3, :3].T
                v = smp(Q)
                kap = np.clip(v["kappa"], 0, 1); bs = np.clip(v["brain"], 0, 1)
                past = np.maximum(-v["depth"], 0.0)
                rd = (v["near"] > 0) & (kap < 1)                                  # where the photograph's own mask is read
                num = np.zeros_like(bs); den = np.zeros_like(bs)
                if rd.any():
                    Qn = nat.f(Q[rd])
                    num[rd] = nat.X.sample(nat.n01, nat.A, Qn, order=1); den[rd] = nat.X.sample(nat.valid, nat.A, Qn, order=1)
                t = (1.0 - kap) * den + kap * bs
                r2 = (1.0 - kap) * np.clip(v["ramp_nr"], 0, 1); phi2 = np.clip(v["phi_nr"], 0, 1)
                a2 = t * r2 * (phi2 / t2 + (1.0 - phi2) / (t2 + s2i))
                b2 = t * (1.0 - r2) * m_i / (t2 + f_s2(past))
                okn = den > POLISH_DEN_MIN
                rec_real = np.where(v["rm"] > POLISH_DEN_MIN, v["real"] / np.maximum(v["rm"], 1e-6), 0.0)
                real = np.where(okn, num / np.maximum(den, 1e-6), rec_real)
                syn = np.where(bs > POLISH_DEN_MIN, v["synth"] / np.maximum(bs, 1e-6), 0.0)
                ws = a2 + b2
                val = np.where(ws > 0, (a2 * real + b2 * syn) / np.maximum(ws, 1e-12), 0.0)
                out["t"][ii, jj, kk] = t; out["w2"][ii, jj, kk] = ws * t2; out["v2"][ii, jj, kk] = val
                out["p2"][ii, jj, kk] = t * r2 * phi2
                if rgb:
                    rr = np.stack([np.where(bs > POLISH_DEN_MIN, v[f"rgb{c}"] / np.maximum(bs, 1e-6), 0.0) for c in range(3)], -1)
                    if okn.any():
                        rr[okn] = nat.rgb_at(nat.f(Q[okn]), den[okn])
                    sy = np.clip((sl_ * syn + ic_)[:, None] * chroma[None, :], 0, 1)
                    orgb[ii, jj, kk] = np.where(ws[:, None] > 0, (a2[:, None] * rr + b2[:, None] * sy) / np.maximum(ws, 1e-12)[:, None], 0.0)
                st["t"] += float(t.sum()); st["t_out_of_record_brain"] += float(t[v["brain"] <= 0.5].sum())
                st["a"] += float(a2.sum()); st["fallback_a"] += float(a2[~okn].sum())
            for k in out:
                mm[k][:, :, k0:k1] = out[k]
            if rgb:
                mm["rgb2"][:, :, k0:k1, :] = orgb
        for k in sorted(mm, key=lambda x: x == "t"):                     # t last: its presence marks the input as done
            mm[k].flush()
            os.replace(f"{od}/{sid}_{k}.tmp.npy", f"{od}/{sid}_{k}.npy")
        mm.clear()
        rows[sid] = dict(seconds=round(time.time() - t1, 1),
                         t_share_outside_record_brain=round(st["t_out_of_record_brain"] / max(st["t"], 1e-12), 5),
                         real_weight_fallback_share=round(st["fallback_a"] / max(st["a"], 1e-12), 5))
        print(f"  [{os.getpid()}] {sid} {rows[sid]}", flush=True)
    return S, dict(inputs=rows, native=nat.how, total_s=round(time.time() - t0, 1),
                   maxrss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2))


def carry(it, rgb, j, brains=None):
    """Every input of the round carried from the photographs (resumes: an input whose t exists is kept)."""
    subj = subjects(); ids = input_ids(brains); od = inputs_dir(it)
    bs = sorted({s.rstrip("f") for s in ids}, key=lambda S: -os.path.getsize(l2_path(S)))         # the largest first
    jobs = [(S, [i for i in ids if i.rstrip("f") == S], it, subj, rgb, str(od)) for S in bs]
    jobs = [jb for jb in jobs if not all(os.path.exists(f"{od}/{sid}_t.npy") for sid in jb[1])]
    t0 = time.time(); res = {}
    with ProcessPoolExecutor(j) as ex:
        for S, r in ex.map(member_job, jobs):
            res[S] = r
    p = itdir(it) / "carry.json"
    old = json.load(open(p)) if p.exists() else {}
    old.update(res); old["_wall_s"] = round(time.time() - t0, 1); old["_workers"] = j
    json.dump(old, open(p, "w"), indent=1)
    print(f"carry iteration {it}: {len(jobs)} brains in {time.time() - t0:.0f} s -> {od}", flush=True)


# ------------------------------------------------------------------------------------------------------------- the average
def symmetrize(v):
    s = 0.5 * (v + np.flip(v, 0))
    assert float(np.abs(s - np.flip(s, 0)).max()) == 0.0
    return s


def combine(it, rgb):
    """The average of the round's inputs: P = mean t (symmetrized), tissue = P >= 0.5, gray = sum w2 v2 / sum w2 (symmetrized)
    x tissue, RGB likewise (clipped to [0, 1]); photo_share = sum p2 / sum w2, real mask = tissue & photo_share >= 0.5."""
    t0 = time.time(); VOX = G["vox"]
    ids = input_ids(); shp, A = grid(); od = inputs_dir(it); N = len(ids)
    keys = ("v2", "w2", "p2", "t")
    mm = {s: {k: np.load(f"{od}/{s}_{k}.npy", mmap_mode="r") for k in keys} for s in ids}
    rgb = rgb and all(os.path.exists(f"{od}/{s}_rgb2.npy") for s in ids)
    if rgb:
        for s in ids:
            mm[s]["rgb"] = np.load(f"{od}/{s}_rgb2.npy", mmap_mode="r")
    acc = {k: np.zeros(shp, np.float32) for k in ("wv", "w", "p", "t")}
    accr = np.zeros(shp + (3,), np.float32) if rgb else None
    for k0 in range(0, shp[2], 32):
        sl = (slice(None), slice(None), slice(k0, min(shp[2], k0 + 32)))
        for s in ids:
            m = mm[s]
            w = m["w2"][sl].astype(np.float32); v = m["v2"][sl].astype(np.float32); p = m["p2"][sl].astype(np.float32)
            t = m["t"][sl].astype(np.float32)
            acc["wv"][sl] += w * v; acc["w"][sl] += w; acc["p"][sl] += p; acc["t"][sl] += t
            if rgb:
                accr[sl] += w[..., None] * m["rgb"][sl].astype(np.float32)
    ok = acc["w"] > 0; dd = np.where(ok, acc["w"], 1.0)
    Pt = symmetrize((acc["t"] / N).astype(np.float32))
    mask = Pt >= 0.5
    ps = symmetrize(np.where(ok, acc["p"] / dd, 0.0).astype(np.float32))
    real = mask & (ps >= 0.5)
    grey = symmetrize(np.where(ok, acc["wv"] / dd, 0.0).astype(np.float32)) * mask
    name = tpl_name(it); dst = itdir(it); os.makedirs(dst, exist_ok=True)
    for k, v in (("grey", grey), ("P", Pt), ("photo_share", ps)):
        nib.save(nib.Nifti1Image(v.astype(np.float32), A), f"{dst}/{name}_{k}.nii.gz")
    for k, m in (("brainmask", mask), ("real_mask", real)):
        assert np.array_equal(m, np.flip(m, 0))
        nib.save(nib.Nifti1Image(m.astype(np.uint8), A), f"{dst}/{name}_{k}.nii.gz")
        np.save(f"{dst}/{name}_{k}.npy", np.asfortranarray(m.astype(np.uint8)))
    np.save(f"{dst}/{name}_grey.npy", np.asfortranarray(grey.astype(np.float32)))
    json.dump(dict(shape=list(shp), affine=A.tolist()), open(f"{dst}/{name}_grid.json", "w"))
    met = dict(name=name, inputs=N, vox_mm=VOX, shape=list(shp),
               brain_cm3=float(mask.sum() * VOX ** 3 / 1000), real_cm3=float(real.sum() * VOX ** 3 / 1000),
               photo_share_below_half_cm3=float((mask & (ps < 0.5)).sum() * VOX ** 3 / 1000),
               photo_share_zero_cm3=float((mask & (ps <= 0)).sum() * VOX ** 3 / 1000),
               sum_w_in_mask=dict(min=float(acc["w"][mask].min()), p1=float(np.percentile(acc["w"][mask], 1)),
                                  median=float(np.median(acc["w"][mask]))))
    del acc
    if rgb:
        r = np.clip(symmetrize(np.where(ok[..., None], accr / dd[..., None], 0.0).astype(np.float32)), 0, 1) * mask[..., None]
        assert float(np.abs(r - np.flip(r, 0)).max()) == 0.0
        nib.save(nib.Nifti1Image(r.astype(np.float32), A), f"{dst}/{name}_rgb.nii.gz"); met["rgb"] = True
    met["wall_s"] = round(time.time() - t0, 1)
    json.dump(met, open(f"{dst}/{name}_combine.json", "w"), indent=1)
    print(json.dumps(met, indent=1), flush=True)


# ------------------------------------------------------------------------------------------------------------------ checks
def xshift(a, b, m, vox):
    """The shift (mm per axis) aligning b to a: FFT cross-correlation of the mean-removed masked volumes, parabolic peak."""
    a = np.where(m, a - a[m].mean(), 0.0); b = np.where(m, b - b[m].mean(), 0.0)
    cc = np.fft.fftshift(np.fft.irfftn(np.fft.rfftn(a) * np.conj(np.fft.rfftn(b)), a.shape, axes=(0, 1, 2)))
    c0 = np.array(np.unravel_index(np.argmax(cc), cc.shape)); out = []
    for k in range(3):
        e = np.eye(3, dtype=int)[k]; y0, y1, y2 = cc[tuple(c0 - e)], cc[tuple(c0)], cc[tuple(c0 + e)]
        den = y0 - 2 * y1 + y2
        out.append(float((c0[k] - a.shape[k] // 2 + (0.5 * (y0 - y2) / den if den != 0 else 0.0)) * vox))
    return out


def sharp(v, m, vox):
    g = np.gradient(v.astype(np.float32), vox)
    return float(np.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2)[m].mean())


def r_coarse(v25, R, bm, sig_mm):
    """r of v and R (both on the 0.25 mm box) after the same Gaussian low-pass, in R's brainmask."""
    from scipy import ndimage as ndi
    f = lambda z: ndi.gaussian_filter(np.asarray(z, np.float32), sig_mm / 0.25, truncate=4.0)     # noqa: E731
    return float(np.corrcoef(f(v25)[bm], f(R)[bm])[0, 1])


def check(it):
    """The round's template against step 8 at 0.25 mm (r unfiltered and after the low-pass, shift), its symmetry, and its
    sharpness in step 8's photographed tissue eroded 1 mm -> <round>/T<k><tag>_check.json."""
    t0 = time.time(); VOX = G["vox"]; name = tpl_name(it)
    T = np.load(f"{tpl(it)}_grey.npy").astype(np.float32)
    R = record_crop("d2bs8_grey.nii.gz").astype(np.float32); bm = record_crop("d2bs8_brainmask.nii.gz") > 0
    T25 = down25(T)
    r = float(np.corrcoef(T25[bm], R[bm])[0, 1]); sh = xshift(R, T25, bm, 0.25)
    rc = r_coarse(T25, R, bm, float(C9["checks"]["lowpass_sigma_mm"]))
    asym = float(np.abs(T - np.flip(T, 0)).max())
    s = {k: sharp(T, m, VOX) for k, m in regions().items()}
    res = dict(name=name, vox_mm=VOX, r_vs_record_025=round(r, 5), r_sigma1_vs_record=round(rc, 5),
               shift_mm=[round(x, 4) for x in sh], max_asym=asym, sharpness=s, wall_s=round(time.time() - t0, 1))
    json.dump(res, open(f"{tpl(it)}_check.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)
    return res


# ------------------------------------------------------------------------------------------------------------ registration
CAP = r'''
import atexit, json, os, runpy, sys, torch
torch.cuda.set_per_process_memory_fraction(float(os.environ["LINC_GPU_CAP_FRAC"]), 0)
def _rep():
    json.dump(dict(max_alloc_gib=torch.cuda.max_memory_allocated(0) / 2**30,
                   max_reserved_gib=torch.cuda.max_memory_reserved(0) / 2**30), open(os.environ["LINC_GPU_MEM_JSON"], "w"))
atexit.register(_rep)
sys.argv = [os.environ["LINC_RUNNER"]] + sys.argv[1:]
runpy.run_path(os.environ["LINC_RUNNER"], run_name="__main__")
'''
RUNNER = REPO / "code" / "blockface" / "registration" / "anatomix_runner.py"


def lin_axis(a, axis, pos):
    """1-D linear interpolation of a along axis at continuous indices pos (clamped to the ends: border values)."""
    n = a.shape[axis]; pos = np.clip(pos, 0, n - 1); i0 = np.minimum(np.floor(pos).astype(int), max(n - 2, 0)); f = (pos - i0).astype(np.float32)
    sh = [1] * a.ndim; sh[axis] = -1; f = f.reshape(sh)
    return np.take(a, i0, axis) * (1 - f) + np.take(a, np.minimum(i0 + 1, n - 1), axis) * f


def lin_resample(v, pos):
    """Trilinear resampling of v (X, Y, Z[, C]) on an axis-aligned grid: pos = 3 arrays of continuous indices (separable)."""
    out = v.astype(np.float32)
    for ax in range(3):
        out = lin_axis(out, ax, pos[ax])
    return out


def to_reg_grid(img, msk, pos):
    """Mask-normalized trilinear resampling onto the registration grid: value = lin(img x msk) / lin(msk) where lin(msk) >= 0.5."""
    m = lin_resample(msk.astype(np.float32), pos); v = lin_resample(img.astype(np.float32) * msk, pos)
    ok = m >= 0.5
    return np.where(ok, v / np.maximum(m, 1e-6), 0.0).astype(np.float32), ok


def folds_of(d, vox):
    """Voxels where x -> x + d(x) folds (det J <= 0, central differences), in z slabs."""
    n = 0
    for z0 in range(0, d.shape[2], 40):
        a0, a1 = max(0, z0 - 1), min(d.shape[2], z0 + 41)
        s = d[:, :, a0:a1].astype(np.float32)
        g = [np.gradient(s[..., c], vox) for c in range(3)]
        J = np.empty(s.shape[:3] + (3, 3), np.float32)
        for c in range(3):
            for ax in range(3):
                J[..., c, ax] = g[c][ax] + (1.0 if c == ax else 0.0)
        det = np.linalg.det(J)[:, :, z0 - a0:z0 - a0 + min(40, d.shape[2] - z0)]
        n += int((det <= 0).sum())
    return n


def gpu_free(gpu):
    """No compute process on the card."""
    q = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                       capture_output=True, text=True)
    if q.returncode != 0:
        return False, q.stderr.strip()
    return (not q.stdout.strip()), q.stdout.strip()


def template_arrays(template):
    """A template's gray, brain mask and real mask: the build's .npy copies, or its NIfTI files (the data folder's)."""
    if Path(f"{template}_grey.npy").exists():
        return (np.load(f"{template}_grey.npy", mmap_mode="r"), np.load(f"{template}_brainmask.npy", mmap_mode="r") > 0,
                np.load(f"{template}_real_mask.npy", mmap_mode="r") > 0)
    L = lambda k: np.asanyarray(nib.load(f"{template}_{k}.nii.gz").dataobj)                     # noqa: E731
    return L("grey").astype(np.float32), L("brainmask") > 0, L("real_mask") > 0


def register_one(sid, template, inputs, field, gpu, cap_gib, scratch, timeout_s=3 * 3600):
    """One input registered whole brain to a template (README of this module: register). Fixed = the template's gray (x its
    mask), cost mask its real mask; moving = the input's start volume v2 x (t >= 0.5), cost mask p2 >= 0.5. The registration grid
    is warp_mm on the box of (template mask | input tissue) plus margin_mm; images go onto it mask-normalized. The engine's
    displacement (ITK LPS, fixed -> moving pull) is written as a RAS field (mm, float16) on the template's whole grid, resampled
    once (trilinear, border values outside the registration grid)."""
    rc = C9["registration"]; t0 = time.time()
    A = np.array(json.load(open(f"{template}_grid.json"))["affine"]); VOX = float(A[0, 0])
    Gt, GB, GR = template_arrays(template)
    T = np.load(f"{inputs}/{sid}_t.npy", mmap_mode="r"); V = np.load(f"{inputs}/{sid}_v2.npy", mmap_mode="r")
    Pp = np.load(f"{inputs}/{sid}_p2.npy", mmap_mode="r")
    shp = Gt.shape
    tb = np.asarray(T).astype(np.float32) >= 0.5
    gb = np.asarray(GB); m = int(round(float(rc["margin_mm"]) / VOX))
    nz = [np.nonzero((gb | tb).any(axis=tuple(x for x in range(3) if x != ax)))[0] for ax in range(3)]
    lo = [max(0, int(z[0]) - m) for z in nz]; hi = [min(n, int(z[-1]) + 1 + m) for z, n in zip(nz, shp)]
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    Ac = A.copy(); Ac[:3, 3] = A[:3, :3] @ np.array(lo, float) + A[:3, 3]
    fx = np.asarray(Gt[sl]).astype(np.float32) * gb[sl]; fr = np.asarray(GR[sl]) & gb[sl]
    mv = np.asarray(V[sl]).astype(np.float32) * tb[sl]; mp = (np.asarray(Pp[sl]).astype(np.float32) >= 0.5) & tb[sl]
    assert float(np.abs(fx[~gb[sl]]).max(initial=0)) == 0.0 and float(np.abs(mv[~tb[sl]]).max(initial=0)) == 0.0
    rv = float(rc["warp_mm"])
    corner = Ac[:3, 3] - VOX / 2; size = np.array([b - a for a, b in zip(lo, hi)]) * VOX
    nr = [int(np.ceil(x / rv - 1e-9)) for x in size]
    Ar = np.diag([rv, rv, rv, 1.0]); Ar[:3, 3] = corner + rv / 2
    pos = [(Ar[d, 3] + rv * np.arange(nr[d]) - Ac[d, 3]) / VOX for d in range(3)]      # registration voxel -> crop index
    gbc = gb[sl]; tbc = tb[sl]
    fx, okf = to_reg_grid(fx, gbc, pos); fr = (lin_resample(fr.astype(np.float32), pos) >= 0.5) & okf
    mv, okm = to_reg_grid(mv, tbc, pos); mp = (lin_resample(mp.astype(np.float32), pos) >= 0.5) & okm
    fx *= okf; mv *= okm
    assert float(np.abs(fx[~okf]).max(initial=0)) == 0.0 and float(np.abs(mv[~okm]).max(initial=0)) == 0.0
    Areg, nreg = Ar, tuple(nr)
    del tb
    sc = Path(scratch) / f"{sid}_{os.getpid()}"; shutil.rmtree(sc, ignore_errors=True); sc.mkdir(parents=True)
    names = dict(fixed=f"{sc}/fixed.nii", moving=f"{sc}/{sid}.nii", fixed_mask=f"{sc}/fixedmask.nii", moving_mask=f"{sc}/{sid}_mask.nii")
    nib.save(nib.Nifti1Image(fx, Areg), names["fixed"]); nib.save(nib.Nifti1Image(mv, Areg), names["moving"])
    nib.save(nib.Nifti1Image(fr.astype(np.uint8), Areg), names["fixed_mask"]); nib.save(nib.Nifti1Image(mp.astype(np.uint8), Areg), names["moving_mask"])
    fclip = float(np.percentile(fx[fr], 99.0))
    with open(f"{sc}/pairs.csv", "w") as fh:
        fh.write("fixed,moving,fixed_mask,moving_mask\n" + ",".join(names[k] for k in ("fixed", "moving", "fixed_mask", "moving_mask")) + "\n")
    del fx, mv, fr, mp
    t_cut = time.time() - t0
    ok, who = gpu_free(gpu)
    if not ok:
        raise SystemExit(f"GPU {gpu} is busy: {who}")
    py = os.environ.get("ANATOMIX_PYTHON") or sys.exit("set ANATOMIX_PYTHON (settings/env/anatomix.yml)")
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               PYTHONNOUSERSITE="1", LINC_RUNNER=str(RUNNER), LINC_GPU_MEM_JSON=f"{sc}/gpu_mem.json")
    tot = float(subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True).stdout.strip()) / 1024
    env["LINC_GPU_CAP_FRAC"] = str(min(0.98, float(cap_gib) / tot))
    env["LINC_UNGATED_FEATURES"] = "1" if rc["ungated_features"] else ""
    env["LINC_FEATURE_RES_MM"] = str(rc["feature_res_mm"])
    outd = sc / "out"; outd.mkdir()
    cmd = [py, "-c", CAP, "--registration-pairs-csv", f"{sc}/pairs.csv", "--output-dir", str(outd), "--fixed-maxclip", repr(fclip)] + \
          [str(x) for x in rc["args"]]
    t1 = time.time()
    od = Path(field).parent; od.mkdir(parents=True, exist_ok=True)
    with open(f"{sc}/engine.log", "w") as log:
        rcode = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s).returncode
    t_reg = time.time() - t1
    shutil.copy(f"{sc}/engine.log", od / f"{sid}_engine.log")
    if rcode != 0:
        raise SystemExit(f"engine exit {rcode}; log {od / (sid + '_engine.log')}")
    stem = os.path.basename(names["moving"]).replace(".nii", "")
    wp = outd / f"reg-warp-{stem}.nii.gz"
    wp = wp if wp.exists() else outd / f"reg-warp-{stem}.nii"
    wimg = nib.load(wp)
    assert np.allclose(wimg.affine, Areg, atol=1e-4), "warp not on the registration grid"
    d = np.asanyarray(wimg.dataobj).astype(np.float32).reshape(nreg + (3,)) * np.array([-1, -1, 1], np.float32)
    nf = folds_of(d, rv)
    posb = [(A[dd, 3] + VOX * np.arange(shp[dd]) - Areg[dd, 3]) / rv for dd in range(3)]          # border values outside
    u = lin_resample(d, posb)
    tmp = str(field).replace(".npy", ".tmp.npy")
    np.save(tmp, np.asfortranarray(u.astype(np.float16))); os.replace(tmp, field)
    mem = json.load(open(f"{sc}/gpu_mem.json")) if os.path.exists(f"{sc}/gpu_mem.json") else {}
    done = dict(sid=sid, template=str(template), gpu_model=subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=name", "--format=csv,noheader"],
                                                                         capture_output=True, text=True).stdout.strip(),
                cap_gib=cap_gib, crop_lo=lo, crop_hi=hi, reg_vox=rv, reg_shape=list(nreg), fixed_maxclip=fclip,
                cut_s=round(t_cut, 1), register_s=round(t_reg, 1), total_s=round(time.time() - t0, 1), gpu_mem=mem,
                folds_engine_grid=nf, max_disp_mm=float(np.sqrt((d ** 2).sum(-1)).max()), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(done, open(str(field).replace(".npy", "_done.json"), "w"), indent=1)
    shutil.rmtree(sc, ignore_errors=True)
    print(json.dumps(done), flush=True)
    return done


def register_round(it, gpu, cap_gib, scratch):
    """Round it's registrations on one card: each input whose field is missing and that no other process has claimed (several
    processes, one per card, may share a round; a claim is a folder under fields_raw/claims)."""
    D = itdir(it); raw = D / "fields_raw"; claims = raw / "claims"; claims.mkdir(parents=True, exist_ok=True)
    tp = tpl(it - 1)
    for f in ("grey.npy", "brainmask.npy", "real_mask.npy", "grid.json"):
        if not Path(f"{tp}_{f}").exists():
            raise SystemExit(f"{tp}_{f} missing: run round {it - 1} first")
    for sid in input_ids():
        if (raw / f"{sid}.npy").exists():
            continue
        try:
            (claims / sid).mkdir()
        except FileExistsError:
            continue
        try:
            register_one(sid, tp, inputs_dir(0), raw / f"{sid}.npy", gpu, cap_gib, scratch)
        except BaseException:
            (claims / sid / "failed").touch()
            raise
    left = [s for s in input_ids() if not (raw / f"{s}.npy").exists()]
    print(f"round {it}: {len(input_ids()) - len(left)} of {len(input_ids())} fields" + (f"; waiting for {left}" if left else ""), flush=True)


def wait_round(it, poll_s=60, timeout_h=48):
    """Until every field of round it exists (other cards may still be working on it); a failed claim stops."""
    raw = itdir(it) / "fields_raw"; t0 = time.time()
    while True:
        failed = [s for s in input_ids() if (raw / "claims" / s / "failed").exists()]
        if failed:
            raise SystemExit(f"round {it}: registration failed for {failed} (logs in {raw}); remove their claims to retry")
        left = [s for s in input_ids() if not (raw / f"{s}.npy").exists()]
        if not left:
            return
        if time.time() - t0 > timeout_h * 3600:
            raise SystemExit(f"round {it}: still waiting for {left} after {timeout_h} h")
        time.sleep(poll_s)


# ------------------------------------------------------------------------------------------------------------------ unbias
def folds_field(U):
    """Folds of x -> x + u(x) over the whole grid, in z slabs (one voxel of overlap for the central differences)."""
    shp = U.shape[:3]; n, dmin = 0, np.inf; VOX = G["vox"]
    for k0 in range(0, shp[2], 32):
        k1 = min(shp[2], k0 + 32); a0, a1 = max(0, k0 - 1), min(shp[2], k1 + 1)
        d = np.asarray(U[:, :, a0:a1, :]).astype(np.float32)
        g = [np.gradient(d[..., c], VOX) for c in range(3)]
        J = np.empty(d.shape[:3] + (3, 3), np.float32)
        for c in range(3):
            for ax in range(3):
                J[..., c, ax] = g[c][ax] + (1.0 if c == ax else 0.0)
        det = np.linalg.det(J)[:, :, k0 - a0:k0 - a0 + (k1 - k0)]
        n += int((det <= 0).sum()); dmin = min(dmin, float(det.min()))
    return n, dmin


def fold_one(args):
    sid, p = args
    nf, dmin = folds_field(np.load(p, mmap_mode="r"))
    return sid, dict(folds=nf, min_det=round(dmin, 4))


def _torch_field(a):
    """(X, Y, Z, 3) mm -> torch (1, 3, X, Y, Z) in voxels."""
    import torch
    return torch.from_numpy(np.ascontiguousarray(np.moveaxis(np.asarray(a, np.float32), -1, 0)) / G["vox"])[None]


class Sampler:
    """Trilinear sampling of a field at x + disp (voxels), border values outside the grid (torch, CPU)."""
    def __init__(self, shp):
        import torch
        self.base = torch.stack(torch.meshgrid(*[torch.arange(n, dtype=torch.float32) for n in shp], indexing="ij"))[None]
        self.scale = torch.tensor([2.0 / (n - 1) for n in shp]).view(1, 3, 1, 1, 1)

    def __call__(self, f, disp):
        import torch.nn.functional as Fn
        g = ((self.base + disp) * self.scale - 1.0)[0].permute(1, 2, 3, 0).flip(-1)[None]   # grid order (W, H, D) = (Z, Y, X)
        return Fn.grid_sample(f, g, mode="bilinear", padding_mode="border", align_corners=True)


def invert_mean(ub_t, smp, tol, rounds):
    """w with y + w(y) = phi_bar^-1(y), damped fixed point; residual max |w + u_bar(y + w)| in mm."""
    w = -ub_t.clone(); hist = []
    for _ in range(rounds):
        g = -smp(ub_t, w); r = float((g - w).abs().max()) * G["vox"]; hist.append(round(r, 6))
        if r < tol:
            break
        w = 0.5 * w + 0.5 * g
    return w, hist


def compose_one(sid, it, w_t, smp):
    """u_i'(y) = w(y) + u_i(y + w(y)): the input's map composed with the inverse mean map (float16, Fortran order)."""
    u = _torch_field(np.load(f"{itdir(it)}/fields_raw/{sid}.npy", mmap_mode="r"))
    up = ((w_t + smp(u, w_t))[0] * G["vox"]).permute(1, 2, 3, 0).numpy(); del u
    os.makedirs(fields_dir(it), exist_ok=True)
    out = np.lib.format.open_memmap(f"{fields_dir(it)}/{sid}_u.tmp.npy", "w+", np.float16, up.shape, fortran_order=True)
    out[...] = up; out.flush(); del out, up
    os.replace(f"{fields_dir(it)}/{sid}_u.tmp.npy", f"{fields_dir(it)}/{sid}_u.npy")


def unbias(it, j):
    """A raw field with any det J <= 0 is left out (the input unmoved in this round; more than max_folding_members stops);
    then the mean field is inverted (damped fixed point) and composed with every field; no composed field may fold."""
    cu = C9["unbias"]; t0 = time.time(); ids = input_ids(); shp, _ = grid(); D = itdir(it)
    missing = [s for s in ids if not (D / "fields_raw" / f"{s}.npy").exists()]
    if missing:
        raise SystemExit(f"round {it}: fields missing for {missing}: run `register --it {it}` first")
    with ProcessPoolExecutor(j) as ex:
        raw = dict(ex.map(fold_one, [(s, f"{D}/fields_raw/{s}.npy") for s in ids]))
    rej = [s for s in ids if raw[s]["folds"] > 0]
    res = dict(it=it, unbias="composition with the inverse mean map", raw_folds=raw, rejected=rej)
    out = D / "unbias.json"
    if len(rej) > int(cu["max_folding_members"]):
        res["stop"] = f"{len(rej)} of {len(ids)} raw fields fold (> {cu['max_folding_members']})"
        json.dump(res, open(out, "w"), indent=1); raise SystemExit("STOP: " + res["stop"])
    for s in rej:                                                    # left out: its folding field kept aside, a zero field used
        os.replace(f"{D}/fields_raw/{s}.npy", f"{D}/fields_raw/{s}_folded.npy")
        np.save(f"{D}/fields_raw/{s}.npy", np.asfortranarray(np.zeros(shp + (3,), np.float16)))
    ub = np.zeros(shp + (3,), np.float32)
    for s in ids:
        u = np.load(f"{D}/fields_raw/{s}.npy", mmap_mode="r")
        for k0 in range(0, shp[2], 32):
            sl = (slice(None), slice(None), slice(k0, min(shp[2], k0 + 32)))
            ub[sl] += np.asarray(u[sl]).astype(np.float32)
    ub /= len(ids)
    bm = np.load(f"{tpl(it - 1)}_brainmask.npy", mmap_mode="r") > 0
    mag = np.sqrt((ub ** 2).sum(-1))[bm]
    res["mean_field_mm"] = dict(mean=float(mag.mean()), p99=float(np.percentile(mag, 99)), max=float(mag.max()))
    import torch
    torch.set_num_threads(16)
    smp = Sampler(shp); ub_t = _torch_field(ub); del ub
    w_t, hist = invert_mean(ub_t, smp, float(cu["inverse_tol_mm"]), int(cu["inverse_rounds"])); del ub_t
    res["inverse"] = dict(rounds=len(hist), residual_mm=hist[-1], residual_per_round=hist)
    if hist[-1] > float(cu["stop_residual_mm"]):
        res["stop"] = f"inverse residual {hist[-1]} mm > {cu['stop_residual_mm']} mm"
        json.dump(res, open(out, "w"), indent=1); raise SystemExit("STOP: " + res["stop"])
    for s in ids:
        compose_one(s, it, w_t, smp)
    del w_t, smp
    with ProcessPoolExecutor(j) as ex:
        fo = dict(ex.map(fold_one, [(s, f"{fields_dir(it)}/{s}_u.npy") for s in ids]))
    res.update(folds=fo, folds_total=int(sum(v["folds"] for v in fo.values())), wall_s=round(time.time() - t0, 1))
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("raw_folds", "folds")}, indent=1), flush=True)
    if res["folds_total"] > 0:
        raise SystemExit(f"STOP: {res['folds_total']} fold voxels after the unbias")


# --------------------------------------------------------------------------------------------------------------- the final
def resample_field(args):
    """A field on the rounds' grid (mm) -> the published grid: trilinear at every center, border values (one resampling)."""
    src, src_grid, dst = args
    t0 = time.time()
    A125 = np.array(json.load(open(src_grid))["affine"])
    pos = [(centers(a) - A125[a, 3]) / A125[a, a] for a in range(3)]
    u = lin_resample(np.load(src, mmap_mode="r"), pos)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    out = np.lib.format.open_memmap(dst[:-4] + ".tmp.npy", "w+", np.float16, u.shape, fortran_order=True)
    out[...] = u; out.flush(); del out
    os.replace(dst[:-4] + ".tmp.npy", dst)
    nf, dmin = folds_field(np.load(dst, mmap_mode="r"))
    return os.path.basename(dst), dict(folds=nf, min_det=round(dmin, 4), wall_s=round(time.time() - t0, 1))


def final_fields(rounds_root, last, j, sids=None, src_dir=None):
    """The last round's fields (rounds_root's iteration_<last>/fields, or src_dir) on the published grid (the current grid):
    resampled once from the rounds' grid, or copied when src_dir already holds them on the published grid (the data folder's
    copies, template/intermediate/construction/step9_fields/<input>_step9_u.npy)."""
    src_it = Path(rounds_root) / "nln_step_09" / f"iteration_{last:02d}"
    sd = Path(src_dir) if src_dir else src_it / "fields"
    ids = [s for s in (sids or input_ids()) if not (fields_dir(last) / f"{s}_u.npy").exists()]
    src = {s: next((sd / n for n in (f"{s}_u.npy", f"{s}_step9_u.npy") if (sd / n).exists()), sd / f"{s}_u.npy") for s in ids}
    shp, _ = grid(); res = {}
    if ids and np.load(src[ids[0]], mmap_mode="r").shape[:3] == shp:
        os.makedirs(fields_dir(last), exist_ok=True)
        for s in ids:
            shutil.copyfile(src[s], fields_dir(last) / f"{s}_u.npy")
            nf, dmin = folds_field(np.load(fields_dir(last) / f"{s}_u.npy", mmap_mode="r"))
            res[f"{s}_u.npy"] = dict(copied=str(src[s]), folds=nf, min_det=round(dmin, 4))
        jobs = []
    else:
        jobs = [(str(src[s]), str(src_it / f"{tpl_name(last)}_grid.json"), str(fields_dir(last) / f"{s}_u.npy")) for s in ids]
    with ProcessPoolExecutor(j) as ex:
        res.update(dict(ex.map(resample_field, jobs)))
    json.dump(dict(rule="trilinear (separable), border values, from the rounds' grid", fields=res,
                   folds_total=int(sum(v.get("folds", 0) for v in res.values()))), open(itdir(last) / "fields_resampled.json", "w"), indent=1)
    print(f"{len(jobs)} fields resampled onto the published grid, {len(ids) - len(jobs)} copied", flush=True)


def checks(last):
    """The step's checks on the published template (settings step9.checks): sharpness against the start, r after the low-pass and
    shift against step 8, exact symmetry of gray, RGB and masks, and the rounds' folds (from each round's unbias.json)."""
    cc = C9["checks"]
    c3, c0 = (json.load(open(f"{tpl(k)}_check.json")) for k in (last, 0))
    s3, s0 = c3["sharpness"]["whole"], c0["sharpness"]["whole"]
    g = {"1_sharpness": dict(final=s3, start=s0, ratio=round(s3 / s0, 4),
                             cubes={k: round(c3["sharpness"][k] / c0["sharpness"][k], 4) for k in CROPS}, passed=s3 >= float(cc["sharpness_ratio_min"]) * s0),
         "2_record": dict(r_lowpass=c3["r_sigma1_vs_record"], shift_mm=c3["shift_mm"], r_unfiltered=c3["r_vs_record_025"],
                          passed=c3["r_sigma1_vs_record"] >= float(cc["r_lowpass_min"]) and max(abs(x) for x in c3["shift_mm"]) < float(cc["shift_max_mm"]))}
    rounds = G["rounds_root"] / "nln_step_09"
    ups = {k: rounds / f"iteration_{k:02d}" / "unbias.json" for k in range(1, last + 1)}
    used = json.load(open(itdir(last) / "fields_resampled.json"))
    fu = int(sum(v.get("folds", 0) for v in used["fields"].values() if "folds" in v))
    if all(p.exists() for p in ups.values()):                    # the rounds ran here: their fields after the unbias
        ub = {k: json.load(open(p)) for k, p in ups.items()}
        g["3_folds"] = dict(per_round={k: dict(folds_total=v["folds_total"], rejected=v["rejected"], residual_mm=v["inverse"]["residual_mm"])
                                       for k, v in ub.items()}, published_grid_folds=fu, passed=all(v["folds_total"] == 0 for v in ub.values()))
    else:                                                        # the fields came from elsewhere: those that were used, folds counted
        n = sum(1 for v in used["fields"].values() if "folds" in v)
        g["3_folds"] = dict(note=f"the rounds did not run here; the fields' folds on this grid ({n} of {len(used['fields'])} counted)",
                            published_grid_folds=fu, passed=fu == 0)
    sym = {}
    for k in ("grey", "brainmask", "real_mask", "rgb"):
        v = np.asanyarray(nib.load(f"{tpl(last)}_{k}.nii.gz").dataobj)
        sym[k] = float(np.abs(v.astype(np.float32) - np.flip(v, 0).astype(np.float32)).max())
    g["4_symmetry"] = dict(max_asym=sym, passed=all(x == 0.0 for x in sym.values()))
    cmb = json.load(open(f"{tpl(last)}_combine.json"))
    res = dict(vox_mm=G["vox"], checks=g, passed=all(v["passed"] for v in g.values()),
               photographed=dict(brain_cm3=cmb["brain_cm3"], photo_share_below_half_cm3=cmb["photo_share_below_half_cm3"],
                                 photo_share_zero_cm3=cmb["photo_share_zero_cm3"]))
    json.dump(res, open(G["root"] / "checks.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)
    if not res["passed"]:
        raise SystemExit("a check failed: " + ", ".join(k for k, v in g.items() if not v["passed"]))


# ----------------------------------------------------------------------------------------------------------------- in-mri
def in_mri(template_prefix, maps, out):
    """The template in the MRI template's frame through the exported pull map <maps>/bftemplate__to__mri-template, one resample
    each (code/template/link/register.py blockface_in_mri's rule: gray, RGB cubic, clipped to [0, 1]; brain mask linear > 0.5;
    photographed fraction linear; synthesized = brain mask & fraction < 0.5), then gray and RGB multiplied by the brain mask."""
    from template.link.common import inputs as link_inputs, product
    t0 = time.time(); os.makedirs(out, exist_ok=True)
    m = X.load_map(f"{maps}/bftemplate__to__mri-template", "full")
    g = nib.load(product(link_inputs()["mri_b0"])); shp, A = g.shape[:3], g.affine

    def pull(path, order):
        i = nib.load(path)
        return X.apply_once(np.asanyarray(i.dataobj).astype(np.float32), i.affine, shp, A, m, order=order)
    T = str(template_prefix)
    mask = pull(f"{T}_brainmask.nii.gz", 1) > 0.5
    frac = pull(f"{T}_photo_share.nii.gz", 1)
    synth = mask & (frac < 0.5)
    grey = np.clip(pull(f"{T}_grey.nii.gz", 3), 0, 1)
    rgb = np.clip(pull(f"{T}_rgb.nii.gz", 3), 0, 1)
    grey = grey * mask; rgb = rgb * mask[..., None]
    for name, v, dt in (("grey", grey, np.float32), ("rgb", rgb, np.float32), ("brainmask", mask, np.uint8),
                        ("photographed_fraction", frac, np.float32), ("synthesized_mask", synth, np.uint8)):
        nib.save(nib.Nifti1Image(np.ascontiguousarray(v.astype(dt)), A), f"{out}/linc_bf_{name}.nii.gz")
    v = 0.25 ** 3 / 1000
    rec = dict(brainmask_voxels=int(mask.sum()), brain_cm3=float(mask.sum() * v), synthesized_voxels=int(synth.sum()),
               synthesized_cm3=float(synth.sum() * v), synthesized_share_of_brain=float(synth.sum() / mask.sum()),
               wall_s=round(time.time() - t0, 1))
    json.dump(rec, open(f"{out}/blockface_in_mri.json", "w"), indent=1)
    print(json.dumps(rec, indent=1), flush=True)


# ------------------------------------------------------------------------------------------------- the new-brain route's template
def on_step8_grid(template_prefix, out):
    """The template averaged exactly onto step 8's 0.25 mm box and zero-padded onto the run-113 grid's 2x refinement, as step 8's
    d2bs8_grey is: the grid `linc-register blockface-template` reads (it coarsens it 2x to 0.5 mm). The coverage is the template's
    brain mask, averaged the same way. Returns the two files."""
    use_grid("mirror", Path(out))
    T = str(template_prefix); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    gi = nib.load(f"{T}_grey.nii.gz"); shp, A = grid()
    assert tuple(gi.shape[:3]) == tuple(shp) and np.allclose(gi.affine, A, atol=1e-6), f"{T}: not on the published grid"
    (bs, bA), (fs, fA) = box(), F.run113_grid(2)
    o = np.rint((bA[:3, 3] - fA[:3, 3]) / np.diag(fA)[:3]).astype(int)
    files = []
    for name, v in (("grey", np.asanyarray(gi.dataobj).astype(np.float32)),
                    ("coverage", (np.asanyarray(nib.load(f"{T}_brainmask.nii.gz").dataobj) > 0).astype(np.float32))):
        full = np.zeros(tuple(fs), np.float32)
        full[o[0]:o[0] + bs[0], o[1]:o[1] + bs[1], o[2]:o[2] + bs[2]] = down25(v)
        f = out / f"blockface_template_step8grid_{name}.nii.gz"
        nib.save(nib.Nifti1Image(full, fA), str(f)); files.append(f)
    json.dump(dict(source=T, rule="exact area average onto step 8's 0.25 mm box, zero-padded onto the run-113 2x grid (no resampling)",
                   offset_vox=o.tolist()), open(out / "blockface_template_step8grid.json", "w"), indent=1)
    return files


def route_template(out):
    """The default template of `linc-register blockface-template`: the published template (products_blockface.yaml
    bftemplate_step9) on step 8's grid, made once into `out` (remade when the template's files change)."""
    T = str(P()["bftemplate_step9"] / tpl_name(int(C9["iterations"]))); out = Path(out)
    stamp = {k: os.path.getsize(f"{T}_{k}.nii.gz") for k in ("grey", "brainmask")}
    rec = out / "blockface_template_step8grid.json"
    files = [out / f"blockface_template_step8grid_{k}.nii.gz" for k in ("grey", "coverage")]
    if not (rec.exists() and all(f.exists() for f in files) and json.load(open(rec)).get("sizes") == stamp and json.load(open(rec))["source"] == T):
        on_step8_grid(T, out)
        r = json.load(open(rec)); r["sizes"] = stamp; json.dump(r, open(rec, "w"), indent=1)
    return [str(f) for f in files]


# ------------------------------------------------------------------------------------------------------- refine (optional)
def new_brain_window(S):
    """The photographs' gray window of a brain outside the paper (blo, bhi): its prepared folder's seam.json when its synthesis ran
    (`linc-complete prepare --out $LINC_WORK/prepared`), else the same rule on its own (code/blockface/synthesis/prepare.py BF_PCT
    percentiles of the iso gray over its photographed tissue)."""
    from blockface.synthesis.prepare import BF_PCT
    for p in (paths.work_root() / "prepared" / S / "seam.json",):
        if p.exists():
            sj = json.load(open(p)); return float(sj["blo"]), float(sj["bhi"]), dict(window=str(p))
    from scipy import ndimage as ndi
    g = np.asanyarray(nib.load(str(paths.subject_dir(S) / "blockface" / f"{S}_blockface_iso_grey.nii.gz")).dataobj).astype(np.float32)
    valid = ndi.binary_fill_holes(g > 0)                         # prepare.py's photographed tissue: filled, its largest component
    lab, n = ndi.label(valid)
    if n > 1:
        valid = lab == (1 + int(np.argmax(ndi.sum(np.ones_like(lab), lab, index=range(1, n + 1)))))
    blo, bhi = (float(np.percentile(g[valid], q)) for q in BF_PCT)
    return blo, bhi, dict(window=f"percentiles {list(BF_PCT)} of the iso gray over the photographed tissue")


def new_brain_job(S, entry, od):
    """A brain outside the paper, carried onto the published grid from its photographs only, through its registration to the
    blockface template (template point -> entry -> its native photographs): t = the photograph's own mask (sampled linearly),
    v2 = its normalized gray / t, p2 = t (all of it photographed). Writes <S>_{t,v2,p2}.npy, the start of its refinement."""
    t0 = time.time(); shp, A = grid(); sup = support()
    blo, bhi, how = new_brain_window(S)
    nat = Native(S, nmap=((lambda Q: Q), blo, bhi, dict(kind="photographs only, through " + str(entry.dir), **how)),
                 l2=paths.subject_dir(S) / "blockface" / f"{S}_blockface_L2.nii.gz")
    os.makedirs(od, exist_ok=True)
    mm = {k: np.lib.format.open_memmap(f"{od}/{S}_{k}.tmp.npy", "w+", np.float16, shp, fortran_order=True) for k in ("v2", "p2", "t")}
    for k0 in range(0, shp[2], KZ):
        k1 = min(shp[2], k0 + KZ); s = sup[:, :, k0:k1]
        ii, jj, kk = np.nonzero(s)
        out = {k: np.zeros(s.shape, np.float32) for k in mm}
        if ii.size:
            Q = entry.pull(np.c_[ii, jj, kk + k0].astype(np.float64) @ A[:3, :3].T + A[:3, 3])
            num = X.sample(nat.n01, nat.A, Q, order=1); den = X.sample(nat.valid, nat.A, Q, order=1)
            out["t"][ii, jj, kk] = den; out["p2"][ii, jj, kk] = den
            out["v2"][ii, jj, kk] = np.where(den > POLISH_DEN_MIN, num / np.maximum(den, 1e-6), 0.0)
        for k in mm:
            mm[k][:, :, k0:k1] = out[k]
    for k in ("v2", "p2", "t"):                                       # t last: its presence marks the start as done
        mm[k].flush(); os.replace(f"{od}/{S}_{k}.tmp.npy", f"{od}/{S}_{k}.npy")
    return dict(native=nat.how, total_s=round(time.time() - t0, 1))


def refine(S, W, gpu, cap_gib, scratch, j, registry=None):
    """Optional: one brain registered whole brain to the finished template with a member's job, as itself (unmirrored).
    A brain of the paper starts from step 8's member warp (as a member does); any other brain from its own registration to the
    blockface template (`linc-register blockface-template`, the entry bf-native__to__bf-template), photographs only. Writes
    final/refine/<S>.npy (the field: RAS mm on the template's grid, y -> y + u(y)) and, for a brain outside the paper, the refined
    entry bf-native__to__bf-template (the field, then the brain's entry) in $LINC_DATA/sub-<ID>/xfm_step9/ (or --registry)."""
    if S is None:
        raise SystemExit("refine: give --subject")
    use_grid("mirror", W / "final"); last = int(C9["iterations"])
    T = tpl(last) if Path(f"{tpl(last)}_grid.json").exists() else P()["bftemplate_step9"] / tpl_name(last)   # built here, or published
    for f in ("grey.nii.gz", "brainmask.nii.gz", "real_mask.nii.gz", "grid.json"):
        if not Path(f"{T}_{f}").exists():
            raise SystemExit(f"{T}_{f} missing: run `final`, or give the data folder's template (products_blockface.yaml bftemplate_step9)")
    A = np.array(json.load(open(f"{T}_grid.json"))["affine"]); shp, Ag = grid()
    assert np.array_equal(A, Ag), f"{T}: not on the published grid"
    rd = G["root"] / "refine"; od = rd / "inputs"
    paper = S in subjects()
    if not (od / f"{S}_t.npy").exists():
        if paper:
            _, r = member_job((S, [S], 0, subjects(), False, str(od)))
        else:
            e = xfm.Entry(str(xfm.xfm_dir("bf-native", "bf-template", S)))
            r = new_brain_job(S, e, str(od))
        json.dump(r, open(od / f"{S}_start.json", "w"), indent=1)
    field = rd / f"{S}.npy"
    register_one(S, T, od, field, gpu, cap_gib, scratch)
    if not paper:
        u = np.load(field).astype(np.float32); _, A = grid()
        un = rd / f"{S}_u.nii.gz"; nib.save(nib.Nifti1Image(u, A), str(un))
        root = Path(registry) if registry else paths.subject_dir(S) / "xfm_step9"      # beside the brain's own xfm/, which it keeps
        paths.refuse_records(root)
        made = refined_entry(S, un, f"{T}_grey.nii.gz", root, rd)
        print(f"refined entry: {made.dir}", flush=True)


def refined_entry(S, u_nii, dst_grid, root, work):
    """The refined transform entry bf-native -> bf-template under `root`: the field (ras_disp), then the components of the brain's
    own entry ($LINC_DATA/sub-<ID>/xfm). The own entry's files are copied aside first, so `root` may be the brain's own registry:
    the own entry is then replaced (code/shared/xfm.py backs it up under $LINC_WORK/xfm_backup)."""
    e = xfm.Entry(str(xfm.xfm_dir("bf-native", "bf-template", S)))
    keep = Path(work) / "own_entry"; shutil.rmtree(keep, ignore_errors=True); keep.mkdir(parents=True)
    comps = [("warp", "ras_disp", Path(u_nii))]
    for c in e.meta["components"]:
        f = keep / Path(c["file"]).name; shutil.copy2(e._resolve(c["file"]), f); comps.append((c["kind"], c["convention"], f))
    sg = e.meta.get("src_grid"); inside = bool(sg) and not Path(sg).is_absolute()     # the grid an FSL warp needs
    if inside:
        shutil.copy2(e._resolve(sg), keep / Path(sg).name); sg = keep / Path(sg).name
    made = xfm.register("bf-native", "bf-template", comps, subject=S, dst_grid=dst_grid, src_grid=sg, root=root, overwrite=True,
                        engine="step 9's whole-brain registration (code/template/blockface_step9.py refine) after the brain's entry",
                        extra=dict(refines=str(e.dir), refines_engine=e.meta.get("engine")))
    if inside:                          # kept inside the refined entry, as in the brain's own, so the work folder can go
        shutil.copy2(sg, made.dir / sg.name)
        m = json.loads((made.dir / "meta.json").read_text()); m["src_grid"] = sg.name
        (made.dir / "meta.json").write_text(json.dumps(m, indent=1))
        made = xfm.Entry(str(made.dir))
    return made


# ------------------------------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["start", "round", "register", "unbias", "carry", "final", "in-mri", "refine"])
    ap.add_argument("--work", default=None, help="the build's folder (default $LINC_WORK/build_blockface_template/step9)")
    ap.add_argument("--it", type=int, default=None, help="the round (register, unbias, carry)")
    ap.add_argument("-j", type=int, default=8, help="CPU workers (each holds one brain's photographs: about 7 GB, up to 15 GB)")
    ap.add_argument("--gpu", type=int, default=0, help="register, refine: the card's index (nvidia-smi order)")
    ap.add_argument("--cap-gib", type=float, default=float(C9["registration"]["cap_gib"]))
    ap.add_argument("--scratch", default=None, help="register, refine: the engine's scratch folder (default <work>/scratch)")
    ap.add_argument("--subject", default=None, help="refine: the brain")
    ap.add_argument("--registry", default=None, help="refine, a brain outside the paper: write the refined entry here (default $LINC_DATA/sub-<ID>/xfm_step9; "
                                                     "the brain's own xfm/ also works: its entry is then replaced, with a backup)")
    ap.add_argument("--fields-from", default=None, help="final: the last round's fields from this folder instead of the build's "
                                                         "(on either grid; e.g. the data folder's step9_fields)")
    a = ap.parse_args(argv)
    W = Path(a.work or (paths.work_root() / "build_blockface_template" / "step9")).resolve()
    paths.refuse_records(W)                       # never write through a link into the records
    use_grid("box", W); G["rounds_root"] = W
    last = int(C9["iterations"]); scratch = a.scratch or str(W / "scratch")
    if a.cmd == "start":
        carry(0, False, a.j); combine(0, False); check(0)
    elif a.cmd == "round":                                       # register (this card) + wait for the others + unbias + carry
        register_round(a.it, a.gpu, a.cap_gib, scratch); wait_round(a.it)
        unbias(a.it, min(a.j, 6))
        rgb = a.it == last
        carry(a.it, rgb, a.j); combine(a.it, rgb); check(a.it)
    elif a.cmd == "register":
        register_round(a.it, a.gpu, a.cap_gib, scratch)
    elif a.cmd == "unbias":
        unbias(a.it, min(a.j, 6))
    elif a.cmd == "carry":
        rgb = a.it == last
        carry(a.it, rgb, a.j); combine(a.it, rgb); check(a.it)
    elif a.cmd == "final":
        use_grid("mirror", W / "final"); G["rounds_root"] = W
        final_fields(W, last, min(a.j, 6), src_dir=a.fields_from)
        carry(0, False, a.j); combine(0, False); check(0)
        carry(last, True, a.j); combine(last, True); check(last)
        checks(last)
    elif a.cmd == "in-mri":
        use_grid("mirror", W / "final")
        in_mri(tpl(last), P()["template_link"], W / "in_mri")
    elif a.cmd == "refine":
        refine(a.subject, W, a.gpu, a.cap_gib, scratch, a.j, a.registry)


if __name__ == "__main__":
    main()
