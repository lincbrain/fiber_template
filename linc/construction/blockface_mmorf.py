"""The MMORF settings of the blockface template's construction: one .ini file per nonlinear step (MMORF, FSL's MultiMOdal
Registration Framework, makes the template's groupwise nonlinear steps).

You need
  configs/blockface_template.yaml (mmorf.ladders: per level, the knot spacing's scaling, the bending-energy weight and the Gaussian
  smoothing); the caller (blockface_build.py) sets the ladder (`set_ladder`) and the channel table (MODALITIES, NATIVE_MM) first.
What it does
  step_ini writes one step's .ini from the ladder, with each channel's smoothing floored at its own measured resolution
  (`fwhm_floor_mm`). The record's two ladders: 7 steps / 8 levels for the first stage (knots 16, 16, 16, 8, 4, 2, 1, 0.5 mm) and 8
  steps / 9 levels for the eighth step (down to 0.25 mm).
You get
  the .ini files, written into the run folders.

Ported unchanged from the analysis project: mri/code/build_template_run.py (`step_ini`, `fwhm_floor_mm` and the constants they
read; the MRI build, linc/construction/mmorf_run.py, writes its runs without the smoothing floor) and bftemplate/build_bf_run.py
(`LADDERS`, `set_ladder`, `link_or_copy`).
"""
import errno
import os
import shutil
from pathlib import Path

import yaml

# the ladders come from configs/blockface_template.yaml (mmorf.ladders, mmorf.warp_res_init_mm): the build reads its parameters from the
# configuration; set_ladder(n) selects one before the configs are written
_CFG = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs" / "blockface_template.yaml").read_text())["mmorf"]
WARP_RES_INIT = _CFG["warp_res_init_mm"]
SCALING, LAMBDA, FWHM = (list(_CFG["ladders"][max(_CFG["ladders"])][k]) for k in ("warp_scaling", "lambda_reg", "fwhm_mm"))

NATIVE_MM = {}                    # per-modality native resolution (mm), set by the caller
MODALITIES = {}                   # per-modality {brain, mask, quant}, set by the caller


def fwhm_floor_mm(native_mm):
    """Smallest smoothing a modality may be given: the Gaussian whose MTF is 0.5 at that
    modality's own effective bandwidth. f_eff was measured from radially-averaged power spectra
    as the 2x-noise-floor crossing: 1.485 cyc/mm at 0.25 mm, 0.785 at 0.50, 0.575 at 0.70 --
    i.e. 74-81 % of each tier's own Nyquist, so header voxel size is a faithful proxy.
    FWHM = 0.4413 / f_eff gives 0.297 mm (0.25), 0.562 mm (0.50), 0.767 mm (0.70).

    Without this floor every modality is driven to 0.05 mm FWHM at the finest levels, which asks
    0.5 mm data to determine 1.5 cyc/mm warp modes -- fitting interpolation, not anatomy.
    """
    f_eff = {0.25: 1.485, 0.50: 0.785, 0.70: 0.575}[round(native_mm, 2)]
    return 0.4413 / f_eff


N_STEPS  = 8
HIRES, IT_LOWRES, IT_HIRES = "1.9", 15, 20
BIAS_RES_INIT, LAMBDA_BIAS = 8, "1e8"


def step_ini(step, lambda_scale=1.0, mods=(("T2", 1.0),), tmods=(), fwhm_floor=False, tensor_cost_mask=True):
    """One MMORF config. Level count is step+1, exactly as run 113 and upstream.

    `lambda_scale` multiplies every `lambda_reg` term EXCEPT the first. The first (4.0e5) is the
    near-rigid coarsest level and is left alone; scaling the rest is a single-factor stiffening of
    the whole deformable schedule, which is what makes an A/B against the baseline interpretable.

    `mods` is [(name, lambda_scalar weight), ...]. Per-modality blocks are emitted in order; the
    order must match the order of `--input_scalar` on the driver command line.
    """
    n = step + 1
    lam = [LAMBDA[0]] + [f"{float(x) * lambda_scale:.4g}" for x in LAMBDA[1:]]
    L = lambda v: " ".join(v[:n])

    def F(name):
        """FWHM ladder for one modality, floored at its own resolution when enabled."""
        if not fwhm_floor:
            return L(FWHM)
        fl = fwhm_floor_mm(NATIVE_MM[name])
        return " ".join(f"{max(float(x), fl):g}" for x in FWHM[:n])
    head = (f"warp_res_init           = {WARP_RES_INIT}\n"
            f"warp_scaling            = {L(SCALING)}\n"
            f"lambda_reg              = {L(lam)}\n"
            f"hires                   = {HIRES}\n"
            f"optimiser_max_it_lowres = {IT_LOWRES}\n"
            f"optimiser_max_it_hires  = {IT_HIRES}\n"
            # masks stay OFF: these are ex-vivo, brain-extracted and intensity-normalised, so the
            # background is exactly zero and the zero-to-tissue edge is the strongest, least noisy
            # feature in the volume. Masking the cost function deletes exactly that signal and the
            # brain edge registers worse. REBUILD_PLAN.md 1.1 -- do not "fix" this.
            )

    # One block per modality for every vector<vector> option.
    body = ""
    for name, w in mods:
        quant = MODALITIES[name]["quant"]
        body += (f"\n; --- scalar modality {name}"
                 f"{' (QUANTITATIVE)' if quant else ''}, lambda_scalar {w} ---\n"
                 f"use_mask_ref_scalar     = {' '.join(['0'] * n)}\n"
                 f"use_mask_mov_scalar     = {' '.join(['0'] * n)}\n"
                 f"fwhm_ref_scalar         = {F(name)}\n"
                 f"fwhm_mov_scalar         = {F(name)}\n"
                 f"lambda_scalar           = {' '.join([f'{w:g}'] * n)}\n"
                 f"lambda_bias_reg         = {' '.join([LAMBDA_BIAS] * n)}\n")

    # Flat per-modality vectors. Bias estimation is OFF for quantitative modalities: a
    # multiplicative bias field would rescale frac1 and frac2 independently and break sum-to-1.
    # Tensor blocks, same [modality][level] shape as the scalar ones. Tensor masks are USED
    # (unlike the scalar ones): the tensor cost is dominated by the strong CSF/background
    # contrast at the brain edge, which is not anatomy.
    for name, w in tmods:
        body += (f"\n; --- tensor modality {name}, lambda_tensor {w} ---\n"
                 f"use_mask_ref_tensor     = {' '.join([str(int(tensor_cost_mask))] * n)}\n"
                 f"use_mask_mov_tensor     = {' '.join([str(int(tensor_cost_mask))] * n)}\n"
                 f"fwhm_ref_tensor         = {F(name)}\n"
                 f"fwhm_mov_tensor         = {F(name)}\n"
                 f"lambda_tensor           = {' '.join([f'{w:g}'] * n)}\n")

    # Flat per-modality vectors: ONE VALUE PER MODALITY on a single line. use_implicit_mask
    # belongs here, not in the header -- MMORF validates n_use_implicit_mask against n_img_ref
    # (MMORFOptions.cpp:185), so a single global 0 fails with "use_implicit_mask count mismatch"
    # as soon as there is more than one modality. Caught by the 3-modality smoke test.
    tail = ("\n; --- per-modality scalars (one value each, in modality order) ---\n"
            f"use_implicit_mask       = {' '.join(['0'] * len(mods))}\n"
            f"estimate_bias           = "
            f"{' '.join('0' if MODALITIES[m]['quant'] else '1' for m, _ in mods)}\n"
            f"bias_res_init           = {' '.join([str(BIAS_RES_INIT)] * len(mods))}\n")
    return head + body + tail


DATA_TREE_HEAD = """ext_nii=.nii.gz
ext_mat=.mat
ext_ini=.ini
ext_txt=.txt
"""


LADDERS = {int(n): dict(N_STEPS=int(n), WARP_RES_INIT=_CFG["warp_res_init_mm"], SCALING=list(v["warp_scaling"]),
                       LAMBDA=list(v["lambda_reg"]), FWHM=list(v["fwhm_mm"]))
           for n, v in _CFG["ladders"].items()}


def link_or_copy(src, dst):
    """Hard-link (the inputs stay one file on disk); across filesystems, where a hard link is impossible (EXDEV, e.g. a run
    tree on a host's local disk: B1 spec amendment 3), copy instead. Same filesystem: exactly the old os.link."""
    try:
        os.link(src, dst)
    except OSError as e:
        if e.errno != errno.EXDEV:
            raise
        shutil.copy2(src, dst)


def set_ladder(n):
    for k, v in LADDERS[n].items():
        globals()[k] = v                  # (the analysis: setattr on build_template_run, whose globals step_ini reads)
