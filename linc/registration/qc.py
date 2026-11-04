"""The paper's acceptance checks of one registration (Methods, "Evaluation of the registration pipeline"; `linc-register qc`).

You need
  the fixed image, the moved image (or the moving image with its warp and linear transform), both brain masks, and the outline
  target; the limits: configs/registration.yaml `qc`.
What it does
    measure(fixed, moved, ...)        on the fixed grid (the formulas of the analysis project's code/regqc.py run_qc, unchanged):
        linear part     the world matrix's polar decomposition (fsl.py decompose): principal scales in [0.9, 1.1] (between
                        animals: anisotropy <= 1.10 and isotropic scale in [0.80, 1.25]; for the cross-modal alignments the
                        anisotropy is only reported), |rot y|, |rot z| < 30 deg after an optional 180-degree turn is taken out, no
                        reflection
        NCC             the normalized cross-correlation between the fixed and moved images inside the evaluation mask (the 1 mm
                        interior of both masks): -NCC >= 0.25 for an inverted-contrast pair (blockface against b0), NCC >= 0.6
                        otherwise; optionally at a matched bandwidth
        reported only   the normalized mutual information (NMI); the Dice coefficient between the fixed and moved brain
                        envelopes; the mean distance between their interior edges (linc/evaluation/regeval.py, edge_mean)
        warp            folds (negative Jacobian determinants) inside the fixed brain mask: none allowed; the displacement's 95th
                        percentile inside it, flagged above 6 mm and failed above 8 mm
    outline(moved_mask, target, ...)  the distance from the moved brain's outline to the target's at each pole (linc/link/qc.py,
                        configs/link.yaml `outline`): fail at p99 > 3.5 mm or max > 5 mm, warn at p99 > 2.5 mm
    photographed_slab(...)            for a blockface -> own b0 registration, the outline target is the b0 brain mask inside the
                        photographed part of the brain (the end faces of the blockface mapped by the 9-dof linear stage)
    render_png(...)                   a three-plane figure (checkerboard; the fixed image with the moved image's interior edges and
                        both envelopes)
You get
  a JSON row with every measure and pass / fail, the PNG, and, given a registry entry, the outcome recorded in it.

Ported from the analysis project's code/regqc.py (run_qc's measures and limits, render_png) and its outline scan of the
blockface -> own b0 registrations (bftemplate/d1b/outline_scan_bf.py).
"""
import os
import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib
import yaml
from scipy import ndimage as ndi

from linc.link import qc as LQ
from linc.registration import fsl

REPO = Path(__file__).resolve().parents[2]


def gate_config():
    return yaml.safe_load((REPO / "configs" / "registration.yaml").read_text())["qc"]


# presets: the flags the paper's QC used for each kind of registration (registration_qc.json families)
PRESETS = {
    "bf2b0": dict(cross_modal=True, allow_180=True, cross_animal=False, aniso_report_only=False),        # blockface -> own b0 (R1)
    "direct": dict(cross_modal=True, allow_180=True, cross_animal=True, aniso_report_only=True),         # blockface -> MRI template
    "bf2bftemplate": dict(cross_modal=False, allow_180=True, cross_animal=True, aniso_report_only=False),  # blockface -> blockface template (same contrast)
    "b02template": dict(cross_modal=False, allow_180=False, cross_animal=True, aniso_report_only=False),  # b0 -> MRI template (R5)
    "template2template": dict(cross_modal=False, allow_180=False, cross_animal=True, aniso_report_only=True),  # the link (b0 -> b0)
}


def itk_to_world(path):
    """ITK AffineTransform text (LPS, fixed->moving pull) -> RAS world matrix moving->fixed (code/regqc.py)."""
    P = F = None
    for line in open(path):
        if line.startswith("Parameters:"):
            P = np.array(line.split()[1:], float)
        elif line.startswith("FixedParameters:"):
            F = np.array(line.split()[1:], float)
    A = P[:9].reshape(3, 3); t = P[9:12]; c = F[:3] if F is not None else np.zeros(3)
    M = np.eye(4); M[:3, :3] = A; M[:3, 3] = t + c - A @ c
    L = np.diag([-1.0, -1.0, 1.0, 1.0])
    return np.linalg.inv(L @ M @ L)


def world_matrix(mat, mat_kind="fsl", mat_src=None, mat_ref=None):
    if mat_kind == "fsl":
        return fsl.fsl_to_world(np.loadtxt(mat), nib.load(str(mat_src)), nib.load(str(mat_ref)))
    if mat_kind == "itk":
        return itk_to_world(mat)
    return np.loadtxt(mat)


def nmi(x, y, bins=64):
    h, _, _ = np.histogram2d(x, y, bins=bins)
    p = h / max(h.sum(), 1)
    px, py = p.sum(1), p.sum(0)
    e = lambda q: -(q[q > 0] * np.log(q[q > 0])).sum()                                    # noqa: E731
    return float((e(px) + e(py)) / max(e(p), 1e-12))


def scale_fails(scales, g, cross_animal, aniso_report_only=False):
    sc = np.abs(np.asarray(scales, float))          # magnitudes: a (declared) reflection is gated separately
    if cross_animal:
        ca = g["cross_animal"]
        aniso = float(sc.max() / sc.min()); iso = float(np.prod(sc) ** (1 / 3))
        out = []
        if aniso > ca["aniso_max"] and not aniso_report_only:       # reported, not gated, for the blockface -> MRI template placements
            out.append(f"scale anisotropy {aniso:.3f} > {ca['aniso_max']}")
        if not ca["iso_lo"] <= iso <= ca["iso_hi"]:
            out.append(f"isotropic scale {iso:.3f} outside [{ca['iso_lo']}, {ca['iso_hi']}]")
        return out
    if not all(g["scale_lo"] <= x <= g["scale_hi"] for x in sc):
        return [f"scales {['%.3f' % x for x in sc]} outside [{g['scale_lo']}, {g['scale_hi']}]"]
    return []


def measure(fixed, moved, *, warp=None, mat=None, mat_kind="fsl", mat_src=None, mat_ref=None, fixed_mask=None, moved_mask=None,
            eval_mask=None, brain_mask=None, cross_modal=False, allow_180=False, cross_animal=False, aniso_report_only=False,
            allow_reflection=False, ncc_fwhm_mm=None, edge_eval=True):
    """The gate measures of one registration (moved already on, or resampled to, the fixed grid). Returns (row, F, M, fm, mm)."""
    g = gate_config()
    gate = dict(scale_lo=g["scale_lo"], scale_hi=g["scale_hi"], rot_yz_deg=g["rot_yz_deg"], folds_max=g["folds_max"],
                ncc_min=g["cross_modal_ncc_min"] if cross_modal else g["ncc_min"])
    fimg = nib.load(str(fixed))
    F, _ = LQ.load(fixed)
    M, resampled = LQ.load(moved, ref=fimg)
    vox = np.abs(np.diag(fimg.affine)[:3])
    fm = (LQ.load(fixed_mask, ref=fimg, order=0)[0] > 0.5) if fixed_mask else (F > 0.02 * np.percentile(F, 99.9))
    mm = (LQ.load(moved_mask, ref=fimg, order=0)[0] > 0.5) if moved_mask else (M > 0.02 * np.percentile(M, 99.9))
    em = ndi.binary_erosion(fm & mm, iterations=max(1, int(round(1.0 / float(vox.min())))))
    if eval_mask:
        em &= LQ.load(eval_mask, ref=fimg, order=0)[0] > 0.5
    row = dict(fixed=str(fixed), moved=str(moved), warp=str(warp) if warp else None, mat=str(mat) if mat else None,
               mat_kind=mat_kind if mat else None, fixed_mask=fixed_mask and str(fixed_mask), moved_mask=moved_mask and str(moved_mask),
               eval_mask=eval_mask and str(eval_mask), moved_resampled_for_qc=resampled, cross_modal=bool(cross_modal),
               allow_180=bool(allow_180), cross_animal=bool(cross_animal), aniso_report_only=bool(aniso_report_only),
               gate_thresholds=gate, eval_mask_vox=int(em.sum()))
    fails = []
    if mat:
        d = fsl.decompose(world_matrix(mat, mat_kind, mat_src, mat_ref), allow_180=allow_180)
        row["linear"] = d
        fails += scale_fails(d["scales"], g, cross_animal, aniso_report_only)
        if d["reflection"] and not allow_reflection:
            fails.append("linear part is a reflection (det<0)")
        rx, ry, rz = d["rot_xyz_deg"]
        if abs(ry) >= gate["rot_yz_deg"] or abs(rz) >= gate["rot_yz_deg"]:
            fails.append(f"rotation y/z {ry:.1f}/{rz:.1f} deg >= {gate['rot_yz_deg']}")
    if ncc_fwhm_mm:
        sg = ncc_fwhm_mm / 2.3548 / vox
        row["ncc_raw"] = LQ.ncc(F, M, em)
        F, M = ndi.gaussian_filter(F, sg), ndi.gaussian_filter(M, sg)
        row["ncc_fwhm_mm"] = float(ncc_fwhm_mm)
    row["ncc"] = LQ.ncc(F, M, em)
    row["nmi"] = nmi(F[em], M[em]) if em.sum() > 100 else float("nan")
    row["envelope_dice"] = LQ.dice(fm, mm)
    row["envelope_dice_kind"] = "mask" if (fixed_mask and moved_mask) else ("support" if not (fixed_mask or moved_mask) else "mixed")
    if brain_mask:
        bm = LQ.load(brain_mask, ref=fimg, order=0)[0] > 0.5
        nb = max(int(bm.sum()), 1)
        row["brain_overlap"] = dict(brain_mask=str(brain_mask), dice=LQ.dice(bm, mm), bf_outside_brain_frac=float((mm & ~bm).sum() / nb),
                                    brain_uncovered_frac=float((bm & ~mm).sum() / nb))
    nccv = -row["ncc"] if cross_modal else row["ncc"]
    row["ncc_gated"] = nccv
    if not (nccv >= gate["ncc_min"]):
        fails.append(f"{'-NCC (inverted contrast)' if cross_modal else 'NCC'} {nccv:.3f} < {gate['ncc_min']}")
    if warp:
        j = LQ.jacobian_stats(warp, mask=fm, mask_affine=fimg.affine)
        row["jacobian"] = j
        if j.get("folds_n_in_mask", j.get("folds_n", 0)) > gate["folds_max"]:
            fails.append(f"folds {j['folds_n_in_mask']} voxels inside the fixed brain mask (total {j['folds_n']}, {j['folds_pct']:.4f} %)")
        dp = j.get("drift_p95_mm")
        if dp is not None:
            row["drift_flag"] = bool(dp > g["drift_flag_p95_mm"])
            if dp > g["drift_fail_p95_mm"]:
                fails.append(f"drift p95 {dp:.1f} mm > {g['drift_fail_p95_mm']} (displacement vs the linear start)")
    if edge_eval:
        from linc.evaluation import regeval
        tmp = None
        if resampled:
            tmp = tempfile.mktemp(suffix=".nii.gz"); nib.save(nib.Nifti1Image(M, fimg.affine), tmp)
        try:
            r = regeval.evaluate(str(fixed), tmp or str(moved), None)
            row["regeval"] = {k: r[k] for k in ("edge_mean", "edge_sym", "edge_frac1mm", "nmi_interior", "n_bf_edges", "n_mri_edges") if k in r}
        finally:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)
    row["gate_pass"] = not fails
    row["gate_fail_reasons"] = fails
    return row, F, M, fm, mm


def photographed_slab(mat, mat_src, mat_ref, iso_mask, grid_img):
    """The blockface's photographed slab on a grid: the end faces of its isotropic mask mapped by the 9-dof linear stage (FLIRT matrix
    of (mat_src, mat_ref)), as linc/registration/to_own_mri.py slab_keep."""
    from linc.registration.to_own_mri import slab_keep
    W = fsl.fsl_to_world(np.loadtxt(mat), nib.load(str(mat_src)), nib.load(str(mat_ref)))
    return slab_keep(W, str(iso_mask), grid_img.affine, grid_img.shape[:3])


def outline(moved_mask, target, affine, declared=()):
    """The outline check of a moved brain mask against the target brain mask on one grid: (status, reasons, regions)."""
    ol = LQ.outline_regions(moved_mask, target, affine, declared=declared)
    status, why = LQ.outline_gate(ol)
    return status, why, ol


def _pnorm(a, m):
    v = a[m] if m is not None and m.any() else a[a != 0]
    if v.size == 0:
        return np.zeros_like(a)
    lo, hi = np.percentile(v, (1, 99.5))
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def render_png(path, F, M, fm, mm, title, lines, ok):
    """regqc's three-plane figure: checkerboard fixed/moved (top); the fixed image with the moved image's interior edges (yellow) and
    the moved / fixed envelopes (green / blue) (bottom)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Fn, Mn = _pnorm(F, fm), _pnorm(M, mm)
    ctr = [int(round(c)) for c in ndi.center_of_mass(fm)] if fm.any() else [s // 2 for s in F.shape]
    g = ndi.gaussian_filter(Mn, 1.0)
    gm = np.sqrt(sum(x ** 2 for x in np.gradient(g)))
    inner = ndi.binary_erosion(mm, iterations=3) if mm.any() else mm
    edge = inner & (gm > np.percentile(gm[inner], 88)) if inner.any() else np.zeros_like(mm)
    env = mm & ~ndi.binary_erosion(mm)
    fenv = fm & ~ndi.binary_erosion(fm)
    fig, ax = plt.subplots(2, 3, figsize=(15, 9.6))
    for c, (nm, axis) in enumerate((("sagittal", 0), ("coronal", 1), ("axial", 2))):
        take = lambda v: np.take(v, ctr[axis], axis=axis).T                                # noqa: E731
        f2, m2 = take(Fn), take(Mn)
        n = max(f2.shape) // 8 or 1
        yy, xx = np.indices(f2.shape)
        chk = ((yy // n + xx // n) % 2).astype(bool)
        ax[0, c].imshow(np.where(chk, f2, m2), cmap="gray", origin="lower", interpolation="nearest")
        ax[0, c].set_title(f"{nm}: checkerboard fixed/moved", fontsize=9)
        rgb = np.stack([f2] * 3, -1)
        rgb[take(edge)] = (1.0, 0.85, 0.0)
        rgb[take(env)] = (0.1, 1.0, 0.2)
        rgb[take(fenv)] = (0.2, 0.5, 1.0)
        ax[1, c].imshow(rgb, origin="lower", interpolation="nearest")
        ax[1, c].set_title(f"{nm}: fixed + moved edges (yellow), moved/fixed envelope (green/blue)", fontsize=8)
        for r in (0, 1):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    fig.suptitle(title, fontsize=12, color=("darkgreen" if ok else "firebrick"), fontweight="bold")
    fig.text(0.01, 0.005, "\n".join(lines), fontsize=8, family="monospace", va="bottom")
    plt.tight_layout(rect=(0, 0.07, 1, 0.95))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def summary_lines(row):
    lines = [f"NCC {row['ncc']:+.3f}  NMI {row['nmi']:.3f}  envDice {row['envelope_dice']:.3f}  eval vox {row['eval_mask_vox']}"]
    if "linear" in row:
        d = row["linear"]
        lines.append(f"rot xyz {d['rot_xyz_deg'][0]:+.1f} {d['rot_xyz_deg'][1]:+.1f} {d['rot_xyz_deg'][2]:+.1f} deg (absorbed {d['absorbed_180']})"
                     f"  scales {' '.join('%.3f' % s for s in d['scales'])}")
    if "jacobian" in row and "folds_n" in row["jacobian"]:
        j = row["jacobian"]
        lines.append(f"det J min {j['jac_min']:.3f} max {j['jac_max']:.2f} std {j['jac_std']:.3f}  folds {j['folds_n']}"
                     + (f"  drift p95 {j['drift_p95_mm']:.1f} mm" if "drift_p95_mm" in j else ""))
    if isinstance(row.get("regeval"), dict) and "edge_mean" in row["regeval"]:
        lines.append(f"edge_mean {row['regeval']['edge_mean']:.3f} mm  frac>1mm {row['regeval']['edge_frac1mm']:.3f}")
    if "outline" in row:
        lines.append(f"outline {row['outline']['status']}" + (": " + "; ".join(row["outline"]["reasons"]) if row["outline"]["reasons"] else ""))
    if row["gate_fail_reasons"]:
        lines.append("FAIL: " + "; ".join(row["gate_fail_reasons"]))
    return lines
