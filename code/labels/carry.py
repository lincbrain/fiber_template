"""Labels on the blockface grid (bf-native) carried into the MRI template with ONE anti-aliased linear resample.

Paper: Methods, "Histology registration and label propagation". Parameters: settings/labels.yaml (carry).

Per label: crop to its bounding box (+ crop_margin_mm); where the blockface slice spacing is finer than the template voxel, widen each
labeled slice along the slice axis to one template voxel (max filter of 2 x round(voxel / spacing / 2) + 1 slices); anti-alias per source
axis (Gaussian FWHM = max(source spacing, template voxel), `aa_sigma`); one linear resample onto the template grid. Labels from
histology can be SPARSE (fiber and terminal annotations), so three outputs per label:
  <S>_<label>_mri-template_density.nii.gz     the carried anti-aliased fraction (float, 0-1)
  <S>_<label>_mri-template_footprint.nii.gz   density >= foot: where annotations land
  <S>_<label>_mri-template.nii.gz             density > solid (0.5): the solid-label rule
QC (report only): fraction of each carried label inside the template brain mask; fraction on the same side (sign of world x, RAS) as the
case's injection site, when one is labeled.

The route is given explicitly, as a list of steps in PULL order (a template point goes through them first to last):
  * a brain with its own MRI: its registry entries (shared.xfm), e.g. bf-native -> b0-native (R1) and b0-native -> mri-template (R5),
    applied with shared.xfm.apply (`carry(..., chain=Chain([...]))`);
  * a brain without MRI: a precomputed pull map (blockface.registration.xfmtools, e.g. mri-template -> blockface template), then the
    member's warp and its rigid placement (`carry(..., pull=[...])`, see `pull_steps`).

Ported from the analysis project's labels/tracer_to_template.py, labels/label_refs.py (aa_sigma) and bftemplate/d2b/tracer_mf191.py.
"""
import json
import os
from pathlib import Path

import numpy as np

from labels.labels import config


def aa_sigma(src_zooms, target_vox):
    """Per source axis: Gaussian FWHM = max(source spacing, target voxel); returned in source voxels (scipy's gaussian_filter)."""
    z = np.asarray(src_zooms, float)[:3]
    return np.maximum(z, float(target_vox)) / 2.355 / z


def crop_sheet(img, margin_mm, target_vox):
    """-> (cropped label as float32, its affine, zooms, number of labeled slices, labeled voxels, widening half-width), or None."""
    import nibabel as nib
    from scipy.ndimage import maximum_filter1d
    a = np.asarray(img.dataobj) > 0
    ijk = np.argwhere(a)
    if not len(ijk):
        return None
    z = np.array(img.header.get_zooms()[:3], float)
    m = np.ceil(margin_mm / z).astype(int)
    lo = np.maximum(ijk.min(0) - m, 0); hi = np.minimum(ijk.max(0) + m + 1, a.shape)
    sub = a[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    n_sections = int(np.unique(ijk[:, 1]).size)                         # axis 1 = the blockface slice axis
    half = int(round(target_vox / z[1] / 2)) if z[1] < target_vox else 0  # widen to one template voxel where slices are thinner
    if half:
        sub = maximum_filter1d(sub.astype(np.uint8), size=2 * half + 1, axis=1) > 0
    aff = img.affine.copy(); aff[:3, 3] = nib.affines.apply_affine(img.affine, lo)
    return sub.astype(np.float32), aff, z, n_sections, int(a.sum()), half


# --------------------------------------------------------------------------- pull steps (route without own MRI)

def pull_steps(specs):
    """Build pull steps from (kind, ...) tuples, in PULL order:
        ("map", dir, name)              an xfmtools pull map <dir>/<name>_L.txt (+ _d.nii.gz)
        ("entry", dir[, inverted])      a registry entry (shared.xfm.Entry), forward or inverted
        ("warp", file, convention[, src_grid])   one warp component (shared.xfm.Warp)
        ("ras_pull", file)              a 4 x 4 RAS matrix, dst point -> src point
    Returns (callable points -> points, description list)."""
    from shared import xfm
    from blockface.registration import xfmtools as X
    fs, desc = [], []
    for s in specs:
        kind = s[0]
        if kind == "map":
            fs.append(X.load_map(str(s[1]), s[2])); desc.append(dict(kind="map", dir=str(s[1]), name=s[2]))
        elif kind == "entry":
            e = xfm.Entry(s[1])
            if len(s) > 2 and s[2]:
                e = xfm.InverseEntry(e)
            fs.append(e.pull); desc.append(dict(kind="entry", dir=str(s[1]), src=e.src, dst=e.dst, inverted=e.inverted,
                                                engine=e.meta.get("engine"), qc=e.qc_status()))
        elif kind == "warp":
            w = xfm.Warp(s[1], s[2], src_grid=s[3] if len(s) > 3 else None)
            fs.append(lambda x, w=w: np.asarray(w.pull(x))); desc.append(dict(kind="warp", file=str(s[1]), convention=s[2]))
        elif kind == "ras_pull":
            M = np.loadtxt(str(s[1]))
            fs.append(lambda x, M=M: x @ M[:3, :3].T + M[:3, 3]); desc.append(dict(kind="ras_pull", file=str(s[1])))
        else:
            raise ValueError(f"unknown pull step {kind}")

    def f(x):
        for g in fs:
            x = g(x)
        return x
    return f, desc


# --------------------------------------------------------------------------- carry

def _load_ref(ref, brain):
    import nibabel as nib
    r = nib.load(str(ref))
    b = np.asanyarray(nib.load(str(brain)).dataobj) > 0 if brain else None
    return r, b


def carry(subject, label_files, *, ref, out_dir, brain=None, chain=None, pull=None, links=None, suffix="_bfnative_L2_demons.nii.gz",
          cfg=None, log=print):
    """Carry each label file (<S>_<label><suffix>) onto the template grid `ref` along `chain` (shared.xfm.Chain) or `pull`
    (points -> points, `pull_steps`). Writes the three outputs per label and tracer_template_<S>.json; returns the case record."""
    import nibabel as nib
    from scipy.ndimage import gaussian_filter
    from shared import xfm
    from blockface.registration import xfmtools as X
    if (chain is None) == (pull is None):
        raise ValueError("give exactly one of chain= or pull=")
    C = (cfg or config())["carry"]
    TV, FOOT, SOLID, MARGIN, SLAB = float(C["template_voxel_mm"]), float(C["foot"]), float(C["solid"]), float(C["crop_margin_mm"]), int(C["slab"])
    refimg, brainm = _load_ref(ref, brain)
    shp, A = refimg.shape[:3], refimg.affine
    xw = A[0, 3] + A[0, 0] * np.arange(shp[0])                              # world x of each template column (RAS)
    od = Path(out_dir); od.mkdir(parents=True, exist_ok=True)
    files = sorted(str(f) for f in label_files)
    classes = [Path(f).name[len(subject) + 1:-len(suffix)] for f in files]
    if links is None:
        links = chain.describe() if chain is not None else None
    crec = dict(route="registry chain" if chain is not None else "pull map", links=links, classes={}, stain=None)
    sheets = {}
    for f, cl in zip(files, classes):
        c = crop_sheet(nib.load(f), MARGIN, TV)
        if c is None:
            continue
        sub, aff, z, nsec, nvox, half = c
        sheets[cl] = (gaussian_filter(sub, aa_sigma(z, TV)), aff, z, nsec, nvox, half)
    P = {}
    if chain is not None:
        for cl, (sm, aff, *_r) in sheets.items():
            tmp = od / f"_tmp_{cl}.nii.gz"; nib.save(nib.Nifti1Image(sm, aff), str(tmp))
            prob = od / f"_prob_{cl}.nii.gz"; xfm.apply(chain, str(tmp), str(ref), str(prob), order=1); os.remove(tmp)
            P[cl] = np.asarray(nib.load(str(prob)).dataobj, np.float32); os.remove(prob)
    else:
        P = {cl: np.zeros(shp, np.float32) for cl in sheets}
        for z0 in range(0, shp[2], SLAB):                                    # one evaluation of the route, every label sampled
            z1 = min(shp[2], z0 + SLAB)
            pts = pull(X.grid_world(shp, A, z0, z1))
            for cl, (sm, aff, *_r) in sheets.items():
                P[cl][:, :, z0:z1] = X.sample(sm, aff, pts, order=1).reshape(shp[0], shp[1], z1 - z0)
    outputs = []
    for cl, (sm, aff, z, nsec, nvox, half) in sheets.items():
        p = P[cl]
        fo = [od / f"{subject}_{cl}_mri-template_density.nii.gz", od / f"{subject}_{cl}_mri-template.nii.gz",
              od / f"{subject}_{cl}_mri-template_footprint.nii.gz"]
        nib.save(nib.Nifti1Image(np.clip(p, 0, 1).astype(np.float32), A), str(fo[0]))
        h = p > SOLID
        nib.save(nib.Nifti1Image(h.astype(np.uint8), A), str(fo[1]))
        m = p >= FOOT
        nib.save(nib.Nifti1Image(m.astype(np.uint8), A), str(fo[2]))
        outputs += fo
        n = int(m.sum())
        crec["classes"][cl] = dict(sections=nsec, voxels_bf=nvox, vol_bf_mm3=round(nvox * float(np.prod(z)), 3), widened_half_slices=half,
                                   voxels_template_footprint=n, voxels_template_gt05=int(h.sum()),
                                   density_sum_mm3=round(float(np.clip(p, 0, 1).sum()) * TV ** 3, 3),
                                   frac_in_brain=(round(float((m & brainm).sum() / n), 4) if n else None) if brainm is not None else None,
                                   x_mean_mm=round(float(xw[np.where(m)[0]].mean()), 3) if n else None)
        log(f"{subject} {cl} {crec['classes'][cl]}")
    inj = crec["classes"].get("Injection_site")
    side = np.sign(inj["x_mean_mm"]) if inj and inj["voxels_template_footprint"] else None
    crec["injection_side"] = None if side is None else ("right (+x)" if side > 0 else "left (-x)")
    for cl, v in crec["classes"].items():
        if side is not None and v["voxels_template_footprint"]:
            mm = np.asarray(nib.load(str(od / f"{subject}_{cl}_mri-template_footprint.nii.gz")).dataobj) > 0
            v["frac_same_side_as_injection"] = round(float((np.sign(xw[np.where(mm)[0]]) == side).mean()), 4)
    crec["n_sections"] = max((v["sections"] for v in crec["classes"].values()), default=0)
    cv = {k: v for k, v in crec["classes"].items() if v["voxels_template_footprint"]}
    crec["qc"] = dict(report_only=True,
                      frac_in_brain={k: v["frac_in_brain"] for k, v in cv.items()},
                      frac_in_brain_min=min((v["frac_in_brain"] for v in cv.values() if v["frac_in_brain"] is not None), default=None),
                      injection_side=crec["injection_side"],
                      frac_same_side_as_injection={k: v.get("frac_same_side_as_injection") for k, v in cv.items()},
                      frac_same_side_min=(min(v["frac_same_side_as_injection"] for v in cv.values()) if side is not None else None),
                      hemisphere_check=("n/a: no injection site recorded" if side is None else
                                        "fraction of each class's carried voxels on the injection's side of x = 0 (projections can be bilateral)"),
                      empty_classes=[k for k, v in crec["classes"].items() if not v["voxels_template_footprint"]])
    (od / f"tracer_template_{subject}.json").write_text(json.dumps(crec, indent=1, default=str))
    return crec, outputs
