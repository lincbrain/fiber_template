"""Solid structures drawn on histology (the LGN and STN), carried from the blockface (bf-native) into the brain's own b0 and into the MRI
template, ONE linear resample per target (paper: Methods, "Histology registration and label propagation"; Results, "Labels in the
template").

Per structure (settings/labels.yaml push.structures):
  crop the label to its bounding box + 2 mm (at least 1 mm along the section axis, axis 1); plain labels (sheets, one blockface slice per
  section) are first rebuilt in 3D by shape-based interpolation between sections (signed distances, ends extended by half the section
  spacing), dense labels (demons-interpolated, *_bfnative_L2_demons) are used as they are; Gaussian anti-aliasing per source axis
  (FWHM = max(source spacing, target voxel), labels.carry.aa_sigma); one linear resample through the target's chain (shared.xfm.apply);
  mask = probability > 0.5. The provenance channel of the dense labels (<S>_labels_demons_prov.nii.gz: 1 = charted section,
  2 = interpolated) goes the same way with nearest neighbor.
Targets:
  mri-template  the MRI template's b0 grid, through the brain's route (settings/labels.yaml template_routes; labels/route.py
                template_chain): the blockface template, or the brain's own MRI (its R1 of record, then its b0's registration)
  b0-native     the brain's b0 grid, through its R1 of record (route.r1_entry)
You get, in the output folder, per structure and target
  <S>_<struct>_bfnative_3d{sfx}.nii.gz, <S>_<struct>_<target>_prob{sfx}.nii.gz (+ provenance), <S>_<struct>_<target>_mask{sfx}.nii.gz,
  <S>_labels_demons_prov_<target>.nii.gz (dense), to_template_<S>{sfx}.json        (sfx = _demons for dense labels)

Ported from the analysis project's labels/labels_to_template.py (plain and --dense), with each brain's route from the settings.
"""
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

from labels.carry import aa_sigma

AA_RULE = "per source axis: Gaussian FWHM = max(source spacing, target voxel), then ONE linear resample, > 0.5 (orchestrator 2026-09-28)"


def fill_between_sections(a, zooms, half_mm=0.6):
    """Plain labels exist only on the blockface slices that carry a histology section (sheets): rebuild the 3D structure by shape-based
    interpolation, the signed distance maps (in-plane, mm) of consecutive labeled slices interpolated linearly over the slices between
    them; the first and last sections extended by half the section spacing. Axis 1 is the slice (y) axis of the bf-native NIfTI."""
    from scipy.ndimage import distance_transform_edt
    zs = [z for z in range(a.shape[1]) if a[:, z, :].any()]
    if not zs:
        return a
    ip = (zooms[0], zooms[2])

    def sdf(m):
        return distance_transform_edt(~m, sampling=ip) - distance_transform_edt(m, sampling=ip)
    out = a.copy()
    for z1, z2 in zip(zs, zs[1:]):
        d1, d2 = sdf(a[:, z1, :]), sdf(a[:, z2, :])
        for z in range(z1 + 1, z2):
            t = (z - z1) / (z2 - z1)
            out[:, z, :] = ((1 - t) * d1 + t * d2) < 0
    ext = int(round(half_mm / zooms[1]))
    for z0, rng in ((zs[0], range(max(0, zs[0] - ext), zs[0])), (zs[-1], range(zs[-1] + 1, min(a.shape[1], zs[-1] + ext + 1)))):
        for z in rng:
            out[:, z, :] = a[:, z0, :]
    return out


def crop(img, dense, margin_mm=2.0):
    """-> (the label in its box + margin_mm as float32, its affine, zooms), or None when it is empty."""
    a = np.asarray(img.dataobj) > 0
    ijk = np.argwhere(a)
    if not len(ijk):
        return None
    z = np.array(img.header.get_zooms()[:3])
    m = np.ceil(margin_mm / z).astype(int)
    m[1] = max(m[1], int(np.ceil(1.0 / z[1])))                      # room for the half-spacing end extension
    lo = np.maximum(ijk.min(0) - m, 0); hi = np.minimum(ijk.max(0) + m + 1, a.shape)
    sub = a[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    sub = (sub if dense else fill_between_sections(sub, z)).astype(np.float32)
    aff = img.affine.copy(); aff[:3, 3] = nib.affines.apply_affine(img.affine, lo)
    return sub, aff, z


def carry_structures(S, src_dir, out_dir, targets, structs, dense, log=print):
    """Carry <src_dir>/<S>_<struct>_bfnative_L2{sfx}.nii.gz for each struct to each target, targets = {name: (ref grid, chain)} in the
    order they are written. Returns the record (also written as to_template_<S>{sfx}.json)."""
    from shared import xfm
    sfx = "_demons" if dense else ""
    src_dir, d = Path(src_dir), Path(out_dir); d.mkdir(parents=True, exist_ok=True)
    rec = {}
    for st in structs:
        src = src_dir / f"{S}_{st}_bfnative_L2{sfx}.nii.gz"
        if not src.exists():
            continue
        c = crop(nib.load(str(src)), dense)
        if c is None:
            continue
        sub, aff, zooms = c
        r = dict(voxels_bf_3d=int(sub.sum()), vol_bf_mm3=round(float(sub.sum() * np.prod(zooms)), 3),
                 src_zooms_mm=[round(float(z), 4) for z in zooms], anti_aliasing=AA_RULE)
        nib.save(nib.Nifti1Image(sub.astype(np.uint8), aff), str(d / f"{S}_{st}_bfnative_3d{sfx}.nii.gz"))
        for dst, (ref, chain) in targets.items():
            vox = float(np.mean(nib.load(str(ref)).header.get_zooms()[:3]))
            sig = aa_sigma(zooms, vox)
            tmp = d / f"_tmp_{st}_{dst}{sfx}.nii.gz"
            nib.save(nib.Nifti1Image(gaussian_filter(sub, sig), aff), str(tmp))
            prob = d / f"{S}_{st}_{dst}_prob{sfx}.nii.gz"
            xfm.apply(chain, str(tmp), str(ref), str(prob), order=1)
            os.remove(tmp)
            p = nib.load(str(prob))
            pa = np.asarray(p.dataobj)
            mask = (pa > 0.5).astype(np.uint8)
            nib.save(nib.Nifti1Image(mask, p.affine), str(d / f"{S}_{st}_{dst}_mask{sfx}.nii.gz"))
            vz = float(np.prod(p.header.get_zooms()[:3]))
            r[dst] = dict(voxels=int(mask.sum()), vol_mm3=round(float(mask.sum() * vz), 3), aa_sigma_src_vox=[round(float(x), 3) for x in sig],
                          prob_sum_mm3=round(float(pa.sum() * vz), 3),
                          centroid_ras=[round(float(x), 2) for x in nib.affines.apply_affine(p.affine, np.argwhere(mask > 0).mean(0))] if mask.any() else None,
                          chain=chain.describe())
            log(f"{S} {st} -> {dst}: {r[dst]['voxels']} vox ({r[dst]['vol_mm3']} mm3; bf {r['vol_bf_mm3']} mm3)")
        rec[st] = r
    pv = src_dir / f"{S}_labels_demons_prov.nii.gz"
    if dense and pv.exists():
        for dst, (ref, chain) in targets.items():
            xfm.apply(chain, str(pv), str(ref), str(d / f"{S}_labels_demons_prov_{dst}.nii.gz"), order=0)
    (d / f"to_template_{S}{sfx}.json").write_text(json.dumps(rec, indent=1))
    return rec


def targets_for(S, route):
    """{"mri-template": (the MRI template's b0, S's chain on `route`), "b0-native": (S's b0 grid, its R1 of record)}, in that order."""
    from shared import paths, xfm
    from labels import route as RT
    P = paths.products("products_blockface")
    meta = json.loads((P["data"] / f"sub-{S}" / "xfm" / "bf-native__to__b0-native" / "meta.json").read_text())
    return {"mri-template": (P["template_mri"] / "linc_mri_b0.nii.gz", RT.template_chain(S, route)),
            "b0-native": (paths.rerooted(meta["dst_grid"]), xfm.Chain([RT.r1_entry(S)], strict=True))}
