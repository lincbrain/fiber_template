#!/usr/bin/env python3
"""The input check of every registration: every input must be brain-masked, so tissue outside the brain (skull, dura, neck) cannot
affect a registration.

    from blockface.registration.inputs import check_masked
    check_masked("A.nii.gz", "Amask.nii.gz", label="A")

You need
  an image and its brain mask, on the same grid.
What it does
  check_masked(image, mask) raises InputNotMasked when real signal lies away from the mask: voxels farther than DIST_MM (2 mm) from
  the mask whose |value| exceeds REL (1 %) of the median |value| inside the brain, over more than `tol` (0.1 %) of the image's
  nonzero voxels. The thin partial-volume shell of a masked image resampled once (<= ~1.5 mm) and float residue pass; tissue
  outside the brain fails. For tensors and multi-channel images a voxel is nonzero if any component is. An override needs an
  explicit, logged reason (allow_reason=...).
You get
  dict(nonzero=..., out_dilated=...), or the InputNotMasked error.

Ported unchanged from the analysis project's code/input_mask_check.py.
"""
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

TOL, DIST_MM, REL = 0.001, 2.0, 0.01


class InputNotMasked(RuntimeError):
    pass


def check_masked(image, mask, tol=TOL, label="", allow_reason=None, dist_mm=DIST_MM, rel=REL):
    im = nib.load(str(image)) if not hasattr(image, "dataobj") else image
    mi = nib.load(str(mask)) if not hasattr(mask, "dataobj") else mask
    d = np.asarray(im.dataobj); m = np.asarray(mi.dataobj) > 0.5
    a = np.sqrt((np.asarray(d, np.float64) ** 2).sum(axis=tuple(range(3, d.ndim)))) if d.ndim > 3 else np.abs(np.asarray(d, np.float64))
    nz = a > 0
    if nz.shape != m.shape:
        raise InputNotMasked(f"{label}: image grid {nz.shape} != mask grid {m.shape} (resample the mask first)")
    n = int(nz.sum()); out = float((nz & ~ndi.binary_dilation(m)).sum() / max(n, 1))
    vox = np.sqrt((np.asarray(im.affine)[:3, :3] ** 2).sum(0))
    far = ndi.distance_transform_edt(~m, sampling=vox) > dist_mm
    med = float(np.median(a[m & nz])) if (m & nz).any() else 0.0
    bad = float((far & (a > rel * med)).sum() / max(n, 1))
    rec = dict(label=label, image=str(image), mask=str(mask), nonzero=n, out_dilated=round(out, 5), far_signal=round(bad, 5),
               rule=f"> {dist_mm} mm from the mask and > {rel:g} x in-brain median, over > {tol:g} of nonzero", tol=tol)
    if bad > tol:
        msg = (f"{label}: {100 * bad:.2f} % of nonzero voxels carry signal (> {100 * rel:g} % of the in-brain median) more than {dist_mm:g} mm "
               f"from the brain mask (> {100 * tol:.2f} %): input not masked")
        if allow_reason:
            print(f"[input_mask_check] OVERRIDDEN ({allow_reason}): {msg}", flush=True)
            rec["override"] = allow_reason
        else:
            raise InputNotMasked(msg + " (multiply the image by its brain mask; see code/blockface/registration/inputs.py)")
    return rec

