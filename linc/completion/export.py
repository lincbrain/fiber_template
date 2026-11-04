"""The completed volume of one brain and its companions, on the brain's 0.25 mm output grid (the products of record; written by
cli/linc-complete). Grey is in the photographs' units (fine/bf.nii.gz); the join is at the clean cut face z_f.

  <ID>_bf_completed        w * photographs + (1 - w) * bf_synth, times the brain (deploy.join)
  <ID>_bf_synth            the prediction mapped to the photographs' grey (one isotonic map), times the brain
  <ID>_bf_real             the photographs where w > 0
  <ID>_seam_ramp           the cross-fade weight w (also as <ID>_seam_ramp_real)
  <ID>_seam_depth          (y_dep - z_f) x 0.25 mm: signed distance along the section normal, >= 0 rostral of the cut, times the brain
  <ID>_photo_fraction      the photographed fraction of each voxel, times the brain;  <ID>_provenance = w x photo fraction
  <ID>_synth_mask          the predicted voxels: the brain minus the photographed tissue rostral of the cut
  <ID>_bf_completed_rgb    color by a constant chroma per brain: the photographs' mean chroma times bf_synth mapped to luminance
                           (least squares, photographed voxels), cross-faded with the photographs' color (nearest photographed
                           voxel beyond the photographs); <ID>_chroma.json holds the chroma and the grey -> luminance line
  <ID>_b0_input_0p5        the network's input (coarse/in.nii.gz), on the 0.5 mm parent grid of the products
With cache_out: <cache_out>/<ID>/deploy/{brain, real_mask}.nii.gz on the product grid and <cache_out>/seam.json (the section
normal per brain), the masks the blockface template's construction reads.
"""
import hashlib
import json
import os
import shutil
import time

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

from . import deploy

md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()  # noqa: E731


def load_real_rgb(path, photo_mask):
    """the photographs' RGB on [0, 1]; a raw uint8 volume (0..255) is scaled and rescaled by the 1st-99th percentile luminance of the
    photographed tissue"""
    im = nib.load(path); r = np.asanyarray(im.dataobj).astype(np.float32)
    if im.get_data_dtype() == np.uint8 or float(r.max()) > 1.5:
        r /= 255.0
        has = r.sum(-1) > 0
        lum = r.mean(-1)[photo_mask & has]
        lo, hi = (np.percentile(lum, (1.0, 99.0)) if lum.size else (0.0, 1.0))
        r = np.clip((r - lo) / max(hi - lo, 1e-6), 0, 1) * has[..., None]
    return r


def nearest_valid_fill(real_bf, real_mask):
    """every voxel outside real_mask takes the value of the nearest voxel inside it"""
    idx = ndi.distance_transform_edt(~real_mask, return_indices=True, return_distances=False)
    if real_bf.ndim == 4:
        return np.stack([real_bf[..., c][tuple(idx)] for c in range(real_bf.shape[-1])], -1)
    return real_bf[tuple(idx)]


def export_brain(net, D, S, out, device="cuda", cache_out=None):
    """write the product set of prepared brain D (LINC ID S) to out/S/; returns (summary row, section-normal record or None)"""
    t0 = time.time()
    rg = deploy.regions(D, "full")
    brain, zf, Aff = rg["brain"], rg["z_f"], rg["affine"]
    x = np.asanyarray(nib.load(f"{D}/coarse/in.nii.gz").dataobj).astype(np.float32)[None]
    raw = deploy.predict(net, x, device) * brain
    grey = np.asanyarray(nib.load(f"{D}/fine/bf.nii.gz").dataobj).astype(np.float32)
    synth, w, completed = deploy.join(raw, grey, rg)
    bf_real = (grey * (w > 0) * brain).astype(np.float32)
    rec = json.load(open(f"{D}/seam.json"))
    ydep = rg["ydep"]; yy = np.rint(ydep)
    sd = (((ydep - zf) * deploy.ISO) * brain).astype(np.float32)
    rm = rg["real_mask"]
    pf = (np.asanyarray(nib.load(f"{D}/fine/photo_fraction.nii.gz").dataobj) * brain).astype(np.float32)
    prov = (w * pf).astype(np.float32)
    synth_mask = (brain & ~(rm & (yy >= zf))).astype(np.uint8)
    od = f"{out}/{S}"; os.makedirs(od, exist_ok=True)
    sv = lambda arr, nm, dt: nib.save(nib.Nifti1Image(np.ascontiguousarray(arr.astype(dt)), Aff), f"{od}/{S}_{nm}.nii.gz")  # noqa: E731
    for arr, nm, dt in ((synth, "bf_synth", np.float32), (completed, "bf_completed", np.float32), (prov, "provenance", np.float32),
                        (synth_mask, "synth_mask", np.uint8), (bf_real, "bf_real", np.float32), (w, "seam_ramp", np.float32),
                        (w, "seam_ramp_real", np.float32), (sd, "seam_depth", np.float32), (pf, "photo_fraction", np.float32)):
        sv(arr, nm, dt)
    chroma_row = None
    rgbp = f"{D}/fine/bf_rgb.nii.gz"
    if os.path.exists(rgbp):
        real_rgb = load_real_rgb(rgbp, rm & brain)
        ph = rm & brain & (real_rgb.sum(-1) > 0)
        lum = real_rgb.mean(-1)
        chroma = (real_rgb[ph] / np.maximum(lum[ph, None], 1e-3)).mean(0); chroma = chroma / chroma.mean()
        (sl, ic), *_ = np.linalg.lstsq(np.c_[grey[ph], np.ones(ph.sum())], lum[ph], rcond=None)
        fill = (sl * synth + ic)[..., None] * chroma[None, None, None, :]
        rf = np.stack([nearest_valid_fill(real_rgb[..., c], rm) for c in range(3)], -1)
        comp_rgb = (w[..., None] * rf + (1 - w[..., None]) * fill) * brain[..., None]
        nib.save(nib.Nifti1Image(np.ascontiguousarray(np.clip(comp_rgb, 0, 1).astype(np.float32)), Aff), f"{od}/{S}_bf_completed_rgb.nii.gz")
        chroma_row = dict(chroma=[round(float(v), 4) for v in chroma], grey_to_luma=[round(float(sl), 4), round(float(ic), 4)])
        json.dump(chroma_row, open(f"{od}/{S}_chroma.json", "w"), indent=1)
    shutil.copy2(f"{D}/coarse/in.nii.gz", f"{od}/{S}_b0_input_0p5.nii.gz")
    normal = None
    if cache_out:
        cd = f"{cache_out}/{S}/deploy"; os.makedirs(cd, exist_ok=True)
        nib.save(nib.Nifti1Image(brain.astype(np.uint8), Aff), f"{cd}/brain.nii.gz")
        nib.save(nib.Nifti1Image((rm & brain).astype(np.uint8), Aff), f"{cd}/real_mask.nii.gz")
        ax = int(np.argmax(np.abs(Aff[:3, :3][1]))); ap_dir = Aff[:3, ax] / np.linalg.norm(Aff[:3, ax])
        gw = np.linalg.solve(Aff[:3, :3].T, np.asarray(rec["seam"]["y_dep"]["g"], float)); gw = gw / np.linalg.norm(gw)
        normal = dict(normal=[float(v) for v in ap_dir], seam_normal_world=[round(float(v), 6) for v in gw],
                      note="the products are not rotated to the sections: normal = the grid's anterior-posterior axis; "
                           "seam_normal_world = the section normal (pointing rostrally)")
    chk = dict(z_f=int(zf), synth_frac=float(synth_mask.sum() / max(brain.sum(), 1)),
               provenance_le_photo_fraction=bool(np.all(prov <= pf + 1e-6)),
               seam_depth_nonneg_exactly_rostral=bool(np.all((sd[brain] >= 0) == (ydep[brain] >= zf))),
               voxel_mm=[round(float(v), 5) for v in np.sqrt((Aff[:3, :3] ** 2).sum(0))])
    row = dict(chk, chroma=chroma_row, mlo=rec["mlo"], mhi=rec["mhi"], wall_s=round(time.time() - t0, 1),
               b0_input=dict(file=f"{S}_b0_input_0p5.nii.gz", md5=md5(f"{od}/{S}_b0_input_0p5.nii.gz")))
    return row, normal
