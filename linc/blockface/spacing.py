"""The calibrated spacing written into a reconstruction's headers: pixels untouched.

The paper's brains were reconstructed at their nominal pixel size and section step (configs/subjects/<ID>.yaml
`reconstruction_spacing_um`) and then given the spacing calibrated against each brain's own MRI (`vx`, `vz`; the pixel sizes and
section steps reported in Methods) by rescaling the headers. A pure world scaling about the origin is exact: in-plane world =
vx * (voxel + 0.5 + offset), with offsets in voxels, and along the sections y(j) = y0 - vz * j with y0 = vz * (photo count - 0.5).
A new (vx', vz') therefore maps every product's world by diag(fx, fz, fx), fx = vx'/vx, fz = vz'/vz: exactly what a rebuild at
(vx', vz') writes for these header quantities.

  OME-Zarr stores (<S>_*.ome.zarr, 02_*.ome.zarr, steps/, _work/): every dataset's scale (rows and columns x fx, sections x fz;
      the translations are in voxels and do not change);
  NIfTIs (<S>_blockface_L2, <S>_mask, <S>_holes, nii/, curated/, steps/): affine <- diag(fx, fz, fx, 1) @ affine.
A file already at the calibrated spacing (e.g. a hand edit of record that carries it) is left as is. For a new brain the two
spacings are equal and nothing changes.

Ported from the analysis project's code/respace_bf.py (the D9 calibration, 2 Oct 2026) and code/respace_vz.py (MF8, MF9, 26 Sep):
the header operations, unchanged; their backups and bookkeeping of the analysis project's curation tools are not part of the release.
"""
import glob
import json
import os

import numpy as np
import nibabel as nib

from linc.blockface import zarrio

NII_GLOBS = ["{S}_blockface_L2.nii.gz", "{S}_blockface_L4.nii.gz", "{S}_holes.nii.gz", "{S}_mask.nii.gz", "nii/*.nii.gz",
             "curated/{S}_*.nii.gz", "steps/*.nii.gz"]
STORE_GLOBS = ["*.ome.zarr", "_work/*.ome.zarr", "steps/*.ome.zarr"]
SKIP = ("03_", "04_", "05_", "06_")                                      # the histology trees (linc/histology)


def _files(root, S):
    stores = sorted(p for g in STORE_GLOBS for p in glob.glob(os.path.join(root, g))
                    if os.path.isdir(p) and not os.path.basename(p).startswith(SKIP))
    niis = sorted(set(p for g in NII_GLOBS for p in glob.glob(os.path.join(root, g.format(S=S))) if os.path.isfile(p)))
    return stores, niis


def _scales(store):
    return [float(x) for x in zarrio.read_group(store).attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][1]["scale"]]


def _nearer(step, a, b):
    """True when `step` is closer to an integer multiple of `a` than of `b` (both > 0)."""
    ra, rb = step / a, step / b
    return abs(ra - round(ra)) <= abs(rb - round(rb))


def _at_reconstruction_spacing(step_inplane, step_slice, cfg):
    """Does a grid with these steps (mm) carry the reconstruction spacing (True) or the calibrated one (False)?"""
    checks = []
    if abs(cfg.vz - cfg.vz_header) > 1e-9:
        checks.append(_nearer(step_slice * 1000.0, cfg.vz, cfg.vz_header))
    if abs(cfg.vx - cfg.vx_header) > 1e-9:
        checks.append(_nearer(step_inplane * 1000.0, cfg.vx, cfg.vx_header))
    return all(checks) if checks else False


def respace(cfg, log_path=None):
    """Rescale the headers of cfg's blockface products from (cfg.vx, cfg.vz) to (cfg.vx_header, cfg.vz_header). Returns the log."""
    import zarr
    S, root = cfg.subject, cfg.root
    vx0, vx1, vz0, vz1 = cfg.vx, cfg.vx_header, cfg.vz, cfg.vz_header
    fx, fz = vx1 / vx0, vz1 / vz0
    log = dict(subject=S, vx_um=[vx0, vx1], vz_um=[vz0, vz1], factors=[fx, fz, fx], stores={}, niftis={})
    if fx == 1.0 and fz == 1.0:
        print(f"{S}: reconstruction spacing = calibrated spacing: nothing to do")
        return log
    stores, niis = _files(root, S)
    Sm = np.diag([fx, fz, fx, 1.0])
    note = f"header-respaced (vx {vx0:g}->{vx1:g}, vz {vz0:g}->{vz1:g} um), voxels unchanged"
    for p in stores:
        s0 = _scales(p)
        if not _at_reconstruction_spacing(abs(s0[1]) / 1000.0, abs(s0[2]) / 1000.0, cfg):        # scales in um
            log["stores"][p] = "already at the calibrated spacing"; continue
        g = zarr.open_group(p, mode="r+")
        ms = g.attrs["multiscales"]
        for d in ms[0]["datasets"]:
            sc = d["coordinateTransformations"][1]["scale"]
            sc[1], sc[2], sc[3] = float(sc[1]) * fx, float(sc[2]) * fz, float(sc[3]) * fx
        ms[0]["description"] = str(ms[0].get("description", "")) + " | " + note
        g.attrs["multiscales"] = ms
        s1 = _scales(p)
        assert np.allclose(s1, [s0[0], s0[1] * fx, s0[2] * fz, s0[3] * fx], atol=1e-6), f"{p}: scales {s1} != scaled {s0}"
        log["stores"][p] = dict(scale_old=s0, scale_new=s1)
    for p in niis:
        im = nib.load(p)
        A_old = im.affine.copy()
        st = np.linalg.norm(A_old[:3, :3], axis=0)
        a, b, zax = min([(0, 1, 2), (0, 2, 1), (1, 2, 0)], key=lambda t: abs(st[t[0]] - st[t[1]]))   # in-plane pair, section axis
        if not _at_reconstruction_spacing(st[a], st[zax], cfg):
            log["niftis"][p] = "already at the calibrated spacing"; continue
        A_new = Sm @ A_old
        raw = np.asanyarray(im.dataobj.get_unscaled()) if hasattr(im.dataobj, "get_unscaled") else np.asanyarray(im.dataobj)
        h = im.header.copy()
        out = nib.Nifti1Image(raw, A_new, h)
        out.header.set_slope_inter(*im.header.get_slope_inter())
        out.set_qform(A_new, int(h["qform_code"]) or 1); out.set_sform(A_new, int(h["sform_code"]) or 1)
        tmp = p + ".tmp_respace.nii.gz"; nib.save(out, tmp); os.replace(tmp, p)
        chk = nib.load(p)                                                  # verify: same pixels, the scaled affine
        same = np.array_equal(np.asanyarray(chk.dataobj.get_unscaled()), raw) and chk.get_data_dtype() == im.get_data_dtype()
        assert same and np.allclose(chk.affine, A_new, atol=1e-5), f"{p}: verification failed"
        log["niftis"][p] = dict(diag_old=np.diag(A_old)[:3].tolist(), diag_new=np.diag(A_new)[:3].tolist(),
                                origin_old=A_old[:3, 3].tolist(), origin_new=A_new[:3, 3].tolist())
    json.dump(log, open(log_path or os.path.join(root, f"{S}_respace.json"), "w"), indent=1)
    n_s = sum(isinstance(v, dict) for v in log["stores"].values()); n_n = sum(isinstance(v, dict) for v in log["niftis"].values())
    print(f"{S}: vx {vx0:g}->{vx1:g}, vz {vz0:g}->{vz1:g} um: {n_s} store(s), {n_n} NIfTI(s) rescaled and verified")
    return log
