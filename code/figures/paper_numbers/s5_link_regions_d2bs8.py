#!/usr/bin/env python3
"""Where the registration between the two templates differs from its construction estimate (Results, "Building the blockface template
and registering it to the MRI template"; Fig. "link").

    python code/figures/paper_numbers/s5_link_regions_d2bs8.py

You need
  the registration of record and the construction estimate (settings/internal/products_mri.yaml `bftemplate_cv`: s5_direct_d2bs8/sym,
  s5_compose_d2bs8_r05_guard, s5_compare_d2bs8.json) and the MRI template's brain mask (settings/link.yaml `inputs.mri_mask`).
What it does
  At every 8th brain voxel of the MRI template, the distance in mm between the two estimates of the corresponding point in the blockface
  template, summarized in four regions of the MRI template, fixed by rule (RAS mm, the template's own brain mask):
    occipital_pole    points within 10 mm of the brain's most posterior point
    dorsal_surface    the remaining points within 2 mm of the brain's outer surface, dorsal to the median height of the brain's voxels
    other_surface     the remaining points within 2 mm of the surface
    interior          every other point
  Per region: points, median, 95th percentile, and the share of its points above the whole brain's 95th percentile. It stops if the
  whole brain's median and 95th percentile do not match s5_compare_d2bs8.json.
You get
  $LINC_NUMBERS/s5_link_regions_d2bs8.json.
"""
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
from shared import paths, provenance  # noqa: E402
from template.link.common import inputs, product  # noqa: E402
from blockface.registration import xfmtools as X  # noqa: E402

OCC_MM, SHELL_MM = 10.0, 2.0
# the record's own labels, verbatim (they are part of the record's data)
LINK = "direct symmetric link of record (bftemplate/cv/s5_direct_d2bs8/sym) vs the composition of the members' warps (bftemplate/cv/s5_compose_d2bs8_r05_guard)"
POINTS = "every 8th brain voxel of template/mri/linc_mri_brainmask, as mri/code/s5_compare.py"


def main():
    C = paths.products("products_mri")["bftemplate_cv"]
    CD, DR, REF, MASK = C / "s5_compose_d2bs8_r05_guard", C / "s5_direct_d2bs8" / "sym", C / "s5_compare_d2bs8.json", product(inputs()["mri_mask"])
    m = nib.load(str(MASK)); M = np.asanyarray(m.dataobj) > 0.5
    Y = X.grid_world(m.shape, m.affine)[M.reshape(-1)][::8]
    ijk = np.argwhere(M)[::8]
    assert np.allclose(ijk @ m.affine[:3, :3].T + m.affine[:3, 3], Y, atol=1e-4)
    D = nib.load(str(CD / "link_ras_disp.nii.gz")); xc = Y + X.sample(np.asanyarray(D.dataobj).astype(np.float32), D.affine, Y, order=1)
    Pst = np.loadtxt(DR / "link_linear_world.txt"); dd, da = X.read_fireants_field(str(DR / "link_warp.nii.gz"))
    xd = X.Map(L=np.linalg.inv(Pst), d=dd, d_affine=da)(Y)
    e = np.linalg.norm(xc - xd, axis=1)
    ref = json.loads(REF.read_text())["composition_vs_direct_sym_mm"]
    whole = dict(n=int(len(e)), p50=float(np.percentile(e, 50)), p95=float(np.percentile(e, 95)))
    assert abs(whole["p50"] - ref["p50"]) < 1e-6 and abs(whole["p95"] - ref["p95"]) < 1e-6, (whole, ref)
    zooms = np.abs(np.diag(m.affine)[:3])
    depth = ndi.distance_transform_edt(M, sampling=zooms)[tuple(ijk.T)]         # mm to the nearest voxel outside the brain
    yb, zb = Y[:, 1], Y[:, 2]
    yall = np.argwhere(M) @ m.affine[:3, :3].T + m.affine[:3, 3]
    occ = yb <= yall[:, 1].min() + OCC_MM
    shell = depth <= SHELL_MM
    dors = ~occ & shell & (zb >= np.median(yall[:, 2]))
    oth = ~occ & shell & ~dors
    inner = ~occ & ~shell
    regions = {}
    for name, sel in (("occipital_pole", occ), ("dorsal_surface", dors), ("other_surface", oth), ("interior", inner)):
        v = e[sel]
        regions[name] = dict(n_points=int(sel.sum()), share_of_points=round(float(sel.mean()), 4), median_mm=round(float(np.median(v)), 3),
                             p95_mm=round(float(np.percentile(v, 95)), 3), share_above_brain_p95=round(float((v > whole["p95"]).mean()), 4))
    assert sum(r["n_points"] for r in regions.values()) == len(e)
    out = dict(link=LINK, points=POINTS, whole_brain=dict(n_points=whole["n"], median_mm=round(whole["p50"], 3), p95_mm=round(whole["p95"], 3)),
               rule=dict(occipital_pole_mm=OCC_MM, surface_shell_mm=SHELL_MM, dorsal="z >= median z of the brain voxels", frame="MRI template RAS mm (y anterior, z dorsal)"),
               regions=regions)
    provenance.write_numbers("s5_link_regions_d2bs8", out, inputs=[MASK, CD / "link_ras_disp.nii.gz", DR / "link_linear_world.txt", DR / "link_warp.nii.gz", REF],
                             md5=False, cmd="python code/figures/paper_numbers/s5_link_regions_d2bs8.py")


if __name__ == "__main__":
    main()
