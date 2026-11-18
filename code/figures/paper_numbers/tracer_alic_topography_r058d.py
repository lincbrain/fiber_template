#!/usr/bin/env python3
"""Where each tracer case's fiber bundles run inside the anterior limb of the internal capsule (ALIC), dorsoventrally, against its
injection site's prefrontal region and the published topography (ventral prefrontal fibers ventral, dorsal ones dorsal: Lehman et
al. 2011; Safadi et al. 2018). Results, tracer labels; Fig. 9 (c).

    python code/figures/paper_numbers/tracer_alic_topography_r058d.py [--labels <products key>] [--name <record>]

You need
  labels   the final tracer labels in the template (settings/internal/products_blockface.yaml `tracer_template_r058d`):
           <S>/<S>_<class>_mri-template_density.nii.gz; bundles = Fiber_{light,moderate,dense}_bundle + Bundle + Fiber_bundles + Unnamed_outline (the "Fibers" strokes
           left out, as in Fig. 9); both rules: density >= 0.1 (the figure's) and > 0.5 (solid).
  capsule  SARM level 6 label 50 (ic) in the template (`atlas_nmt`/SARM_level6_in_template.nii.gz); also + 0.5 mm (2 dilations), as
           Fig. 9 (c) draws it.
  region   CHARM in NMT v2 (CHARM_in_NMT_v2.0_sym, levels 1-6; key CHARM_key_all.txt), from your copy of the public NMT v2.0
           symmetric archive (https://afni.nimh.nih.gov/pub/dist/atlases/macaque/nmt/NMT_v2.0_sym.tgz): the image under
           settings/link.yaml inputs.atlas_reference, the key from settings/internal/products_blockface.yaml `nmt_label_keys` (the archive's
           tables_CHARM/ folder). Not redistributed here.
What it does
  The injection site (Injection_site, both rules) is transformed to NMT v2 by the template's registration of record
  (`atlas_nmt`/run; code/template/link/symreg.maps) and looked up in CHARM (nearest; the majority label per level). Measures, per case and
  rule, in the injection's hemisphere (x > 0, right; MF191 x < 0, left), template world y in [20, 24] mm (Fig. 9 c's three planes,
  anterior of the AC at y = 19.6 mm):
  n            bundle voxels inside the capsule
  rel_dv       the mean over those voxels of (z - zmin) / (zmax - zmin), zmin / zmax = the capsule's ventral and dorsal ends in the
               voxel's coronal plane (same side): 0 = the capsule's ventral end, 1 = its dorsal end
  dz_mm        the voxels' mean z minus the mean z of the capsule itself (same side and y range): < 0 ventral of the capsule's
               middle
  per_plane    the same at y = 20, 22, 24 mm (+- 0.75 mm, Fig. 9 c's slabs)
  The verdict rule (stated before the numbers): a site in CHARM's OFC or vmPFC groups (ventral prefrontal) is expected ventral
  (rel_dv < 0.5), a site in dorsal prefrontal, dorsal cingulate or medial motor areas (dlPFC, ACC dorsal, area 8, SMA/preSMA)
  dorsal (rel_dv > 0.5); and across cases the ventral-site bundles below the dorsal-site bundles. A case with no annotated site gets
  its position only.
You get
  $LINC_NUMBERS/tracer_alic_topography_r058d.json.

Ported from the analysis project's labels/tracer_alic_topography_r058.py (run there as --labels labels/tracer_template_r058d
--name tracer_alic_topography_r058d).
"""
import argparse
import json
import sys
from glob import glob
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
from shared import paths, provenance  # noqa: E402
from template.link import symreg  # noqa: E402
from template.link.common import inputs, product  # noqa: E402

NAMES = Path(__file__).resolve().parents[3] / "settings" / "specimen_labels.json"
IC_LABEL = 50
CASES = ["MR243", "MR252", "MR256", "MF191"]
SIDE = {"MR243": 1, "MR252": 1, "MR256": 1, "MF191": -1}          # injection hemisphere (RAS x sign); MF191 left (tube side)
RULES = {"footprint": lambda d: d >= 0.1, "solid": lambda d: d > 0.5}
RULE_TEXT = {"footprint": "density >= 0.1 (what Fig. 9 shows)", "solid": "density > 0.5"}
BUNDLES = ("Fiber_light_bundle", "Fiber_moderate_bundle", "Fiber_dense_bundle", "Bundle", "Fiber_bundles", "Unnamed_outline")
Y_RANGE, PLANES, HALF = (20.0, 24.0), (20.0, 22.0, 24.0), 0.75
VENTRAL_PFC = {"OFC", "med_OFC", "lat_OFC", "caudal_OFC", "vmPFC", "area_25", "area_32", "area_14", "area_10", "areas_11/13"}
DORSAL = {"dlPFC", "area_8A", "area_8B", "SMA/preSMA", "SMA", "preSMA", "MCC", "area_9", "area_46", "motor", "M1/PM"}
MIN_N = 20
PUBLISHED = ("ventral prefrontal (vmPFC, OFC) fibers ventral, dorsal prefrontal and cingulate fibers dorsal in the ALIC "
             "(Lehman et al. 2011; Safadi et al. 2018)")


def load(p):
    return np.asanyarray(nib.load(str(p)).dataobj)


def charm_key(path):
    k = {}
    for line in Path(path).read_text().splitlines():
        p = line.split("\t")
        if p and p[0].isdigit():
            k[int(p[0])] = p[1]
    return k


def site_region(LAB, S, rule, a2b, C, CA, key):
    f = LAB / S / f"{S}_Injection_site_mri-template_density.nii.gz"
    if not f.exists():
        return None
    im = nib.load(str(f)); m = rule(np.asanyarray(im.dataobj))
    P = np.argwhere(m) @ im.affine[:3, :3].T + im.affine[:3, 3]
    q = a2b(P); V = np.linalg.inv(CA); q = q @ V[:3, :3].T + V[:3, 3]
    levels = {}
    for L in range(C.shape[3]):
        lab = map_coordinates(C[..., L].astype(np.float32), q.T, order=0, cval=0).astype(int)
        u, c = np.unique(lab, return_counts=True); o = np.argsort(-c)
        levels[f"level_{L + 1}"] = [dict(label=key.get(int(u[k]), "outside CHARM cortex" if u[k] == 0 else int(u[k])),
                                         share=round(float(c[k] / len(lab)), 3)) for k in o[:3]]
    return dict(voxels=int(m.sum()), centroid_template_mm=[round(float(v), 2) for v in P.mean(0)], charm=levels)


def position(m, ic, A, side, yr):
    """Dorsoventral position of mask m inside capsule ic (injection side, world y in yr, closed), per coronal plane: rel_dv =
    (z - zmin) / (zmax - zmin), zmin/zmax = the capsule's ventral and dorsal ends in that coronal plane (the ALIC is an oblique band,
    so its ends in the plane, not in a vertical column, give the position along the capsule)."""
    X = A[0, 3] + A[0, 0] * np.arange(ic.shape[0]); Y = A[1, 3] + A[1, 1] * np.arange(ic.shape[1]); Z = A[2, 3] + A[2, 2] * np.arange(ic.shape[2])
    sx = np.sign(X) == side; sy = (Y >= yr[0] - 1e-6) & (Y <= yr[1] + 1e-6)
    icS = ic & sx[:, None, None] & sy[None, :, None]; b = m & icS; n = int(b.sum())
    if n == 0:
        return dict(n=0)
    i, j, k = np.nonzero(b); ii, jj, kk = np.nonzero(icS)
    zmin = np.full(ic.shape[1], np.nan); zmax = np.full(ic.shape[1], np.nan)
    for jp in np.unique(jj):
        zp = Z[kk[jj == jp]]; zmin[jp], zmax[jp] = zp.min(), zp.max()
    ext = zmax[j] - zmin[j]
    rel = (Z[k] - zmin[j]) / ext
    planes_ext = (zmax - zmin)[np.isfinite(zmax)]
    return dict(n=n, rel_dv=round(float(rel.mean()), 3), rel_dv_iqr=[round(float(v), 3) for v in np.percentile(rel, [25, 75])],
                share_ventral_half=round(float((rel < 0.5).mean()), 3), capsule_dv_extent_mm=round(float(np.median(planes_ext)), 2),
                z_mean_mm=round(float(Z[k].mean()), 2), capsule_z_mean_mm=round(float(Z[kk].mean()), 2),
                dz_mm=round(float(Z[k].mean() - Z[kk].mean()), 2),
                x_abs_mean_mm=round(float(np.abs(X[i]).mean()), 2), capsule_x_abs_mean_mm=round(float(np.abs(X[ii]).mean()), 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="tracer_template_r058d", help="label folder: a key of configs/products_blockface.yaml")
    ap.add_argument("--name", default="tracer_alic_topography_r058d", help="record name ($LINC_NUMBERS/<name>.json)")
    a = ap.parse_args()
    P = paths.products("products_blockface")
    LAB, SARM6, RUN = P[a.labels], P["atlas_nmt"] / "SARM_level6_in_template.nii.gz", P["atlas_nmt"] / "run"
    CHARM = product(inputs()["atlas_reference"]) / "NMT_v2.0_sym" / "NMT_v2.0_sym" / "CHARM_in_NMT_v2.0_sym.nii.gz"
    CHARM_KEY = P["nmt_label_keys"] / "CHARM_key_all.txt"
    names = json.loads(NAMES.read_text())["labels"]; key = charm_key(CHARM_KEY)
    sarm_img = nib.load(str(SARM6)); A = sarm_img.affine; ic0 = np.asanyarray(sarm_img.dataobj) == IC_LABEL
    caps = {"ic": ic0, "ic_plus_0p5mm": ndi.binary_dilation(ic0, iterations=2)}
    a2b, _ = symreg.maps(RUN)
    ci = nib.load(str(CHARM)); C = np.asanyarray(ci.dataobj); C = C.reshape(C.shape[:3] + (-1,))
    rec = dict(rules=RULE_TEXT, y_range_mm=list(Y_RANGE), planes_mm=list(PLANES), plane_half_mm=HALF, capsule=f"SARM level 6 label {IC_LABEL}",
               injection_side=SIDE, min_voxels_for_a_verdict=MIN_N, ventral_groups=sorted(VENTRAL_PFC), dorsal_groups=sorted(DORSAL),
               published_order=PUBLISHED, cases={})
    ins = [SARM6, RUN / "H.txt", RUN / "Hinv.txt", RUN / "F_ras_disp.nii.gz", RUN / "G_ras_disp.nii.gz", CHARM, CHARM_KEY, NAMES]
    for S in CASES:
        files = [f for c in BUNDLES for f in glob(str(LAB / S / f"{S}_{c}_mri-template_density.nii.gz"))]
        ins += [Path(f) for f in files]
        case = dict(label=names.get(S, S), side="right" if SIDE[S] > 0 else "left", site={}, alic={})
        for r, rule in RULES.items():
            case["site"][r] = site_region(LAB, S, rule, a2b, C, ci.affine, key)
            m = np.zeros(ic0.shape, bool)
            for f in files:
                m |= rule(load(f))
            case["alic"][r] = {cn: dict(range=position(m, cap, A, SIDE[S], Y_RANGE),
                                        per_plane={str(y): position(m, cap, A, SIDE[S], (y - HALF, y + HALF)) for y in PLANES})
                               for cn, cap in caps.items()}
        # verdict on the figure's rule and capsule (footprint, + 0.5 mm), then checked on the others
        site = case["site"]["footprint"]
        lv = {k: v[0]["label"] for k, v in site["charm"].items()} if site else {}
        grp = "ventral" if VENTRAL_PFC & set(lv.values()) else "dorsal" if DORSAL & set(lv.values()) else None
        pos = {f"{r}_{cn}": case["alic"][r][cn]["range"].get("rel_dv") for r in RULES for cn in caps}
        ns = {f"{r}_{cn}": case["alic"][r][cn]["range"].get("n", 0) for r in RULES for cn in caps}
        obs = {k: (None if ns[k] < MIN_N or v is None else "ventral" if v < 0.5 else "dorsal") for k, v in pos.items()}
        case["verdict"] = dict(site_group=grp, site_levels=lv, observed=obs, n=ns,
                               consistent={k: (None if grp is None or o is None else o == grp) for k, o in obs.items()})
        rec["cases"][S] = case
        print(S, case["label"], grp, lv.get("level_3"), lv.get("level_6"), {k: (pos[k], ns[k]) for k in pos})
    rec["labels"] = str(LAB.relative_to(paths.products_root()))       # the label folder, relative to the products (as in the record)
    provenance.write_numbers(a.name, rec, inputs=ins, md5=False,
                             cmd=" ".join(["python code/figures/paper_numbers/tracer_alic_topography_r058d.py"] + sys.argv[1:]))


if __name__ == "__main__":
    main()
