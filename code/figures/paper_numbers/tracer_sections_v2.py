#!/usr/bin/env python3
"""Transform each charted histology section to the template on its own, without interpolation (Fig. 10's sections view; S2), to
show what was drawn before any interpolation between sections.

You need (settings/internal/products_mri.yaml, settings/internal/products_figures.yaml)
  push_tracer_r058d: the pushed sheets of record per class; tracer_v2_bfnative: the group sheets; numbers_of_record: the
  blockface -> b0 registrations of record (r1_current_2026-10-03.json); $LINC_DATA/sub-<ID>/xfm/b0-native__to__mri-template;
  for MF191 the blockface template link and its member warp (bft_pull_map, bft_member_warps, mf8_rigid).

What it does, per case
  1. transforms each class sheet and group sheet with the resampling rule of record (crop, anti-aliasing, one linear resample);
  2. transforms a section index (each charted slice filled with its section number) by nearest neighbor;
  3. fits each section's plane and measures the spacing between consecutive sections; lists every footprint voxel with its section.

You get
  $LINC_WORK/tracer_sections_v2<tag>/<ID>/ (<tag> = settings/labels.yaml template_chain_tag): <ID>_<label>_mri-template_sections_{density,footprint}.nii.gz,
  <ID>_section_index_mri-template.nii.gz, <ID>_sections.json, <ID>_sections_points.csv; and $LINC_NUMBERS/tracer_sections_v2<tag>.json
  (the four cases)

    python code/figures/paper_numbers/tracer_sections_v2.py [--cases MR243 ...]
"""
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
from shared import paths, provenance, xfm  # noqa: E402
from labels import carry as CA  # noqa: E402
from labels.labels import config  # noqa: E402
from blockface.registration import xfmtools as X  # noqa: E402

NAME = "tracer_sections_v2" + paths.template_chain_tag()
GROUPS = {"Bundles": ["Fiber_light_bundle", "Fiber_moderate_bundle", "Fiber_dense_bundle", "Bundle"],
          "Terminals": ["Terminal_light", "Terminal_moderate", "Terminal_dense"], "Injection_site": ["Injection_site"], "Fibers": ["Fibers"]}
from labels import route as RT  # noqa: E402
ROUTE = {S: ("A" if r == "blockface" else "B") for S, r in RT.routes().items()}   # A = blockface-based, B = MRI-based (settings/labels.yaml template_routes)
NOTE = ("a slice holding pieces of two sections is indexed by the lower section number Spacing = |(centroid_k - centroid_{k-1}) . mean normal|; "
        "section numbers increase toward the posterior (decreasing template y).")
MF8 = "MF191"
# the record's method text, fixed (it was this script's docstring when the record was made; the docstring may change)
PROTOCOL = (
    "Per-section tracer labels in the MRI template, WITHOUT any interpolation between sections (paper: Fig. 10's sections panel; the\n"
    "placement record's `sections` rows; Supplementary S2): the sheets of record (products_mri `push_tracer_r058d`: one blockface slice per\n"
    'charted section, per grade; `tracer_v2_bfnative`: the group unions Bundles / Terminals / Injection_site / Fibers) transformed by the\n'
    'registrations of record (MR243 / MR252 / MR256, the MRI-based alignment: the blockface -> b0 registration of record named in the R1 manifest of\n'
    '`numbers_of_record`/r1_current_2026-10-03.json, then the b0 -> template entry $LINC_DATA/sub-<S>/xfm/b0-native__to__mri-template;\n'
    "MF191, the blockface-based alignment: the blockface-template -> MRI-template pull map, MF8's member warp and the inverse of its rigid placement, products_figures\n"
    '`bft_pull_map`, `bft_member_warps`, `mf8_rigid`), with the resampling rule of record (linc/histology/carry.py: crop, per-axis anti-aliasing,\n'
    'one linear resample; configs/labels.yaml carry).\n'
    'Outputs, $LINC_WORK/tracer_sections_v2/<S>/ (what the released volumes `tracer_sections_v2` hold):\n'
    '  <S>_<label>_mri-template_sections_density.nii.gz   float 0-1: the transformed fraction of the sheet (a section = a ~0.25-0.4 mm slab)\n'
    '  <S>_<label>_mri-template_sections_footprint.nii.gz uint8: density >= foot\n'
    "  <S>_section_index_mri-template.nii.gz                uint16: the section number of record at every template voxel a charted section's\n"
    '      slice lands on (nearest neighbor through the same chain; the slice plane filled, then cut to the template brain). A slice that\n'
    '      holds parts of two sections (a hemisphere part of each) is indexed by the lower number; the JSON lists these slices\n'
    '  <S>_sections.json                                     per charted section: number, blockface slice, voxels in the template, plane centroid\n'
    '      (mm), unit normal (PCA), spacing to the previous section along the mean normal (mm)\n'
    '  <S>_sections_points.csv                               x_mm,y_mm,z_mm,group,grade,section: every footprint voxel of every grade (world RAS,\n'
    '      MF191 unmirrored), the section from the index volume (nearest filled)\n'
    'and the record $LINC_NUMBERS/tracer_sections_v2_h9.json (the four cases), as the analysis wrote\n'
    'it (per case: route (A = blockface-based, B = MRI-based alignment), links, per-label counts, sections, spacing: the absolute spacing between consecutive charted sections along their\n'
    'mean normal, median / p90 / max, point rows).\n'
)


def placeholders(d):
    """A chain link's description with its paths written through the roots ($LINC_DATA/..., $LINC_PRODUCTS/...), so the record holds
    no machine path (shared.paths.rerooted reads them back)."""
    roots = [("$LINC_DATA", str(paths.data_root())), ("$LINC_TEMPLATE", str(paths.template_root())), ("$LINC_PRODUCTS", str(paths.products_root()))]
    out = {}
    for k, v in d.items():
        if isinstance(v, str) and v.startswith("/"):
            for var, root in roots:
                if v == root or v.startswith(root.rstrip("/") + "/"):
                    v = var + v[len(root.rstrip("/")):]; break
        out[k] = v
    return out


def carrier(S, P2, P3, ref, slab):
    """A function (moving NIfTI path, order, tmpdir) -> array on the template grid, and the route's description."""
    refimg = nib.load(str(ref)); shp, A = refimg.shape[:3], refimg.affine
    if ROUTE[S] == "B":
        man = json.load(open(P2["numbers_of_record"] / "r1_current_2026-10-03.json")); B = (man.get("data") or man)["brains"]
        assert B[S]["qc"] == "pass", (S, B[S]["qc"])
        chain = xfm.Chain([xfm.Entry(paths.rerooted(B[S]["registry"])),
                           xfm.Entry(paths.data_root() / f"sub-{S}" / "xfm" / "b0-native__to__mri-template")], strict=True)

        def go(src, order, tmpdir):
            out = tmpdir / f"_carry_{os.getpid()}.nii.gz"
            xfm.apply(chain, str(src), str(ref), str(out), order=order)
            a = np.asarray(nib.load(str(out)).dataobj, np.float32); os.remove(out)
            return a
        return go, [str(placeholders(e)) for e in chain.describe()]
    f, links = RT.blockface_pull(S, P3)

    def go(src, order, tmpdir):
        img = nib.load(str(src)); V = np.asarray(img.dataobj, np.float32)
        p = np.zeros(shp, np.float32)
        for z0 in range(0, shp[2], slab):
            z1 = min(shp[2], z0 + slab)
            pts = f(X.grid_world(shp, A, z0, z1))
            p[:, :, z0:z1] = X.sample(V, img.affine, pts, order=order).reshape(shp[0], shp[1], z1 - z0)
        return p
    return go, links


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="+", default=list(ROUTE)); a = ap.parse_args()
    P2, P3 = paths.products("products_mri"), paths.products("products_figures")
    C = config()["carry"]; TV, FOOT, MARGIN, SLAB = float(C["template_voxel_mm"]), float(C["foot"]), float(C["crop_margin_mm"]), int(C["slab"])
    ref, brain_p = paths.template_root() / C["ref"], paths.template_root() / C["brain"]
    SHEETS, UNION = P2["push_tracer_r058d"], P2["tracer_v2_bfnative"]; OUT = paths.work_root() / NAME
    refimg = nib.load(str(ref)); shp, A = refimg.shape[:3], refimg.affine
    brain = np.asanyarray(nib.load(str(brain_p)).dataobj) > 0
    ijk = np.indices(shp).reshape(3, -1).T
    recs = {}
    inputs = [ref, brain_p]
    for S in a.cases:
        od = OUT / S; od.mkdir(parents=True, exist_ok=True)
        go, links = carrier(S, P2, P3, ref, SLAB)
        crec = dict(route=ROUTE[S], links=links, labels={})
        # 1. the sheets per group and per grade
        files = {g: UNION / S / f"{S}_{g}_bfnative_L2.nii.gz" for g in GROUPS}
        for g, cls in GROUPS.items():
            for c in cls:
                files[c] = SHEETS / S / f"{S}_{c}_bfnative_L2.nii.gz"
        dens = {}
        for lab, f in files.items():
            if not f.exists():
                continue
            inputs.append(f)
            img = nib.load(str(f)); c = CA.crop_sheet(img, MARGIN, TV)
            if c is None:
                continue
            sub, aff, z, nsec, nvox, half = c
            sm = gaussian_filter(sub, CA.aa_sigma(z, TV))
            tmp = od / f"_tmp_{lab}.nii.gz"; nib.save(nib.Nifti1Image(sm.astype(np.float32), aff), str(tmp))
            p = np.clip(go(tmp, 1, od), 0, 1); os.remove(tmp)
            nib.save(nib.Nifti1Image(p.astype(np.float32), A), str(od / f"{S}_{lab}_mri-template_sections_density.nii.gz"))
            nib.save(nib.Nifti1Image((p >= FOOT).astype(np.uint8), A), str(od / f"{S}_{lab}_mri-template_sections_footprint.nii.gz"))
            dens[lab] = p
            crec["labels"][lab] = dict(sections=nsec, voxels_bf=nvox, widened_half_slices=half, voxels_template_footprint=int((p >= FOOT).sum()),
                                       density_sum_mm3=round(float(p.sum()) * TV ** 3, 3))
            print(S, lab, crec["labels"][lab], flush=True)
        # 2. the section index: every charted section's blockface slice, the plane filled with its number, nearest neighbor
        pjp = SHEETS / S / f"push_tracer_{S}.json"; pj = json.load(open(pjp)); inputs.append(pjp)
        pieces = pj["pieces"] if isinstance(pj, dict) and "pieces" in pj else pj
        sheet0 = nib.load(str(files["Bundles"] if files["Bundles"].exists() else next(v for v in files.values() if v.exists())))
        idxv = np.zeros(sheet0.shape[:3], np.float32)
        sec_of_slice = {}
        for pc in pieces:
            n = int(re.search(r"_s(\d+)", pc["section"]).group(1))
            j = int(pc["bf_idx"])
            if 0 <= j < idxv.shape[1] and any(v.get("px_L2", 0) > 0 for v in pc.get("classes", {}).values()):
                sec_of_slice.setdefault(j, set()).add(n)
        multi = {int(j): sorted(ns) for j, ns in sec_of_slice.items() if len(ns) > 1}     # pieces of two sections on one slice
        for j, ns in sec_of_slice.items():
            idxv[:, j, :] = min(ns)                                              # the index volume carries the lower section number
        tmp = od / "_tmp_index.nii.gz"; nib.save(nib.Nifti1Image(idxv, sheet0.affine), str(tmp))
        ix = np.rint(go(tmp, 0, od)).astype(np.uint16); os.remove(tmp)
        ix[~brain] = 0
        nib.save(nib.Nifti1Image(ix, A), str(od / f"{S}_section_index_mri-template.nii.gz"))
        secs = []
        for n in sorted(set(int(v) for v in np.unique(ix) if v > 0)):
            pts = ijk[ix.ravel() == n] @ A[:3, :3].T + A[:3, 3]
            cen = pts.mean(0); u, s_, vt = np.linalg.svd(pts - cen, full_matrices=False); normal = vt[2]
            if normal[1] < 0:
                normal = -normal
            js = [j for j, ns in sec_of_slice.items() if min(ns) == n]
            secs.append(dict(section=n, bf_slice=js[0], also_on_slice=sorted(set(sec_of_slice[js[0]]) - {n}), voxels=int(len(pts)),
                             centroid_mm=[round(float(x), 3) for x in cen], normal=[round(float(x), 4) for x in normal]))
        mean_n = np.mean([s["normal"] for s in secs], 0); mean_n /= np.linalg.norm(mean_n)
        for k, s in enumerate(secs):
            s["spacing_to_previous_mm"] = round(abs(float(np.dot(np.array(s["centroid_mm"]) - np.array(secs[k - 1]["centroid_mm"]), mean_n))), 3) if k else None
        sp = [s["spacing_to_previous_mm"] for s in secs[1:]]
        summary = dict(median=round(float(np.median(sp)), 3), p90=round(float(np.percentile(sp, 90)), 3), max=round(float(np.max(sp)), 3)) if sp else None
        json.dump(dict(subject=S, mean_normal=[round(float(x), 4) for x in mean_n], sections=secs,
                       slices_with_two_sections=multi, note=NOTE, spacing_summary_mm=summary),
                  open(od / f"{S}_sections.json", "w"), indent=1)
        # 3. points: every footprint voxel of every grade, the section from the nearest filled index
        _, near = ndi.distance_transform_edt(ix == 0, return_indices=True)
        ixf = ix[near[0], near[1], near[2]]
        with open(od / f"{S}_sections_points.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["x_mm", "y_mm", "z_mm", "group", "grade", "section"])
            nrow = 0
            for g, cls in GROUPS.items():
                for c in cls:
                    if c not in dens:
                        continue
                    m = dens[c] >= FOOT
                    pts = ijk[m.ravel()] @ A[:3, :3].T + A[:3, 3]
                    for (x, y, z), sec in zip(pts, ixf[m]):
                        w.writerow([f"{x:.3f}", f"{y:.3f}", f"{z:.3f}", g, c, int(sec)]); nrow += 1
        crec["sections"] = len(secs); crec["points_rows"] = nrow
        crec["section_spacing_mm"] = summary
        crec["spacing_note"] = "absolute spacing along the mean section normal; see <S>_sections.json"
        recs.setdefault(NAME, dict(protocol=PROTOCOL, template_voxel_mm=TV, foot=FOOT, cases={}))["cases"][S] = crec
        print(S, "sections", len(secs), "spacing", crec["section_spacing_mm"], "points", nrow, flush=True)
    cmd = " ".join(["python code/figures/paper_numbers/tracer_sections_v2.py"] + sys.argv[1:])
    for name, r in recs.items():                                              # one record, the four cases, as the analysis wrote it
        print(provenance.write_numbers(name, r, inputs=inputs, md5=False, cmd=cmd))


if __name__ == "__main__":
    main()
