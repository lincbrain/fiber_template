#!/usr/bin/env python3
"""The settings of the blockface reconstruction, as one record (paper: Supplementary Section "Blockface reconstruction").

    python code/figures/paper_numbers/blockface_params_r073.py

You need
  this repository only: settings/blockface.yaml (reconstruction, volume_segmentation), settings/tirl/camera_jumps.yml,
  settings/registration.yaml, settings/brains/<ID>.yaml, settings/specimen_labels.json, and the defaults of
  code/blockface/destep.py detect_steps and code/blockface/prepare.py filter_components_3d.
What it does
  It copies the settings the paper's volumes were made with (pyramid levels, the curated section step, the crop, the camera-jump
  registration's stages and each brain's resolution, the demons schedule, the 3D-mask component threshold, the illumination field, the de-step
  and color settings, the tube rules, the 0.15 mm registration copy) and counts each brain's listed camera jumps.
  The camera jumps are registered at each brain's own photograph pixel: code/blockface/jumps.py tirl_jumps gives both images
  the brain's reconstruction vx (settings/brains/<ID>.yaml reconstruction_spacing_um, else vx, as code/blockface/subject.py),
  over the resolution in settings/tirl/camera_jumps.yml; the record gives that per brain with jumps (checked in jumps.py's source).
  (Until 12 Nov 2026 it gave that file's 0.034 mm, which no jump ran at: the paper's runs of MR263, the ones kept, read 0.02.)
You get
  $LINC_NUMBERS/blockface_params_r073.json.

Ported from the analysis project's paper/r073_l1/blockface_params_r073.py, which read the same files of this repository.
"""
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
from shared import provenance  # noqa: E402
from blockface import destep, prepare  # noqa: E402

LABELS = REPO / "settings" / "specimen_labels.json"


def find(o, key):
    """The first value of `key` in a nested YAML structure."""
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key:
                return v
            r = find(v, key)
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r = find(v, key)
            if r is not None:
                return r
    return None


def main():
    bf_p = REPO / "settings/blockface.yaml"; bf = yaml.safe_load(bf_p.read_text())
    tj_p = REPO / "settings/tirl/camera_jumps.yml"; tj = yaml.safe_load(tj_p.read_text())
    rg_p = REPO / "settings/registration.yaml"; rg = yaml.safe_load(rg_p.read_text())
    R = bf["reconstruction"]; V = bf["volume_segmentation"]; D = R["demons"]; T = R["tube_inpainting"]
    ds = {k: v.default for k, v in inspect.signature(destep.detect_steps).parameters.items() if v.default is not inspect.Parameter.empty}
    f3 = float(inspect.signature(prepare.filter_components_3d).parameters["frac"].default)   # x the largest component
    lab = json.loads(LABELS.read_text())["labels"]
    jumps, cf, ins = {}, {}, [bf_p, tj_p, rg_p, LABELS]
    jp = REPO / "code/blockface/jumps.py"
    assert jp.read_text().count('["resolution"] = cfg.vx / 1000') == 2, "jumps.py no longer registers at the brain's vx: update this record"
    res = {}                                                         # mm: the pixel each brain's jumps are registered at
    for p in sorted((REPO / "settings/brains").glob("M*.yaml")):
        c = yaml.safe_load(p.read_text()); S = c.get("subject", p.stem)
        if S in lab and "jumps" in c:
            jumps[S] = len(c.get("jumps") or []); cf[S] = c.get("curation_f"); ins.append(p)
            if jumps[S]:
                rs = c.get("reconstruction_spacing_um") or {}
                res[S] = round(float(rs.get("vx", c["vx"])) / 1000, 6)
    assert len(jumps) == 21, len(jumps)
    jv = np.array(list(jumps.values()), float)
    shading = {S: str(yaml.safe_load((REPO / f"settings/brains/{S}.yaml").read_text()).get("shading", "")) for S in jumps}
    out = dict(
        levels=dict(final_level=R["level"], view_level=R["view_level"], reduction=dict(L1=2, L2=4, L4=16)),
        curation_z_target_um=R["curation_z_target_um"], curation_f=cf,
        crop_margin_mm=R["crop_margin_mm"], crop_multiple=R["crop_multiple"],
        camera_jumps=dict(stages=find(tj, "stages"),
                          resolution_mm=dict(per_brain=res, values=sorted(set(res.values())),
                                             rule="each brain's photographs at full resolution, at their own pixel size (the reconstruction vx)"),
                          per_brain=jumps,
                          summary=dict(min=float(jv.min()), max=float(jv.max()), median=float(np.median(jv)), total=int(jv.sum()),
                                       brains_with_jumps=int((jv > 0).sum()))),
        demons=dict(shrink=D["shrink"], sigma=D["sigma"], iterations=D["iters"], field_sd=D["dev"], mask_dilation_vox_L4=D["dilate"]),
        mask3d_component_min_fraction=f3,
        illumination=dict(poly_degree=2, zref=R["zref"], zref_brains=[S for S in jumps if "zref" in shading[S]]),
        destep=dict(window_sections=ds["win"], ratio=ds["rel"], abs_jump=ds["absjump"], edge_sections=ds["edge"], harmonize_mm=R["harmonize_mm"]),
        gain_clip=R["gain_clip"], w_lum=R["w_lum"],
        tubes=dict(left_tolerance_vox=V["tubes_left_tolerance_vox"], min_voxels=V["tubes_min_voxels"], halo_frac=T["halo_frac"],
                   dilate_vox=T["dilate"], margin_vox=T["margin"], feather_band_vox=T["feather_band"]),
        registration_iso_mm=rg["to_own_mri"]["iso_mm"],
        note="settings of record from the release's configs (values as used for the paper's volumes); counts per brain from configs/subjects")
    print(json.dumps({k: v for k, v in out.items() if k != "curation_f"}, indent=1, default=str))
    ins += [REPO / "code/blockface/destep.py", REPO / "code/blockface/prepare.py", jp]
    provenance.write_numbers("blockface_params_r073", out, inputs=ins, md5=False, cmd="python code/figures/paper_numbers/blockface_params_r073.py")


if __name__ == "__main__":
    main()
