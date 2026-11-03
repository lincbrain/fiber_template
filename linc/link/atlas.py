#!/usr/bin/env python3
"""Mappings between the MRI template and three published templates: NMT v2 (symmetric), INIA19 and CIVM-R (paper: Methods,
"Relation to published templates", first part).

Port of the analysis project's mri/code/atlas_v2.py (outputs of record mri/tc/atlas_v2/<atlas>/; numbers atlas_final_v2_<atlas>.json).
  affine  S = the log-Euclidean mean of two ANTs affines (rigid + affine, MI): fwd (fixed = our template channel, moving = atlas) and
          bwd (fixed = atlas, moving = our template channel); S maps atlas world -> template world (RAS).
  solve   the symmetric engine (linc/link/symreg.py): A = our template channel (T2; b0 for CIVM-R), B = the atlas image, start S,
          0.25 mm, warp smoothing 1.25 voxels, no landmarks.
  score   brain-mask Dice and the outline check of each mask carried across (both directions); folds and the symmetric round trip (the
          engine's metrics.json); Jacobian range of both maps inside their brains; CIVM-R: pointwise agreement with the one-way direct
          b0 registration inside the CIVM-R brain; NMT: SARM level 6 carried into the template (labels for figures).

    python -m linc.link.atlas affine <atlas> --out DIR      (CPU; ANTs, version-checked)
    python -m linc.link.atlas solve  <atlas> --out DIR      (GPU; the symmetric engine)
    python -m linc.link.atlas score  <atlas> --out DIR      (CPU; DIR/<atlas>/score.json and $LINC_NUMBERS/atlas_final_v2_<atlas>.json)
<atlas> = NMT_v2.0_sym | INIA19 | CIVM-R; DIR defaults to $LINC_WORK/atlas_v2. ONPRC18 and CIVM-R images are read, never written or
copied (their licenses do not allow redistribution); only transforms and numbers are products.
Differences from the analysis script: `cmd` (printed the GPU command) is replaced by `solve` (runs it); paths come from configs/link.yaml.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.linalg import expm, logm

from .. import paths, provenance, xfm
from ..registration import xfmtools as X
from . import qc
from .common import chunked, config, inputs, product


def spec(name):
    """Image, mask, our channel (and SARM for NMT) of a published template, from configs/link.yaml."""
    I = inputs(); a = dict(config()["atlas"]["atlases"][name])
    fmt = dict(ref=str(product(I["atlas_reference"])), v1=str(product(I["atlas_v1_xfm"])))
    for k in ("img", "msk", "sarm"):
        if k in a:
            a[k] = Path(a[k].format(**fmt))
    return a


def template_channel(name):
    I = inputs()
    return product(I["mri_t2"] if spec(name)["channel"] == "T2" else I["mri_b0"])


def tmask():
    return product(inputs()["mri_mask"])


def outdir(out, name):
    return Path(out or paths.work_root() / "atlas_v2") / name


def affine(name, out=None):
    from .ants import affine_init
    a = spec(name); d = outdir(out, name); (d / "fwd").mkdir(parents=True, exist_ok=True); (d / "bwd").mkdir(parents=True, exist_ok=True)
    tch = template_channel(name)
    affine_init(str(tch), str(a["img"]), str(tmask()), str(a["msk"]), str(d / "fwd" / "affine_"))   # moving atlas -> fixed template
    affine_init(str(a["img"]), str(tch), str(a["msk"]), str(tmask()), str(d / "bwd" / "affine_"))   # moving template -> fixed atlas
    return average_affines(d)


def average_affines(d):
    """S (atlas world -> template world) = exp(mean(log(inv(fwd)), log(bwd))) and prep.json, from d/fwd and d/bwd."""
    d = Path(d)
    S1 = np.linalg.inv(xfm.read_itk_affine(d / "fwd" / "affine_0GenericAffine.mat")); S2 = xfm.read_itk_affine(d / "bwd" / "affine_0GenericAffine.mat")
    S = np.real(expm(0.5 * (np.real(logm(S1)) + np.real(logm(S2))))); np.savetxt(d / "S_ras_atlas_to_template.txt", S)
    sv = lambda M: np.linalg.svd(M[:3, :3], compute_uv=False).round(4).tolist()  # noqa: E731
    json.dump(dict(S1_scales=sv(S1), S2_scales=sv(S2), S_scales=sv(S)), open(d / "prep.json", "w"), indent=1); print(d.name, "S scales", sv(S))
    return S


def solve(name, out=None, device="cuda:0"):
    from . import symreg
    a = spec(name); d = outdir(out, name); e = config()["atlas"]["engine"]
    symreg.main([str(template_channel(name)), str(tmask()), str(a["img"]), str(a["msk"]), str(d / "S_ras_atlas_to_template.txt"), str(d / "run"),
                 "--res", str(e["res"]), "--device", device, "--smooth-warp-sigma", str(e["smooth_warp_sigma"]),
                 "--channel-chunk", str(e["channel_chunk"]), "--max-gpu-frac", str(e["max_gpu_frac"])])


def score(name, out=None, run=None, numbers=True):
    """score.json of one mapping (see the module docstring); `run` overrides the engine folder (default DIR/<atlas>/run)."""
    from scipy.ndimage import map_coordinates
    from . import symreg
    sc = config()["atlas"]["score"]
    a = spec(name); d = outdir(out, name); run = Path(run) if run else d / "run"
    prep = Path(run).parent / "prep.json" if not (d / "prep.json").exists() else d / "prep.json"
    a2b, b2a = symreg.maps(run)                              # A = template, B = atlas: a2b pulls template point -> atlas point
    met = json.load(open(run / "metrics.json"))
    ti = nib.load(str(tmask())); T = np.asanyarray(ti.dataobj) > 0.5; ai = nib.load(str(a["msk"])); Am = np.asanyarray(ai.dataobj).reshape(ai.shape[:3]) > 0.5
    ch = lambda f, P: chunked(f, P, 2_000_000)  # noqa: E731

    def carry(mask, maff, shape, gaff, pull):
        P = X.grid_world(shape, gaff); Q = ch(pull, P); V = np.linalg.inv(maff)
        return (map_coordinates(mask.astype(np.float32), (Q @ V[:3, :3].T + V[:3, 3]).T, order=1, cval=0) > 0.5).reshape(shape)
    at = carry(Am, ai.affine, T.shape, ti.affine, a2b)        # atlas mask in template space
    ta = carry(T, ti.affine, Am.shape, ai.affine, b2a)        # template mask in atlas space
    dice = lambda x, y: round(float(2 * (x & y).sum() / (x.sum() + y.sum())), 4)  # noqa: E731
    ol1 = qc.outline_regions(at, T, ti.affine); ol2 = qc.outline_regions(ta, Am, ai.affine)
    s1, w1 = qc.outline_gate(ol1); s2, w2 = qc.outline_gate(ol2)

    def detj(pull, mask, aff, every=sc["detj_every"], h=sc["detj_h_mm"]):
        P = (np.argwhere(mask)[::every]) @ aff[:3, :3].T + aff[:3, 3]; J = np.zeros((len(P), 3, 3))
        for k in range(3):
            e = np.zeros(3); e[k] = h; J[:, :, k] = (ch(pull, P + e) - ch(pull, P - e)) / (2 * h)
        dj = np.linalg.det(J); return dict(min=round(float(dj.min()), 3), p1=round(float(np.percentile(dj, 1)), 3), p99=round(float(np.percentile(dj, 99)), 3),
                                           folds=int((dj <= 0).sum()), n=int(len(dj)))
    res = dict(atlas=name, template_channel=a["channel"], engine=met.get("engine"), settings={k: met["settings"].get(k) for k in ("smooth_warp_sigma", "scales", "stage2")},
               mask_dice=dict(atlas_mask_in_template_vs_template=dice(at, T), template_mask_in_atlas_vs_atlas=dice(ta, Am)),
               outline=dict(atlas_in_template=dict(status=s1, reasons=w1), template_in_atlas=dict(status=s2, reasons=w2)),
               folds_fraction=met.get("folds_fraction"), roundtrip_mm={k: met[k] for k in met if k.startswith("roundtrip")},
               detJ=dict(template_to_atlas_in_template=detj(a2b, T, ti.affine), atlas_to_template_in_atlas=detj(b2a, Am, ai.affine)),
               linear_scales=json.load(open(prep)))
    if name == "CIVM-R":                                       # agreement with the one-way direct b0 registration (same pair, other engine/start)
        I = inputs()
        Rig = np.array(json.loads(product(I["civmr_rigid"]).read_text())["applied_rigid_world"])
        g = product(I["civmr_direct_link"]); Pst = np.loadtxt(g / "link_linear_world.txt"); dd, da = X.read_fireants_field(str(g / "link_warp.nii.gz"))
        direct = X.Map(L=np.linalg.inv(Pst), d=dd, d_affine=da)                       # pull: placed CIVM-R point -> our template point
        N = np.argwhere(Am)[::sc["civmr_every"]] @ ai.affine[:3, :3].T + ai.affine[:3, 3]   # native CIVM-R brain points
        y1 = ch(b2a, N); y2 = ch(direct, N @ Rig[:3, :3].T + Rig[:3, 3]); e = np.linalg.norm(y1 - y2, axis=1)
        res["vs_item11_anatomix_direct_mm"] = dict(p50=round(float(np.median(e)), 3), p95=round(float(np.percentile(e, 95)), 3), max=round(float(e.max()), 3), n=int(len(e)))
    if "sarm" in a:                                            # SARM level into the template (labels for figures; nearest)
        si = nib.load(str(a["sarm"])); S6 = np.asanyarray(si.dataobj).reshape(si.shape[:3] + (-1,))[..., a["sarm_level"] - 1]
        P = X.grid_world(T.shape, ti.affine); Q = ch(a2b, P); V = np.linalg.inv(si.affine)
        lab = map_coordinates(S6.astype(np.float32), (Q @ V[:3, :3].T + V[:3, 3]).T, order=0, cval=0).reshape(T.shape).astype(np.int16)
        d.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(lab, ti.affine), str(d / "SARM_level6_in_template.nii.gz")); res["sarm6_in_template"] = str(d / "SARM_level6_in_template.nii.gz")
    d.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(d / "score.json", "w"), indent=1, default=float); print(json.dumps(res, default=float)[:1500])
    if numbers:
        provenance.write_numbers(f"atlas_final_v2_{name}", res, inputs=[run / "metrics.json", template_channel(name), tmask(), a["img"], a["msk"],
                                                                         d / "S_ras_atlas_to_template.txt"],
                                 md5=False, cmd=f"python -m linc.link.atlas score {name}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["affine", "solve", "score"])
    ap.add_argument("atlas", choices=list(config()["atlas"]["atlases"]))
    ap.add_argument("--out", default=None, help="default: $LINC_WORK/atlas_v2")
    ap.add_argument("--run", default=None, help="score: the engine folder (default: OUT/<atlas>/run)")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(argv)
    if a.step == "affine":
        affine(a.atlas, a.out)
    elif a.step == "solve":
        solve(a.atlas, a.out, a.device)
    else:
        score(a.atlas, a.out, a.run)


if __name__ == "__main__":
    main()
