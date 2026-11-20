"""The registrations a brain's MRI build needs: its T2 and microstructure sessions onto its b0, and NMT v2 onto its b0; and, after the
build, the brain's links up to the template.

You need
  the brain's staged MRI folder, $LINC_MRI_DATA/sub-<ID>/mri (mri.mri_prep stage makes it): dwi/b0_brain_n4, t2/echo_n4_brain,
  micro/b0_brain_n4 and the three session masks (masks/); for nmt, NMT v2.0 sym with its SARM labels (settings/link.yaml
  `atlas_reference`: the published templates are downloaded, not redistributed); ANTs 2.5.4 (ANTSPATH); settings/env/anatomix.yml
  ($ANATOMIX_PYTHON) and a GPU.

What it does
  Every registration starts from an ANTs rigid, then affine registration of the two images inside their masks (mutual
  information), and refines it with a deformable registration of anatomix features (FireANTs), as for the paper's brains:
  t2     the T2 echo onto the b0 (blockface.registration.featurereg, 0.25 mm, warp smoothing 1.25 voxels); its field holds the affine;
  micro  the microstructure session's b0 onto the b0 (the symmetric registration, template.link.symreg, 0.25 mm, warp smoothing
         0.75 voxels); its two halves and the affine are composed into one field on the b0 grid;
  nmt    NMT v2 onto the b0 (fixed = the brain; blockface.registration.featurereg, 0.25 mm, second stage at scales 2 and 1); the SARM
         level-6 labels are pulled onto the b0 grid. The canonical frame takes the AC and PC from them and the pitch from the
         affine;
  links  registers what the build already made (no registration): the frame (landmarks/canonical.json) and the final warp of the
         finished run (--run; the warp the template channels use).

You get
  t2, micro  the registry entries $LINC_MRI_DATA/sub-<ID>/xfm/t2-native__to__b0-native/ and micro-native__to__b0-native/ (one ITK
             displacement field on the b0 grid each) and the run files in --out
  nmt        $LINC_MRI_DATA/sub-<ID>/mri/landmarks/nmt_affine_0GenericAffine.mat and SARM6_in_subject.nii.gz
  links      the registry entries b0-native__to__mri-canonical/ and mri-canonical__to__mri-template/, so that python -m shared.xfm moves
             images between the brain's spaces and the template
  A registry entry that exists is refused before any work is done; --overwrite replaces it, after a backup of the old one under
  $LINC_WORK/xfm_backup/.

    python -m mri.mri_register t2 --subject MF283 [--out <dir>] [--device cuda:0] [--overwrite]
    python -m mri.mri_register micro --subject MF283
    python -m mri.mri_register nmt --subject MF283
    python -m mri.mri_register links --subject MF283 --run $LINC_WORK/build_mri_channels/runs/rb_final
The MRI build works in its own working folder, $LINC_MRI_DATA: run these with LINC_DATA=$LINC_MRI_DATA in front, as the steps of
code/template/build_mri_channels.py are (docs/tutorials/01_mri.md, part A); --mri-root and --xfm-root default to the brain's folders there.

Paper: Methods, "Image registration and segmentation" and "Preprocessing".
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from shared import paths, xfm
from mri.mri_prep import Brain, remove_own
from template.tools import REPO, ants, recipe, run

ENGINE = "anatomix + FireANTs (settings/env/anatomix.yml)"


def affine_init(fixed, moving, fmask, mmask, prefix):
    """ANTs rigid then affine, moving -> fixed, mutual information inside both masks, centers of mass aligned first."""
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    sched = ["-c", "[1000x500x250x100,1e-6,10]", "-f", "8x4x2x1", "-s", "3x2x1x0vox"]
    run([ants("antsRegistration"), "-d", 3, "--float", 0, "-v", 0, "-o", f"[{prefix},{prefix}warped.nii.gz]",
         "--interpolation", "Linear", "--winsorize-image-intensities", "[0.005,0.995]", "--use-histogram-matching", 1,
         "--initial-moving-transform", f"[{fixed},{moving},1]", "-x", f"[{fmask},{mmask}]",
         "-t", "Rigid[0.1]", "-m", f"MI[{fixed},{moving},1,32,Regular,0.25]", *sched,
         "-t", "Affine[0.1]", "-m", f"MI[{fixed},{moving},1,32,Regular,0.25]", *sched])
    return f"{prefix}0GenericAffine.mat"


def _free(b, src, dst, overwrite):
    """Stop before any work when the registry entry exists, unless it is to be replaced. A folder without meta.json is what a write
    that stopped partway left: not an entry, and in the way of the next one, so with overwrite it is removed."""
    d = xfm.xfm_dir(src, dst, b.subject, b.xfm_root)
    if (d / "meta.json").exists():
        if not overwrite:
            sys.exit(f"{d} exists: --overwrite replaces it (the old entry is backed up under $LINC_WORK/xfm_backup/)")
    elif d.exists() or d.is_symlink():
        if not overwrite:
            sys.exit(f"{d} holds an entry that was not finished: --overwrite removes it")
        remove_own(d)


def _entry(b, src, warp, engine, sigma, overwrite=False):
    return xfm.register(src, "b0-native", [("warp", "itk_lps_disp", warp)], subject=b.subject, dst_grid=b("b0"), engine=engine,
                        config=dict(resolution_mm=0.25, warp_sigma=sigma), root=b.xfm_root, overwrite=overwrite)


def t2(b, out, device, overwrite=False):
    from blockface.registration import featurereg
    _free(b, "t2-native", "b0-native", overwrite)
    aff = affine_init(b("b0_n4_brain"), b("t2_brain"), b("dwi_mask"), b("t2_mask"), out / "affine_")
    featurereg.register(str(b("b0_n4_brain")), str(b("t2_brain")), str(b("dwi_mask")), str(b("t2_mask")), str(out), affine=aff,
                        resolution=0.25, device=device, loss="cc", lr=1.0, smooth_grad_sigma=1.5, smooth_warp_sigma=1.25,
                        channel_chunk=8)
    return _entry(b, "t2-native", out / "warp_mri_to_psoct.nii.gz", f"featurereg, {ENGINE}", 1.25, overwrite)


class _Pull:
    """A pull chain of transforms for xfm.compose (no registry entries in it)."""
    links = []

    def __init__(self, comps):
        self.comps = comps

    def pull(self, X):
        for c in self.comps:
            X = c.pull(X)
        return X

    def describe(self):
        return [c.label for c in self.comps]


def micro(b, out, device, overwrite=False):
    _free(b, "micro-native", "b0-native", overwrite)
    micro_b0 = b.mri_root / "micro" / "b0_brain_n4.nii.gz"
    aff = affine_init(b("b0_n4_brain"), micro_b0, b("dwi_mask"), b("micro_mask"), out / "affine_")
    S = np.linalg.inv(xfm.read_itk_affine(aff))                       # micro world -> b0 world (RAS, push): the symmetric start
    np.savetxt(out / "S_ras_B_to_A.txt", S)
    py = os.environ.get("ANATOMIX_PYTHON") or sys.exit("set ANATOMIX_PYTHON (settings/env/anatomix.yml)")
    dev = device.split(":")[-1]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=dev, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    subprocess.run([py, "-m", "template.link.symreg", b("b0_n4_brain"), b("dwi_mask"), micro_b0, b("micro_mask"), out / "S_ras_B_to_A.txt",
                    out / "sym", "--res", "0.25", "--device", "cuda:0", "--smooth-warp-sigma", "0.75", "--channel-chunk", "8"],
                   check=True, env=env, cwd=REPO)
    sym = out / "sym"
    Hinv = xfm.Affine(np.loadtxt(sym / "Hinv.txt"))
    chain = _Pull([Hinv, xfm.Warp(sym / "F_ras_disp.nii.gz", "ras_disp"), Hinv])
    xfm.compose(chain, b("b0"), out / "warp_mri_to_psoct.nii.gz", fmt="itk")
    return _entry(b, "micro-native", out / "warp_mri_to_psoct.nii.gz", f"symreg, {ENGINE}", 0.75, overwrite)


def nmt(b, out, device):
    from template.link.atlas import spec
    from blockface.registration import featurereg
    a = spec("NMT_v2.0_sym")
    lm = b("nmt_affine").parent; lm.mkdir(parents=True, exist_ok=True)
    aff = affine_init(b("b0_n4_brain"), a["img"], b("dwi_mask"), a["msk"], lm / "nmt_affine_")
    featurereg.register(str(b("b0_n4_brain")), str(a["img"]), str(b("dwi_mask")), str(a["msk"]), str(out), affine=aff,
                        resolution=0.25, device=device, stage2_scales="2,1", stage2_cc="5,3", stage2_iterations="200,200")
    import nibabel as nib
    si = nib.load(str(a["sarm"]))                                     # SARM's levels as volumes; the config's level (6)
    level = np.asanyarray(si.dataobj).reshape(si.shape[:3] + (-1,))[..., a["sarm_level"] - 1]
    nib.save(nib.Nifti1Image(level, si.affine, si.header), str(out / "SARM_level_in_NMT.nii.gz"))
    run([ants("antsApplyTransforms"), "-d", 3, "-i", out / "SARM_level_in_NMT.nii.gz", "-r", b("b0_n4_brain"),
         "-o", b("sarm_in_subject"), "-n", "NearestNeighbor", "-t", out / "warp_mri_to_psoct.nii.gz"])
    return b("sarm_in_subject")


def links(b, run, out, overwrite=False):
    for src, dst in (("b0-native", "mri-canonical"), ("mri-canonical", "mri-template")):
        _free(b, src, dst, overwrite)
    R = recipe(); k = R["channels"]; mod = R["scalar_modalities"][0]["name"]; run = Path(run)
    grid = run / "standardspace.nii.gz"                               # the canonical grid, which is also the template's
    T = out / "T_subject_to_canonical.txt"
    M = np.asarray(json.loads((b.mri_root / "landmarks" / "canonical.json").read_text())["T_subject_to_canonical"], float)
    np.savetxt(T, M)
    e = xfm.register("b0-native", "mri-canonical", [("affine", "ras_push", T)], subject=b.subject, dst_grid=grid, src_grid=b("b0"),
                     engine="the canonical frame (mri.canonical)", root=b.xfm_root, overwrite=overwrite)
    off = float(np.abs(M[:3, :3].T @ M[:3, :3] - np.eye(3)).max())    # rigid: its rotation part is orthonormal
    xfm.set_qc(e.dir, "pass" if off < 1e-6 else "fail", check="the frame is rigid", orthonormality_error=off)
    it = run / "out" / f"nln_step_{k['step']:02d}" / f"iteration_{k['iteration']:02d}"
    warp = it / f"{b.subject}_scalar_mod_01_to_template_warp_resampled_unbiased_full.nii.gz"
    e = xfm.register("mri-canonical", "mri-template", [("warp", "fsl_rel", warp)], subject=b.subject, dst_grid=grid, src_grid=grid,
                     engine="MMORF 1.0.1 through the patched template-construction driver", root=b.xfm_root, overwrite=overwrite)
    moved = out / "t2_input_in_template.nii.gz"                       # the entry against the driver's own resampling of this input
    xfm.apply(xfm.Chain([e]), run / "data" / mod / b.subject / mod / f"{mod}_unbiased_brain.nii.gz", grid, moved)
    x = np.asanyarray(nib.load(str(moved)).dataobj).astype(np.float64)
    y = np.asanyarray(nib.load(str(it / f"{b.subject}_scalar_mod_01_brain_to_template.nii.gz")).dataobj).astype(np.float64)
    r = float(np.corrcoef(x[y > 0], y[y > 0])[0, 1])
    xfm.set_qc(e.dir, "pass" if r >= 0.95 else "fail", check="r with the driver's resampling of the brain's T2 input", r=round(r, 4))
    return e


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["t2", "micro", "nmt", "links"])
    ap.add_argument("--subject", required=True); ap.add_argument("--mri-root"); ap.add_argument("--xfm-root")
    ap.add_argument("--out", help="the run files (default $LINC_WORK/mri_register/<ID>/<what>)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--run", help="links: the finished MMORF run's folder")
    ap.add_argument("--overwrite", action="store_true", help="t2, micro, links: replace an existing registry entry (the old one is backed up)")
    a = ap.parse_args(argv)
    os.umask(0o002)
    b = Brain(a.subject, a.mri_root, a.xfm_root)
    out = Path(a.out) if a.out else paths.work_root() / "mri_register" / a.subject / a.what
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.what == "links":
        made = links(b, a.run, out, a.overwrite)
    elif a.what == "nmt":
        made = nmt(b, out, a.device)
    else:
        made = {"t2": t2, "micro": micro}[a.what](b, out, a.device, a.overwrite)
    print(made, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
