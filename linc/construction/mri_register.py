"""The registrations a brain's MRI build needs: its T2 and microstructure sessions onto its b0, and NMT v2 onto its b0.

You need
  the brain's staged MRI folder (linc.construction.mri_prep stage, or the data release): dwi/b0_brain_n4, t2/echo_n4_brain,
  micro/b0_brain_n4 and the three hand masks (masks/); for nmt, NMT v2.0 sym with its SARM labels (configs/link.yaml
  `atlas_reference`: the published templates are downloaded, not redistributed); ANTs 2.5.4 (ANTSPATH); env/anatomix.yml
  ($ANATOMIX_PYTHON) and a GPU.

What it does
  Every registration starts from an ANTs rigid, then affine registration of the two images inside their masks (mutual
  information), and refines it with a deformable registration of anatomix features (FireANTs), as for the paper's brains:
  t2     the T2 echo onto the b0 (linc.registration.featurereg, 0.25 mm, warp smoothing 1.25 voxels); its field holds the affine;
  micro  the microstructure session's b0 onto the b0 (the symmetric registration, linc.link.symreg, 0.25 mm, warp smoothing
         0.75 voxels); its two halves and the affine are composed into one field on the b0 grid;
  nmt    NMT v2 onto the b0 (fixed = the brain; linc.registration.featurereg, 0.25 mm, second stage at scales 2 and 1); the SARM
         level-6 labels are pulled onto the b0 grid. The canonical frame takes the AC and PC from them and the pitch from the
         affine.

You get
  t2, micro  the registry entries $LINC_DATA/sub-<ID>/xfm/t2-native__to__b0-native/ and micro-native__to__b0-native/ (one ITK
             displacement field on the b0 grid each) and the run files in --out
  nmt        <MRI folder>/landmarks/nmt_affine_0GenericAffine.mat and SARM6_in_subject.nii.gz

    python -m linc.construction.mri_register t2 --subject MF283 [--out <dir>] [--device cuda:0]
    python -m linc.construction.mri_register micro --subject MF283
    python -m linc.construction.mri_register nmt --subject MF283

Paper: Methods, "Image registration" and "MRI preprocessing".
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .. import paths, xfm
from .mri_prep import Brain
from .tools import REPO, ants, recipe, run

ENGINE = "anatomix + FireANTs (env/anatomix.yml)"


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


def _entry(b, src, warp, engine, sigma):
    return xfm.register(src, "b0-native", [("warp", "itk_lps_disp", warp)], subject=b.subject, dst_grid=b("b0"), engine=engine,
                        config=dict(resolution_mm=0.25, warp_sigma=sigma), root=b.xfm_root)


def t2(b, out, device):
    from ..registration import featurereg
    aff = affine_init(b("b0_n4_brain"), b("t2_brain"), b("dwi_mask"), b("t2_mask"), out / "affine_")
    featurereg.register(str(b("b0_n4_brain")), str(b("t2_brain")), str(b("dwi_mask")), str(b("t2_mask")), str(out), affine=aff,
                        resolution=0.25, device=device, loss="cc", lr=1.0, smooth_grad_sigma=1.5, smooth_warp_sigma=1.25,
                        channel_chunk=8)
    return _entry(b, "t2-native", out / "warp_mri_to_psoct.nii.gz", f"featurereg, {ENGINE}", 1.25)


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


def micro(b, out, device):
    micro_b0 = b.mri_root / "micro" / "b0_brain_n4.nii.gz"
    aff = affine_init(b("b0_n4_brain"), micro_b0, b("dwi_mask"), b("micro_mask"), out / "affine_")
    S = np.linalg.inv(xfm.read_itk_affine(aff))                       # micro world -> b0 world (RAS, push): the symmetric start
    np.savetxt(out / "S_ras_B_to_A.txt", S)
    py = os.environ.get("ANATOMIX_PYTHON") or sys.exit("set ANATOMIX_PYTHON (env/anatomix.yml)")
    dev = device.split(":")[-1]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=dev, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    subprocess.run([py, "-m", "linc.link.symreg", b("b0_n4_brain"), b("dwi_mask"), micro_b0, b("micro_mask"), out / "S_ras_B_to_A.txt",
                    out / "sym", "--res", "0.25", "--device", "cuda:0", "--smooth-warp-sigma", "0.75", "--channel-chunk", "8"],
                   check=True, env=env, cwd=REPO)
    sym = out / "sym"
    Hinv = xfm.Affine(np.loadtxt(sym / "Hinv.txt"))
    chain = _Pull([Hinv, xfm.Warp(sym / "F_ras_disp.nii.gz", "ras_disp"), Hinv])
    xfm.compose(chain, b("b0"), out / "warp_mri_to_psoct.nii.gz", fmt="itk")
    return _entry(b, "micro-native", out / "warp_mri_to_psoct.nii.gz", f"symreg, {ENGINE}", 0.75)


def nmt(b, out, device):
    from ..link.atlas import spec
    from ..registration import featurereg
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["t2", "micro", "nmt"])
    ap.add_argument("--subject", required=True); ap.add_argument("--mri-root"); ap.add_argument("--xfm-root")
    ap.add_argument("--out", help="the run files (default $LINC_WORK/mri_register/<ID>/<what>)")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(argv)
    os.umask(0o002)
    b = Brain(a.subject, a.mri_root, a.xfm_root)
    out = Path(a.out) if a.out else paths.work_root() / "mri_register" / a.subject / a.what
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print({"t2": t2, "micro": micro, "nmt": nmt}[a.what](b, out, a.device), f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
