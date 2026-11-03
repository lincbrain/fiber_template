#!/usr/bin/env python3
"""Our MRI template as a registration target, against NMT v2 and CIVM-R (Table 3).

You need
  the registrations of record of every brain to the three targets (configs/products_blockface.yaml `r060_runs`), our template
  (template_mri), and your own copies of NMT v2 and CIVM-R (configs/targets.yaml says where to get them and where to put them).

What it does
  1. prep: masks the targets' channels and every brain's images into $LINC_WORK/r060 (skipped when present);
  2. measure: pulls each brain's b0 and mask onto each target's grid through its registration of record, then computes the pairwise
     b0 correlation and mask Dice (CPU, about 76 warps, --jobs in parallel).

You get
  $LINC_NUMBERS/r060_target_comparison.json; nothing derived from the published templates' intensities leaves $LINC_WORK

    python reproduce/numbers/r060_target_comparison.py [--jobs 4] [--plan]

The record's free-text `method` field is kept verbatim from the analysis run (METHOD below); every other field is regenerated
(paper: Methods, "The template as a registration target").
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from linc import paths, provenance  # noqa: E402
from linc.evaluation import targets  # noqa: E402

# the record's method text, verbatim (the analysis project's mri/code/r060_targets.py docstring; its paths are the analysis project's)
METHOD = """The target comparison (30 Oct 2026, with the T2 + tensor arm): our MRI template as a registration target for ex vivo brains,
against NMT v2 (sym, in vivo T1w) and CIVM-R (ex vivo b0), with ONE engine and ONE setting for all three targets.

Brains
  sectioned (13; out of sample for every target; the headline table): data/sub-<ID>/mri/<ID>_mri_b0_native x <ID>_mri_mask_native
  template  (4, MF283-MF286; in sample for ours): mri/data/sub-<ID>/canonical_rebuild_handfrac/_native/b0_head_n4 x
            model_mask_source_uint8 (the author's final masks; the construction's own b0 inputs)
Targets (as the paper's mappings used them: mri/code/atlas_v2.py ATL; 30 Oct 2026), every channel multiplied by its brain mask here
  nmt    NMT_v2.0_sym_SS (T1w) + NMT_v2.0_sym_brainmask
  civmr  CIVM-R_b0_brain + CIVM-R_brain_mask; FA = CIVM-R_DTI_FA. Local use only: NOT_FOR_SHARING applies to its intensities and
         anything derived from them (the warped images under mri/tc/r060/*/civmr/).
  ours   template/mri/linc_mri_b0 + linc_mri_brainmask; T2 = linc_mri_T2, FA = linc_mri_FA
Arms
  b0     every brain: its b0 <-> the target's image (nmt: b0 <-> T1w)
  multi  the 4 template brains with the channels each target has, so NOT like-for-like across targets:
         nmt: brain T2 <-> T1w;  civmr: b0 <-> b0 + FA <-> FA;  ours: b0 <-> b0 + T2 <-> T2 + FA <-> FA.
         Brain T2 = t2/echo_n4 pulled to b0-native by reg/t2_to_dwi_v3 (of record), x the final mask; FA = dwi/dti_FA x the final mask.
Engine (the recipe of the template <-> atlas mappings of record, atlas_v2.py, with the brains placed upright first)
  upright P0 (rigid, native b0 world -> the canonical NMT-v2-stereotaxic frame of our template), as the R5 recipe places a b0:
         sectioned brains: the frame bridge (template/bridge_run113_to_canonical) @ the FLIRT 6-dof placement of record
         (data/sub-<ID>/mri/<ID>_mri2template.mat with <ID>_mri.nii.gz); template brains: landmarks/canonical.json T_subject_to_canonical.
         Needed because ANTs starts from the centres of mass without a rotation search: five brains' b0 headers are 34-56 deg from
         upright (the others 3-15 deg), and their starts failed in a first run (mri/tc/r060/start, b0: kept, not used).
  start  S (brain world -> target world) = S' @ P0. S' = the best of three candidates from the two ANTs MI rigid+affine registrations
         (preprocess_mri.affine_init) of the upright b0 (header P0 @ affine, no resampling), target fixed (fwd) and brain fixed (bwd):
         fwd, bwd and their log-Euclidean mean (the mappings' rule); best = the highest Dice of the brain mask carried by the candidate
         with the target's brain mask (target grid, every 2nd voxel). The mean alone failed where the two directions disagree (7 of 51
         pairs by 5-14 mm: e.g. civmr MR256 fwd 0.93, bwd 0.58, mean 0.74). The multi arm reuses its pair's b0-arm start.
         The solve gets the upright images RESAMPLED ONCE onto an axis-aligned 0.25 mm grid around the brain (mask box + 4 mm; images
         cubic, clipped at 0 and masked; masks linear > 0.5), as the R5 recipe resamples its input once: the engine sizes its halfway
         grid from the corners of each mask's VOXEL box, so a tilted voxel grid (native, or upright by header only) inflated it ~1.4x
         per axis for the 34-56 deg brains (NMT MR253: 288-318 x 342-392 x 343-390) and ran out of memory on 48 GB cards. Those two
         runs are in mri/tc/r060_up/_superseded_nativeB and _superseded_uprightheader, not used. The native b0 is reached by P0^-1.
  solve  mri/code/symreg_upstream.py A = target, B = upright brain, S', --res 0.25 --smooth-warp-sigma 1.25 --max-gpu-frac 0.95; channel chunk 8
         (4 on the 20 GB cards: memory only, the loss is summed over all chunks either way)
Folders: targets/ and inputs/ in mri/tc/r060; everything from the upright starts on in mri/tc/r060_up (place/, start/, b0/, multi/,
xfm/, logs/).
Registry entries (for the LGN/STN carries): mri/tc/r060_up/xfm/sub-<ID>/b0-native__to__{atlas-NMT_v2.0_sym, atlas-CIVM-R, mri-template}
  = [affine ras_pull Hinv, warp ras_disp F, affine ras_pull Hinv, affine ras_pull P0^-1] (a2b of linc.link.symreg.maps, then upright
  -> native; checked against it), files in place.
Measures (each target's own 0.25 mm grid; pairs of different brains)
  b0_r       Pearson r of the two warped b0 images (cubic) inside both warped masks (linear > 0.5), minus a 1.5 mm rim (6 erosions at
             0.25 mm: the rule of the paper's pairwise variant measures)
  mask_dice  Dice of the two warped brain masks
  per target and group: median, IQR (25th-75th percentile); per pair whether ours is higher than nmt / civmr. Groups: the 13 sectioned
  brains (78 pairs, the headline), the 4 template brains (6 pairs, in sample for ours). Reference beside ours: the 13 brains through the
  R5 of record (data/sub-<ID>/xfm/b0-native__to__mri-template, the b0_to_template engine, 0.35 mm).
    <linc_template python> mri/code/r060_targets.py prep | place | affine <ID> <tgt> | choose | grid | jobs | cmd <arm> <tgt> <ID> [chunk] | registry | warps [arm] [jobs] | measure
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--plan", action="store_true", help="print what this script reads, writes and needs; run nothing")
    a = ap.parse_args()
    W = targets.Work(paths.work_root() / "r060", paths.products("products_blockface")["r060_runs"])
    SECTIONED, TEMPLATE = targets.brains()
    need = [W.tgt_mask(t) for t in targets.config()["targets"]] + [W.inp(ID) for ID in SECTIONED + TEMPLATE]
    if a.plan:
        from linc.evaluation import plan
        import yaml
        ref = paths.products_root() / yaml.safe_load((Path(__file__).resolve().parents[2] / "configs" / "link.yaml").read_text())["inputs"]["atlas_reference"]
        return plan.show("r060_target_comparison", env="env/linc.yml", est=f"about 15 min with --jobs 4 (CPU); the prep of the targets runs once",
                         inputs={"registrations of record": paths.products("products_blockface")["r060_runs"], "published targets": [ref / "NMT_v2.0_sym", ref / "CIVM-R_T2_DTI"]},
                         outputs=[paths.numbers_root() / "r060_target_comparison.json", paths.work_root() / "r060"],
                         needs=["FSLDIR", "NMT v2 and CIVM-R downloaded as configs/targets.yaml says (not redistributed)"],
                         notes=[f"brains: {', '.join(SECTIONED)} (sectioned) and {', '.join(TEMPLATE)} (the template's)",
                                "the prepared targets and inputs stay under $LINC_WORK/r060 (lgnstn_targets_r060c reads them); nothing derived from the published templates leaves it"])
    if not all(p.exists() for p in need):
        targets.prep(W)
    out, ins = targets.measure(W, a.jobs, method=METHOD)
    provenance.write_numbers("r060_target_comparison", out, inputs=ins, md5=False,
                             cmd=" ".join(["python reproduce/numbers/r060_target_comparison.py"] + sys.argv[1:]))


if __name__ == "__main__":
    main()
