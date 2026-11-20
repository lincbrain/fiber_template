"""Caudal completion: the blockface tissue that a brain's photographs lack, predicted from its b0 image and joined to the photographs
(paper: Methods, "Synthesis of missing sections"; the network: Gonzales et al. 2026, ISBI).

  network   the 3D U-Net: b0 at 0.5 mm in, blockface grey at 0.25 mm out
  data      a prepared brain folder (the network's input, the photographs, the masks, the join plane) and training patches
  augment   the paired two-grid augmentation of training
  losses    masked l1, gradient l1, multi-scale SSIM
  metrics   SSIM / l1 / PSNR over a region, high-frequency ratio
  deploy    sliding-window prediction, the regions around the cut, the intensity match and the cross-fade
  export    the completed volume and its companions for one brain (code/blockface/synthesis/linc-complete)
  train     the training recipe of the networks of record

A prepared brain folder (one per brain, LINC ID as the folder name):
  seam.json                  the join plane: z_f, the slab size dstar_planes, the plane coordinate y_dep(k) = y0 + g . k of every
                             output voxel k, the input normalization (mlo, mhi), the input box
  coarse/in.nii.gz           the network input: b0 (own or stand-in) on the 0.5 mm box, normalized to [0, 1], 0 outside the brain
  coarse/brain.nii.gz        the completion brain on the same box
  fine/bf.nii.gz             the photographs' grey on the 0.25 mm box (its center-aligned 2x refinement), bf_rgb.nii.gz the color
  fine/real_mask.nii.gz      photographed tissue; photo_fraction.nii.gz its photographed fraction; brain.nii.gz the completion brain;
  fine/valid.nii.gz          the training targets (real_mask minus each column's caudal-most 0.25 mm)
Parameters: settings/completion.yaml (CFG).
"""
from pathlib import Path

import yaml

CFG = yaml.safe_load((Path(__file__).resolve().parents[3] / "settings" / "completion.yaml").read_text())
