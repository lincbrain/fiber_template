"""linc.segment: one recipe and one code path for the project's segmentation models (2D and 3D), with no nnU-Net dependency.

  net       nnU-Net-like U-Net (state_dict + JSON spec)
  io        NIfTI (nibabel) and PNG/TIFF (Pillow) in; masks out in the input's grid
  data      task specs, subject-grouped folds, the per-case cache, patch sampling
  augment   augmentation on the GPU
  losses    Dice + cross-entropy, deep supervision
  metrics   Dice, ASSD and HD95 in mm
  train     python -m linc.segment.train --task <spec> --fold <k|all>
  predict   python -m linc.segment.predict --model <run folder> --out <mask> IMAGE [...]
Dependencies: torch, numpy, nibabel, scipy (metrics, largest component); Pillow for 2D PNG/TIFF.
"""
