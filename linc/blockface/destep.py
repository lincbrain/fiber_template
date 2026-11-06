#!/usr/bin/env python3
"""The brightness steps between imaging sessions, removed (Methods, "Blockface reconstruction"; used by reconstruct.py).

You need
  L, the brain's brightness per section, in section order.
What it does
  It models L[i] = T[i] (a smooth tissue trend) x G[session(i)] (one exposure gain per session) x N[i] (flicker):
  detect_steps(L)            the change-points p between sessions (a jump in level between sections p-1 and p)
  destep_reference(L, ...)   each session's gain divided out (every session brought to the level of the longest), then smoothed
You get
  (ref, steps, gains): ref[i] is section i's target level, with the session jumps and the flicker removed and the gentle
  anatomical trend kept.

Ported unchanged from the analysis project's code/destep.py.
"""
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter1d


def detect_steps(L, win=8, rel=1.05, absjump=6.0, edge=6):
    """Change-points where the robust level ratio across a window exceeds `rel` (>=5% fractional jump) AND
    the absolute jump exceeds `absjump` units. Both conditions guard against false positives: a gentle
    anatomical gradient fails the sharp-window test (small right-left over the window), a tiny wiggle fails
    absjump. Ignores the first/last `edge` slices (tiny-brain, unreliable level). Returns ascending,
    de-duplicated step positions (the jump is between p-1 and p)."""
    n = len(L)
    Ls = median_filter(L, size=3, mode="nearest")
    cand = []
    for i in range(edge, n - edge):
        left = np.median(Ls[max(0, i - win):i]); right = np.median(Ls[i:i + win])
        ratio = max(right, 1e-3) / max(left, 1e-3)
        if (ratio > rel or ratio < 1.0 / rel) and abs(right - left) > absjump:
            cand.append((abs(np.log(ratio)), i))
    # non-maximum suppression: keep the strongest in each cluster of nearby candidates
    cand.sort(reverse=True); steps = []
    for _, i in cand:
        if all(abs(i - j) > win for j in steps):
            steps.append(i)
    return sorted(steps)


def destep_reference(L, smooth=20, smoother="gauss", **kw):
    """Target level per slice = LOW-PASS(de-stepped L). De-stepping scales each session (segment between
    change-points) by a constant so its robust median matches the LARGEST segment's - the bulk of the brain
    keeps its natural level, and the outlier sessions (a small bright anterior pole, a differently-exposed
    posterior block) are aligned to it. The low-pass then estimates the smooth underlying trend:
      smoother='gauss'  (default): gaussian_filter1d(sigma=`smooth`) - a LINEAR low-pass, so it also smooths
              out the broad mid-frequency undulation (and any small undetected step ramps) that an
              edge-preserving median would keep. `smooth` = sigma in slices (bigger = smoother trend).
      smoother='median': median_filter(size=`smooth`) - edge-preserving (keeps structure slower than the
              window); the previous behaviour.
    Detected SHARP steps are still removed sharply: the de-stepped Ld is continuous at each boundary, so the
    smoothed reference is continuous there while the gain = ref/L jumps where the raw level jumps.
    Returns (ref, steps, seg_gain)."""
    L = np.asarray(L, float); n = len(L)
    steps = detect_steps(L, **kw)
    bounds = [0] + steps + [n]                              # segment j = [bounds[j], bounds[j+1])
    segs = [(bounds[j], bounds[j + 1]) for j in range(len(bounds) - 1)]
    med = [np.median(L[a:b]) for a, b in segs]
    ref_seg = int(np.argmax([b - a for a, b in segs]))     # largest segment = the level anchor
    seg_gain = np.ones(n)
    for j, (a, b) in enumerate(segs):
        seg_gain[a:b] = med[ref_seg] / max(med[j], 1e-3)   # scale this session to the anchor's level
    Ld = L * seg_gain                                       # de-stepped level (smooth, aligned to anchor)
    if smooth <= 0:
        ref = np.full(n, np.median(Ld))
    elif smoother == "median":
        ref = median_filter(Ld, size=int(smooth), mode="nearest")
    else:
        ref = gaussian_filter1d(Ld, sigma=float(smooth), mode="nearest")
    return ref, steps, seg_gain
