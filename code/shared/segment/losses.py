"""Dice + cross-entropy (nnU-Net's loss), with deep supervision.

Soft Dice over the foreground classes, computed over the whole batch (batch Dice), smooth 1e-5; plus cross-entropy; summed with equal
weight. With deep supervision, each coarser head's target is the one-hot label averaged over the head's blocks (2^l voxels per axis),
used as soft class probabilities: centered on the block, with no tie rule. (Picking one voxel per block, as nearest downsampling does,
offsets the target by (2^l - 1)/2 fine voxels; thresholding the block mean sends 4-of-8 ties to one class.) The heads are weighted
1, 1/2, 1/4, ... (the lowest resolution head gets 0), normalized to sum 1, as nnU-Net does.
"""
import torch
import torch.nn.functional as F


def dice_ce(logits, y, smooth=1e-5):
    """y: class indices (B, *S), or class probabilities (B, K, *S) float"""
    p = logits.float().softmax(1)
    if y.is_floating_point():
        ce = F.cross_entropy(logits.float(), y)
        oh = y
    else:
        ce = F.cross_entropy(logits, y)
        oh = F.one_hot(y, logits.shape[1]).movedim(-1, 1).float()
    ax = [0] + list(range(2, logits.ndim))
    inter = (p * oh).sum(ax)[1:]; den = (p.sum(ax) + oh.sum(ax))[1:]
    dice = (2 * inter + smooth) / (den + smooth)
    return ce + (1 - dice.mean())


def ds_weights(n):
    w = torch.tensor([1 / 2 ** i for i in range(n)])
    if n > 1:
        w[-1] = 0
    return w / w.sum()


def ds_target(y, K, size):
    """the one-hot label (B, *S) averaged over blocks down to `size`: soft class probabilities (B, K, *size), block-centered"""
    f = [s // t for s, t in zip(y.shape[1:], size)]
    assert all(s == t * k for s, t, k in zip(y.shape[1:], size, f)), (tuple(y.shape[1:]), tuple(size))
    oh = F.one_hot(y, K).movedim(-1, 1).float()
    return (F.avg_pool2d if y.ndim == 3 else F.avg_pool3d)(oh, f, f)


def loss(outs, y):
    if not isinstance(outs, (list, tuple)):
        return dice_ce(outs, y)
    w = ds_weights(len(outs))
    tot = 0.0
    for wi, o in zip(w.tolist(), outs):
        if wi == 0:
            continue
        yi = y if o.shape[2:] == y.shape[1:] else ds_target(y, o.shape[1], o.shape[2:])
        tot = tot + wi * dice_ce(o, yi)
    return tot
