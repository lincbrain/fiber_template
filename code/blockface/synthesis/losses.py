"""Training losses, all masked to the training targets: l1, l1 on the spatial gradients, and 1 - multi-scale SSIM."""
import torch
import torch.nn.functional as F
from monai.losses import SSIMLoss

from . import CFG


def masked_l1(pred, tgt, mask):
    c = pred.shape[1] if mask.shape[1] == 1 else 1        # a C-channel prediction against a 1-channel mask counts C elements
    return (torch.abs(pred - tgt) * mask).sum() / (mask.sum() * c).clamp_min(1.0)


def masked_grad_l1(pred, tgt, mask):
    """l1 on the finite differences along every spatial axis; a difference counts only if both of its voxels are in the mask"""
    def d(t, dim):
        return t.narrow(dim, 1, t.size(dim) - 1) - t.narrow(dim, 0, t.size(dim) - 1)
    loss = 0.0
    for dim in range(2, pred.dim()):
        pg, tg = d(pred, dim), d(tgt, dim)
        mg = mask.narrow(dim, 1, mask.size(dim) - 1) * mask.narrow(dim, 0, mask.size(dim) - 1)
        c = pred.shape[1] if mg.shape[1] == 1 else 1
        loss = loss + (torch.abs(pg - tg) * mg).sum() / (mg.sum() * c).clamp_min(1.0)
    return loss


class MSSSIMTerm:
    """1 - SSIM at three scales (average-pooled images, max-pooled mask), weighted; invalid voxels zeroed in both images"""
    def __init__(self, spatial_dims=3, weights=tuple(CFG["training"]["ms_ssim_weights"])):
        self.fn = SSIMLoss(spatial_dims=spatial_dims, data_range=1.0)
        self.d = spatial_dims; self.w = weights

    def __call__(self, pred, tgt, mask):
        pool = F.avg_pool2d if self.d == 2 else F.avg_pool3d
        mpool = F.max_pool2d if self.d == 2 else F.max_pool3d
        p, t, m = pred, tgt, mask; loss = 0.0
        for i, w in enumerate(self.w):
            if i > 0:
                p, t, m = pool(p, 2), pool(t, 2), mpool(m, 2)
            loss = loss + w * self.fn(p * m, t * m)
        return loss / sum(self.w)
