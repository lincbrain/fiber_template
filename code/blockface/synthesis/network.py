"""The completion network (paper: Methods, caudal completion; Gonzales et al. 2026, ISBI).

MONAI's BasicUNet at the input grid (0.5 mm; features 32/32/64/128/256/32, LeakyReLU 0.1, instance norm) with 32 output features,
a x2 transposed convolution (kernel 2, stride 2) to the output grid (0.25 mm), two 3x3x3 convolution blocks (instance norm,
LeakyReLU 0.1), a 1x1x1 convolution and a sigmoid. Input: one b0 channel on [0, 1]. Output: blockface grey on [0, 1].
A checkpoint of record is {"G": the EMA weights, "cfg": the training configuration, "epoch": the last epoch}.
"""
import torch
import torch.nn as nn
from monai.networks.nets import BasicUNet


class CompletionNet(nn.Module):
    def __init__(self, in_channels=1, features=(32, 32, 64, 128, 256, 32), head=32):
        super().__init__()
        self.unet = BasicUNet(spatial_dims=3, in_channels=in_channels, out_channels=head, features=features,
                              act=("LeakyReLU", {"negative_slope": 0.1}), norm=("instance", {}), dropout=0.0)
        self.up = nn.ConvTranspose3d(head, head, kernel_size=2, stride=2)
        blk = lambda: [nn.Conv3d(head, head, 3, padding=1), nn.InstanceNorm3d(head), nn.LeakyReLU(0.1)]  # noqa: E731
        self.refine = nn.Sequential(*blk(), *blk(), nn.Conv3d(head, 1, 1))

    def forward(self, x):
        return torch.sigmoid(self.refine(self.up(self.unet(x))))


def build(cfg):
    """the network described by a training configuration (features, head, in_channels)"""
    return CompletionNet(in_channels=cfg.get("in_channels", 1), features=tuple(cfg["features"]), head=cfg.get("head", 32))


def load(path, device="cuda"):
    """(network in eval mode on `device`, its training configuration) from a checkpoint of record"""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    net = build(ck["cfg"]).to(device)
    net.load_state_dict(ck["G"])
    net.eval()
    return net, ck["cfg"]
