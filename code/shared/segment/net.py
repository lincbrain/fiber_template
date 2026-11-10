"""A plain 2D/3D U-Net with nnU-Net's topology (no nnU-Net import).

Each stage has two 3x3(x3) convolutions, each followed by instance norm and leaky ReLU 0.01. The encoder downsamples with a strided first
convolution, the decoder upsamples with a transposed convolution and concatenates the skip. With deep supervision, every decoder stage
has its own 1x1 head; forward() then returns the logits of every head, full resolution first (training), else only the full-resolution
logits. The architecture is fully described by its spec (to_spec / from_spec), so a model ships as state_dict + JSON.
"""
import torch
import torch.nn as nn


def _ops(dim):
    return (nn.Conv2d, nn.ConvTranspose2d, nn.InstanceNorm2d) if dim == 2 else (nn.Conv3d, nn.ConvTranspose3d, nn.InstanceNorm3d)


class Stage(nn.Module):
    def __init__(self, dim, ci, co, stride):
        super().__init__()
        Conv, _, Norm = _ops(dim)
        self.body = nn.Sequential(Conv(ci, co, 3, stride=stride, padding=1), Norm(co, affine=True, eps=1e-5),
                                  nn.LeakyReLU(0.01, inplace=True),
                                  Conv(co, co, 3, padding=1), Norm(co, affine=True, eps=1e-5), nn.LeakyReLU(0.01, inplace=True))

    def forward(self, x):
        return self.body(x)


class UNet(nn.Module):
    def __init__(self, dim, in_channels, n_classes, features=(32, 64, 128, 256, 320, 320), strides=None, deep_supervision=True):
        super().__init__()
        assert dim in (2, 3)
        n = len(features)
        strides = strides or [1] + [2] * (n - 1)
        strides = [tuple([s] * dim) if isinstance(s, int) else tuple(s) for s in strides]
        assert len(strides) == n and all(len(s) == dim for s in strides)
        self.dim, self.in_channels, self.n_classes = dim, in_channels, n_classes
        self.features, self.strides, self.deep_supervision = list(features), [list(s) for s in strides], deep_supervision
        Conv, ConvT, _ = _ops(dim)
        self.enc = nn.ModuleList([Stage(dim, in_channels if i == 0 else features[i - 1], f, strides[i]) for i, f in enumerate(features)])
        self.up = nn.ModuleList([ConvT(features[i], features[i - 1], strides[i], stride=strides[i]) for i in range(n - 1, 0, -1)])
        self.dec = nn.ModuleList([Stage(dim, 2 * features[i - 1], features[i - 1], 1) for i in range(n - 1, 0, -1)])
        self.heads = nn.ModuleList([Conv(features[i - 1], n_classes, 1) for i in range(n - 1, 0, -1)])

    def forward(self, x):
        skips = []
        for st in self.enc:
            x = st(x); skips.append(x)
        x = skips.pop()
        outs = []
        for up, dec, head in zip(self.up, self.dec, self.heads):
            x = dec(torch.cat([up(x), skips.pop()], 1))
            outs.append(head(x))
        outs = outs[::-1]                                                  # full resolution first
        return outs if (self.training and self.deep_supervision) else outs[0]

    def divisor(self):
        """the input size along each axis must be a multiple of this (the product of the strides)"""
        d = [1] * self.dim
        for s in self.strides:
            d = [a * b for a, b in zip(d, s)]
        return d

    def to_spec(self):
        return dict(dim=self.dim, in_channels=self.in_channels, n_classes=self.n_classes, features=self.features,
                    strides=self.strides, deep_supervision=self.deep_supervision)

    @classmethod
    def from_spec(cls, s):
        return cls(s["dim"], s["in_channels"], s["n_classes"], tuple(s["features"]), s["strides"], s.get("deep_supervision", True))
