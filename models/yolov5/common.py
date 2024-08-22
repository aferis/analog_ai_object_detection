"""
Common YOLOv5 modules
"""

import logging
import math
import warnings
import torch
import torch.nn as nn

LOGGER = logging.getLogger("MainLogger")

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, ch_in, ch_out, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(ch_out)
        self.activation = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.activation(self.conv(x))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, ch_in, ch_out, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(ch_out * e)  # hidden channels
        self.cv1 = Conv(ch_in, c_, 1, 1)
        self.cv2 = Conv(c_, ch_out, 3, 1, g=g)
        self.add = shortcut and ch_in == ch_out

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, ch_in, ch_out, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, num_modules, shortcut, groups, expansion
        super().__init__()
        c_ = int(ch_out * e)  # hidden channels
        self.cv1 = Conv(ch_in, c_, 1, 1)
        self.cv2 = Conv(ch_in, c_, 1, 1)
        self.cv3 = Conv(2 * c_, ch_out, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, ch_in, ch_out, k=(5, 9, 13)):
        super().__init__()
        c_ = ch_in // 2  # hidden channels
        self.cv1 = Conv(ch_in, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), ch_out, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, ch_in, ch_out, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(ch_in * 4, ch_out, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)