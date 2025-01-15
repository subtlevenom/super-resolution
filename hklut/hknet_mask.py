import torch
import torch.nn as nn

from .units import *
from .hknet import HKNet
from .blocks import MaskBlock
from .att import AttBlock


class HKNetMask(nn.Module):

    def __init__(self, msb='hdb', lsb='hd', nf=64, upscale=1, act=nn.ReLU):
        super(HKNetMask, self).__init__()

        self.upscale = upscale

        self.hknet = HKNet(msb, lsb, nf, upscale, act, intermediate=True)
        self.mask = AttBlock(in_channels=3, out_channels=1)

    def _upscale(self, x):
        return F.interpolate(x, scale_factor=self.upscale, mode='bilinear')

    def _downscale(self, x):
        return F.interpolate(x, scale_factor=1./self.upscale, mode='bilinear')

    def forward(self, x, mask=1.0):
        y, cb, cr, sb = self.hknet(x)
        y = torch.concat([y + mask * sb, cb, cr], axis=1)
        return y
