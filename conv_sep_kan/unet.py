""" Full assembly of the parts to form the complete network """

import torch
from .unet_blocks import *


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.proj = DoubleConv(in_channels, 4)

        self.down1 = Down(4, 8)
        self.down2 = Down(8, 16)
        self.down3 = Down(16, 32)
        self.down4 = Down(32, 64)

        self.up1 = Up(64, 32)
        self.up2 = Up(32, 16)
        self.up3 = Up(16, 8)
        self.up4 = Up(8, 4)

        self.att1 = SelfAttention(dim=4, num_heads=2)
        self.att2 = SelfAttention(dim=8, num_heads=2)
        self.att3 = SelfAttention(dim=16, num_heads=4)
        self.att4 = SelfAttention(dim=32, num_heads=4)

        self.reproj = OutConv(4, out_channels)

    def forward(self, x):
        x1 = self.proj(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        y = self.up1(x5, self.att4(x4))
        y = self.up2(y, self.att3(x3))
        y = self.up3(y, self.att2(x2))
        y = self.up4(y, self.att1(x1))

        return self.reproj(y)
