import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .kan_layer import KANLayer


class ConvSepKanAttention(nn.Module):

    def __init__(self, in_channels:int, mid_channels:int, out_channels:int):
        super(ConvSepKanAttention, self).__init__()

        # q: what we want to attend to (spatial information)
        self.q_proj = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.q_dw = nn.Conv2d(mid_channels,
                              mid_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              groups=mid_channels)
        # k: what we use to calculate attention (channel information)
        self.k_proj = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.k_dw = nn.Conv2d(mid_channels,
                              mid_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              groups=mid_channels)
        # v: what we use to calculate the output (channel information)
        self.v_proj = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.v_dw = nn.Conv2d(mid_channels,
                              mid_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              groups=mid_channels)

        # output projection
        self.a_proj = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        self.a_dw = nn.Conv2d(out_channels,
                              out_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              groups=out_channels)

    def forward(self, x, q=None, k=None, v=None):

        b, c, h, w = x.shape

        if q is None:
            q = self.q_dw(self.q_proj(x))
        if k is None:
            k = self.k_dw(self.k_proj(x))
        if v is None:
            v = self.v_dw(self.v_proj(x))

        q = rearrange(q, 'b c h w -> b c (h w)')
        k = rearrange(k, 'b c h w -> b c (h w)')
        v = rearrange(v, 'b c h w -> b c (h w)')

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        a = (q @ k.transpose(-2, -1)).softmax(dim=-1)

        x = a @ v
        x = rearrange(x, 'b c (h w) -> b c h w', h=h, w=w)
        x = self.a_proj(x)

        return x