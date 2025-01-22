""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False), nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,
                      out_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False), nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels,
                                         in_channels // 2,
                                         kernel_size=2,
                                         stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SelfAttention(nn.Module):

    def __init__(self, dim, num_heads, bias=False):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature_a = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature_v = nn.Parameter(torch.ones(num_heads, 1, 1))

        # q: what we want to attend to (spatial information)
        self.q_proj = nn.Conv2d(dim,
                                dim,
                                kernel_size=3,
                                padding=1,
                                stride=2,
                                padding_mode='reflect',
                                groups=dim,
                                bias=bias)
        # k: what we use to calculate attention (channel information)
        self.k_proj = nn.Conv2d(dim,
                                dim,
                                kernel_size=3,
                                padding=1,
                                stride=2,
                                padding_mode='reflect',
                                bias=bias)
        # v: what we use to calculate the output (channel information)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # a: anchor information (reduced spatial information and channel information)
        self.a_proj = nn.Sequential(
            nn.Conv2d(dim,
                      dim,
                      kernel_size=3,
                      padding=1,
                      stride=2,
                      padding_mode='reflect',
                      groups=dim,
                      bias=bias), nn.Conv2d(dim, dim // 2, kernel_size=1))
        # output projection
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, w=None):
        b, c, h, w = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        if w is not None:
            v = v * w
        a = self.a_proj(x)

        q = rearrange(q,
                      'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k,
                      'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v,
                      'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        a = rearrange(a,
                      'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        a = torch.nn.functional.normalize(a, dim=-1)

        # Q - C×(H/s×W/s), K - C×(H/s×W/s), V - C×(H×W), A - C/r×(H/s×W/s)

        # transposed self-attention with attention map of shape (C×C)
        attn_a = (q @ a.transpose(-2, -1)) * self.temperature_a
        attn_a = attn_a.softmax(dim=-1)

        attn_k = (a @ k.transpose(-2, -1)) * self.temperature_v
        attn_k = attn_k.softmax(dim=-1)

        out_v = (attn_k @ v)

        out = (attn_a @ out_v)

        out = rearrange(out,
                        'b head c (h w) -> b (head c) h w',
                        head=self.num_heads,
                        h=h,
                        w=w)

        out = self.project_out(out)
        return out

class _SelfAttention(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int = 5):
        super().__init__()
        self.att = nn.MultiheadAttention(embed_dim=embed_dim,
                                         num_heads=num_heads,
                                         batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        x = rearrange(x, 'b c h w -> b (h w) c')
        x, _ = self.att(x, x, x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x


class LayerNorm(nn.Module):

    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = nn.LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.body(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x


class FFN(nn.Module):
    """
    Feed-forward Network with Depth-wise Convolution
    """

    def __init__(self, in_features, out_features=None, hidden_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.pointwise1 = nn.Conv2d(in_features,
                                    hidden_features,
                                    kernel_size=1)
        self.depthwise = nn.Conv2d(hidden_features,
                                   hidden_features,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   dilation=1,
                                   groups=hidden_features)
        self.pointwise2 = nn.Conv2d(hidden_features,
                                    out_features,
                                    kernel_size=1)
        self.act_layer = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pointwise1(x)
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise2(x)
        return x


class GatedFFN(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        act_layer,
    ):
        super().__init__()
        mlp_channels = 2 * in_channels

        self.fn_1 = nn.Sequential(
            nn.Conv2d(in_channels, mlp_channels, kernel_size=1, padding=0),
            act_layer,
        )
        self.fn_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.gate = nn.Conv2d(in_channels,
                              in_channels,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2,
                              groups=mlp_channels // 2)

    def forward(self, x: torch.Tensor):
        x = self.fn_1(x)
        x, gate = torch.chunk(x, 2, dim=1)

        gate = self.gate(gate)
        x = x * gate

        x = self.fn_2(x)
        return x
