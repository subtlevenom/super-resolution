import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .kan_layer import KANLayer


class ConvSepKanAttentionCell(nn.Module):

    def __init__(self, in_channels):
        super(ConvSepKanAttentionCell, self).__init__()

        # q: what we want to attend to (spatial information)
        self.q_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.q_dw = nn.Conv2d(in_channels,
                              in_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              groups=in_channels)
        # k: what we use to calculate attention (channel information)
        self.k_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k_dw = nn.Conv2d(in_channels,
                              in_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              groups=in_channels)
        # v: what we use to calculate the output (channel information)
        self.v_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v_dw = nn.Conv2d(in_channels,
                              in_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              groups=in_channels)

        # output projection
        self.a_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x, k=None, v=None):

        b, c, h, w = x.shape

        if k is None:
            k = self.k_dw(self.k_proj(x))
        if v is None:
            v = self.v_dw(self.v_proj(x))

        q = self.q_dw(self.q_proj(x))

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


class ConvSepKanInputCell(nn.Module):

    def __init__(self, in_channels, res_channels):
        super(ConvSepKanInputCell, self).__init__()
        if res_channels > 0:
            self.rot_r = nn.Conv2d(res_channels, in_channels, kernel_size=1)
        else:
            self.rot_r = None
        self.rot = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x, xr):
        """
            xr: (b, in_channels, h, w)
            x: (b, out_channels, h, w)
        """
        if xr is None:
            return F.relu(self.rot(x))
        return F.relu(self.rot(x) + self.rot_r(xr))


class ConvSepKanWeightCell(nn.Module):

    def __init__(self, in_channels, res_channels):
        super(ConvSepKanWeightCell, self).__init__()
        if res_channels > 0:
            self.rot_r = nn.Conv2d(res_channels, in_channels, kernel_size=1)
        self.rot = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, w, wr):
        """
            w: (b, in_channels, h, w)
            wr: (b, res_channels, h, w)
        """
        if wr is None:
            return F.relu(self.rot(w))
        return self.rot(w) + self.rot_r(wr)


class ConvSepKanCell(torch.nn.Module):

    def __init__(self, in_channels, out_channels, grid_size, spline_order,
                 residual_std, grid_range):
        super(ConvSepKanCell, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kan_layer = KANLayer(in_dim=in_channels,
                                  out_dim=out_channels,
                                  grid_size=grid_size,
                                  spline_order=spline_order,
                                  residual_std=residual_std,
                                  grid_range=grid_range)

        # Arbitrary layers configuration fc
        self._kan_params_indices = [0]

        coef_len = np.prod(self.kan_layer.activation_fn.coef_shape)
        univariate_weight_len = np.prod(
            self.kan_layer.residual_layer.univariate_weight_shape)
        residual_weight_len = np.prod(
            self.kan_layer.residual_layer.residual_weight_shape)
        self._kan_params_indices.extend(
            [coef_len, univariate_weight_len, residual_weight_len])
        self._kan_params_indices = np.cumsum(self._kan_params_indices)

        self.size = self._kan_params_indices[-1]

    def forward(self, x, w):

        B, C, H, W = x.shape

        # weights (b * h * w, kan_size)
        w = w.permute(0, 2, 3, 1).reshape(-1, self.size)
        # img (b * h * w, kan_size)
        x = x.permute(0, 2, 3, 1).reshape(-1, self.in_channels)

        i, j = self._kan_params_indices[0], self._kan_params_indices[1]
        coef = w[:, i:j].view(-1, *self.kan_layer.activation_fn.coef_shape)
        i, j = self._kan_params_indices[1], self._kan_params_indices[2]
        univariate_weight = w[:, i:j].view(
            -1, *self.kan_layer.residual_layer.univariate_weight_shape)
        i, j = self._kan_params_indices[2], self._kan_params_indices[3]
        residual_weight = w[:, i:j].view(
            -1, *self.kan_layer.residual_layer.residual_weight_shape)
        x = self.kan_layer(x, coef, univariate_weight, residual_weight)
        x = x.view(B, H, W, self.out_channels).permute(0, 3, 1, 2)

        return x
