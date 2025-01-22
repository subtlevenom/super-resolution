import torch
import torch.nn as nn
import numpy as np
import math
import copy
import torch.nn.functional as F
from .resnet import resnet18
from .unet_blocks import FFN, GatedFFN
from einops import rearrange


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src):
        output = src
        for layer in self.layers:
            output = layer(output)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, in_channels, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(in_channels, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(in_channels, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, in_channels)

        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, src):
        # src_mask: Optional[Tensor] = None,
        # src_key_padding_mask: Optional[Tensor] = None):
        # pos: Optional[Tensor] = None):

        q = k = src
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class GazeTR(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(GazeTR, self).__init__()

        mid_channels = 64
        nhead = 2
        dim_feedforward = 32
        dropout = 0.1
        num_layers = 2

        # self.base_model = resnet18(in_channels=in_channels, out_channels=mid_channels, pretrained=False)
        self.ffn1 = FFN(in_features=in_channels, out_features=mid_channels)
        self.ffn2 = GatedFFN(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, act_layer=nn.ReLU())
        self.ffn3 = FFN(in_features=mid_channels, out_features=mid_channels)
        self.ffn4 = GatedFFN(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, act_layer=nn.ReLU())

        # d_model: dim of Q, K, V
        # nhead: seq num
        # dim_feedforward: dim of hidden linear layers
        # dropout: prob

        encoder_layer = TransformerEncoderLayer(mid_channels, nhead, dim_feedforward, dropout)

        encoder_norm = nn.LayerNorm(mid_channels)
        # num_encoder_layer: deeps of layers

        self.encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        self.feed = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

    def forward(self, x):

        x1 = self.ffn1(x)
        x2 = self.ffn2(x1)
        x3 = F.sigmoid(x2)
        x4 = self.ffn3(x3)
        x5 = self.ffn4(x4)
        x6 = F.sigmoid(x5)

        B, C, H, W = x6.shape
        x6 = rearrange(x6, 'b c h w -> (h w) b c')

        # feature is [HW, batch, channel]
        x = self.encoder(x6)

        x = rearrange(x, '(h w) b c -> b c h w', h=H, w=W)

        gaze = self.feed(x)

        return gaze
