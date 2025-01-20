from typing import Tuple, List
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import UNet
from .unet_blocks import LayerNorm, GatedFFN


class ConvSepKanEncoder(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.encoder = UNet(in_channels, out_channels)
        self.ffn = GatedFFN(out_channels, out_channels, kernel_size=3, act_layer=nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.encoder(x)
        # x = self.norm(x)
        x = self.ffn(x)
        return x
