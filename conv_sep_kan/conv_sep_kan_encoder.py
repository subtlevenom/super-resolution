from typing import Tuple, List
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import UNet
from .unet_blocks import LayerNorm, GatedFFN, FFN
from .gazetr import GazeTR


class ConvSepKanEncoder(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.encoder = GazeTR(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.encoder(x)
        return x
