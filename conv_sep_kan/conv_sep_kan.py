import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv_sep_kan_layer import ConvSepKanLayer


class ConvSepKan(torch.nn.Module):
    """ Input features BxCxN """

    def __init__(self, in_channels:list, out_channels:list, kernel_sizes:list, grid_size, spline_order, residual_std, grid_range, upscale):
        super(ConvSepKan, self).__init__()

        kan_channels = [s for s in zip(in_channels, out_channels)]
        self.upscale = upscale

        self.layers = nn.ModuleList()
        self.shuffle = nn.PixelShuffle(self.upscale)

        for in_ch, out_ch in kan_channels:
            layer = ConvSepKanLayer(in_channels=in_ch,
                                    out_channels=out_ch,
                                    grid_size=grid_size,
                                    spline_order=spline_order,
                                    residual_std=residual_std,
                                    grid_range=grid_range)
            self.layers.append(layer)

    def forward(self, x:torch.Tensor):
        cb = x[:,1:2,:,:]
        cr = x[:,2:,:,:]

        cb = F.interpolate(cb, scale_factor=self.upscale, mode='bilinear')
        cr = F.interpolate(cr, scale_factor=self.upscale, mode='bilinear')

        for layer in self.layers:
            y = layer(x[:,:1])
        y = self.shuffle(y)

        x = torch.concat([y, cb, cr], dim=1)

        return x
