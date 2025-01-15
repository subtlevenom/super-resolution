import torch
import torch.nn.functional as F
from .conv_sep_kan_encoder import ConvSepKanEncoder
from .conv_sep_kan_cells import ConvSepKanCell


class ConvSepKanLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, grid_size, spline_order,
                 residual_std, grid_range):
        super(ConvSepKanLayer, self).__init__()

        self.kan_cell = ConvSepKanCell(in_channels, out_channels, grid_size,
                                       spline_order, residual_std, grid_range)
        self.encoder = ConvSepKanEncoder(in_channels, self.kan_cell.size)

    def forward(self, x):

        w = self.encoder(x)
        x = self.kan_cell(x, w)

        return x
