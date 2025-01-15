import torch
import torch.nn as nn
from einops import rearrange


class AttBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 n_features=1024,
                 f_channels=11):
        super(AttBlock, self).__init__()

        self.q_proj = nn.Conv2d(in_channels, f_channels, kernel_size=1)

        self.keys = nn.Parameter(torch.rand(1, f_channels, n_features))
        self.basis = nn.Parameter(torch.rand(1, n_features, out_channels))

    def forward(self, x):

        B, C, H, W = x.shape

        q = self.q_proj(x)
        q = rearrange(q, 'b c h w -> b (h w) c')
        q = torch.nn.functional.normalize(q, dim=-1)

        k = (q @ self.keys).softmax(dim=-1)
        v = k @ self.basis

        v = rearrange(v, 'b (h w) c -> b c h w', h=H, w=W)

        return v
