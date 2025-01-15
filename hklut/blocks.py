import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms.functional import gaussian_blur


class MaskBlock(nn.Module):

    def __init__(self):
        super(MaskBlock, self).__init__()

        self.alpha = nn.Parameter(torch.tensor(7.))
        self.betta = nn.Parameter(torch.tensor(1.))
        self.sigma = 2.0

    def forward(self, x):

        y = gaussian_blur(x, kernel_size=9, sigma=self.sigma)
        z = torch.abs(x - y)

        z = z - torch.min(z)
        if torch.max(z) > 0:
            z = z / torch.max(z)

        z = self.alpha * (z - F.relu(self.betta))
        z = 2 * F.sigmoid(z) - 1

        return z
