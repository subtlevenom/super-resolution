import torch
from torch import nn
import random
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import RandomCrop
from torchvision.transforms import Compose
from typing import Tuple

class PairTransform(nn.Module):
    def __init__(self, crop_size: int = 256, p: float = 0.5, seed: int = 42, upscale: int = 1) -> None:
        super().__init__()
        self.p = p
        self.upscale = upscale
        self.source_crop_size = int(crop_size / self.upscale)
        random.seed(seed)

    def forward(self, source_image: torch.Tensor, target_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        h = int(target_image.shape[-2] / self.upscale)
        w = int(target_image.shape[-1] / self.upscale)
        source_image = F.center_crop(source_image, output_size=(h, w))

        i_source, j_source, h_source, w_source = RandomCrop.get_params(
            source_image, output_size=(self.source_crop_size, self.source_crop_size))

        if self.upscale == 1:
            i_target, j_target, h_target, w_target = i_source, j_source, h_source, w_source
        else:
            i_target = i_source * self.upscale
            j_target = j_source * self.upscale
            h_target = h_source * self.upscale
            w_target = w_source * self.upscale

        source_image = F.crop(source_image, i_source, j_source, h_source, w_source)
        target_image = F.crop(target_image, i_target, j_target, h_target, w_target)

        if random.random() > self.p:
            source_image = F.hflip(source_image)
            target_image = F.hflip(target_image)

        if random.random() > self.p:
            source_image = F.vflip(source_image)
            target_image = F.vflip(target_image)

        return source_image, target_image
