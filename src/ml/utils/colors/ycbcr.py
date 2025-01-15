from __future__ import annotations

import torch
from einops import rearrange

O = torch.Tensor([16., 128., 128.]) / 255.
T = torch.Tensor([[0.256788235294118, 0.504129411764706, 0.097905882352941],
                  [-0.148223529411765, -0.290992156862745, 0.439215686274510],
                  [0.439215686274510, -0.367788235294118,
                   -0.071427450980392]]).T


def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YCbCr. SDTV range conversion.

    .. image:: _static/img/rgb_to_ycbcr.png

    Args:
        image: RGB Image in range [0,1] to be converted to YCbCr with shape :math:`(*, 3, H, W)`.

    Returns:
        YCbCr version of the image with shape :math:`(*, 3, H, W)`.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_ycbcr(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(
            f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    image = image.to(torch.float32)

    C, H, W = image.shape[-3:]

    rgb = image.view(-1, C, H, W).permute(0, 2, 3, 1).view(-1, C)
    ycbcr = rgb @ T + O
    ycbcr = ycbcr.view(-1, H, W, C).permute(0, 3, 1, 2).view(image.shape)

    return ycbcr


def ycbcr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an YCbCr image to RGB. SDTV range conversion.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: YCbCr Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = ycbcr_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(
            f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    image = image.to(torch.float32).cpu()

    C, H, W = image.shape[-3:]

    ycbcr = image.view(-1, C, H, W).permute(0, 2, 3, 1).reshape(-1, C)
    rgb = (ycbcr - O) @ T.inverse()
    rgb = rgb.reshape(-1, H, W, C).permute(0, 3, 1, 2).view(image.shape)

    return rgb.to(device=image.device)


class RgbToYcbcr(torch.nn.Module):
    r"""Convert an image from RGB to YCbCr.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        YCbCr version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> ycbcr = RgbToYcbcr()
        >>> output = ycbcr(input)  # 2x3x4x5
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return rgb_to_ycbcr(image)


class YcbcrToRgb(torch.nn.Module):
    r"""Convert an image from YCbCr to Rgb.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        RGB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = YcbcrToRgb()
        >>> output = rgb(input)  # 2x3x4x5
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return ycbcr_to_rgb(image)
