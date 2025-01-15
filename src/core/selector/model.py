from pathlib import Path
import torch
from ..config.model import ModelType
from ..config import Config
from typing import Union
from src.ml.models import (ConvSepKan, HKNet, HKLut, HKNetMask)
from hklut.weights import load_weights


class ModelSelector:

    def select(config: Config) -> Union[HKNet]:
        match config.model.type:
            case ModelType.conv_sep_kan:
                  return ConvSepKan(
                      in_channels=config.model.params.in_dims,
                      out_channels=config.model.params.out_dims,
                      kernel_sizes = config.model.params.kernel_sizes,
                      grid_size=config.model.params.grid_size,
                      spline_order=config.model.params.spline_order,
                      residual_std=config.model.params.residual_std,
                      grid_range=config.model.params.grid_range,
                      upscale=config.model.params.upscale
                  )
            case ModelType.hkmask:
                return HKNetMask(msb=config.model.params.msb,
                                 lsb=config.model.params.lsb,
                                 nf=config.model.params.n_filters,
                                 upscale=config.model.params.upscale)
            case ModelType.hknet:
                return HKNet(
                    msb=config.model.params.msb,
                    lsb=config.model.params.lsb,
                    nf=config.model.params.n_filters,
                    upscale=config.model.params.upscale,
                )
            case ModelType.hklut:
                lut_path = Path(config.save_dir).joinpath(
                    config.experiment).joinpath('logs/luts/')
                msb_weights, lsb_weights = load_weights(
                    msb=config.model.params.msb,
                    lsb=config.model.params.lsb,
                    upscale=config.model.params.upscale,
                    lut_path=lut_path)
                return HKLut(
                    msb_weights=msb_weights,
                    lsb_weights=lsb_weights,
                    msb=config.model.params.msb,
                    lsb=config.model.params.lsb,
                    upscale=config.model.params.upscale,
                )
            case _:
                raise ValueError(f'Unupported model type f{config.model.type}')
