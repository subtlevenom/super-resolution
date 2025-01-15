from pathlib import Path
import torch
from ..config.model import ModelType
from ..config import Config
from typing import Union
from src.ml.models import (HKNet, HKLut, HKNetMask)
from hklut.weights import load_weights


class ModelSelector:

    def select(config: Config) -> Union[HKNet]:
        match config.model.type:
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
