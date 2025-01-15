from ..config.pipeline import PipelineType
from ..config import Config
from typing import Union
from src.ml.pipelines import (
    DefaultPipeline,
    GanPipeline,
    NetLutPipeline,
    CalGanPipeline,
    MaskPipeline,
)
from src.ml.models import (HKNet, HKNetMask, HKLut)


class PipelineSelector:

    def select(
        config: Config, model: Union[HKNet, HKNetMask, HKLut]
    ) -> Union[DefaultPipeline, MaskPipeline]:
        match config.pipeline.type:
            case PipelineType.mask:
                return MaskPipeline(
                    model=model,
                    optimiser=config.pipeline.params.optimizer,
                    lr=config.pipeline.params.lr,
                    weight_decay=config.pipeline.params.weight_decay,
                    stage=config.pipeline.params.stage)
            case PipelineType.default:
                return DefaultPipeline(
                    model=model,
                    optimiser=config.pipeline.params.optimizer,
                    lr=config.pipeline.params.lr,
                    weight_decay=config.pipeline.params.weight_decay)
            case PipelineType.gan:
                return GanPipeline(
                    model=model,
                    optimiser=config.pipeline.params.optimizer,
                    lr=config.pipeline.params.lr,
                    weight_decay=config.pipeline.params.weight_decay)
            case PipelineType.netlut:
                return NetLutPipeline(
                    model=model,
                    optimiser=config.pipeline.params.optimizer,
                    lr=config.pipeline.params.lr,
                    weight_decay=config.pipeline.params.weight_decay,
                    hklut_loss_coef=config.lut_transfer.params.hklut_loss_coef,
                    main_loss_coef=config.lut_transfer.params.main_loss_coef,
                    msb_order=config.model.params.msb,
                    lsb_order=config.model.params.lsb,
                    accelerator=config.accelerator,
                    num_acc=config.devices[0],
                    upscale=config.model.params.upscale)
            case PipelineType.calgan:
                return CalGanPipeline(
                    model=model,
                    optimiser=config.pipeline.params.optimizer,
                    lr=config.pipeline.params.lr,
                    weight_decay=config.pipeline.params.weight_decay)
            case _:
                raise ValueError(
                    f'Unupported pipeline type f{config.pipeline.type}')
