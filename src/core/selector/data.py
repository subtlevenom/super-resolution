from ..config.data import DataType
from ..config import Config
from typing import Union
from src.ml.datasets import (
    HuaweiDataModule,
)


class DataSelector:
    def select(config: Config) -> Union[HuaweiDataModule, HuaweiDataModule]:
        match config.data.type:
            case DataType.huawei:
                return HuaweiDataModule(
                    train_a=config.data.train.source,
                    train_b=config.data.train.target,
                    val_a=config.data.val.source,
                    val_b=config.data.val.target,
                    test_a=config.data.test.source,
                    test_b=config.data.test.target,
                    predict_a=config.data.predict.source,
                    upscale=config.model.params.upscale,
                    train_batch_size=config.pipeline.params.batch_size,
                    val_batch_size=config.pipeline.params.val_batch_size,
                    test_batch_size=config.pipeline.params.test_batch_size,
                    predict_batch_size=config.pipeline.params.predict_batch_size,
                )
            case _:
                raise ValueError(f'Unupported data type f{config.data.type}')
