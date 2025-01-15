from pydantic import BaseModel
from enum import Enum
from typing import List, Union


class ModelType(str, Enum):
    hknet = 'hknet'
    hklut = 'hklut'
    hkmask = 'hkmask'


class HKNetModelParams(BaseModel):
    upscale: int = 1
    msb: str = 'hdb'
    lsb: str = 'hd'
    n_filters: int = 64


class Model(BaseModel):
    type: ModelType
    params: Union[
        HKNetModelParams,
    ]
