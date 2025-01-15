from pydantic import BaseModel
from enum import Enum
from typing import List, Union


class ModelType(str, Enum):
    hknet = 'hknet'
    hklut = 'hklut'
    hkmask = 'hkmask'
    conv_sep_kan = 'conv_sep_kan'

class ConvSepKanModelParams(BaseModel):
      in_dims: List[int]
      out_dims: List[int]
      kernel_sizes: List[int]
      grid_size: int
      spline_order: int
      residual_std: float
      upscale: int
      grid_range: List[float]

class HKNetModelParams(BaseModel):
    upscale: int = 1
    msb: str = 'hdb'
    lsb: str = 'hd'
    n_filters: int = 64

class Model(BaseModel):
    type: ModelType
    params: Union[
        ConvSepKanModelParams,
        HKNetModelParams,
    ]
