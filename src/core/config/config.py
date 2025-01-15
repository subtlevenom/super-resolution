from typing import Union, List
from typing_extensions import Optional
from pydantic import BaseModel

from rich.syntax import Syntax
from rich import print

from .data import Data
from .model import Model
from .pipeline import Pipeline
from .hklut import HklutTransfer

import yaml


class Config(BaseModel):
    experiment: str = 'huawei'
    save_dir: str = 'experiments'
    resume: bool = False
    model: Model
    data: Data
    pipeline: Pipeline = Pipeline()
    accelerator: Union[str,int] = 'gpu'
    devices: Union[int,List[int]] = [0,1]
    lut_transfer: Optional[HklutTransfer] = None

    def print(self) -> None:
        str = yaml.dump(self.model_dump())
        syntax = Syntax(str, "yaml")
        print(syntax)
