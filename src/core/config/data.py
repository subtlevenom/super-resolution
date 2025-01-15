from pydantic import BaseModel
from enum import Enum


class DataType(str, Enum):
    huawei = 'huawei' 


class DataPathes(BaseModel):
    source: str
    target: str = None


class Data(BaseModel):
    type: DataType = DataType.huawei
    train: DataPathes
    val: DataPathes
    test: DataPathes
    predict: DataPathes