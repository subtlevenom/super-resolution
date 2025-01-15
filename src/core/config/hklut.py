from pydantic import BaseModel


class HklutParams(BaseModel):
    hklut_loss_coef: float = 0.5
    main_loss_coef: float = 0.5


class HklutTransfer(BaseModel):
    params: HklutParams
