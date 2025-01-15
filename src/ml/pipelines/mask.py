import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from torch import optim
from ..models import HKNet, MaskBlock, HKNetMask
from src.core import Logger
from ..metrics import (
    PSNR,
    SSIM,
    DeltaE
)
from ..losses import GANFeatLoss


class MaskPipeline(L.LightningModule):
    def __init__(self,
        model: HKNetMask,
        optimiser: str = 'adam',
        lr: float = 1e-3,
        weight_decay: float = 0,
        stage: int = 0
    ) -> None:
        super(MaskPipeline, self).__init__()

        self.model = model
        self.optimizer_type = optimiser
        self.lr = lr
        self.weight_decay = weight_decay
        self.stage = stage
        self.mae_loss = nn.L1Loss(reduction='mean')
        self.ssim_loss = SSIM(data_range=(0, 1))
        self.de_metric = DeltaE()
        self.ssim_metric = SSIM(data_range=(0, 1))
        self.psnr_metric = PSNR(data_range=(0, 1))

        self.save_hyperparameters(ignore=['model'])

    def setup(self, stage: str) -> None:
        '''
        Initialize model weights before training
        '''
        if stage == 'fit' or stage is None:
            for m in self.model.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

        Logger.info('Initialized model weights with [bold green]HKNet[/bold green] pipeline.')

    def configure_optimizers(self):
        if self.optimizer_type == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f'unsupported optimizer_type: {self.optimizer_type}')
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=500, T_mult=1, eta_min=1e-5
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def forward(self, x: torch.Tensor, y: torch.Tensor=None) -> torch.Tensor:

        # prediction
        if y is None:
            return self.model(x, mask=1.0)

        # train mask
        if self.stage == 1:
            # hknet
            with torch.no_grad():
                xc, cb, cr, sb = self.model.hknet(x)
            # mask
            x = self.model._upscale(x)
            d = torch.abs(x - y)
            mask = F.relu(self.model.mask(d))
            # ycbcr
            return torch.concat([xc + mask * sb, cb, cr], axis=1)

        # tune hknet
        if self.stage == 2:
            # mask
            with torch.no_grad():
                y = self.model._downscale(y)
                d = torch.abs(x - y)
                mask = F.relu(self.model.mask(d))
                # mx = torch.max(mask)
                # if mx > 0:
                    # mask = mask / mx
            # new x
            x = mask * x + (1 - mask) * y  #mask * x + (1 - mask) * y

            # model
            return self.model(x)

            """
            import os
            import torchvision
            from src.ml.utils.colors.ycbcr import ycbcr_to_rgb
            outputs = F.relu(torch.concat([mask,mask,mask],dim=1))
            save_path = os.path.join('/home/korepanov/work/huawei/lut-python/.experiments/huawei.hknet.mask/logs/figures', f"train_x.png")
            torchvision.utils.save_image(outputs[0], save_path)
            """

        # train hklut
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs, targets)

        mae_loss = self.mae_loss(predictions, targets)
        ssim_loss = self.ssim_loss(predictions, targets)
        loss = mae_loss + (1 - ssim_loss) * 0.15

        self.log('train_loss', loss, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs, targets)

        mae_loss = self.mae_loss(predictions, targets)
        ssim_loss = self.ssim_loss(predictions, targets)
        loss = mae_loss + (1 - ssim_loss) * 0.15

        psnr_metric = self.psnr_metric(predictions, targets)
        ssim_metric = self.ssim_metric(predictions, targets)
        de_metric = self.de_metric(predictions, targets)

        self.log('val_psnr', psnr_metric, prog_bar=True, logger=True)
        self.log('val_ssim', ssim_metric, prog_bar=True, logger=True)
        self.log('val_de', de_metric, prog_bar=True, logger=True)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs, targets)

        mae_loss = self.mae_loss(predictions, targets)
        ssim_loss = self.ssim_loss(predictions, targets)
        loss = mae_loss + (1 - ssim_loss) * 0.15

        psnr_metric = self.psnr_metric(predictions, targets)
        ssim_metric = self.ssim_metric(predictions, targets)
        de_metric = self.de_metric(predictions, targets)

        self.log('test_psnr', psnr_metric, prog_bar=True, logger=True)
        self.log('test_ssim', ssim_metric, prog_bar=True, logger=True)
        self.log('test_de', de_metric, prog_bar=True, logger=True)
        self.log('test_loss', loss, prog_bar=True, logger=True)
        return {'loss': loss}

    def predict_step(self, batch, batch_idx):
        inputs = batch
        output = self(inputs)
        return output
