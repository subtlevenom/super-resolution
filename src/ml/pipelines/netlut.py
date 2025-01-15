import torch
from torch import nn
import lightning as L
from torch import optim
from ..models import HKNet, HKLut
from src.core import Logger
from hklut.utils import decode_bit_mask
from hklut.weights import get_input_tensor
from ..metrics import (
    PSNR,
    SSIM,
    DeltaE
)
from ..losses import GANFeatLoss

pixel_dict = {'hdtb': 4, 'hdb': 3, 'hd': 2}


class NetLutPipeline(L.LightningModule):
    def __init__(self,
        model: HKNet,
        optimiser: str = 'adam',
        lr: float = 1e-3,
        weight_decay: float = 0,
        hklut_loss_coef: float = 0.5,
        main_loss_coef: float = 0.5,
        bit_mask: str = '11110000',
        msb_order: str = "hdb",
        lsb_order: str = "hd",
        accelerator: str = "gpu",
        num_acc: int = 0,
        upscale: int = 0
    ) -> None:
        super(NetLutPipeline, self).__init__()

        self.model = model
        self.optimizer_type = optimiser
        self.lr = lr
        self.weight_decay = weight_decay
        self.mae_loss = nn.L1Loss(reduction='mean')
        self.ssim_loss = SSIM(data_range=(0, 1))
        self.de_metric = DeltaE()
        self.ssim_metric = SSIM(data_range=(0, 1))
        self.psnr_metric = PSNR(data_range=(0, 1))

        self.save_hyperparameters(ignore=['model'])

        self.hklut_loss_coef = hklut_loss_coef
        self.main_loss_coef = main_loss_coef
        self.msb_order = msb_order
        self.lsb_order = lsb_order

        self.msb_bits, self.lsb_bits, self.msb_step, self.lsb_step = decode_bit_mask(bit_mask)
        self.devices = f'cuda:{num_acc}' if accelerator == 'gpu' else 'cpu'
        self.upscale = upscale

        self.hklut = None

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred = self.model(x)
        return pred

    def training_step(self, batch, batch_idx):
        self.update_hklut()

        inputs, targets = batch
        predictions = self(inputs)
        mae_loss = self.mae_loss(predictions, targets)
        ssim_loss = self.ssim_loss(predictions, targets)
        loss = mae_loss + (1 - ssim_loss) * 0.15

        with torch.no_grad():
            lut_predictions = torch.stack([self.hklut(inputs[i:i+1]) for i in range(inputs.size(0))], dim=0).squeeze(1)
            lut_mae_loss = self.mae_loss(lut_predictions, targets)
            lut_ssim_loss = self.ssim_loss(lut_predictions, targets)
            lut_loss = lut_mae_loss + (1 - lut_ssim_loss) * 0.15

            loss *= self.main_loss_coef
            loss += lut_loss * self.hklut_loss_coef

        self.log('train_loss', loss, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        self.update_hklut()
        inputs, targets = batch
        predictions = self(inputs)
        mae_loss = self.mae_loss(predictions, targets)
        ssim_loss = self.ssim_loss(predictions, targets)
        loss = mae_loss + (1 - ssim_loss) * 0.15

        with torch.no_grad():
            lut_predictions = torch.stack([self.hklut(inputs[i:i+1]) for i in range(inputs.size(0))], dim=0).squeeze(1)
            lut_mae_loss = self.mae_loss(lut_predictions, targets)
            lut_ssim_loss = self.ssim_loss(lut_predictions, targets)
            lut_loss = lut_mae_loss + (1 - lut_ssim_loss) * 0.15

            loss *= self.main_loss_coef
            loss += lut_loss * self.hklut_loss_coef

        psnr_metric = self.psnr_metric(lut_predictions, targets)
        ssim_metric = self.ssim_metric(lut_predictions, targets)
        de_metric = self.de_metric(lut_predictions, targets)

        self.log('val_psnr', psnr_metric, prog_bar=True, logger=True)
        self.log('val_ssim', ssim_metric, prog_bar=True, logger=True)
        self.log('val_de', de_metric, prog_bar=True, logger=True)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        self.update_hklut()
        inputs, targets = batch
        predictions = self(inputs)
        mae_loss = self.mae_loss(predictions, targets)
        ssim_loss = self.ssim_loss(predictions, targets)
        loss = mae_loss + (1 - ssim_loss) * 0.15

        with torch.no_grad():
            lut_predictions = torch.stack([self.hklut(inputs[i:i+1]) for i in range(inputs.size(0))], dim=0).squeeze(1)
            lut_mae_loss = self.mae_loss(lut_predictions, targets)
            lut_ssim_loss = self.ssim_loss(lut_predictions, targets)
            lut_loss = lut_mae_loss + (1 - lut_ssim_loss) * 0.15

            loss *= self.main_loss_coef
            loss += lut_loss * self.hklut_loss_coef

        psnr_metric = self.psnr_metric(lut_predictions, targets)
        ssim_metric = self.ssim_metric(lut_predictions, targets)
        de_metric = self.de_metric(lut_predictions, targets)

        self.log('test_psnr', psnr_metric, prog_bar=True, logger=True)
        self.log('test_ssim', ssim_metric, prog_bar=True, logger=True)
        self.log('test_de', de_metric, prog_bar=True, logger=True)
        self.log('test_loss', loss, prog_bar=True, logger=True)
        return {'loss': loss}

    def predict_step(self, batch, batch_idx):
        inputs = batch
        output = self(inputs)
        return output

    def process_lut(self, model, kind, bits, step, bit_order, device):
        lut_tensor = {}
        luts = filter(lambda a: kind in a and 'lut' in a , dir(model))

        for lut in luts:
            unit = model.__getattr__(lut)
            lut_input = get_input_tensor(bits, step, n_pixels=pixel_dict[bit_order])

            lut_input = unit.get_lut_input(lut_input).to(device)

            batch_size = 1000
            num_batches = (lut_input.size(0) + batch_size -1) // batch_size
            outputs = []

            for b in range(num_batches):
                batch_input = lut_input[b * batch_size:(b + 1) * batch_size]
                batch_output = unit(batch_input)
                results = torch.floor(torch.clamp(batch_output, -1, 1) * 127).to(torch.int8).cpu()
                outputs.append(results)

            results = torch.cat(outputs, dim=0).numpy()
            lut_key = lut[-1]
            lut_tensor[lut_key] = results

        return lut_tensor

    def update_hklut(self):
        msb_luts_tensor = self.process_lut(
            model=self.model,
            kind='msb',
            bits=self.msb_bits,
            step=self.msb_step,
            bit_order=self.msb_order,
            device=self.devices
        )

        lsb_luts_tensor = self.process_lut(
            model=self.model,
            kind='lsb',
            bits=self.lsb_bits,
            step=self.lsb_step,
            bit_order=self.lsb_order,
            device=self.devices
        )

        msb_weights = [msb_luts_tensor[ktype] for ktype in self.msb_order]
        lsb_weights = [lsb_luts_tensor[ktype] for ktype in self.lsb_order]
        self.hklut = HKLut(msb_weights, lsb_weights, msb=self.msb_order, lsb=self.lsb_order, upscale=self.upscale).to(self.devices)
