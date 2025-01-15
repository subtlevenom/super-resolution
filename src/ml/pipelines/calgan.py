import torch
from torch import nn
import lightning as L
from torch import optim
from ..models import HKNet, MOD
from src.core import Logger
from ..metrics import (
    PSNR,
    SSIM,
    DeltaE
)
from ..losses import (
    L1Loss,
    PerceptualLoss,
    GANLoss,
    cos_similarity,
    lda_loss,
    load_balancing_loss
)


class CalGanPipeline(L.LightningModule):
    def __init__(self,
                 model: HKNet,
                 optimiser: str = 'adam',
                 lr: float = 1e-3,
                 weight_decay: float = 0,
                 ) -> None:
        super(CalGanPipeline, self).__init__()

        self.generator = model
        self.discriminator = MOD()

        self.optimizer_type = optimiser
        self.lr_G = lr
        self.lr_D = 5e-4
        self.weight_decay = weight_decay

        self.pixel_loss = L1Loss(loss_weight=1e-2)
        self.perceptual_loss = PerceptualLoss(
            layer_weights={
                'conv1_2': 0.1,
                'conv2_2': 0.1,
                'conv3_4': 1,
                'conv4_4': 1,
                'conv5_4': 1
            },
            vgg_type='vgg19',
            use_input_norm=True,
            range_norm=False,
            perceptual_weight=1.0,
            style_weight=0.0,
            criterion='l1'
        )
        self.gan_loss = GANLoss(gan_type='vanilla', real_label_val=1.0,
                                fake_label_val=0.0, loss_weight=5e-3)

        self.cos_similarity_loss = cos_similarity
        self.lda_loss = lda_loss
        self.load_balancing_loss = load_balancing_loss

        self.ssim_metric = SSIM(data_range=(0, 1))
        self.de_metric = DeltaE()
        self.psnr_metric = PSNR(data_range=(0, 1))

        self.automatic_optimization = False

        self.save_hyperparameters(ignore=['model'])

    def setup(self, stage: str) -> None:
        '''
        Initialize model weights before training
        '''
        if stage == 'fit' or stage is None:
            for m in self.generator.modules():
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
        optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr_G, weight_decay=self.weight_decay)
        optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.lr_D)
        scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, milestones=[100000, 150000], gamma=0.1)
        scheduler_D = optim.lr_scheduler.MultiStepLR(optimizer_D, milestones=[100000, 150000], gamma=0.1)
        return [optimizer_G, optimizer_D], [scheduler_G, scheduler_D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred = self.generator(x)
        return pred

    def training_step(self, batch, batch_idx):
        inputs, targets = batch

        optimizer_G, optimizer_D = self.optimizers()

        self.toggle_optimizer(optimizer_G)
        optimizer_G.zero_grad()

        fake_imgs = self.generator(inputs)
        l_g_pix = self.pixel_loss(fake_imgs, targets)

        l_g_percep, l_g_style = self.perceptual_loss(fake_imgs, targets)
        if l_g_style is not None:
            l_g_percep += l_g_style

        # GAN loss
        pred_fake_g, routing, _, _ = self.discriminator(fake_imgs)
        with torch.no_grad():
            pred_real_d, _, _, _ = self.discriminator(targets)

        l_g_real = self.gan_loss(pred_real_d - torch.mean(pred_fake_g), False, is_disc=False)
        l_g_fake = self.gan_loss(pred_fake_g - torch.mean(pred_real_d), True, is_disc=False)
        l_g_gan = (l_g_real + l_g_fake) / 2

        l_g_total = l_g_pix + l_g_percep + l_g_gan

        self.manual_backward(l_g_total)
        optimizer_G.step()
        self.untoggle_optimizer(optimizer_G)

        self.log('train_loss_G', l_g_total, prog_bar=True, logger=True)

        self.toggle_optimizer(optimizer_D)
        optimizer_D.zero_grad()

        pred_real_d, routing, feature, weight = self.discriminator(targets)
        pred_fake_d, _, _, _ = self.discriminator(fake_imgs.detach(), routing.detach())

        l_d_real = self.gan_loss(pred_real_d - torch.mean(pred_fake_d), True, is_disc=True) * 0.5
        l_d_real += self.cos_similarity_loss(weight) * 10.
        l_d_real += self.lda_loss(feature) * 10.
        l_d_real += self.load_balancing_loss(routing) * 0.05
        l_d_fake = self.gan_loss(pred_fake_d - torch.mean(pred_real_d.detach()), False, is_disc=True) * 0.5

        l_d_total = l_d_real + l_d_fake

        self.manual_backward(l_d_total)
        optimizer_D.step()
        self.untoggle_optimizer(optimizer_D)

        self.log('train_loss_D', l_d_total, prog_bar=True, logger=True)
        return {'loss': l_d_total}

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self.generator(inputs)
        mae_loss = self.pixel_loss(predictions, targets)
        l_g_percep, l_g_style = self.perceptual_loss(predictions, targets)
        if l_g_style is not None:
            l_g_percep += l_g_style
        loss = mae_loss + l_g_percep

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
        predictions = self.generator(inputs)
        mae_loss = self.pixel_loss(predictions, targets)
        l_g_percep, l_g_style = self.perceptual_loss(predictions, targets)
        if l_g_style is not None:
            l_g_percep += l_g_style
        loss = mae_loss + l_g_percep

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
        output = self.generator(inputs)
        return output
