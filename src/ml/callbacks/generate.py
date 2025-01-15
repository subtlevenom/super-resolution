from lightning.pytorch.callbacks import Callback
from lightning import LightningModule, Trainer
import torch
import torch.nn.functional as F
import torchvision
import os
from src.ml.utils.colors.ycbcr import ycbcr_to_rgb

class GenerateCallback(Callback):
    def __init__(
            self,
            every_n_epochs=1
        ) -> None:
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.input_imgs = None
        self.save_dir = None
        self.target_imgs = None

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        dataloader = trainer.val_dataloaders
        self.input_imgs, self.target_imgs = next(iter(dataloader))
        self.input_imgs = self.input_imgs.to(pl_module.device)
        self.target_imgs = self.target_imgs.to(pl_module.device)
        self.save_dir = os.path.join(trainer.log_dir, 'figures')

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(self.input_imgs)
                pl_module.train()
            # Plot and add to tensorboard
            target_size = reconst_imgs.shape[-2:]
            input_imgs_resized = F.interpolate(self.input_imgs, size=target_size, mode='bilinear', align_corners=False)
            target_imgs_resized = F.interpolate(self.target_imgs, size=target_size, mode='bilinear', align_corners=False)
            imgs = torch.stack([input_imgs_resized, target_imgs_resized, reconst_imgs], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=3)
            # Save image
            save_path = os.path.join(self.save_dir, f"reconst_{trainer.current_epoch}.png")
            os.makedirs(self.save_dir, exist_ok=True)
            torchvision.utils.save_image(ycbcr_to_rgb(grid), save_path)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        dataloader = trainer.test_dataloaders
        self.input_imgs, self.target_imgs = next(iter(dataloader))
        self.input_imgs = self.input_imgs.to(pl_module.device)
        self.target_imgs = self.target_imgs.to(pl_module.device)
        self.save_dir = os.path.join(trainer.log_dir, 'figures')

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(self.input_imgs)
                pl_module.train()
            # Plot and add to tensorboard
            target_size = reconst_imgs.shape[-2:]
            input_imgs_resized = F.interpolate(self.input_imgs, size=target_size, mode='bilinear', align_corners=False)
            target_imgs_resized = F.interpolate(self.target_imgs, size=target_size, mode='bilinear', align_corners=False)
            imgs = torch.stack([input_imgs_resized, target_imgs_resized, reconst_imgs], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=3)
            grid = ycbcr_to_rgb(grid)
            # Save image
            save_path = os.path.join(self.save_dir, f"test_{trainer.current_epoch}.png")
            os.makedirs(self.save_dir, exist_ok=True)
            torchvision.utils.save_image(grid, save_path)

    def on_predict_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: torch.Any, batch_idx: int) -> None:
        self.input_imgs = batch.to(pl_module.device)
        self.save_dir = os.path.join(trainer.log_dir, 'figures')

    def on_predict_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: torch.Any, batch: torch.Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        # Reconstruct images
        with torch.no_grad():
            pl_module.eval()
            outputs = pl_module(self.input_imgs)
            outputs = ycbcr_to_rgb(outputs)
            pl_module.train()
        # Save image
        for i in range(outputs.shape[0]):
            save_path = os.path.join(self.save_dir, f"predict_{batch_idx}_{i}.png")
            os.makedirs(self.save_dir, exist_ok=True)
            torchvision.utils.save_image(outputs[i], save_path)
