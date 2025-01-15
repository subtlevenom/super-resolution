import os
import random
import torch
import lightning as L
from torchvision.transforms.v2 import (
    Compose,
    ToImage,
    ToDtype,
    CenterCrop,
)
from torch.utils.data import DataLoader
from typing import Tuple
from .dataset import Image2ImageDataset
from src.core import Logger
from src.ml.transforms.pair_transform import PairTransform
from src.ml.utils.colors.ycbcr import RgbToYcbcr

CROP = 512

class HuaweiDataModule(L.LightningDataModule):
    def __init__(
            self,
            train_a: str,
            train_b: str,
            val_a: str,
            val_b: str,
            test_a: str,
            test_b: str,
            predict_a: str,
            upscale: int,
            train_batch_size: int = 32,
            val_batch_size: int = 32,
            test_batch_size: int = 32,
            predict_batch_size: int = 32,
            num_workers: int = min(12, os.cpu_count() - 1),
            img_exts: Tuple[str] = (".png", ".jpg"),
            seed: int = 42,
    ) -> None:
        super().__init__()
        self.test_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.predict_dataset = None
        self.upscale = upscale

        random.seed(seed)

        # train
        paths_a = [
            os.path.join(train_a, fname)
            for fname in os.listdir(train_a)
            if fname.endswith(img_exts)
        ]
        paths_b = [
            os.path.join(train_b, fname)
            for fname in os.listdir(train_b)
            if fname.endswith(img_exts)
        ]
        self.train_paths_a = sorted(paths_a)
        self.train_paths_b = sorted(paths_b)

        # valid
        paths_a = [
            os.path.join(val_a, fname)
            for fname in os.listdir(val_a)
            if fname.endswith(img_exts)
        ]
        paths_b = [
            os.path.join(val_b, fname)
            for fname in os.listdir(val_b)
            if fname.endswith(img_exts)
        ]
        self.val_paths_a = sorted(paths_a)
        self.val_paths_b = sorted(paths_b)

        # test
        paths_a = [
            os.path.join(test_a, fname)
            for fname in os.listdir(test_a)
            if fname.endswith(img_exts)
        ]
        paths_b = [
            os.path.join(test_b, fname)
            for fname in os.listdir(test_b)
            if fname.endswith(img_exts)
        ]
        self.test_paths_a = sorted(paths_a)
        self.test_paths_b = sorted(paths_b)

        # predict
        paths_a = [
            os.path.join(predict_a, fname)
            for fname in os.listdir(predict_a)
            if fname.endswith(img_exts)
        ]
        self.predict_paths_a = sorted(paths_a)

        self.batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.predict_batch_size = predict_batch_size

        self.train_image_p_transform = PairTransform(
            crop_size=CROP, p=0.5, seed=seed, upscale=upscale,
        )
        self.val_image_p_transform = PairTransform(
            crop_size=CROP, p=0.0, seed=seed, upscale=upscale,
        )
        self.test_image_p_transform = PairTransform(
            crop_size=CROP, p=1.0, seed=seed, upscale=upscale,
        )

        self.image_train_transform = Compose([
            ToImage(),
            CenterCrop(1024),
            ToDtype(dtype=torch.float32, scale=True),
            RgbToYcbcr(),
        ])
        self.image_val_transform = Compose([
            ToImage(),
            CenterCrop(1024),
            ToDtype(dtype=torch.float32, scale=True),
            RgbToYcbcr(),
        ])
        self.image_test_transform = Compose([
            ToImage(),
            CenterCrop(1024),
            ToDtype(dtype=torch.float32, scale=True),
            RgbToYcbcr(),
        ])
        self.image_predict_transform = Compose([
            ToImage(),
            ToDtype(dtype=torch.float32, scale=True),
            RgbToYcbcr(),
        ])
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        if stage == 'fit' or stage is None:
            self.train_dataset = Image2ImageDataset(
                self.train_paths_a, self.train_paths_b, self.image_train_transform, self.train_image_p_transform,
            )
            self.val_dataset = Image2ImageDataset(
                self.val_paths_a, self.val_paths_b, self.image_val_transform, self.val_image_p_transform,
            )
        if stage == 'test' or stage is None:
            self.test_dataset = Image2ImageDataset(
                self.test_paths_a, self.test_paths_b, self.image_test_transform, self.test_image_p_transform,
            )
        if stage == 'predict' or stage is None:
            self.predict_dataset = Image2ImageDataset(
                self.predict_paths_a, None, self.image_predict_transform,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict_dataset,
            batch_size=self.predict_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )
