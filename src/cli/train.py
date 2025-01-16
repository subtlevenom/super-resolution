import argparse
import yaml
from ..core.selector import (
    ModelSelector,
    DataSelector,
    PipelineSelector
)
from ..core.config import Config
import lightning as L
import os
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    LearningRateMonitor,
)
from src.ml.callbacks import GenerateCallback
from lightning.pytorch.loggers import CSVLogger
from src import cli


def add_parser(subparser: argparse) -> None:
    parser = subparser.add_parser(
        "train",
        help="Train color transfer model",
        formatter_class=cli.ArgumentDefaultsRichHelpFormatter,
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Path to config file",
        default="config.yaml",
        required=False,
    )

    parser.set_defaults(func=train)


def train(args: argparse.Namespace) -> None:
    print(f"Loading config from {args.config}")
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    config = Config(**config)
    print('Config:')
    config.print()
    
    dm = DataSelector.select(config)
    model = ModelSelector.select(config)
    pipeline = PipelineSelector.select(config, model)

    logger = CSVLogger(
        save_dir=os.path.join(config.save_dir, config.experiment),
        name='logs',
        version='',
    )

    trainer = L.Trainer(
        logger=logger,
        default_root_dir=os.path.join(config.save_dir, config.experiment),
        max_epochs=config.pipeline.params.epochs,
        devices=config.devices,
        callbacks=[
            ModelCheckpoint(
                filename="{epoch}-{val_de:.2f}",
                monitor='val_de',
                save_top_k=5,
                save_last=True,
            ),
            RichModelSummary(),
            RichProgressBar(),
            LearningRateMonitor(
                logging_interval='epoch',
            ),
            GenerateCallback(
                every_n_epochs=1,
            ),
        ],
    )

    ckpt_path = os.path.join(config.save_dir, config.experiment, 'logs/checkpoints/last.ckpt')

    """
    # load hknet
    import torch
    _path = os.path.join(config.save_dir, config.experiment, 'logs/checkpoints/model_G_S0_i220000.pth')
    state_dict = torch.load(_path, weights_only=True, map_location=torch.device(config.devices[0]))
    model.hknet.load_state_dict(state_dict)
    """

    trainer.fit(
        model=pipeline, 
        datamodule=dm,
        ckpt_path=ckpt_path if config.resume and os.path.exists(ckpt_path) else None,
    )
