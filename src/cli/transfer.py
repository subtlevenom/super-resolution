import argparse
from pathlib import Path
import yaml
import torch
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
from hklut.weights import transfer_weights


def add_parser(subparser: argparse) -> None:
    parser = subparser.add_parser(
        "transfer",
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

    parser.set_defaults(func=transfer)


def transfer(args: argparse.Namespace) -> None:
    print(f"Loading config from {args.config}")
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    config = Config(**config)
    print('Config:')
    config.print()
    
    model = ModelSelector.select(config)

    ckpt_path = os.path.join(config.save_dir, config.experiment, 'logs/checkpoints/last.ckpt')
    if not os.path.exists(ckpt_path):
        raise Exception('Checkpoint is missing for prediction')

    state_dict = torch.load(ckpt_path, weights_only=True)['state_dict']
    state_dict = {key.replace('model.',''): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict)

    lut_path = Path(config.save_dir).joinpath(config.experiment).joinpath('logs/luts/')
    lut_path.mkdir(exist_ok=True, parents=True)

    transfer_weights(model.hknet, lut_path)
