import argparse
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import os
from pathlib import Path
import random
import numpy as np
from rich.progress import Progress
from typing import List
import imageio
import asyncio
from .utils import concurrent
from src import cli

THREADS = 1


def add_parser(subparser: argparse) -> None:
    parser = subparser.add_parser(
        "create-image-data",
        help="Create dataset",
        formatter_class=cli.ArgumentDefaultsRichHelpFormatter,
    )
    parser.add_argument(
        "-s",
        "--source",
        type=str,
        help="Path to input source directory",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        help="Path to input target directory",
        required=True,
    )
    parser.add_argument(
        "-f",
        "--feature",
        type=str,
        help="Path to feature directory",
        default=None,
        required=False,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=os.path.join('data', 'huawei'),
        help="Path to output directory",
        required=False,
    )
    parser.add_argument(
        "-r",
        "--seed",
        type=int,
        default=42,
        help="Seed",
        required=False,
    )
    parser.add_argument(
        "-c",
        "--crop_size",
        type=int,
        default=1024,
        help="Crop size, set 0 to skip cropping",
        required=False,
    )
    parser.set_defaults(func=generate_dataset)


def parallel(f):

    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(
            None, f, *args, **kwargs)

    return wrapped


def _crop_image(image: np.ndarray, crop_size: int) -> List[np.ndarray]:
    if crop_size == 0:
        return [image]
    h, w, c = image.shape
    crop_list = []
    for y in range(crop_size, h, crop_size):
        for x in range(crop_size, w, crop_size):
            crop = image[y - crop_size:y, x - crop_size:x, 0:c]
            crop_list.append(crop)
    return crop_list


@concurrent
def _prepare_data(
    input_src_img_dir: Path,
    input_ref_img_dir: Path,
    save_train_src_dir: Path,
    save_train_ref_dir: Path,
    name: str,
    args,
):

    source_path = input_src_img_dir.joinpath(name)
    if not source_path.is_file():
        raise Exception('No source file')

    target_path = input_ref_img_dir.joinpath(name)
    if not target_path.is_file():
        raise Exception('No target file')

    image = imageio.v3.imread(source_path)
    crop_list = [image]
    # crop_list = _crop_image(image, args.crop_size)
    for (i, image) in enumerate(crop_list):
        save_name =Path(name).stem + f'_{i}' + Path(name).suffix
        imageio.v3.imwrite(save_train_src_dir.joinpath(save_name), image)

    image = imageio.v3.imread(target_path)
    crop_list = [image]# _crop_image(image, args.crop_size)
    # crop_list = _crop_image(image, args.crop_size)
    for (i, image) in enumerate(crop_list):
        save_name =Path(name).stem + f'_{i}' + Path(name).suffix
        imageio.v3.imwrite(save_train_ref_dir.joinpath(save_name), image)


def generate_dataset(args: argparse.Namespace) -> None:
    input_src_img_dir = Path(args.source)
    input_ref_img_dir = Path(args.target)
    output_dir = Path(args.output)

    if not input_src_img_dir.is_dir():
        raise Exception(f'No such directory: {input_src_img_dir}')
    if not input_ref_img_dir.is_dir():
        raise Exception(f'No such directory: {input_ref_img_dir}')

    save_test_src_dir = output_dir.joinpath('test', 'source')
    save_test_ref_dir = output_dir.joinpath('test', 'target')
    save_val_src_dir = output_dir.joinpath('val', 'source')
    save_val_ref_dir = output_dir.joinpath('val', 'target')
    save_train_src_dir = output_dir.joinpath('train', 'source')
    save_train_ref_dir = output_dir.joinpath('train', 'target')

    save_test_src_dir.mkdir(parents=True, exist_ok=True)
    save_test_ref_dir.mkdir(parents=True, exist_ok=True)
    save_val_src_dir.mkdir(parents=True, exist_ok=True)
    save_val_ref_dir.mkdir(parents=True, exist_ok=True)
    save_train_src_dir.mkdir(parents=True, exist_ok=True)
    save_train_ref_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_src_img_dir.glob('*.[jpg png bmp]*'))
    random.seed(args.seed)
    random.shuffle(files)

    n = len(files)
    split = np.cumsum([int(0.7 * n), int(0.1 * n)])
    train_files = files[:split[0]]
    val_files = files[split[0]:split[1]]
    test_files = files[split[1]:]

    with Progress() as progress:
        train_pb = progress.add_task("[cyan]Train features",
                                     total=len(train_files))
        val_pb = progress.add_task("[cyan]Val images", total=len(val_files))
        test_pb = progress.add_task("[cyan]Test images", total=len(test_files))

        with ThreadPoolExecutor(max_workers=THREADS) as executor:
            train_tasks = [
                _prepare_data(
                    executor,
                    input_src_img_dir,
                    input_ref_img_dir,
                    save_train_src_dir,
                    save_train_ref_dir,
                    filename.name,
                    args,
                ) for filename in train_files
            ]
            val_tasks = [
                _prepare_data(
                    executor,
                    input_src_img_dir,
                    input_ref_img_dir,
                    save_val_src_dir,
                    save_val_ref_dir,
                    filename.name,
                    args,
                ) for filename in val_files
            ]
            test_tasks = [
                _prepare_data(
                    executor,
                    input_src_img_dir,
                    input_ref_img_dir,
                    save_test_src_dir,
                    save_test_ref_dir,
                    filename.name,
                    args,
                ) for filename in test_files
            ]

            for task in train_tasks:
                task.add_done_callback(
                    lambda _: progress.update(train_pb, advance=1))
            for task in val_tasks:
                task.add_done_callback(
                    lambda _: progress.update(val_pb, advance=1))
            for task in test_tasks:
                task.add_done_callback(
                    lambda _: progress.update(test_pb, advance=1))

            _, not_done = wait(train_tasks + val_tasks + test_tasks,
                               return_when=ALL_COMPLETED)

            if len(not_done) > 0:
                print(f'[Warn] Skipped {len(not_done)} image pairs.')
