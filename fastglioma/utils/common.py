"""Common modules for FastGlioma training, evaluation, and inference.

Copyright (c) 2024 University of Michigan. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

import os
import math
import logging
import argparse
from shutil import copy2
from datetime import datetime
from functools import partial
from typing import Tuple, Dict, Optional, Any

import uuid

import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR, LambdaLR
from torchvision.transforms import Compose

import pytorch_lightning as pl


def get_optimizer_func(cf: Dict[str, Any]) -> callable:
    """Return a optimizer callable based on config value"""
    lr = cf["training"]["learn_rate"]
    if cf["training"]["optimizer"] == "adamw":
        return partial(optim.AdamW, lr=lr)
    elif cf["training"]["optimizer"] == "adam":
        return partial(optim.Adam, lr=lr)
    elif cf["training"]["optimizer"] == "sgd":
        return partial(optim.SGD, lr=lr, momentum=0.9)
    else:
        raise NotImplementedError()


def get_scheduler_func(cf: Dict[str, Any],
                       num_it_per_ep: int = 0) -> Optional[callable]:
    """Return a scheduler callable based on config value."""
    if "scheduler" not in cf["training"]:
        return None

    if cf["training"]["scheduler"]["which"] == "step_lr":
        step_size = convert_epoch_to_iter(
            cf["training"]["scheduler"]["params"]["step_unit"],
            cf["training"]["scheduler"]["params"]["step_size"], num_it_per_ep)
        return partial(StepLR,
                       step_size=step_size,
                       gamma=cf["training"]["scheduler"]["params"]["gamma"])
    elif cf["training"]["scheduler"]["which"] == "cos_warmup":
        num_epochs = cf['training']['num_epochs']

        num_warmup_steps = cf['training']['scheduler']['params'][
            'num_warmup_steps']
        if isinstance(num_warmup_steps, float):  # fraction of total train
            cf['training']['scheduler']['params']['num_warmup_steps'] = int(
                num_warmup_steps * num_epochs * num_it_per_ep)

        return partial(get_cosine_schedule_with_warmup,
                       num_training_steps=num_it_per_ep * num_epochs,
                       **cf["training"]["scheduler"]["params"])
    else:
        raise NotImplementedError()


def convert_epoch_to_iter(unit: str, steps: int, num_it_per_ep: int) -> int:
    """Converts number of epochs / iterations to number of iterations."""
    if unit == "epoch":
        return num_it_per_ep * steps  # per epoch
    elif unit == "iter":
        return steps
    else:
        NotImplementedError("unit must be one of [epoch, iter]")


def get_cosine_schedule_with_warmup(optimizer: torch.optim.Optimizer,
                                    num_warmup_steps: int,
                                    num_training_steps: int,
                                    num_cycles: float = 0.5,
                                    last_epoch: int = -1):
    """Create cosine learn rate scheduler with linear warm up built in."""

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        return max(
            0.0, 0.5 *
            (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def config_loggers(exp_root):
    """Config logger for the experiments

    Sets string format and where to save.
    """

    logging_format_str = "[%(levelname)-s|%(asctime)s|%(name)s|" + \
        "%(filename)s:%(lineno)d|%(funcName)s] %(message)s"
    logging.basicConfig(level=logging.INFO,
                        format=logging_format_str,
                        datefmt="%H:%M:%S",
                        handlers=[
                            logging.FileHandler(
                                os.path.join(exp_root, 'train.log')),
                            logging.StreamHandler()
                        ],
                        force=True)
    logging.info("Exp root {}".format(exp_root))

    formatter = logging.Formatter(logging_format_str, datefmt="%H:%M:%S")
    logger = logging.getLogger("pytorch_lightning.core")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(os.path.join(exp_root, 'train.log')))
    for h in logger.handlers:
        h.setFormatter(formatter)


def setup_ddp_exp_name(exp_name: str):
    if pl.utilities.rank_zero.rank_zero_only.rank != 0:
        return os.path.join(exp_name, "high_rank")
    else:
        return exp_name


def setup_output_dirs(cf: Dict, get_exp_name: callable,
                      cmt_append: str) -> Tuple[str, str, callable]:
    """Get name of the ouput dirs and create them in the file system."""
    log_root = cf["infra"]["log_dir"]
    instance_name = "_".join([get_exp_name(cf), cmt_append])
    exp_name = setup_ddp_exp_name(cf["infra"]["exp_name"])
    exp_root = os.path.join(log_root, exp_name, instance_name)

    model_dir = os.path.join(exp_root, 'models')
    config_dir = os.path.join(exp_root, 'config')

    for dir_name in [model_dir, config_dir]:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    return exp_root, model_dir, partial(copy2, dst=config_dir)


def parse_args():
    """Get config file handle from command line argument."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        type=argparse.FileType('r'),
                        required=True,
                        help='config file for training')
    args = parser.parse_args()
    return args.config


def get_exp_name(cf):
    """Generate experiment name with a hash, time, and comments in config."""
    time = datetime.now().strftime("%b%d-%H-%M-%S")
    return "-".join([uuid.uuid4().hex[:8], time, cf["infra"]["comment"]])


def get_num_worker():
    """Estimate number of cpu workers."""
    try:
        num_worker = len(os.sched_getaffinity(0))
    except Exception:
        num_worker = os.cpu_count()

    if num_worker > 1:
        return num_worker - 1
    else:
        return torch.cuda.device_count() * 4