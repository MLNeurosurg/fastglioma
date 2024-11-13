"""Save embeddings script.

Copyright (c) 2024 University of Michigan. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

import os
import logging
from shutil import copy2
from functools import partial
from typing import List, Union, Dict, Any

import gzip
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torchvision.transforms import Compose

import pytorch_lightning as pl

from fastglioma.datasets.srh_dataset import PatchDataset #SlideDataset, slide_collate_fn
from fastglioma.datasets.improc import get_transformations
from fastglioma.utils.common import (parse_args, get_exp_name, config_loggers,
                                     get_num_worker)

from fastglioma.models.resnet import resnet_backbone
from fastglioma.models.cnn import MLP, ContrastiveLearningNetwork
from fastglioma.models.mil import MIL_forward, MIL_Classifier, TransformerMIL

from fastglioma.train.train_hidisc import HiDiscSystem

from fastglioma.utils.format_slide_embedding import prediction_to_slide_embedding


def get_predictions(
        cf: Dict[str, Any],
        exp_root: str) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """Run forward pass on the dataset, and generate embeddings and logits"""
    _, valid_xform = get_transformations(cf)

    dset = PatchDataset(
        data_root=cf["data"]["db_root"],
        studies=cf["data"]["studies"],
        transform=valid_xform,
        balance_patch_per_class=False,
        use_patient_class=cf["data"]["use_patient_class"])

    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=cf["eval"]["predict_batch_size"],
        drop_last=False,
        pin_memory=True,
        num_workers=get_num_worker(),
        # collate_fn=slide_collate_fn,
        persistent_workers=True)

    # load lightning checkpoint
    ckpt_path = os.path.join(cf["infra"]["log_dir"], cf["infra"]["exp_name"],
                             cf["eval"]["ckpt_path"])
                             
    # Load model from ckpt
    model = HiDiscSystem.load_from_checkpoint(ckpt_path,
                                              cf=cf,
                                              num_it_per_ep=0)

    # Create trainer
    trainer = pl.Trainer(accelerator="gpu",
                         devices=1,
                         max_epochs=-1,
                         default_root_dir=exp_root,
                         enable_checkpointing=False,
                         logger=False)

    predictions = trainer.predict(model, dataloaders=loader)

    def process_predictions(predictions):
        # Combine predictions into a single dictionary
        pred = {}
        for k in predictions[0].keys():
            if k == "path":
                pred[k] = [pk for p in predictions for pk in p[k][0]]
            else:
                pred[k] = torch.cat([p[k] for p in predictions])

        return pred

    predictions = process_predictions(predictions)
    return predictions


def setup_eval_paths(cf, get_exp_name, cmt_append):
    """Get name of the ouput dirs and create them in the file system."""
    log_root = cf["infra"]["log_dir"]
    exp_name = cf["infra"]["exp_name"]
    instance_name = cf["eval"]["ckpt_path"].split("/")[0]
    eval_instance_name = "_".join([get_exp_name(cf), cmt_append])
    exp_root = os.path.join(log_root, exp_name, instance_name, "evals",
                            eval_instance_name)

    # Generate needed folders
    pred_dir = os.path.join(exp_root, 'predictions')
    config_dir = os.path.join(exp_root, 'config')
    for dir_name in [pred_dir, config_dir]:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    return exp_root, pred_dir, partial(copy2, dst=config_dir)


def main():
    """Driver script for pipeline."""
    cf_fd = parse_args()
    cf = yaml.load(cf_fd, Loader=yaml.FullLoader)
    exp_root, pred_dir, cp_config = setup_eval_paths(cf, get_exp_name, "")
    pl.seed_everything(cf["infra"]["seed"])

    # Logging and copying config files
    cp_config(cf_fd.name)
    config_loggers(exp_root)

    logging.info("Generating predictions")
    predictions = get_predictions(cf, exp_root)

    # save embeddings
    prediction_to_slide_embedding(
        saving_dir=cf["eval"]["save_by_slide"]["saving_dir"],
        tag=cf["eval"]["save_by_slide"]["tag"],
        predictions=predictions)


if __name__ == "__main__":
    main()
