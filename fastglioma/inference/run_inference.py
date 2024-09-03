"""Inference modules and script.

Copyright (c) 2024 University of Michigan. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

import os
import logging
from shutil import copy2
from functools import partial
from typing import List, Union, Dict, Any

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torchvision.transforms import Compose

import pytorch_lightning as pl

from fastglioma.datasets.srh_dataset import SlideDataset, slide_collate_fn
from fastglioma.datasets.improc import get_srh_base_aug
from fastglioma.utils.common import (parse_args, get_exp_name, config_loggers,
                                     get_num_worker)

from fastglioma.models.resnet import resnet_backbone
from fastglioma.models.cnn import MLP, ContrastiveLearningNetwork
from fastglioma.models.mil import MIL_forward, MIL_Classifier, TransformerMIL

from huggingface_hub import hf_hub_download


class FastGliomaInferenceSystem(pl.LightningModule):
    """Lightning system for FastGlioma inference on OpenSRH."""

    def __init__(self, cf: Dict[str, Any], num_it_per_ep: int):
        super().__init__()
        self.cf_ = cf

        if cf["model"]["patch"]["backbone"]["which"] == "resnet34":
            bb = partial(
                resnet_backbone,
                arch=cf["model"]["patch"]["backbone"]["which"],
                num_channel_in=cf["model"]["patch"]["backbone"]["params"].get(
                    "num_channel_in", 3))
        else:
            raise NotImplementedError()

        if cf["model"]["slide"]["mil"]["which"] == "transformer":
            mil = partial(MIL_forward,
                          mil=partial(TransformerMIL,
                                      **cf["model"]["slide"]["mil"]["params"]))
        else:
            raise NotImplementedError()

        mlp = partial(MLP,
                      n_in=mil().num_out,
                      hidden_layers=cf["model"]["slide"]["mlp_hidden"],
                      n_out=1)
        self.model = MIL_Classifier(bb, mil, mlp)

        self.criterion = self.train_loss = self.val_loss = None
        self.num_it_per_ep_ = num_it_per_ep

    @staticmethod
    def get_kth_view(data: List[List[torch.Tensor]], k: int):
        return [d[k] for d in data]

    def forward(self, batch):
        return self.model(self.get_kth_view(batch["image"], 0),
                          coords=self.get_kth_view(batch["coords"], 0))

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx):
        out = self.forward(batch)

        return {
            "path": [batch["path"]],
            "label": batch["label"],
            "logits": out["logits"],
            "embeddings": out["embeddings"]
        }


def get_predictions(
        cf: Dict[str, Any],
        exp_root: str) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """Run forward pass on the dataset, and generate embeddings and logits"""
    dset = SlideDataset(
        data_root=cf["data"]["db_root"],
        studies=cf["data"]["studies"],
        transform=Compose(
            get_srh_base_aug(
                base_aug=("three_channels" if cf["data"]["patch_input"] == "highres" else "ch2_only"), #yapf:disable
                y_skip=(0 if cf["data"]["patch_input"] == "highres" else 5))), #yapf:disable
        balance_slide_per_class=False,
        use_patient_class=cf["data"]["use_patient_class"])

    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=cf["eval"]["predict_batch_size"],
        drop_last=False,
        pin_memory=True,
        num_workers=get_num_worker(),
        collate_fn=slide_collate_fn,
        persistent_workers=True)

    # Load model from huggingface repo
    ckpt_path = hf_hub_download(repo_id=cf["infra"]["hf_repo"],
                                filename=cf["eval"]["ckpt_path"])
    model = FastGliomaInferenceSystem.load_from_checkpoint(ckpt_path,
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

        pred["logits"] = pred["logits"].squeeze(1)
        pred["scores"] = torch.sigmoid(pred["logits"])
        pred["label"] = [{v: k for k, v in dset.class_to_idx_.items()}[l.item()] for l in pred["label"]] #yapf:disable
        pred["slide"] = ["/".join(imp[0].split("/")[:9]) for imp in pred["path"]] #yapf:disable

        # Sort predictions by slide name
        sorted_indices = sorted(range(len(pred['slide'])),
                                key=lambda k: pred['slide'][k])

        # Apply the same ordering to all keys in pred
        for key in pred:
            if isinstance(pred[key], list):
                pred[key] = [pred[key][i] for i in sorted_indices]
            elif isinstance(pred[key], torch.Tensor):
                pred[key] = pred[key][sorted_indices]

        del pred["path"]

        return pred

    predictions = process_predictions(predictions)
    return predictions


def setup_eval_paths(cf, get_exp_name, cmt_append):
    """Get name of the ouput dirs and create them in the file system."""
    log_root = cf["infra"]["log_dir"]
    exp_name = cf["infra"]["exp_name"]
    eval_instance_name = "_".join([get_exp_name(cf), cmt_append])
    exp_root = os.path.join(log_root, exp_name, eval_instance_name)

    # Generate needed folders
    pred_dir = os.path.join(exp_root, 'predictions')
    config_dir = os.path.join(exp_root, 'config')
    for dir_name in [pred_dir, config_dir]:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    return exp_root, pred_dir, partial(copy2, dst=config_dir)


def main():
    """Driver script for inference pipeline."""
    cf_fd = parse_args()
    cf = yaml.load(cf_fd, Loader=yaml.FullLoader)
    exp_root, pred_dir, cp_config = setup_eval_paths(cf, get_exp_name, "")
    pl.seed_everything(cf["infra"]["seed"])

    # Logging and copying config files
    cp_config(cf_fd.name)
    config_loggers(exp_root)

    logging.info("Generating predictions")
    predictions = get_predictions(cf, exp_root)
    torch.save(predictions, os.path.join(pred_dir, "predictions.pt"))


if __name__ == "__main__":
    main()
