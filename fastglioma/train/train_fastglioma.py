"""FastGlioma training script.

Copyright (c) 2024 University of Michigan. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

import yaml
import logging
from functools import partial
from typing import Dict, Any

import torch

import pytorch_lightning as pl
import torchmetrics

from fastglioma.models import MLP, MIL_Classifier
from fastglioma.train.common import (setup_output_dirs, parse_args,
                                     get_exp_name, config_loggers,
                                     get_optimizer_func, get_scheduler_func,
                                     get_dataloaders)
from fastglioma.losses.ordmet import OrdinalMetricLoss


class FastGliomaSystem(pl.LightningModule):
    """Lightning system for fastglioma experiments."""

    def __init__(self, cf: Dict[str, Any], num_it_per_ep: int):
        super().__init__()
        self.cf_ = cf

        mil = partial(MIL_forward,
                      mil=partial(TransMIL, **cf["model"]["mil"]["params"]))
        mlp = partial(MLP,
                      n_in=mil().num_out,
                      hidden_layers=cf["model"]["mlp_hidden"],
                      n_out=1)
        self.model = MIL_Classifier(mil, mlp)

        if "training" in cf:
            self.criterion = OrdinalMetricLoss()
            self.train_loss = torchmetrics.MeanMetric()
            self.val_loss = torchmetrics.MeanMetric()
        else:
            self.criterion = self.train_loss = self.val_loss = None

        self.num_it_per_ep_ = num_it_per_ep

    @staticmethod
    def get_kth_view(data: List[List[torch.Tensor]], k: int):
        return [d[k] for d in data]

    def forward(self, batch):
        return self.model(self.get_kth_view(batch["image"], 0),
                          coords=self.get_kth_view(batch["coords"], 0))

    def training_step(self, batch, batch_idx):
        pred = self.forward(batch)
        loss = self.criterion(pred["logits"], batch["label"])
        bs = batch["image"][0].shape[0] * torch.cuda.device_count()

        self.log("train/ordmet",
                 loss,
                 on_step=True,
                 on_epoch=True,
                 batch_size=bs,
                 sync_dist=True,
                 rank_zero_only=True)
        self.train_loss.update(loss, weight=bs)

    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch)
        loss = self.criterion(pred["logits"], batch["label"])
        bs = batch["image"][0].shape[0] * torch.cuda.device_count()

        self.log("val/ordmet",
                 loss,
                 on_step=True,
                 on_epoch=True,
                 batch_size=bs,
                 sync_dist=True,
                 rank_zero_only=True)
        self.val_loss.update(loss, weight=bs)

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx):
        out = self.forward(batch)

        return {
            "path": [batch["path"]],
            "label": batch["label"],
            "logits": out["logits"],
            "embeddings": out["embeddings"]
        }

    def on_train_epoch_end(self):
        train_loss = self.train_loss.compute()
        self.log("train/ordmet_manualepoch",
                 train_loss,
                 on_epoch=True,
                 sync_dict=True,
                 rank_zero_only=True)
        logging.info(f"train/ordmet_manualepoch")
        self.train_loss.reset()

    def on_validation_epoch_end(self):
        val_loss = self.val_loss.compute()
        self.log("val/ordmet_manualepoch",
                 val_loss,
                 on_epoch=True,
                 sync_dict=True,
                 rank_zero_only=True)
        logging.info(f"val/ordmet_manualepoch")
        self.val_loss.reset()

    def configure_optimizers(self):
        # if not training, no optimizer
        if "training" not in self.cf_:
            return None

        # get optimizer
        opt = get_optimizer_func(self.cf_)(self.model.parameters())

        # check if use a learn rate scheduler
        sched_func = get_scheduler_func(self.cf_, self.num_it_per_ep_)
        if not sched_func:
            return opt

        # get learn rate scheduler
        lr_scheduler_config = {
            "scheduler": sched_func(opt),
            "interval": "step",
            "frequency": 1,
            "name": "lr"
        }

        return [opt], lr_scheduler_config


def main():
    cf_fd = parse_args()
    cf = yaml.load(cf_fd, Loader=yaml.FullLoader)
    exp_root, model_dir, cp_config = setup_output_dirs(cf, get_exp_name, "")
    pl.seed_everything(cf["infra"]["seed"])

    # logging and copying config files
    cp_config(cf_fd.name)
    config_loggers(exp_root)

    train_loader, valid_loader = get_dataloaders(cf)
    system_func = FastGliomaSystem

    logging.info(f"num devices: {torch.cuda.device_count()}")
    logging.info(f"num workers in dataloader: {train_loader.num_workers}")

    num_it_per_ep = len(train_loader)
    if torch.cuda.device_count() > 1:
        num_it_per_ep //= torch.cuda.device_count()

    exp = system_func(cf, num_it_per_ep)

    # config loggers
    logger = [
        pl.loggers.TensorBoardLogger(save_dir=exp_root, name="tb"),
        pl.loggers.CSVLogger(save_dir=exp_root, name="csv")
    ]

    # config callbacks
    epoch_ckpt = pl.callbacks.ModelCheckpoint(
        dirpath=model_dir,
        save_top_k=-1,
        every_n_epochs=cf["training"]["eval_ckpt_ep_freq"],
        filename="ckpt-epoch{epoch}",
        auto_insert_metric_name=False)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step",
                                                  log_momentum=False)

    # create trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        default_root_dir=exp_root,
        strategy=pl.strategies.DDPStrategy(find_unused_parameters=False,
                                           static_graph=True),
        logger=logger,
        log_every_n_steps=10,
        callbacks=[epoch_ckpt, lr_monitor],
        max_epochs=cf["training"]["num_epochs"],
        check_val_every_n_epoch=cf["training"]["eval_ckpt_ep_freq"],
        num_nodes=1)
    trainer.fit(exp,
                train_dataloaders=train_loader,
                val_dataloaders=valid_loader)


if __name__ == '__main__':
    main()
