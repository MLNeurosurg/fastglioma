"""Slide SSL with VICReg pretraining script.

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

from fastglioma.models import MLP, VICRegNetwork
from fastglioma.train.common import (setup_output_dirs, parse_args,
                                     get_exp_name, config_loggers,
                                     get_optimizer_func, get_scheduler_func,
                                     get_dataloaders)
from fastglioma.losses.vicreg import VICRegLoss


class SlideVICRegSystem(pl.LightningModule):
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
        self.model = VICRegNetwork(mil, mlp)

        if "training" in cf:
            self.criterion = VICRegLoss(
                embedding_dim=cf["model"]["num_embedding_out"],
                **cf["training"]["objective"]["params"])
            self.train_loss = torch.nn.ModuleDict({
                n: torchmetrics.MeanMetric()
                for n in GeneralVICRegLoss.get_loss_names()
            })

            self.val_loss = torch.nn.ModuleDict({
                n: torchmetrics.MeanMetric()
                for n in GeneralVICRegLoss.get_loss_names()
            })
        else:
            self.criterion = self.train_loss = self.val_loss = None

        self.num_it_per_ep_ = num_it_per_ep

    @staticmethod
    def get_kth_view(data: List[List[torch.Tensor]], k: int):
        return [d[k] for d in data]

    def forward(self, batch):
        pred = [
            self.model(self.get_kth_view(batch["image"], 0),
                       coords=self.get_kth_view(batch["coords"], 0)),
            self.model(self.get_kth_view(batch["image"], 1),
                       coords=self.get_kth_view(batch["coords"], 1))
        ]
        pred = torch.stack(pred, dim=1)
        pred_gather = self.all_gather(pred, sync_grads=True)
        pred_gather = pred_gather.reshape(-1, *pred_gather.shape[-2:])

        losses = self.criterion(pred_gather)

        return losses

    def training_step(self, batch, batch_idx):
        losses = self.forward(batch)
        bs = batch["image"][0].shape[0] * torch.cuda.device_count()

        for k in self.train_loss:
            self.log(f"train/{k}",
                     losses[k],
                     on_step=True,
                     on_epoch=False,
                     batch_size=bs,
                     rank_zero_only=True)
            self.train_loss[k].update(losses[k], weight=bs)

    def validation_step(self, batch, batch_idx):
        losses = self.forward(batch)
        bs = batch["image"][0].shape[0] * torch.cuda.device_count()

        for k in self.val_loss:
            self.log(f"val/{k}",
                     losses[k],
                     on_step=True,
                     on_epoch=False,
                     batch_size=bs,
                     rank_zero_only=True)
            self.val_loss[k].update(losses[k], weight=bs)

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx):
        pred = [
            self.model(self.get_kth_view(batch["image"], 0),
                       coords=self.get_kth_view(batch["coords"], 0)),
            self.model(self.get_kth_view(batch["image"], 1),
                       coords=self.get_kth_view(batch["coords"], 1))
        ]
        pred = torch.stack(pred, dim=1)
        pred_gather = self.all_gather(pred, sync_grads=True)
        pred_gather = pred_gather.reshape(-1, *pred_gather.shape[-2:])

        return {
            "path": [batch["path"]],
            "label": batch["label"],
            "embeddings": pred_gather
        }

    def on_train_epoch_end(self):
        losses = {}
        for k in self.train_loss.keys():
            losses[k] = self.train_loss[k].compute()
            self.log(f"train/{k}_manualepoch",
                     losses[k],
                     on_epoch=True,
                     sync_dist=True,
                     rank_zero_only=True)
            self.train_loss[k].reset()
        logging.info(f"train/manualepoch {losses}")

    def on_validation_epoch_end(self):
        losses = {}
        for k in self.val_loss.keys():
            losses[k] = self.val_loss[k].compute()
            self.log(f"val/{k}_manualepoch",
                     losses[k],
                     on_epoch=True,
                     sync_dist=True,
                     rank_zero_only=True)
            self.val_loss[k].reset()
        logging.info(f"val/manualepoch {losses}")

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
    system_func = SlideVICRegSystem

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
