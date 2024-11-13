"""Patch SSL with HiDisc pretraining script.

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

from fastglioma.models.resnet import resnet_backbone
from fastglioma.models.cnn import MLP, ContrastiveLearningNetwork
from fastglioma.utils.common import (setup_output_dirs, parse_args,
                                     get_exp_name, config_loggers,
                                     get_optimizer_func, get_scheduler_func,
                                     get_num_worker)
from fastglioma.losses.hidisc import HiDiscLoss


class HiDiscSystem(pl.LightningModule):
    """Lightning system for hidisc experiments."""

    def __init__(self, cf: Dict[str, Any], num_it_per_ep: int):
        super().__init__()
        self.cf_ = cf

        if "resnet" in cf["model"]["backbone"]["which"]:
            bb = partial(
                resnet_backbone,
                arch=cf["model"]["backbone"]["which"],
                num_channel_in=cf["model"]["backbone"]["params"].get(
                    "num_channel_in", 3))
        else:
            raise NotImplementedError()

        mlp = partial(MLP,
                      n_in=bb().num_out,
                      hidden_layers=cf["model"]["mlp_hidden"],
                      n_out=cf["model"]["num_embedding_out"])
        self.model = ContrastiveLearningNetwork(bb, mlp)

        if "training" in cf:
            crit_params = cf["training"]["objective"]["params"]
            self.criterion = HiDiscLoss(
                lambda_patient=crit_params["lambda_patient"],
                lambda_slide=crit_params["lambda_slide"],
                lambda_patch=crit_params["lambda_patch"],
                supcon_loss_params=crit_params["supcon_params"])
            self.train_loss = torch.nn.ModuleDict({
                "patient_loss": torchmetrics.MeanMetric(),
                "slide_loss": torchmetrics.MeanMetric(),
                "patch_loss": torchmetrics.MeanMetric(),
                "sum_loss": torchmetrics.MeanMetric()
            }) # yapf: disable
            self.val_loss = torch.nn.ModuleDict({
                "patient_loss": torchmetrics.MeanMetric(),
                "slide_loss": torchmetrics.MeanMetric(),
                "patch_loss": torchmetrics.MeanMetric(),
                "sum_loss": torchmetrics.MeanMetric()
            })  #yapf: disable
        else:
            self.criterion = self.train_loss = self.val_loss = None

        self.num_it_per_ep_ = num_it_per_ep

    def forward(self, batch):
        im_reshaped = batch["image"].reshape(-1, *batch["image"].shape[-3:])
        pred = self.model(im_reshaped)
        return pred.reshape(*batch["image"].shape[:4], pred.shape[-1])

    def training_step(self, batch, _):
        im_reshaped = batch["image"].reshape(-1, *batch["image"].shape[-3:])
        pred = self.model(im_reshaped)
        pred = pred.reshape(*batch["image"].shape[:4], pred.shape[-1])

        pred_gather = self.all_gather(pred, sync_grads=True)
        pred_gather = pred_gather.reshape(-1, *pred_gather.shape[2:])
        label_gather = self.all_gather(batch["label"]).reshape(-1, 1)

        losses = self.criterion(pred_gather, label_gather)

        bs = batch["image"][0].shape[0] * torch.cuda.device_count()
        log_partial = partial(self.log,
                              on_step=True,
                              on_epoch=True,
                              batch_size=bs,
                              sync_dist=True,
                              rank_zero_only=True)
        for k in self.train_loss:
            log_partial(f"train/{k}", losses[k])
            self.train_loss[k].update(losses[k], weight=bs)

        return losses["sum_loss"]

    def validation_step(self, batch, batch_idx):
        im_reshaped = batch["image"].reshape(-1, *batch["image"].shape[-3:])
        pred = self.model(im_reshaped)
        pred = pred.reshape(*batch["image"].shape[:4], pred.shape[-1])

        pred_gather = self.all_gather(pred, sync_grads=True)
        pred_gather = pred_gather.reshape(-1, *pred_gather.shape[2:])
        label_gather = self.all_gather(batch["label"]).reshape(-1, 1)

        losses = self.criterion(pred_gather, label_gather)

        bs = batch["image"][0].shape[0] * torch.cuda.device_count()
        for k in self.val_loss:
            self.val_loss[k].update(losses[k], weight=bs)

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx):
        self.model.eval()
        
        if isinstance(batch["image"], torch.Tensor):
            assert len(batch["image"].shape) == 4
            out = self.model.bb(batch["image"])
            return {
                "path": batch["path"],
                "label": batch["label"],
                "embeddings": out
            }
        else:
            out = self.model.bb(batch["image"][0][0])
            return {
                "path": batch["path"][0],
                "label": batch["label"][0],
                "embeddings": out
            }


    def on_train_epoch_end(self):
        for k in self.train_loss:
            train_loss_k = self.train_loss[k].compute()
            self.log(f"train/{k}_manualepoch",
                     train_loss_k,
                     on_epoch=True,
                     sync_dist=True,
                     rank_zero_only=True)
            logging.info(f"train/{k}_manualepoch {train_loss_k}")
            self.train_loss[k].reset()

    def on_validation_epoch_end(self):
        for k in self.val_loss:
            val_loss_k = self.val_loss[k].compute()
            self.log(f"val/{k}_manualepoch",
                     val_loss_k,
                     on_epoch=True,
                     sync_dist=True,
                     rank_zero_only=True)
            logging.info(f"val/{k}_manualepoch {val_loss_k}")
            self.val_loss[k].reset()

    def configure_ddp(self, *args, **kwargs):
        logging.basicConfig(level=logging.INFO)
        return super().configure_ddp(*args, **kwargs)

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

from fastglioma.datasets.srh_dataset import HiDiscDataset
from fastglioma.datasets.improc import get_transformations

def get_dataloaders(cf):
    """Create dataloader for contrastive experiments."""
    train_xform, valid_xform = get_transformations(cf)

    logging.info(f"train_xform\n{train_xform}")
    logging.info(f"valid_xform\n{valid_xform}")

    train_dset = HiDiscDataset(
        data_root=cf["data"]["db_root"],
        studies="train",
        transform=train_xform,
        balance_study_per_class=cf["data"]["balance_study_per_class"],
        num_slide_samples=cf["data"]["hidisc"]["num_slide_samples"],
        num_patch_samples=cf["data"]["hidisc"]["num_patch_samples"],
        num_transforms=cf["data"]["hidisc"]["num_transforms"])
    val_dset = HiDiscDataset(
        data_root=cf["data"]["db_root"],
        studies="val",
        transform=valid_xform,
        balance_study_per_class=False,
        num_slide_samples=cf["data"]["hidisc"]["num_slide_samples"],
        num_patch_samples=cf["data"]["hidisc"]["num_patch_samples"],
        num_transforms=cf["data"]["hidisc"]["num_transforms"])

    dataloader_callable = partial(torch.utils.data.DataLoader,
                                  batch_size=cf['training']['batch_size'],
                                  drop_last=False,
                                  pin_memory=True,
                                  num_workers=get_num_worker(),
                                  persistent_workers=True)

    return dataloader_callable(train_dset,
                               shuffle=True), dataloader_callable(val_dset,
                                                                  shuffle=True)


def main():
    cf_fd = parse_args()
    cf = yaml.load(cf_fd, Loader=yaml.FullLoader)
    exp_root, model_dir, cp_config = setup_output_dirs(cf, get_exp_name, "")
    pl.seed_everything(cf["infra"]["seed"])

    # logging and copying config files
    cp_config(cf_fd.name)
    config_loggers(exp_root)

    train_loader, valid_loader = get_dataloaders(cf)

    logging.info(f"num devices: {torch.cuda.device_count()}")
    logging.info(f"num workers in dataloader: {train_loader.num_workers}")

    num_it_per_ep = len(train_loader)
    if torch.cuda.device_count() > 1:
        num_it_per_ep //= torch.cuda.device_count()

    exp = HiDiscSystem(cf, num_it_per_ep)

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
        precision=cf["training"].get("amp", "32"),
        deterministic=cf["training"].get("deterministic", False),
        num_nodes=1)
    trainer.fit(exp,
                train_dataloaders=train_loader,
                val_dataloaders=valid_loader)


if __name__ == '__main__':
    main()
