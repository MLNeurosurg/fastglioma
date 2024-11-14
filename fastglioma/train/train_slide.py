"""Slide SSL with SCM/VICReg pretraining script.

Copyright (c) 2024 University of Michigan. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

import yaml
import logging
from functools import partial
from typing import Dict, Any, List

import torch

import pytorch_lightning as pl
import torchmetrics

from fastglioma.models.cnn import MLP, VICRegNetwork
from fastglioma.models.mil import TransformerMIL, MIL_forward
from fastglioma.utils.common import (setup_output_dirs, parse_args,
                                     get_exp_name, config_loggers,
                                     get_optimizer_func, get_scheduler_func,
                                     get_num_worker)
from fastglioma.losses.vicreg import GeneralVICRegLoss


class SlideSSLSystem(pl.LightningModule):
    """Lightning system for slide ssl experiments."""

    def __init__(self, cf: Dict[str, Any], num_it_per_ep: int):
        super().__init__()
        self.cf_ = cf
        self.num_it_per_ep_ = num_it_per_ep

        if cf["model"]["backbone"]["which"] == "transformer":
            mil = partial(MIL_forward,
                          mil=partial(TransformerMIL,
                                      **cf["model"]["backbone"]["params"]))
        else:
            raise NotImplementedError()

        mlp = partial(MLP,
                      n_in=mil().num_out,
                      hidden_layers=cf["model"]["mlp_hidden"],
                      n_out=cf["model"]["num_embedding_out"])
        self.model = VICRegNetwork(mil, mlp)

        if "training" in cf:
            self.criterion = GeneralVICRegLoss(
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
            self.train_loss = self.val_loss = None

    @staticmethod
    def get_kth_view(data: List[List[torch.Tensor]], k: int):
        return [d[k] for d in data]

    def forward(self, batch):
        pred = [
            self.model(self.get_kth_view(batch["embeddings"], 0),
            coords=self.get_kth_view(batch["coords"], 0)), 
            self.model(self.get_kth_view(batch["embeddings"], 1),
            coords=self.get_kth_view(batch["coords"], 1))
            ]
            
        pred = torch.stack(pred, dim=1)
        pred_gather = self.all_gather(pred, sync_grads=True)
        pred_gather = pred_gather.reshape(-1, *pred_gather.shape[-2:])

        losses = self.criterion(pred_gather)

        return losses

    def training_step(self, batch, _):
        losses = self.forward(batch)
        bs = len(batch['embeddings']) * torch.distributed.get_world_size()

        for k in self.train_loss:
            self.log(f"train/{k}",
                     losses[k],
                     on_step=True,
                     on_epoch=False,
                     batch_size=bs,
                     rank_zero_only=True)
            self.train_loss[k].update(losses[k], weight=bs)

        return losses["loss"]

    @torch.inference_mode()
    def validation_step(self, batch, _):
        losses = self.forward(batch)
        bs = len(batch['embeddings']) * torch.distributed.get_world_size()
        for k in self.val_loss:
            self.val_loss[k].update(losses[k], weight=bs)

    def on_train_epoch_end(self):
        torch.cuda.empty_cache()
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

    @torch.inference_mode()
    def on_validation_epoch_end(self):
        losses = {}
        for k in self.val_loss.keys():
            losses[k] = self.val_loss[k].compute()
            self.log(f"valid/{k}_manualepoch",
                     losses[k],
                     on_epoch=True,
                     sync_dist=True,
                     rank_zero_only=True)
            self.val_loss[k].reset()
        logging.info(f"valid/manualepoch {losses}")

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx):
        out = self.model.bb(
            self.get_kth_view(batch["embeddings"], 0),
            coords=self.get_kth_view(batch["coords"], 0))

        return {
            "path": [batch["path"]],
            "label": batch["label"],
            "embeddings": out
        }

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

from fastglioma.datasets.embedding_dataset import SlideEmbeddingDataset
from fastglioma.datasets.emb_proc import get_emb_transformations, emb_collate_fn
def get_dataloaders(cf):
    """Create dataloader for contrastive experiments."""
    train_xform, valid_xform = get_emb_transformations(cf)

    logging.info(f"train_xform\n{train_xform}")
    logging.info(f"valid_xform\n{valid_xform}")

    train_dset = SlideEmbeddingDataset(
        data_root=cf["data"]["db_root"],
        embedding_root=cf["data"]["embedding_root"],
        tag=cf["data"]["tag"],
        studies="train",
        transform=train_xform,
        balance_slide_per_class=cf["data"]["balance_study_per_class"],
        num_transforms=cf["data"]["num_transforms"])
    val_dset = SlideEmbeddingDataset(
        data_root=cf["data"]["db_root"],
        embedding_root=cf["data"]["embedding_root"],
        tag=cf["data"]["tag"],
        studies="val",
        transform=valid_xform,
        balance_slide_per_class=False,
        num_transforms=cf["data"]["num_transforms"])

    dataloader_callable = partial(torch.utils.data.DataLoader,
                                  batch_size=cf['training']['batch_size'],
                                  drop_last=False,
                                  pin_memory=True,
                                  num_workers=get_num_worker(),
                                  persistent_workers=True,
                                  collate_fn=emb_collate_fn)

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

    exp = SlideSSLSystem(cf, num_it_per_ep)

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
