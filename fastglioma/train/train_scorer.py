"""Slide-level training with ordmet script.

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

from fastglioma.models.cnn import MLP
from fastglioma.models.mil import TransformerMIL, MIL_forward, MIL_Classifier
from fastglioma.utils.common import (setup_output_dirs, parse_args,
                                     get_exp_name, config_loggers,
                                     get_optimizer_func, get_scheduler_func,
                                     get_num_worker)
from fastglioma.losses.ordmet import OrdinalMetricLoss


class SlideOrdMetSystem(pl.LightningModule):
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
                      n_out=1)
        self.model = MIL_Classifier(None, mil, mlp)

        if "training" in cf:
            self.criterion = OrdinalMetricLoss(**self.cf_["training"]["objective"]["params"])
            self.train_loss = torchmetrics.MeanMetric()
            self.val_loss = torchmetrics.MeanMetric()
        else:
            self.criterion = self.train_loss = self.val_loss = None

    @staticmethod
    def get_kth_view(data: List[List[torch.Tensor]], k: int):
        return [d[k] for d in data]

    def training_step(self, batch, batch_idx):
        pred = self.model(self.get_kth_view(batch["embeddings"], 0),
                       coords=self.get_kth_view(batch["coords"], 0))["logits"]
        pred_gather = self.all_gather(pred, sync_grads=True)
        pred_gather = pred_gather.reshape(-1, 1)
        # pred_gather = pred_gather.reshape(-1, *pred_gather.shape[-2:])
        label_gather = self.all_gather(batch["label"]).reshape(-1, 1)

        loss = self.criterion(pred_gather, label_gather)["loss"]
        bs = len(batch["embeddings"]) * torch.distributed.get_world_size()
        self.log("train/loss",
                 loss,
                 on_step=True,
                 on_epoch=True,
                 batch_size=bs,
                 rank_zero_only=True)
        self.train_loss.update(loss, weight=bs)

        return loss
    
    @torch.inference_mode()
    def validation_step(self, batch, batch_idx):
        bs = len(batch["embeddings"]) * torch.distributed.get_world_size()

        pred = self.model(self.get_kth_view(batch["embeddings"], 0),
                       coords=self.get_kth_view(batch["coords"], 0))["logits"]
        pred_gather = self.all_gather(pred, sync_grads=True)
        pred_gather = pred_gather.reshape(-1, 1)
        # pred_gather = pred_gather.reshape(-1, *pred_gather.shape[-2:])
        label_gather = self.all_gather(batch["label"]).reshape(-1, 1)

        loss = self.criterion(pred_gather, label_gather)["loss"]
        bs = len(batch["embeddings"]) * torch.distributed.get_world_size()
        self.log("val/loss",
                 loss,
                 on_step=True,
                 on_epoch=True,
                 batch_size=bs,
                 rank_zero_only=True)
        self.val_loss.update(loss, weight=bs)

    def on_train_epoch_end(self):
        torch.cuda.empty_cache()

        # compute metrics
        train_loss = self.train_loss.compute()

        # log metrics
        self.log("train/loss",
                 train_loss,
                 on_epoch=True,
                 sync_dist=True,
                 rank_zero_only=True)
        self.train_loss.reset()

        log_metrics = {"ap": {}, "auroc": {}}

    @torch.inference_mode()
    def on_validation_epoch_end(self):
        # compute metrics
        val_loss = self.val_loss.compute()

        # log metrics
        self.log("val/loss",
                 val_loss,
                 on_epoch=True,
                 sync_dist=True,
                 rank_zero_only=True)
        self.val_loss.reset()

    def predict_step(self, batch, batch_idx):
        out = self.model(self.get_kth_view(batch["embeddings"], 0),
            coords=self.get_kth_view(batch["coords"], 0))

        return {
            "path": [batch["path"]],
            "label": batch["label"],
            "logits": out["logits"],
            "embeddings": out["embeddings"]
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
        use_patient_class=cf["data"]["use_patient_class"],
        meta_fname=cf["data"]["meta_fname"],
        num_transforms=cf["data"]["num_transforms"])
    val_dset = SlideEmbeddingDataset(
        data_root=cf["data"]["db_root"],
        embedding_root=cf["data"]["embedding_root"],
        tag=cf["data"]["tag"],
        studies="val",
        transform=valid_xform,
        balance_slide_per_class=False,
        use_patient_class=cf["data"]["use_patient_class"],
        meta_fname=cf["data"]["meta_fname"],
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

    exp = SlideOrdMetSystem(cf, num_it_per_ep)

    if "load_backbone" in cf["training"]:
        # load lightning checkpint
        ckpt_dict = torch.load(cf["training"]["load_backbone"].get("ckpt_path", None),
                               map_location="cpu")

        mil_state_dict = {
            k.removeprefix("model.bb.mil."): ckpt_dict["state_dict"][k]
            for k in ckpt_dict["state_dict"] if "model.bb.mil" in k
        }
        
        exp.model.bb.mil.load_state_dict(mil_state_dict)

        if not cf["training"]["load_backbone"].get("finetune", True):
            for param in exp.model.bb.mil.parameters():
                param.requires_grad = False
            exp.model.bb.mil.eval()

        logging.info(f"Loaded checkpoint {cf['training']['load_backbone'].get('ckpt_path', None)}")

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
