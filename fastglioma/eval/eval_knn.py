"""kNN evaluation modules and script.

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
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import torch
from torchvision.transforms import Compose

import pytorch_lightning as pl
from torchmetrics import AveragePrecision, Accuracy

from fastglioma.datasets.srh_dataset import PatchDataset
from fastglioma.datasets.embedding_dataset import SlideEmbeddingDataset
from fastglioma.datasets.improc import get_transformations
from fastglioma.datasets.emb_proc import get_emb_transformations, emb_collate_fn
from fastglioma.utils.common import (parse_args, get_exp_name, config_loggers,
                                 get_num_worker)
from fastglioma.train.train_patch import HiDiscSystem
from fastglioma.train.train_slide import SlideSSLSystem


# code for kNN prediction is from the github repo IgorSusmelj/barlowtwins
# https://github.com/IgorSusmelj/barlowtwins/blob/main/utils.py
def knn_predict(feature, feature_bank, feature_labels, classes: int,
                knn_k: int, knn_t: float):
    """Helper method to run kNN predictions on features from a feature bank.

    Args:
        feature: Tensor of shape [N, D] consisting of N D-dimensional features
        feature_bank: Tensor of a database of features used for kNN
        feature_labels: Labels for the features in our feature_bank
        classes: Number of classes (e.g. 10 for CIFAR-10)
        knn_k: Number of k neighbors used for kNN
        knn_t: Temperature
    """
    # cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1),
                              dim=-1,
                              index=sim_indices)
    # we do a reweighting of the similarities
    sim_weight = (sim_weight / knn_t).exp()
    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k,
                                classes,
                                device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1,
                                          index=sim_labels.view(-1, 1),
                                          value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) *
                            sim_weight.unsqueeze(dim=-1),
                            dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels, pred_scores


def get_embeddings_patch(cf: Dict[str, Any],
                   exp_root: str) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """Run forward pass on the dataset, and generate embeddings and logits"""
    train_xform, valid_xform = get_transformations(cf)

    logging.info(f"train_xform \n{train_xform}")
    logging.info(f"valid_xform \n{valid_xform}")

    # get dataset / loader
    train_dset = PatchDataset(data_root=cf["data"]["db_root"],
                              studies="train",
                              transform=train_xform,
                              balance_patch_per_class=False)

    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=cf["eval"]["predict_batch_size"],
        drop_last=False,
        pin_memory=True,
        num_workers=get_num_worker(),
        persistent_workers=True)

    val_dset = PatchDataset(data_root=cf["data"]["db_root"],
                            studies="val",
                            transform=valid_xform,
                            balance_patch_per_class=False)

    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=cf["eval"]["predict_batch_size"],
        drop_last=False,
        pin_memory=True,
        num_workers=get_num_worker(),
        persistent_workers=True)

    # load lightning checkpoint
    ckpt_path = os.path.join(cf["infra"]["log_dir"], cf["infra"]["exp_name"],
                             cf["eval"]["ckpt_path"])

    model = HiDiscSystem.load_from_checkpoint(ckpt_path,
                                              cf=cf,
                                              num_it_per_ep=0,
                                              max_epochs=-1,
                                              nc=0)

    # create trainer
    trainer = pl.Trainer(accelerator="gpu",
                         devices=1,
                         max_epochs=-1,
                         default_root_dir=exp_root,
                         enable_checkpointing=False,
                         logger=False)

    # generate predictions
    train_predictions = trainer.predict(model, dataloaders=train_loader)
    val_predictions = trainer.predict(model, dataloaders=val_loader)

    def process_predictions(predictions):
        pred = {}
        for k in predictions[0].keys():
            if k == "path":
                pred[k] = [pk for p in predictions for pk in p[k][0]]
            else:
                pred[k] = torch.cat([p[k] for p in predictions])
        return pred

    train_predictions = process_predictions(train_predictions)
    val_predictions = process_predictions(val_predictions)

    train_embs = torch.nn.functional.normalize(train_predictions["embeddings"],
                                               p=2,
                                               dim=1).T
    val_embs = torch.nn.functional.normalize(val_predictions["embeddings"],
                                             p=2,
                                             dim=1)

    # knn evaluation
    batch_size = cf["eval"]["knn"]["batch_size"]
    all_scores = []
    for k in tqdm(range(val_embs.shape[0] // batch_size + 1)):
        start_coeff = batch_size * k
        end_coeff = min(batch_size * (k + 1), val_embs.shape[0])
        val_embs_k = val_embs[start_coeff:end_coeff]  # 1536 x 2048

        pred_labels, pred_scores = knn_predict(
            val_embs_k,
            train_embs,
            train_predictions["label"],
            len(train_loader.dataset.classes_),
            knn_k=cf["eval"]["knn"]["k"],
            knn_t=cf["eval"]["knn"]["t"])

        all_scores.append(
            torch.nn.functional.normalize(pred_scores, p=1, dim=1))
        torch.cuda.empty_cache()

    val_predictions["logits"] = torch.vstack(all_scores)
    return val_predictions


def get_embeddings_slide(cf: Dict[str, Any],
                   exp_root: str) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """Run forward pass on the dataset, and generate embeddings and logits"""
    train_xform, valid_xform = get_emb_transformations(cf)

    logging.info(f"train_xform \n{train_xform}")
    logging.info(f"valid_xform \n{valid_xform}")

    # get dataset / loader
    train_dset = SlideEmbeddingDataset(data_root=cf["data"]["db_root"],
                                       embedding_root=cf["data"]["embedding_root"],
                                       tag=cf["data"]["tag"],
                                       studies="train",
                                       transform=train_xform,
                                       balance_slide_per_class=False,
                                       num_transforms=1)

    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=cf["eval"]["predict_batch_size"],
        drop_last=False,
        pin_memory=True,
        num_workers=get_num_worker(),
        persistent_workers=True,
        collate_fn=emb_collate_fn)

    val_dset = SlideEmbeddingDataset(data_root=cf["data"]["db_root"],
                                     embedding_root=cf["data"]["embedding_root"],
                                     tag=cf["data"]["tag"],
                                     studies="val",
                                     transform=valid_xform,
                                     balance_slide_per_class=False,
                                     num_transforms=1)

    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=cf["eval"]["predict_batch_size"],
        drop_last=False,
        pin_memory=True,
        num_workers=get_num_worker(),
        persistent_workers=True,
        collate_fn=emb_collate_fn)

    # load lightning checkpoint
    ckpt_path = os.path.join(cf["infra"]["log_dir"], cf["infra"]["exp_name"],
                             cf["eval"]["ckpt_path"])

    model = SlideSSLSystem.load_from_checkpoint(ckpt_path,
                                              cf=cf,
                                              num_it_per_ep=0,
                                              max_epochs=-1,
                                              nc=0)

    # create trainer
    trainer = pl.Trainer(accelerator="gpu",
                         devices=1,
                         max_epochs=-1,
                         default_root_dir=exp_root,
                         enable_checkpointing=False,
                         logger=False)

    # generate predictions
    train_predictions = trainer.predict(model, dataloaders=train_loader)
    val_predictions = trainer.predict(model, dataloaders=val_loader)

    def process_predictions(predictions):
        pred = {}
        for k in predictions[0].keys():
            if k == "path":
                pred[k] = [pk for p in predictions for pk in p[k][0]]
            else:
                pred[k] = torch.cat([p[k] for p in predictions])
        return pred

    train_predictions = process_predictions(train_predictions)
    val_predictions = process_predictions(val_predictions)

    train_embs = torch.nn.functional.normalize(train_predictions["embeddings"],
                                               p=2,
                                               dim=1).T
    val_embs = torch.nn.functional.normalize(val_predictions["embeddings"],
                                             p=2,
                                             dim=1)

    # knn evaluation
    batch_size = cf["eval"]["knn"]["batch_size"]
    all_scores = []
    for k in tqdm(range(val_embs.shape[0] // batch_size + 1)):
        start_coeff = batch_size * k
        end_coeff = min(batch_size * (k + 1), val_embs.shape[0])
        val_embs_k = val_embs[start_coeff:end_coeff]  # 1536 x 2048

        pred_labels, pred_scores = knn_predict(
            val_embs_k,
            train_embs,
            train_predictions["label"],
            len(train_loader.dataset.classes_),
            knn_k=cf["eval"]["knn"]["k"],
            knn_t=cf["eval"]["knn"]["t"])

        all_scores.append(
            torch.nn.functional.normalize(pred_scores, p=1, dim=1))
        torch.cuda.empty_cache()

    val_predictions["logits"] = torch.vstack(all_scores)
    return val_predictions


def make_specs_patch(predictions: Dict[str, Union[torch.Tensor, List[str]]]) -> None:
    """Compute all specs for an experiment"""

    # aggregate prediction into a dataframe
    pred = pd.DataFrame.from_dict({
        "path":
        predictions["path"],
        "labels": [l.item() for l in list(predictions["label"])],
        "logits": [l.tolist() for l in list(predictions["logits"])]
    })
    pred["logits"] = pred["logits"].apply(
        lambda x: torch.nn.functional.softmax(torch.tensor(x), dim=0))

    # add patient and slide info from patch paths
    pred["patient"] = pred["path"].apply(lambda x: x.split("/")[-4])
    pred["slide"] = pred["path"].apply(lambda x: "/".join(
        [x.split("/")[-4], x.split("/")[-3]]))

    # aggregate logits
    get_agged_logits = lambda pred, mode: pd.DataFrame(
        pred.groupby(by=[mode, "labels"])["logits"].apply(
            lambda x: [sum(y) for y in zip(*x)])).reset_index()

    slides = get_agged_logits(pred, "slide")
    patients = get_agged_logits(pred, "patient")

    normalize_f = lambda x: torch.nn.functional.normalize(x, dim=1, p=1)
    patch_logits = normalize_f(torch.tensor(np.vstack(pred["logits"])))
    slides_logits = normalize_f(torch.tensor(np.vstack(slides["logits"])))
    patient_logits = normalize_f(torch.tensor(np.vstack(patients["logits"])))

    patch_label = torch.tensor(pred["labels"])
    slides_label = torch.tensor(slides["labels"])
    patient_label = torch.tensor(patients["labels"])

    # generate metrics
    def get_all_metrics(logits, label):
        map = AveragePrecision(task="multiclass", num_classes=7)
        acc = Accuracy(task="multiclass", num_classes=7)
        t2 = Accuracy(task="multiclass", num_classes=7, top_k=2)
        t3 = Accuracy(task="multiclass", num_classes=7, top_k=3)
        mca = Accuracy(task="multiclass", num_classes=7, average="macro")

        acc_val = acc(logits, label)
        t2_val = t2(logits, label)
        t3_val = t3(logits, label)
        mca_val = mca(logits, label)
        map_val = map(logits, label)

        return torch.stack((acc_val, t2_val, t3_val, mca_val, map_val))

    all_metrics = torch.vstack((get_all_metrics(patch_logits, patch_label),
                                get_all_metrics(slides_logits, slides_label),
                                get_all_metrics(patient_logits,
                                                patient_label)))
    all_metrics = pd.DataFrame(all_metrics,
                               columns=["acc", "t2", "t3", "mca", "map"],
                               index=["patch", "slide", "patient"])

    # generate confusion matrices
    patch_conf = confusion_matrix(y_true=patch_label,
                                  y_pred=patch_logits.argmax(dim=1))

    slide_conf = confusion_matrix(y_true=slides_label,
                                  y_pred=slides_logits.argmax(dim=1))

    patient_conf = confusion_matrix(y_true=patient_label,
                                    y_pred=patient_logits.argmax(dim=1))

    print("\nmetrics")
    print(all_metrics)
    print("\npatch confusion matrix")
    print(patch_conf)
    print("\nslide confusion matrix")
    print(slide_conf)
    print("\npatient confusion matrix")
    print(patient_conf)

    return


def make_specs_slide(predictions: Dict[str, Union[torch.Tensor, List[str]]]) -> None:
    """Compute all specs for an experiment"""

    # aggregate prediction into a dataframe
    pred = pd.DataFrame.from_dict({
        "path":
        predictions["path"],
        "labels": [l.item() for l in list(predictions["label"])],
        "logits": [l.tolist() for l in list(predictions["logits"])]
    })
    pred["logits"] = pred["logits"].apply(
        lambda x: torch.nn.functional.softmax(torch.tensor(x), dim=0))

    # add patient and slide info from patch paths
    pred["patient"] = pred["path"].apply(lambda x: x.split("/")[-3])

    # aggregate logits
    get_agged_logits = lambda pred, mode: pd.DataFrame(
        pred.groupby(by=[mode, "labels"])["logits"].apply(
            lambda x: [sum(y) for y in zip(*x)])).reset_index()

    patients = get_agged_logits(pred, "patient")

    normalize_f = lambda x: torch.nn.functional.normalize(x, dim=1, p=1)
    slides_logits = normalize_f(torch.tensor(np.vstack(pred["logits"])))
    patient_logits = normalize_f(torch.tensor(np.vstack(patients["logits"])))

    slides_label = torch.tensor(pred["labels"])
    patient_label = torch.tensor(patients["labels"])

    # generate metrics
    def get_all_metrics(logits, label):
        map = AveragePrecision(task="multiclass", num_classes=7)
        acc = Accuracy(task="multiclass", num_classes=7)
        t2 = Accuracy(task="multiclass", num_classes=7, top_k=2)
        t3 = Accuracy(task="multiclass", num_classes=7, top_k=3)
        mca = Accuracy(task="multiclass", num_classes=7, average="macro")

        acc_val = acc(logits, label)
        t2_val = t2(logits, label)
        t3_val = t3(logits, label)
        mca_val = mca(logits, label)
        map_val = map(logits, label)

        return torch.stack((acc_val, t2_val, t3_val, mca_val, map_val))

    all_metrics = torch.vstack((get_all_metrics(slides_logits, slides_label),
                                get_all_metrics(patient_logits,
                                                patient_label)))
    all_metrics = pd.DataFrame(all_metrics,
                               columns=["acc", "t2", "t3", "mca", "map"],
                               index=["slide", "patient"])

    # generate confusion matrices
    slide_conf = confusion_matrix(y_true=slides_label,
                                  y_pred=slides_logits.argmax(dim=1))

    patient_conf = confusion_matrix(y_true=patient_label,
                                    y_pred=patient_logits.argmax(dim=1))

    print("\nmetrics")
    print(all_metrics)
    print("\nslide confusion matrix")
    print(slide_conf)
    print("\npatient confusion matrix")
    print(patient_conf)

    return


def setup_eval_paths(cf, get_exp_name, cmt_append):
    """Get name of the ouput dirs and create them in the file system."""
    log_root = cf["infra"]["log_dir"]
    exp_name = cf["infra"]["exp_name"]
    instance_name = cf["eval"]["ckpt_path"].split("/")[0]
    eval_instance_name = "_".join([get_exp_name(cf), cmt_append])
    exp_root = os.path.join(log_root, exp_name, instance_name, "evals",
                            eval_instance_name)

    # generate needed folders, evals will be embedded in experiment folders
    pred_dir = os.path.join(exp_root, 'predictions')
    config_dir = os.path.join(exp_root, 'config')
    for dir_name in [pred_dir, config_dir]:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    # if there is a previously generated prediction, also return the
    # prediction filename so we don't have to predict again
    if cf["eval"].get("eval_predictions", None):
        other_eval_instance_name = cf["eval"]["eval_predictions"]
        pred_fname = os.path.join(log_root, exp_name, instance_name, "evals",
                                  other_eval_instance_name, "predictions",
                                  "predictions.pt")
    else:
        pred_fname = None

    return exp_root, pred_dir, partial(copy2, dst=config_dir), pred_fname


def main():
    """Driver script for evaluation pipeline."""
    cf_fd = parse_args()
    cf = yaml.load(cf_fd, Loader=yaml.FullLoader)
    exp_root, pred_dir, cp_config, pred_fname = setup_eval_paths(
        cf, get_exp_name, "")
    pl.seed_everything(cf["infra"]["seed"])

    # logging and copying config files
    cp_config(cf_fd.name)
    config_loggers(exp_root)

    # get predictions
    if not cf["eval"].get("eval_predictions", None):
        logging.info("generating predictions")
        if cf["model"]["train_alg"] == "hidisc":
            predictions = get_embeddings_patch(cf, exp_root)
        elif cf["model"]["train_alg"] == "scm":
            predictions = get_embeddings_slide(cf, exp_root)
        else:
            raise NotImplementedError(f"train_alg {cf['model']['train_alg']} not implemented")
        torch.save(predictions, os.path.join(pred_dir, "predictions.pt"))
    else:
        logging.info("loading predictions")
        predictions = torch.load(pred_fname)

    # generate specs
    if cf["model"]["train_alg"] == "hidisc":
        make_specs_patch(predictions)
    elif cf["model"]["train_alg"] == "scm":
        make_specs_slide(predictions)
    else:
        raise NotImplementedError(f"train_alg {cf['model']['train_alg']} not implemented")


if __name__ == "__main__":
    main()