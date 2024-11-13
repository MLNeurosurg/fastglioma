import sys
import logging
import os
import copy
import shutil
import gzip
import numpy as np
import pandas as pd
from typing import Optional, List, TypedDict, Any, NamedTuple
from collections import defaultdict
from tqdm import tqdm
from glob import glob
import torch


def nio_patchpath_to_coord(patch_name: str):
    """Converts SRH patch pathname to its 300x300 coordinate on the WSI
    Example: `/.../NIO_UM_135-6-0_1000_300_600.tif` -> (1, 5)
    """
    ix, jx, iy, jy = patch_name.split("/")[-1].split("-")[-1].split(
        ".")[0].split("_")
    return (3 * (int(ix) // 1000) + int(iy) // 300,
            3 * (int(jx) // 1000) + int(jy) // 300)


def prediction_to_slide_embedding(saving_dir: str,
                                  tag: str,
                                  embed_path: str = "",
                                  predictions=None):

    """Preassign data indices by either slide/patient."""
    if embed_path and not predictions:
        logging.info(f"Loading {embed_path}")
        if embed_path.endswith(".gz"):
            with gzip.open(embed_path) as f:
                predictions = torch.load(f)
        else:
            predictions = torch.load(embed_path)
        logging.info(f"Loading {embed_path} - OK")

        assert ((len(predictions["path"]) == len(predictions["label"])) or
                (len(predictions["label"]) == len(predictions["embeddings"])))

    if not (embed_path or predictions):
        raise ValueError("embed_path or predictions should be specified.")

    # make dictionary for all slides from the patch predictions
    slide_instances_ = defaultdict(list)
    for idx in tqdm(range(len(predictions["path"]))):
        path_i = predictions["path"][idx]
        label_i = predictions["label"][idx]
        embedding_i = predictions["embeddings"][idx]
        patient_name = path_i.split("/")[-4]
        slide_name = path_i.split("/")[-3]
        slide_instances_[patient_name + "." + str(slide_name)].append(
            [path_i, label_i, embedding_i])

    # process each slide and save
    for slide_id in tqdm(slide_instances_):
        patches = slide_instances_[slide_id]
        coords = [nio_patchpath_to_coord(p[0]) for p in patches]

        # sort patch order
        sorted_indices = sorted(enumerate(coords), key=lambda x: x[1])
        ordered_idx = [index for index, _ in sorted_indices]
        patches_path = []
        patches_label = []
        patches_embeddings = []
        patches_coords = []
        for idx in ordered_idx:
            patches_path.append(patches[idx][0])
            patches_label.append(patches[idx][1])
            patches_embeddings.append(patches[idx][2])
            patches_coords.append(coords[idx])

        patches_label = torch.stack(patches_label)
        patches_embeddings = torch.stack(patches_embeddings)
        slide_data = {
            "path": patches_path,
            "labels": patches_label,
            "embeddings": patches_embeddings,
            "coords": patches_coords
        }

        # save
        institute = patches[0][0].split("/")[-5]
        slide_data_dir = os.path.join(saving_dir, institute,
                                      slide_id.split(".")[0],
                                      slide_id.split(".")[1])
        if not os.path.exists(slide_data_dir):
            os.makedirs(slide_data_dir)
        slide_npath = os.path.join(slide_data_dir, slide_id + f"-{tag}.pt")
        torch.save(slide_data, slide_npath)

        logging.debug(f"{slide_id} DONE")
