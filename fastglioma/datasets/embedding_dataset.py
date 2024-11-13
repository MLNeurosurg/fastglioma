"""PyTorch embedding datasets designed to work with OpenSRH.

Copyright (c) 2024 University of Michigan. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

import os
import json
import logging
from collections import Counter
from typing import Optional, List, Union, TypedDict, Tuple
import random

from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import is_image_file
from torchvision.transforms import Compose


class SlideEmbeddingDataset(Dataset):
    """OpenSRH embedding dataset"""

    def __init__(self,
                 embedding_root: str,
                 tag: List[str],
                 data_root: str,
                 studies: Union[str, List[str]],
                 transform: callable = None,
                 target_transform: callable = torch.tensor,
                 balance_slide_per_class: bool = False,
                 use_patient_class: bool = True,
                 check_images_exist: bool = False,
                 meta_fname: str = "opensrh.json",
                 num_transforms: int = 1) -> None:
            """Inits the OpenSRH dataset, where each instance is a slide."""

            self.embed_root_ = embedding_root
            self.tag_ = tag
            
            self.data_root_ = data_root
            self.transform_ = transform
            self.target_transform_ = target_transform
            self.use_patient_class_ = use_patient_class
            self.check_images_exist_ = check_images_exist
            self.meta_fname_ = meta_fname
            self.num_transforms_ = num_transforms
            self.get_all_meta()
            self.get_study_list(studies)

            # Walk through each study
            self.instances_ = []
            for p in tqdm(self.studies_):
                self.instances_.extend(self.get_study_instances(p))

            if balance_slide_per_class:
                self.replicate_balance_instances()
            self.get_weights()

    def get_all_meta(self):
        """Read in all metadata files."""

        try:
            with open(os.path.join(self.data_root_,
                                   f"meta/{self.meta_fname_}")) as fd:
                self.metadata_ = json.load(fd)
        except Exception as e:
            logging.critical("Failed to locate dataset.")
            raise e

        logging.info(f"Locate OpenSRH dataset at {self.data_root_}")
        return

    def get_study_list(self, studies):
        """Get a list of studies from default split or list of IDs."""

        if isinstance(studies, str):
            try:
                with open(
                        os.path.join(self.data_root_,
                                     "meta/train_val_split.json")) as fd:
                    train_val_split = json.load(fd)
            except Exception as e:
                logging.critical("Failed to locate preset train/val split.")
                raise e

            if studies == "train":
                self.studies_ = train_val_split["train"]
            elif studies in ["valid", "val"]:
                self.studies_ = train_val_split["val"]
            else:
                return ValueError(
                    "studies split must be one of [\"train\", \"val\"]")
        elif isinstance(studies, List):
            self.studies_ = studies
        else:
            raise ValueError("studies must be a string representing " +
                             "train/val split or a list of study numbers")
        return

    def get_study_instances(self, patient: str):
        """Get all instances from one study."""

        study_instances = []
        logging.debug(patient)
        
        for s in self.metadata_[patient]["slides"]:
            slide_tag_instances = []
            for tag in self.tag_:
                tag_path = os.path.join(self.embed_root_, 
                                        "studies", 
                                        patient, 
                                        s, 
                                        f"{patient}.{s}-{tag}.pt")
                if os.path.exists(tag_path):
                    slide_tag_instances.append(tag_path)

            slide_label = (
                self.metadata_[patient]["slides"][s].get(
                    "slide_class", self.metadata_[patient]["class"])  #yapf:disable
                if not self.use_patient_class_ else
                self.metadata_[patient]["class"])

            slide_instance = (f"{patient}/{s}", slide_label, slide_tag_instances) #yapf:disable

            logging.debug(f"slide {patient}/{s} tags {len(slide_tag_instances)}") #yapf:disable
            study_instances.append(slide_instance)

        logging.debug(f"patient {patient} slides {len(study_instances)}")
        return study_instances

    def process_classes(self):
        """Look for all the labels in the dataset.

        Creates the classes_, and class_to_idx_ attributes"""
        all_labels = [i[1] for i in self.instances_]
        self.classes_ = sorted(set(all_labels))
        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}
        logging.info("Labels: {}".format(self.classes_))
        return

    def get_weights(self):
        """Count number of instances for each class, and computes weights."""

        # Get classes
        self.process_classes()
        all_labels = [self.class_to_idx_[i[1]] for i in self.instances_]

        # Count number of slides in each class
        count = Counter(all_labels)
        count = torch.Tensor([count[i] for i in range(len(count))])
        logging.info("Count: {}".format(count))

        # Compute weights
        inv_count = 1 / count
        self.weights_ = inv_count / torch.sum(inv_count)
        logging.debug("Weights: {}".format(self.weights_))
        return self.weights_

    def replicate_balance_instances(self):
        """resample the instances list to balance each class."""
        all_labels = [i[1] for i in self.instances_]
        val_sample = max(Counter(all_labels).values())

        all_instances_ = []
        for l in sorted(set(all_labels)):
            instances_l = [i for i in self.instances_ if i[1] == l]
            random.shuffle(instances_l)
            instances_l = instances_l * (val_sample // len(instances_l) + 1)
            all_instances_.extend(sorted(instances_l[:val_sample]))

        self.instances_ = all_instances_
        return

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.instances_)

    def __getitem__(self, idx):
        """Retrieve a list of slides specified by idx"""

        slide, target, tag_list = self.instances_[idx]
        target = self.class_to_idx_[target]

        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        instance = {
            "embeddings": [None for _ in range(self.num_transforms_)],
            "coords": [None for _ in range(self.num_transforms_)],
            "path": tag_list[0],
            "label": target,
        } #yapf:disable

        for transform_idx in range(self.num_transforms_):
            pt_path = random.choice(tag_list)
            inst_ = torch.load(pt_path)

            instance["embeddings"][transform_idx] = inst_["embeddings"]
            instance["coords"][transform_idx] = torch.tensor(inst_["coords"])

            del inst_

        if self.transform_:
            instance = self.transform_(instance)

        return instance
