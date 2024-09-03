"""PyTorch datasets designed to work with OpenSRH.

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

from fastglioma.datasets.improc import process_read_im, get_srh_base_aug


class PatchData(TypedDict):
    image: Optional[torch.Tensor]
    label: Optional[torch.Tensor]
    path: Optional[List[str]]


class SlideData(TypedDict):
    image: Optional[List[torch.Tensor]]
    label: Optional[torch.Tensor]
    path: Optional[List[str]]


class PatchDataset(Dataset
                   ):  # TODO give option to either use slide or patient class
    """OpenSRH classification dataset - used for evaluation"""

    def __init__(self,
                 data_root: str,
                 studies: Union[str, List[str]],
                 transform: callable = Compose(get_srh_base_aug()),
                 target_transform: callable = torch.tensor,
                 balance_patch_per_class: bool = False,
                 use_patient_class: bool = True,
                 check_images_exist: bool = False) -> None:
        """Inits the OpenSRH dataset, where each instance is a patch.
        
        Populate each attribute and walk through slides to look for patches.

        Args:
            data_root: root OpenSRH directory
            studies: either a string in {"train", "val"} for the default
                train/val dataset split, or a list of strings representing
                patient IDs
            transform: a callable object for image transformation
            target_transform: a callable object for label transformation
            balance_patch_per_class: balance the patches in each class
            use_patient_class: use the patient class for the label instead of the slide class
            check_images_exist: a flag representing whether to check every
                image file exists in data_root. Turn this on for debugging,
                turn it off for speed.
        """

        self.data_root_ = data_root
        self.transform_ = transform
        self.target_transform_ = target_transform
        self.check_images_exist_ = check_images_exist
        self.get_all_meta()
        self.get_study_list(studies)

        # Walk through each study
        self.instances_ = []
        for p in tqdm(self.studies_):
            self.instances_.extend(self.get_study_instances(p))

        if balance_patch_per_class:
            self.replicate_balance_instances()
        self.get_weights()

    def get_all_meta(self):
        """Read in all metadata files."""

        try:
            with open(os.path.join(self.data_root_,
                                   "meta/opensrh.json")) as fd:
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

        slide_instances = []
        logging.debug(patient)
        if self.check_images_exist_:
            tiff_file_exist = lambda im_p: (os.path.exists(im_p) and
                                            is_image_file(im_p))
        else:
            tiff_file_exist = lambda _: True

        def check_add_patches(patches: List[str]):
            for p in patches:
                im_p = os.path.join(self.data_root_, p)
                if tiff_file_exist(im_p):
                    slide_instances.append(
                        (im_p, self.metadata_[patient]["class"]
                         if self.use_patient_class_ else
                         self.metadata_[patient]["slides"][s].get(
                             "class", self.metadata_[patient]["class"])))
                else:
                    logging.warning(f"Bad patch: unable to locate {im_p}")

        for s in self.metadata_[patient]["slides"]:
            for patch_type in [
                    "normal_patches", "tumor_patches", "nondiagnostic_patches"
            ]:
                if patch_type in self.metadata_[patient]["slides"][s]:
                    check_add_patches(
                        self.metadata_[patient]["slides"][s][patch_type])
        logging.debug(f"patient {patient} patches {len(slide_instances)}")
        return slide_instances

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

    def __getitem__(self, idx: int) -> PatchData:
        """Retrieve a patch specified by idx"""

        imp, target = self.instances_[idx]
        target = self.class_to_idx_[target]

        # Read image
        im: torch.Tensor = process_read_im(imp)

        # Perform transformations
        if self.transform_ is not None:
            im = self.transform_(im)
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return {"image": im, "label": target, "path": [imp]}


def nio_patchpath_to_coord(patch_name: str):
    """Converts SRH patch pathname to its 300x300 coordinate on the WSI
    Example: `/.../NIO_UM_135-6-0_1000_300_600.tif` -> (1, 5)
    """
    ix, jx, iy, jy = patch_name.split("/")[-1].split("-")[-1].split(
        ".")[0].split("_")
    return torch.tensor([
        3 * (int(ix) // 1000) + int(iy) // 300,
        3 * (int(jx) // 1000) + int(jy) // 300
    ])


def slide_collate_fn(batch):
    return {
        "image": [data['image'] for data in batch],
        "label": torch.stack([data['label'] for data in batch]),
        "path": [data['path'] for data in batch],
        "coords": [data['coords'] for data in batch]
    }


class SlideDataset(Dataset):
    """OpenSRH slide-level classification dataset"""

    def __init__(self,
                 data_root: str,
                 studies: Union[str, List[str]],
                 transform: callable = Compose(get_srh_base_aug()),
                 target_transform: callable = torch.tensor,
                 balance_slide_per_class: bool = False,
                 use_patient_class: bool = True,
                 check_images_exist: bool = False) -> None:
        """Inits the OpenSRH dataset, where each instance is a slide.
        
        Populate each attribute and walk through studies to look for slides and corresponding patches.

        Args:
            data_root: root OpenSRH directory
            studies: either a string in {"train", "val"} for the default
                train/val dataset split, or a list of strings representing
                patient IDs
            transform: a callable object for image transformation
            target_transform: a callable object for label transformation
            balance_slide_per_class: balance the slides in each class
            check_images_exist: a flag representing whether to check every
                image file exists in data_root. Turn this on for debugging,
                turn it off for speed.
        """

        self.data_root_ = data_root
        self.transform_ = transform
        self.target_transform_ = target_transform
        self.check_images_exist_ = check_images_exist
        self.use_patient_class_ = use_patient_class
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
                                   "meta/opensrh.json")) as fd:
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
        if self.check_images_exist_:
            tiff_file_exist = lambda im_p: (os.path.exists(im_p) and
                                            is_image_file(im_p))
        else:
            tiff_file_exist = lambda _: True

        def check_add_patches(patches: List[str]):
            slide_instances = []
            for p in patches:
                im_p = os.path.join(self.data_root_, p)
                if tiff_file_exist(im_p):
                    slide_instances.append(im_p)
                else:
                    logging.warning(f"Bad patch: unable to locate {im_p}")
            return slide_instances

        for s in self.metadata_[patient]["slides"]:
            slide_patch_instances = []
            for patch_type in ["normal", "tumor", "nondiagnostic"]:
                slide_patch_instances += check_add_patches(
                    self.metadata_[patient]["slides"][s]
                    [f"{patch_type}_patches"])

            slide_label = (
                self.metadata_[patient]["slides"][s].get(
                    "class", self.metadata_[patient]["class"])  #yapf:disable
                if self.use_patient_class_ else
                self.metadata_[patient]["class"])

            slide_instance = (f"{patient}/{s}", slide_label, slide_patch_instances) #yapf:disable

            logging.debug(f"slide {patient}/{s} patches {len(slide_patch_instances)}") #yapf:disable
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

    def __getitem__(self, idx: int) -> PatchData:
        """Retrieve a patch specified by idx"""

        slide, target, imp_list = self.instances_[idx]
        target = self.class_to_idx_[target]

        # Read images
        im_list: List[torch.Tensor] = [
            process_read_im(imp) for imp in imp_list
        ]

        # Perform transformations
        if self.transform_ is not None:
            im_list = [self.transform_(im) for im in im_list]
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return {
            "image": [torch.stack(im_list)],
            "label": target,
            "path": [imp_list[0]],
            "coords": [torch.stack([nio_patchpath_to_coord(imp) for imp in imp_list])]
        } #yapf:disable
