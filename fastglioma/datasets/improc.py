"""Image processing functions designed to work with OpenSRH datasets.

Copyright (c) 2024 University of Michigan. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

from typing import Optional, List, Tuple, Dict, Callable
from functools import partial

import random
import tifffile
import numpy as np

import torch
from torchvision.transforms import functional as F
from torch.nn import ModuleList
from torchvision.transforms import (
    Normalize, RandomApply, Compose, RandomHorizontalFlip, RandomVerticalFlip,
    Resize, RandAugment, RandomErasing, RandomAutocontrast, Grayscale,
    RandomSolarize, ColorJitter, RandomAdjustSharpness, GaussianBlur,
    RandomAffine, RandomResizedCrop)

# Base augmentation modules
class GetThirdChannel(torch.nn.Module):
    """Computes the third channel of SRH image

    Compute the third channel of SRH images by subtracting CH3 and CH2. The
    channel difference is added to the subtracted_base.

    """

    def __init__(self,
                 mode: str = "three_channels",
                 subtracted_base: float = 5000 / 65536.0):
        super().__init__()

        self.subtracted_base = subtracted_base
        aug_func_dict = {
            "three_channels": self.get_third_channel_,
            "ch2_only": self.get_ch2_,
            "ch3_only": self.get_ch3_,
            "diff_only": self.get_diff_
        }
        if mode in aug_func_dict:
            self.aug_func = aug_func_dict[mode]
        else:
            raise ValueError("base_augmentation must be in " +
                             f"{aug_func_dict.keys()}")

    def get_third_channel_(self, im2: torch.Tensor) -> torch.Tensor:
        ch2 = im2[0, :, :]
        ch3 = im2[1, :, :]
        ch1 = ch3 - ch2 + self.subtracted_base
        return torch.stack((ch1, ch2, ch3), dim=0)

    def get_ch2_(self, im2: torch.Tensor) -> torch.Tensor:
        return im2[0, :, :].unsqueeze(0)

    def get_ch3_(self, im2: torch.Tensor) -> torch.Tensor:
        return im2[1, :, :].unsqueeze(0)

    def get_diff_(self, im2: torch.Tensor) -> torch.Tensor:
        ch2 = im2[0, :, :]
        ch3 = im2[1, :, :]
        ch1 = ch3 - ch2 + self.subtracted_base

        return ch1.unsqueeze(0)

    def forward(self, two_channel_image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            two_channel_image: a 2 channel np array in the shape H * W * 2

        Returns:
            A 1 or 3 channel np array in the shape 3 * H * W
        """
        return self.aug_func(two_channel_image)


class MinMaxChop(torch.nn.Module):
    """Clamps the images to float (0,1) range."""

    def __init__(self, min_val: float = 0.0, max_val: float = 1.0):
        super().__init__()
        self.min_ = min_val
        self.max_ = max_val

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image.clamp(self.min_, self.max_)


class GaussianNoise(torch.nn.Module):
    """Adds guassian noise to images."""

    def __init__(self, min_var: float = 0.01, max_var: float = 0.1):
        super().__init__()
        self.min_var = min_var
        self.max_var = max_var

    def __call__(self, tensor):

        var = random.uniform(self.min_var, self.max_var)
        noisy = tensor + torch.randn(tensor.size()) * var
        noisy = torch.clamp(noisy, min=0., max=1.)
        return noisy


# Strong augmentation modules
class InpaintRows(torch.nn.Module):

    def __init__(self, y_skip: int = 2, image_size: int = 300):
        self.y_skip = y_skip
        self.image_size = image_size

    def __call__(self, img):
        self.original_y = img.shape[1]
        mask = np.arange(0, self.original_y, self.y_skip)
        img_trans = img[:, mask, :]
        img_trans = Resize(
            size=(self.image_size, self.image_size),
            interpolation=F.InterpolationMode.BILINEAR,
            antialias=True)(img_trans)
        return img_trans

    def __repr__(self):
        return self.__class__.__name__ + '()'


def process_read_im(imp: str) -> torch.Tensor:
    """Read in two channel image

    Args:
        imp: a string that is the path to the tiff image

    Returns:
        A 2 channel torch Tensor in the shape 2 * H * W
    """
    # reference: https://github.com/pytorch/vision/blob/49468279d9070a5631b6e0198ee562c00ecedb10/torchvision/transforms/functional.py#L133

    return torch.from_numpy(tifffile.imread(imp).astype(
        np.float32)).contiguous()


# helpers
def get_srh_base_aug(base_aug: str = "three_channels") -> List:
    """Base processing augmentations for all SRH images

    Args:
        base_aug: specifies which channel subset should be used ('three_channel', 'ch2_only', 'ch3_only', 'diff_only')

    Returns:
        An augmented 1 or 3 torch Tensor in the shape of 3 * H * W
    """
    u16_min = (0, 0)
    u16_max = (65536, 65536)  # 2^16

    # if y_skip != 0:
    #     xform_list = [Normalize(mean=u16_min, std=u16_max), GetThirdChannel(mode=base_aug), MinMaxChop(), InpaintRows(y_skip=y_skip)]
    # else:
    xform_list = [Normalize(mean=u16_min, std=u16_max), GetThirdChannel(mode=base_aug), MinMaxChop()]

    return xform_list


def get_strong_aug(augs, rand_prob) -> List:
    """Strong augmentations for training"""
    rand_apply = lambda which, **kwargs: RandomApply(
        ModuleList([which(**kwargs)]), p=rand_prob)

    callable_dict = {
        "resize": Resize,
        "inpaint_rows_always_apply": InpaintRows,
        "inpaint_rows": partial(rand_apply, which=InpaintRows),
        "random_horiz_flip": partial(RandomHorizontalFlip, p=rand_prob),
        "random_vert_flip": partial(RandomVerticalFlip, p=rand_prob),
        "gaussian_noise": partial(rand_apply, which=GaussianNoise),
        "color_jitter": partial(rand_apply, which=ColorJitter),
        "random_autocontrast": partial(RandomAutocontrast, p=rand_prob),
        "random_solarize": partial(RandomSolarize, p=rand_prob),
        "random_sharpness": partial(RandomAdjustSharpness, p=rand_prob),
        "drop_color": partial(rand_apply, which=Grayscale),
        "gaussian_blur": partial(rand_apply, GaussianBlur),
        "random_erasing": partial(RandomErasing, p=rand_prob),
        "random_affine": partial(rand_apply, RandomAffine),
        "random_resized_crop": partial(rand_apply, RandomResizedCrop)
    }

    return [callable_dict[a["which"]](**a["params"]) for a in augs]


def get_srh_aug_list(augs, base_aug: str = "three_channels", rand_prob=0.5) -> List:
    """Combine base and strong augmentations for training"""
    return get_srh_base_aug(base_aug=base_aug) + get_strong_aug(augs, rand_prob)


def get_transformations(
        cf: Optional[Dict] = None,
        strong_aug: Callable = get_strong_aug) -> Tuple[Compose, Compose]:

    if cf:
        train_augs = cf["data"]["train_augmentation"]
        val_augs = cf["data"]["valid_augmentation"]
        base_aug = cf["data"]["srh_base_augmentation"]
        aug_prob = cf["data"]["rand_aug_prob"]
    else:
        train_augs = []
        val_augs = []
        base_aug = "three_channels"
        aug_prob = 0

    if val_augs == "same":
        val_augs = train_augs

    train_xform = Compose(get_srh_aug_list(train_augs, base_aug=base_aug, rand_prob=aug_prob))
    valid_xform = Compose(get_srh_aug_list(val_augs, base_aug=base_aug, rand_prob=aug_prob))

    return train_xform, valid_xform