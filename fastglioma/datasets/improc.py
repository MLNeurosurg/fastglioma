"""Image processing functions designed to work with OpenSRH datasets.

Copyright (c) 2024 University of Michigan. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

from typing import Optional, List, Tuple, Dict
from functools import partial

import random
import tifffile
import numpy as np

import torch
from torch.nn import ModuleList
from torchvision.transforms import (
    Normalize, RandomApply, Compose, RandomHorizontalFlip, RandomVerticalFlip,
    Resize, RandAugment, RandomErasing, RandomAutocontrast, Grayscale,
    RandomSolarize, ColorJitter, RandomAdjustSharpness, GaussianBlur,
    RandomAffine, RandomResizedCrop)

# TODO: add FastSRH skipping line aug
class GetThirdChannel(torch.nn.Module):
    """Computes the third channel of SRH image

    Compute the third channel of SRH images by subtracting CH3 and CH2. The
    channel difference is added to the subtracted_base.

    """

    def __init__(self, subtracted_base: float = 5000 / 65536.0):
        super().__init__()
        self.subtracted_base = subtracted_base

    def __call__(self, two_channel_image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            two_channel_image: a 2 channel np array in the shape H * W * 2
            subtracted_base: an integer to be added to (CH3 - CH2)

        Returns:
            A 3 channel np array in the shape H * W * 3
        """
        ch2 = two_channel_image[0, :, :]
        ch3 = two_channel_image[1, :, :]
        ch1 = ch3 - ch2 + self.subtracted_base

        return torch.stack((ch1, ch2, ch3), dim=0)


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


def get_srh_base_aug() -> List:
    """Base processing augmentations for all SRH images"""
    u16_min = (0, 0)
    u16_max = (65536, 65536)  # 2^16
    return [Normalize(u16_min, u16_max), GetThirdChannel(), MinMaxChop()]


def get_srh_vit_base_aug() -> List:
    """Base processing augmentations for all SRH images, with resize to 224"""
    u16_min = (0, 0)
    u16_max = (65536, 65536)  # 2^16
    return [
        Normalize(u16_min, u16_max),
        GetThirdChannel(),
        MinMaxChop(),
        Resize((224, 224))
    ]


def get_strong_aug(augs, rand_prob) -> List:
    """Strong augmentations for OpenSRH training"""
    rand_apply = lambda which, **kwargs: RandomApply(
        ModuleList([which(**kwargs)]), p=rand_prob)

    callable_dict = {
        "resize": Resize,
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


def get_srh_aug_list(augs, rand_prob=0.5) -> List:
    """Combine base and strong augmentations for OpenSRH training"""
    return get_srh_base_aug() + get_strong_aug(augs, rand_prob)