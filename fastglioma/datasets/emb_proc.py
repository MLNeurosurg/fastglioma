import random
import logging
from functools import partial
from typing import List, Tuple, Dict, Optional, Any
import math

import numpy as np

import torch
from torch.nn import ModuleList

import torchvision
from torchvision.transforms import Compose, RandomApply
from torchvision.transforms import functional as F

from torchvision.transforms.transforms import _setup_angle, _check_sequence_input
from torch import Tensor


def emb_collate_fn(batch):
    return {
        "embeddings": [data['embeddings'] for data in batch],
        "label": torch.stack([data['label'] for data in batch]),
        "path": [data['path'] for data in batch],
        "coords": [data['coords'] for data in batch]
    }


def get_emb_transformations(
        cf: Optional[Dict] = None) -> Tuple[Compose, Compose]:

    if cf:
        train_dict = cf["data"]["train_augmentation"]
        valid_dict = cf["data"]["valid_augmentation"]
        aug_prob = cf["data"]["rand_aug_prob"]
    else:
        train_dict = []
        valid_dict = []
        aug_prob = 0

    if valid_dict == "same":
        valid_dict = train_dict

    rand_apply_p = lambda which, **kwargs: RandomApply(
        ModuleList([which(**kwargs)]), p=aug_prob)
    rand_apply = lambda which, p, **kwargs: RandomApply(
        ModuleList([which(**kwargs)]), p=p)

    callable_dict = {
        "k_random_region_partition_always_apply": NViewKRandomPartition,
        "one_random_cropping_always_apply": NViewOneRandomCropping,
        "random_splitting": partial(rand_apply_p, NViewRandomSplitting),
        "random_masking": partial(rand_apply_p, NViewRandomMasking),
        "random_cropping": partial(rand_apply_p, NViewRandomCropping),
    }

    def validate_aug_names(aug_cf):
        aug_names = [a["which"] for a in aug_cf]
        aug_set = set(aug_names)
        if ("random_splitting" in aug_set) or ("random_partitioning"
                                               in aug_set):
            assert (
                ("random_splitting" in aug_set) ^
                ("random_partitioning" in aug_set)
            ), "random_splitting and random_partitioning are mutually exclusive"

    validate_aug_names(train_dict)
    validate_aug_names(valid_dict)

    train_xform = Compose(
        [callable_dict[a["which"]](**a["params"]) for a in train_dict])
    valid_xform = Compose(
        [callable_dict[a["which"]](**a["params"]) for a in valid_dict])

    logging.info(f"train_xform\n{train_xform}")
    logging.info(f"valid_xform\n{valid_xform}")

    return train_xform, valid_xform


class NViewRandomSplitting(torch.nn.Module):
    """Randomly split all the tokens into N random views."""

    def __init__(self,
                 masking_ratio: List[float] = [0.7, 0.3],
                 fixed_order=False):
        super().__init__()
        assert sum(masking_ratio) == 1
        self.n_views_ = len(masking_ratio)
        self.splitting_ratio_ = torch.tensor(masking_ratio)

        self.idx_f_ = self.get_two_idx if self.n_views_ == 2 else self.get_n_idx
        self.asgmt_order_f_ = torch.arange if fixed_order else torch.randperm

    def get_n_idx(self, length):
        elt_sizes = (self.splitting_ratio_ * length).floor()
        elt_sizes[-1] = length - elt_sizes[:-1].sum()
        end = elt_sizes.to(int).cumsum(dim=0)
        start = torch.hstack((torch.tensor([0]), end[:-1]))
        all_idx = torch.randperm(length)
        return [all_idx[i:j] for i, j in zip(start, end)]

    def get_two_idx(self, length):
        sample_size = int(self.splitting_ratio_[0] * length)
        all_idx = torch.randperm(length)
        return [all_idx[:sample_size], all_idx[sample_size:]]

    def forward(self, inst: Dict[str, Any]):
        length = len(inst['embeddings'][0])
        assert len(inst['embeddings']) == self.n_views_, (
            f"length of embedding is {len(inst['embeddings'])}, " +
            f"n_views is {self.n_views_}")
        idxs = self.idx_f_(length)
        asgmt_order = self.asgmt_order_f_(self.n_views_)

        inst["embeddings"] = [
            emb[idxs[asgmt_order[i]], :]
            for i, emb in enumerate(inst["embeddings"])
        ]
        inst["coords"] = [
            emb[idxs[asgmt_order[i]], :]
            for i, emb in enumerate(inst["coords"])
        ]

        return inst


class NViewRandomPartitioning(torch.nn.Module):

    def __init__(self):
        raise NotImplementedError()

    def forward(self, inst: Dict[str, Any]):
        raise NotImplementedError()


class NViewRandomMasking(torch.nn.Module):

    def __init__(self,
                 masking_ratio_ranges: List[List[float]] = [[0.8, 1],
                                                            [0.3, 0.7]],
                 max_num_tokens: Optional[List[int]] = None,
                 fixed_order=False):
        super().__init__()
        self.n_views_ = len(masking_ratio_ranges)
        mrr = torch.tensor(masking_ratio_ranges)
        self.masking_range_ = torch.diff(mrr, dim=1).squeeze(dim=1)
        self.masking_min_ = mrr[:, 0]
        if max_num_tokens:
            self.num_max_token_ = torch.tensor(max_num_tokens)
        else:
            self.num_max_token_ = None
        print(self.num_max_token_)
        self.asgmt_order_f_ = torch.arange if fixed_order else torch.randperm

    def forward(self, inst: Dict[str, Any]):
        lengths = torch.tensor([len(e) for e in inst['embeddings']])
        assert len(inst['embeddings']) == self.n_views_, (
            f"length of embedding is {len(inst['embeddings'])}, " +
            f"n_views is {self.n_views_}")

        asgmt_ord = self.asgmt_order_f_(self.n_views_)
        sizes = ((torch.rand(self.n_views_) * self.masking_range_[asgmt_ord] +
                  self.masking_min_[asgmt_ord]) * lengths).round().to(int)

        if self.num_max_token_ is not None:
            sizes = torch.minimum(sizes, self.num_max_token_)

        idxs = [torch.randperm(l)[:s] for s, l in zip(sizes, lengths)]

        inst["embeddings"] = [
            emb[idxs[i], :] for i, emb in enumerate(inst["embeddings"])
        ]
        inst["coords"] = [
            emb[idxs[i], :] for i, emb in enumerate(inst["coords"])
        ]

        return inst


class NViewOneRandomCropping(torch.nn.Module):

    def __init__(self,
                 masking_size_ranges: List[int] = [1500, 1700],
                 masking_aspect_ratio_range: List[float] = [1, 1],
                 min_crop_area_thres=1600):
        super().__init__()
        msr = torch.tensor(masking_size_ranges)
        self.masking_area_range_ = msr[1] - msr[0]
        self.masking_area_min_ = msr[0]

        marr = torch.tensor(masking_aspect_ratio_range)
        self.aspect_range_ = marr[1] - marr[0]
        self.aspect_min_ = marr[0]

        self.min_crop_area_thres_ = min_crop_area_thres
        assert min_crop_area_thres >= masking_size_ranges[-1]

    def forward(self, inst: Dict[str, Any]):
        num_tokens = [len(e) for e in inst['embeddings']]
        assert len(set(num_tokens)) == 1
        num_tokens = num_tokens[0]
        if num_tokens <= self.min_crop_area_thres_: return inst

        coords_uncropped = inst["coords"][0]

        # attempt to exclude some edge regions for a center-ish crop
        min_r, min_c = coords_uncropped.min(dim=0).values
        max_r, max_c = coords_uncropped.max(dim=0).values
        exclude_region_side = torch.sqrt(
            torch.tensor(self.min_crop_area_thres_)) // 2

        if max_r - min_r > exclude_region_side * 2 + 1:
            max_r = max_r - exclude_region_side
            min_r = min_r + exclude_region_side
        if max_c - min_c > exclude_region_side * 2 + 1:
            max_c = max_c - exclude_region_side
            min_c = min_c + exclude_region_side
        coords_filt = coords_uncropped[
            filt_coords(coords_uncropped, min_r, max_r, min_c, max_c), :]

        if len(coords_filt) == 0:
            logging.warning(
                f"bug found when computing one crop for {inst['path']}." +
                f"coords_filt shape {coords_filt.shape}"
                f"uncropped min {coords_uncropped.min(dim=0).values}; " +
                f"uncropped max {coords_uncropped.max(dim=0).values}; " +
                f"(min_r, min_c, max_r, max_c) ({(min_r, min_c, max_r, max_c)});"
                + f"exclude_region_side {exclude_region_side}")
            coords_filt = coords_uncropped

        centroid_idx = (len(coords_filt) * torch.rand(1)).to(int)
        centroid = coords_filt[centroid_idx, :].squeeze()

        # get bbox size
        area = ((torch.rand(1) * self.masking_area_range_ +
                 self.masking_area_min_)).round().to(int)
        aspect = (torch.rand(1) * self.aspect_range_ + self.aspect_min_)
        dr = torch.sqrt(area / aspect)
        dc = (dr * aspect)
        dr = (dr / 2).round()
        dc = (dc / 2).round()

        r0, r1 = centroid[0] - dr, centroid[0] + dr
        c0, c1 = centroid[1] - dc, centroid[1] + dc

        idxs = filt_coords(coords_uncropped, r0, r1, c0, c1)

        inst["embeddings"] = [emb[idxs, :] for emb in inst["embeddings"]]
        inst["coords"] = [emb[idxs, :] for emb in inst["coords"]]

        return inst


def filt_coords(coords, min_r, max_r, min_c, max_c):
    return torch.logical_and(
        torch.logical_and(coords[:, 0] > min_r, coords[:, 0] < max_r),
        torch.logical_and(coords[:, 1] > min_c, coords[:, 1] < max_c))


class NViewRandomCropping(torch.nn.Module):

    def __init__(self,
                 masking_size_ranges: List[List[int]] = [[100, 900],
                                                         [100, 900]],
                 masking_aspect_ratio_range: List[List[float]] = [[0.3, 3],
                                                                  [0.3, 3]],
                 fixed_order=False):
        super().__init__()
        assert len(masking_size_ranges) == len(masking_aspect_ratio_range)
        self.n_views_ = len(masking_size_ranges)

        msr = torch.tensor(masking_size_ranges)
        self.masking_area_range_ = torch.diff(msr, dim=1).squeeze(dim=1)
        self.masking_area_min_ = msr[:, 0]

        marr = torch.tensor(masking_aspect_ratio_range)
        self.aspect_range_ = torch.diff(marr, dim=1).squeeze(dim=1)
        self.aspect_min_ = marr[:, 0]

        self.msr_ = masking_size_ranges
        self.marr_ = masking_aspect_ratio_range

        self.asgmt_order_f_ = torch.arange if fixed_order else torch.randperm

    def forward(self, inst: Dict[str, Any]):
        lengths = torch.tensor([len(e) for e in inst['embeddings']])
        assert len(inst['embeddings']) == self.n_views_, (
            f"length of embedding is {len(inst['embeddings'])}, " +
            f"n_views is {self.n_views_}")

        # randomly picking a centroid on the slide
        centroid_idx = (lengths * torch.rand(self.n_views_)).to(int)
        centroids = torch.stack(
            [i[j, :] for i, j in zip(inst["coords"], centroid_idx)])

        # get bbox size
        asgmt_ord = self.asgmt_order_f_(self.n_views_)
        areas = (
            (torch.rand(self.n_views_) * self.masking_area_range_[asgmt_ord] +
             self.masking_area_min_[asgmt_ord])).round().to(int)
        aspects = (torch.rand(self.n_views_) * self.aspect_range_[asgmt_ord] +
                   self.aspect_min_[asgmt_ord])
        dr = torch.sqrt(areas / aspects)
        dc = (dr * aspects)
        dr = (dr / 2).round()
        dc = (dc / 2).round()

        r_min, r_max = centroids[:, 0] - dr, centroids[:, 0] + dr
        c_min, c_max = centroids[:, 1] - dc, centroids[:, 1] + dc

        idxs = [
            filt_coords(coords, r0, r1, c0, c1)
            for (coords, r0, r1, c0,
                 c1) in zip(inst["coords"], r_min, r_max, c_min, c_max)
        ]

        inst["embeddings"] = [
            emb[idxs[i], :] for i, emb in enumerate(inst["embeddings"])
        ]
        inst["coords"] = [
            emb[idxs[i], :] for i, emb in enumerate(inst["coords"])
        ]

        return inst