"""Ordinal Metric Learning loss module.

Copyright (c) 2024 University of Michigan. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

import torch
import torch.nn as nn

import numpy as np
from typing import Optional

cat_data = lambda x: torch.cat([x[0], x[1]], dim=0)
cat_label = lambda x: torch.cat([x, x], dim=0)


def uncat_data(emb):
    half_sz = int(emb.shape[0] / 2)
    f1, f2 = torch.split(emb, [half_sz, half_sz], dim=0)
    return torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class OrdinalMetricLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR.
	https://github.com/HobbitLong/SupContrast source code here
    NOTE: Loss expects the representations have already been normalized!!
	"""

    def __init__(self, pos_label: Optional[str] = "none", **kwargs):
        super(OrdinalMetricLoss, self).__init__()\

        self.pos_label = pos_label
        self.crit = torch.nn.BCEWithLogitsLoss(reduction="none")

    @staticmethod
    def get_loss_names():
        return ["loss"]

    def forward(self, scores, labels=None):
        device = (torch.device('cuda')
                  if scores.is_cuda else torch.device('cpu'))

        scores = scores.reshape(-1, 1)

        batch_size = scores.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')

        upper_mask = torch.gt(labels, labels.T).repeat(1, 1)
        lower_mask = torch.lt(labels, labels.T).repeat(1, 1)

        logits_mask = torch.scatter(  # mask-out self-contrast cases
            torch.ones_like(upper_mask), 1,
            torch.arange(batch_size * 1).view(-1, 1).to(device), 0).float()

        diff_mat = scores.repeat(1, batch_size) - scores.repeat(1,
                                                                batch_size).T

        if self.pos_label == "none":
            neg_mask = (upper_mask | lower_mask)
            loss = (self.crit(diff_mat, upper_mask.float()) *
                    neg_mask).sum(1) / neg_mask.sum(1)
        elif self.pos_label == "same":
            mask = torch.eq(labels, labels.T).float().repeat(1, 1)
            label_mat = torch.where(upper_mask, upper_mask.float(), mask * 0.5)
            loss = (self.crit(diff_mat, label_mat) *
                    logits_mask).sum(1) / logits_mask.sum(1)
        elif self.pos_label == "lower":
            loss = (self.crit(diff_mat, upper_mask.float()) *
                    logits_mask).sum(1) / logits_mask.sum(1)
        elif self.pos_label == "upper":
            loss = (self.crit(diff_mat, (~lower_mask).float()) *
                    logits_mask).sum(1) / logits_mask.sum(1)
        elif self.pos_label == "both":
            loss = torch.div(
                (self.crit(diff_mat, upper_mask.float()) * logits_mask).sum(1)
                / logits_mask.sum(1) +
                (self.crit(diff_mat,
                           (~lower_mask).float()) * logits_mask).sum(1) /
                logits_mask.sum(1), 2.)
        else:
            raise ValueError(
                "`pos_label` must be one of {none, same, lower, upper, both}")

        loss = loss.view(1, batch_size).mean()

        return {"loss": loss}
