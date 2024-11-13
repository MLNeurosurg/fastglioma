"""VICReg loss module.

Copyright (c) 2024 University of Michigan. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

import torch
from torch import nn
import torch.nn.functional as F


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class GeneralVICRegLoss(nn.Module):
    """VICReg Loss"""

    def __init__(self, embedding_dim: int, sim_coeff: float=25, std_coeff: float=25,
                 cov_coeff: float=1, epsilon: float=1.e-4):
        super(GeneralVICRegLoss, self).__init__()
        self.embedding_dim = embedding_dim
        self.sim_coeff_ = sim_coeff
        self.std_coeff_ = std_coeff
        self.cov_coeff_ = cov_coeff
        self.epsilon_ = epsilon

    @staticmethod
    def get_loss_names():
        return ["inv", "var", "cov", "loss"]

    def var_loss(self, x):
        std_x = torch.sqrt(x.var(dim=0) + self.epsilon_)
        std_x = torch.mean(F.relu(1 - std_x))
        return std_x

    def cov_loss(self, x):
        cov_x = (x.T @ x) / (x.shape[0] - 1)
        return off_diagonal(cov_x).pow_(2).sum().div(self.embedding_dim)

    def forward(self, x, _=None):  # _=None for future supervised version

        # inv loss
        n_views = x.shape[1]
        if n_views == 1:
            repr_loss = 0
        else:
            repr_loss = torch.mean(
                torch.stack([
                    F.mse_loss(x[:, i, :], x[:, j, :]) for i in range(n_views)
                    for j in range(i + 1, n_views)
                ]))

        x = x - x.mean(dim=0)

        # var, cov loss
        std_loss = torch.mean(
            torch.stack(
                [self.var_loss(x_.squeeze()) for x_ in x.split(1, dim=1)]))
        cov_loss = torch.mean(
            torch.stack(
                [self.cov_loss(x_.squeeze()) for x_ in x.split(1, dim=1)]))

        loss = (self.sim_coeff_ * repr_loss + self.std_coeff_ * std_loss +
                self.cov_coeff_ * cov_loss)

        return {
            "inv": repr_loss,
            "var": std_loss,
            "cov": cov_loss,
            "loss": loss
        }