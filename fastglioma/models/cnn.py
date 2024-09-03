"""Model wrappers.

Copyright (c) 2024 University of Michigan. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

from typing import Dict, List
from itertools import chain

import torch
from torch import nn as nn


class MLP(nn.Module):
    """MLP for classification head.

    Forward pass returns a tensor.
    """

    def __init__(self, n_in: int, hidden_layers: List[int],
                 n_out: int) -> None:
        super().__init__()
        layers_in = [n_in] + hidden_layers
        layers_out = hidden_layers + [n_out]

        layers_list = list(
            chain.from_iterable((nn.Linear(a, b), nn.ReLU())
                                for a, b in zip(layers_in, layers_out)))[:-1]
        self.layers = nn.Sequential(*layers_list)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.layers.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ContrastiveLearningNetwork(torch.nn.Module):
    """A network consists of a backbone and projection head.

    Forward pass returns the normalized embeddings after a projection layer.
    """

    def __init__(self, backbone: callable, proj: callable):
        super(ContrastiveLearningNetwork, self).__init__()
        self.bb = backbone()
        self.proj = proj()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bb_out = self.bb(x)
        #bb_out_norm = torch.nn.functional.normalize(bb_out, p=2.0, dim=1)
        proj_out = self.proj(bb_out)
        proj_out_norm = torch.nn.functional.normalize(proj_out, p=2.0, dim=1)

        return proj_out_norm.unsqueeze(1)


class VICRegNetwork(torch.nn.Module):
    """A network consists of a backbone and projection head.

    Forward pass returns the normalized embeddings after a projection layer.
    """

    def __init__(self, backbone: callable, proj: callable):
        super(VICRegNetwork, self).__init__()
        self.bb = backbone()
        self.proj = proj()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.proj(self.bb(x, **kwargs))