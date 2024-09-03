"""HiDisc loss module.

Copyright (c) 2024 University of Michigan. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

from typing import Tuple, Dict, Optional, Any
import torch
from torch import nn

from fastglioma.losses.supcon import SupConLoss


class HiDiscLoss(nn.Module):
    """Computes the HiDisc loss

    Input representation needs to be normalized
    """

    def __init__(self,
                 lambda_patient: Optional[float] = 1.0,
                 lambda_slide: Optional[float] = 1.0,
                 lambda_patch: Optional[float] = 1.0,
                 supcon_loss_params: Optional[Dict] = {}):
        super(HiDiscLoss, self).__init__()
        self.criterion = SupConLoss(**supcon_loss_params)
        self.lambda_patient_ = lambda_patient
        self.lambda_slide_ = lambda_slide
        self.lambda_patch_ = lambda_patch

    def forward(self, features, labels=None):
        emb_sz = features.shape[-1]
        sz_prod = lambda x: torch.prod(torch.tensor(x))
        feat_shape = features.shape

        patient_emb = features.reshape(feat_shape[0], -1, emb_sz)
        slide_emb = features.reshape(sz_prod(feat_shape[0:2]), -1, emb_sz)
        patch_emb = features.reshape(sz_prod(feat_shape[0:3]), -1, emb_sz)

        patient_loss = self.criterion(patient_emb, None)
        slide_loss = self.criterion(slide_emb, None)
        patch_loss = self.criterion(patch_emb, None)

        loss = ((self.lambda_patient_ * patient_loss) +
                (self.lambda_slide_ * slide_loss) +
                (self.lambda_patch_ * patch_loss))

        return {
            "patient_loss": patient_loss,
            "slide_loss": slide_loss,
            "patch_loss": patch_loss,
            "sum_loss": loss
        }