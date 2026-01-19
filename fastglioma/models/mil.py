"""Whole slide transformer model.

Copyright (c) 2024 University of Michigan. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from typing import List, Optional
from torch.nn.init import trunc_normal_
from fastglioma.models.vit import Block
from fastglioma.models.cnn import MLP


class FFPEG(nn.Module):
    """Fourier feature positional embedding generator module.

    References:
        Learnable Fourier Features for Multi-Dimensional Spatial Positional Encoding, Li et al. NeurIPS 2021

    Attributes:
        embed_dim: dimension of the embedding
        dim_ff: dimension of the fourier features
        dim_mlp: dimension of the mlp
        gamma: std of the normal distribution used to initialize the fourier features
        prefix_len: number of registers + cls token
        pos_emb_grad: whether to allow the fourier features to be optimized
    """
    def __init__(self,
                 embed_dim: int,
                 dim_ff: int = 96,
                 dim_mlp: int = 36,
                 gamma: float = .25,
                 prefix_len: int = 0,
                 pos_emb_grad: bool = True,
                 **kwargs):
        super(FFPEG, self).__init__()
        self.dim_ff_ = dim_ff
        self.dim_mlp_ = dim_mlp
        self.gamma_ = gamma
        self.embed_dim_ = embed_dim
        self.prefix_len = prefix_len

        self.cls_pos_emb = nn.Parameter(torch.zeros(1, self.prefix_len, embed_dim), #yapf:disable
                                        requires_grad=True)

        self.num_pos_ = 1 # G
        self.pos_dim_ = 2 # M

        self._ff_embed = nn.Linear(self.pos_dim_, dim_ff // 2, bias=False)
        torch.nn.init.normal_(self._ff_embed.weight, mean=0, std=gamma)
        if not pos_emb_grad:
            for param in self._ff_embed.parameters():
                param.requires_grad = False

        self._mlp = nn.Sequential(*[
            nn.LayerNorm(dim_ff),
            nn.Linear(dim_ff, dim_mlp),
            nn.GELU(),
            nn.LayerNorm(dim_mlp),
            nn.Linear(dim_mlp, embed_dim // self.num_pos_)
        ])

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self._mlp.apply(init_weights)

    def forward(self, H, coords, return_ff: bool = False):
        bsz, n = H.shape[0], H.shape[1]
        n = n - self.prefix_len

        x = coords.unsqueeze(0).float().unsqueeze(-2)  # NxGxM (G=1, M=2)
        x = x.to(H.device)

        ff_vec = self._ff_embed(x)  # NxGx(F/2)

        f = torch.cat([torch.cos(ff_vec), torch.sin(ff_vec)], axis=-1)
        f = 1 / np.sqrt(self.dim_ff_) * f  # NxGxF

        if return_ff: return f

        pe = self._mlp(f).reshape(bsz, n, self.embed_dim_)
        pe = torch.cat((self.cls_pos_emb.repeat(bsz, 1, 1), pe), dim=1)

        return H + pe


class MIL_forward(nn.Module):
    '''MIL module for batch forward

    Attributes:
        mil: process bag of instance embeddings (list) to produce a single bag embedding (tensor).
    '''

    def __init__(self, mil: callable):
        super().__init__()
        self.mil = mil()
        # AdditiveMIL uses mlp_out_dim instead of dim_out
        if hasattr(self.mil, 'dim_out'):
            self.num_out = self.mil.dim_out
        elif hasattr(self.mil, 'mlp_out_dim'):
            self.num_out = self.mil.mlp_out_dim
        else:
            raise AttributeError(f"MIL model {type(self.mil)} must have either 'dim_out' or 'mlp_out_dim' attribute")

    def forward_mil(self,
                    bag_embed: list,
                    return_embed: bool = False,
                    **kwargs):
        '''Forward function for bag input

        Attributes:
            bag: A batch of bags, each bag will have the various number of instances.
            return_embed: return a batch of bag embeddings.
        '''
        if 'coords' in kwargs and kwargs['coords'] is not None:
            batch_embed = torch.stack([
                self.mil(insta, coords=coords)
                for insta, coords in zip(bag_embed, kwargs['coords'])
            ]).squeeze(1)  # bsz * bagsize * emb_di -> bsz * emb_dim
        else:
            batch_embed = torch.stack([
                self.mil(insta) for insta in bag_embed
            ]).squeeze(1)  # bsz * bagsize * emb_di -> bsz * emb_dim
        return batch_embed

    def forward(self, bag, return_embed: bool = False, **kwargs):
        return self.forward_mil(bag, **kwargs)


class MIL_Classifier(nn.Module):
    '''MIL module for classification task. 

    Attributes:
        backbone: process instances (list) into instances embeddings (list).
        mil: process bag of instance embeddings (list) to produce a single bag embedding (tensor).
        head: process the bag embedding (tensor) to bag logits (tensor).
    '''

    def __init__(self, backbone: callable, mil: callable, head: callable):
        super().__init__()
        if backbone:
            self.backbone = backbone()
        if head:
            self.head = head()
        self.bb = mil()

    def forward(self, bag: list, return_embed: bool = False, **kwargs):
        ''' forward for bag input

        Attributes:
            bag: A batch of bags, each bag will have the various number of instances.
            return_embed: return a batch of bag embeddings.
        '''
        bag_embed = bag
        if hasattr(self, 'backbone'):
            bag_embed = [
                self.backbone(insta) for insta in bag
            ] # bsz * bagsize * input_dim -> bsz * bagsize * emb_dim

        batch_embed = self.bb.forward(bag_embed, coords=kwargs.get('coords', None))
        if return_embed:
            return {"embeddings": batch_embed}

        if hasattr(self, 'head'):
            batch_logits = self.head(batch_embed)  # bsz * emb_dim -> bsz
            return {"logits": batch_logits, "embeddings": batch_embed}
        
        # If no head, check if the MIL model outputs logits directly (e.g., AdditiveMIL)
        # This happens when the output dimension is 1 (binary classification logits)
        if batch_embed.shape[-1] == 1:
            # Treat as logits (e.g., AdditiveMIL outputs logits directly)
            return {"logits": batch_embed.squeeze(-1), "embeddings": batch_embed}
        
        return {"embeddings": batch_embed}


class Identity(nn.Identity):

    def __init__(self):
        super().__init__()

    def forward(self, x, **kwargs):
        return x


class TransformerMIL(torch.nn.Module):
    """Transformer module for MIL.

    Attributes:
        global_pool: global pooling method
        embed_dim: dimension of the embedding
        depth: number of layers
        num_heads: number of attention heads
        mlp_ratio: ratio of the MLP hidden dimension to the embedding dimension
        qkv_bias: whether to use bias in the QKV linear layer
        pos_emb_type: type of positional embedding
    """
    def __init__(self,
                 global_pool='token',
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 pos_emb_type=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()
        self.global_pool = global_pool
        assert self.global_pool in ['', 'avg', 'token']
        assert embed_dim % num_heads == 0, "embed_dim should be divisiable by num_heads in transformer"
        self.cls_token = nn.Parameter(torch.zeros(1, kwargs.get("prefix_len", 1), embed_dim))

        self.pos_embed = Identity()
        if pos_emb_type:
            self.pos_embed = FFPEG(seq_len=30 * 30,
                                   embed_dim=embed_dim,
                                   **kwargs)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[i],
                  norm_layer=norm_layer) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.dim_out = embed_dim

    def forward_features(self, x, **kwargs):
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x),
                          dim=1)
        x = self.pos_embed(x, **kwargs)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        return x

    def forward_attention(self, x, **kwargs):
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x),
                          dim=1)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        x = self.pos_embed(x, **kwargs)

        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                return blk(x, return_attention=True)

    @torch.inference_mode()
    def forward_attention_all_blocks(self, x, **kwargs):
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x),
                          dim=1)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        x = self.pos_embed(x, **kwargs)

        out = []
        for _, blk in enumerate(self.blocks):
            out.append(blk(x, return_attention=True))
            x = blk(x)

        return out

    def forward(self, x, **kwargs):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        x = self.forward_features(x, **kwargs)
        return x


class AdditiveMIL(nn.Module):
    """Additive MIL model that outputs logits directly via patch-level MLP.
    
    This model uses additive attention with gating and includes a per-patch MLP
    that produces logits which are then pooled using attention weights.
    """
    def __init__(
        self, 
        dim: int,
        hidden_dim: Optional[int] = None,
        W_out: int = 1,                 # number of attention heads / output channels
        mlp_hidden_dims: Optional[List[int]] = None,
        mlp_out_dim: Optional[int] = None,  # per-patch logits dim
        drop_rate: float = 0.0,
        init_std: float = 0.02,
        use_gating: bool = True,
        use_entmax: bool = False,       # for now we fallback to softmax even if True
        use_norm: bool = False,
        use_output_bias_scale: bool = True, # kept for API symmetry; unused
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()

        self.dim = dim
        self.W_out = W_out
        self.use_gating = use_gating
        self.use_entmax = use_entmax
        self.init_std = init_std

        if hidden_dim is None:
            hidden_dim = dim

        # Additive attention network (V)
        self.attention_V = nn.Linear(dim, hidden_dim, bias=True)
        self.use_gating = use_gating
        if use_gating:
            self.gating_V = nn.Linear(dim, hidden_dim, bias=True)
        else:
            self.gating_V = None

        # Final projection to W_out attention channels
        self.W = nn.Linear(hidden_dim, W_out, bias=True)

        self.dropout = nn.Dropout(p=drop_rate)
        self.use_norm = use_norm
        if use_norm:
            self.norm_attn = norm_layer(dim)
            self.norm_mlp = norm_layer(dim)
        else:
            self.norm_attn = None
            self.norm_mlp = None

        # Per-patch MLP that produces logits/features to pool
        if mlp_out_dim is None:
            mlp_out_dim = W_out
        self.mlp_out_dim = mlp_out_dim

        # For this simple "attention-per-class" layout, keep them equal
        assert mlp_out_dim == W_out, \
            f"mlp_out_dim ({mlp_out_dim}) must equal W_out ({W_out}) for this implementation."

        self.mlp = MLP(n_in=dim, n_out=mlp_out_dim, hidden_layers=mlp_hidden_dims)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # -------- Core single-bag logic --------

    def forward_features(self, H: torch.Tensor, **kwargs):
        """
        H: [N, dim] for a *single* bag.
        Returns:
            bag_logits: [1, W_out]
            A:          [W_out, N]  (attention maps per head, like ABMIL's KxN)
        """

        # Pre-norm for attention path
        if self.norm_attn is not None:
            H_attn = self.norm_attn(H)
        else:
            H_attn = H

        # Pre-norm for MLP path
        if self.norm_mlp is not None:
            H_mlp = self.norm_mlp(H)
        else:
            H_mlp = H

        # Dropout after norm
        H_attn = self.dropout(H_attn)
        H_mlp = self.dropout(H_mlp)

        # Additive + gated attention
        attn_feat = torch.tanh(self.attention_V(H_attn))        # [N, hidden_dim]
        if self.gating_V is not None:
            gate_feat = torch.sigmoid(self.gating_V(H_attn))    # [N, hidden_dim]
            score_feat = attn_feat * gate_feat
        else:
            score_feat = attn_feat

        scores = self.W(score_feat)                             # [N, W_out]

        # Per-patch logits/features to be pooled
        patch_logits = self.mlp(H_mlp)                          # [N, W_out]

        # Compute attention over *instances* for each head/class:
        # scores: [N, W_out] -> [W_out, N]
        scores_T = scores.transpose(0, 1)                       # [W_out, N]

        A = F.softmax(scores_T, dim=1)                      # [W_out, N]

        # Pool per-patch logits with per-head attention
        # A: [W_out, N], patch_logits^T: [W_out, N]
        bag_logits = (A * patch_logits.transpose(0, 1)).sum(dim=1)  # [W_out]
        bag_logits = bag_logits.unsqueeze(0)                        # [1, W_out]

        return bag_logits, A, patch_logits.transpose(0, 1)  # A is [W_out, N], like ABMIL's KxN

    def get_attention_maps(self, H: torch.Tensor, **kwargs):
        _, A, _ = self.forward_features(H, **kwargs)
        return A

    def forward(self, H: torch.Tensor, **kwargs):
        bag_logits, _, _ = self.forward_features(H, **kwargs)
        return bag_logits