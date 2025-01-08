"""Whole slide transformer model.

Copyright (c) 2024 University of Michigan. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from fastglioma.models.vit import Block


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
        self.num_out = self.mil.dim_out

    def forward_mil(self,
                    bag_embed: list,
                    return_embed: bool = False,
                    **kwargs):
        '''Forward function for bag input

        Attributes:
            bag: A batch of bags, each bag will have the various number of instances.
            return_embed: return a batch of bag embeddings.
        '''
        if 'coords' in kwargs:
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

        batch_embed = self.bb.forward(bag_embed, coords=kwargs['coords'])
        if return_embed:
            return {"embeddings": batch_embed}

        if hasattr(self, 'head'):
            batch_logits = self.head(batch_embed)  # bsz * emb_dim -> bsz
            return {"logits": batch_logits, "embeddings": batch_embed}
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