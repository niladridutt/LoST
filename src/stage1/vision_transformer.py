# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
import torch
import torch.nn as nn
from diffusers.models.embeddings import get_2d_sincos_pos_embed_from_grid
from functools import partial
from src.stage1.fused_attention import Attention
import numpy as np


__all__ = ['VisionTransformer', 'vit_tiny_patch16', 'vit_small_patch16',
           'vit_base_patch16', 'vit_large_patch16', 'vit_huge_patch14']


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


def get_2d_sincos_pos_embed(
    embed_dim, grid_size, cls_token=False, extra_tokens=0, interpolation_scale=1.0, base_size=16
):
    """
    Create 2D sin/cos positional embeddings.
    """
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)
    
    if isinstance(base_size, int):
        base_size = (base_size, base_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size[0]) / interpolation_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size[1]) / interpolation_scale
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])

    grid_tensor = torch.from_numpy(grid)
    
    pos_embed_pt = get_2d_sincos_pos_embed_from_grid(embed_dim, grid_tensor, output_type="pt")
    
    pos_embed = pos_embed_pt.numpy()

    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, init_values=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, attn_mask=None):
        y = self.attn(self.norm1(x), attn_mask=attn_mask)
        if self.gamma_1 is None:
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(32,96), patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}x{W}) doesn't match model ({self.img_size[0]}x{self.img_size[1]})."
        return self.proj(x)


class VisionTransformer(nn.Module):
    """ Vision Transformer """

    def __init__(self, img_size=[32,96], patch_size=16, in_chans=16, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 init_values=0, num_slots=16):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.num_slots = num_slots

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.slot_embed = nn.Parameter(torch.zeros(1, num_slots, embed_dim))

        grid_size = self.patch_embed.grid_size
        patch_pos_embed_np = get_2d_sincos_pos_embed(
            embed_dim, grid_size, base_size=grid_size
        )
        patch_pos_embed = torch.from_numpy(patch_pos_embed_np).float().unsqueeze(0)
        self.register_buffer("patch_pos_embed", patch_pos_embed, persistent=False)
        
        self.abstract_pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_slots, embed_dim))


        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)

        nn.init.trunc_normal_(self.abstract_pos_embed, std=.02)

        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.trunc_normal_(self.slot_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1 - self.num_slots
        
        h0_orig, w0_orig = self.patch_embed.grid_size
        N_orig = h0_orig * w0_orig

        h0 = h // self.patch_embed.patch_size[0]
        w0 = w // self.patch_embed.patch_size[1]
        
        if npatch == N_orig and w == self.patch_embed.img_size[1] and h == self.patch_embed.img_size[0]:
            patch_pos_embed = self.patch_pos_embed
        else:
            patch_pos_embed_2d = self.patch_pos_embed.reshape(1, h0_orig, w0_orig, self.embed_dim).permute(0, 3, 1, 2)
            
            patch_pos_embed_2d = nn.functional.interpolate(
                patch_pos_embed_2d,
                size=(h0, w0),
                mode='bicubic',
                align_corners=False,
            )
            patch_pos_embed = patch_pos_embed_2d.permute(0, 2, 3, 1).view(1, -1, self.embed_dim)
        
        cls_pos_embed = self.abstract_pos_embed[:, :1]
        slots_pos_embed = self.abstract_pos_embed[:, 1:]
        
        return torch.cat((cls_pos_embed, patch_pos_embed, slots_pos_embed), dim=1)
        

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x, self.slot_embed.expand(B, -1, -1)), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        return self.pos_drop(x)

    def forward(self, x, is_causal=True):
        x = self.prepare_tokens(x)
        if is_causal:
            attn_mask = torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool)
            causal_mask = torch.ones(self.num_slots, self.num_slots, device=x.device, dtype=torch.bool).tril(diagonal=0)
            attn_mask[-self.num_slots:, -self.num_slots:] = causal_mask
            attn_mask[:-self.num_slots, -self.num_slots:] = False
        else:
            attn_mask = None

        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)

        x = self.norm(x)
        outcome = x[:, -self.num_slots:] 
        return outcome

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def vit_tiny_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=2, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=2, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=2, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=2, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=2, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
