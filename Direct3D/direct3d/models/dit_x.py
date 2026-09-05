# Modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_2d.py

from typing import Optional
import os
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.models.embeddings import PatchEmbed, PixArtAlphaTextProjection, get_2d_sincos_pos_embed_from_grid
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.attention import FeedForward


class ClassCombinedTimestepSizeEmbeddings(nn.Module):

    def __init__(self, embedding_dim, class_emb_dim):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, 
                                                   time_embed_dim=embedding_dim,
                                                   cond_proj_dim=class_emb_dim)

    def forward(self, timestep, hidden_dtype, class_embedding=None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype), 
                                               condition=class_embedding)  # (N, D)
        return timesteps_emb
    

class AdaLayerNormClassEmb(nn.Module):

    def __init__(self, embedding_dim: int, class_emb_dim: int):
        super().__init__()

        self.emb = ClassCombinedTimestepSizeEmbeddings(
            embedding_dim, class_emb_dim
        )
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)

    def forward(
        self,
        timestep: torch.Tensor,
        class_embedding: torch.Tensor = None,
        hidden_dtype: Optional[torch.dtype] = None,
    ):
        embedded_timestep = self.emb(timestep, 
                                     class_embedding=class_embedding,
                                     hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep
    

def get_2d_sincos_pos_embed(
    embed_dim, grid_size, cls_token=False, extra_tokens=0, interpolation_scale=1.0, base_size=16
):
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)
    
    if isinstance(base_size, int):
        base_size = (base_size, base_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size[0]) / interpolation_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size[1]) / interpolation_scale
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


class PatchEmbed(nn.Module):

    def __init__(
        self,
        height=224,
        width=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
        interpolation_scale=1,
    ):
        super().__init__()

        self.flatten = flatten
        self.layer_norm = layer_norm

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
        )
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None

        self.patch_size = patch_size
        # See:
        # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L161
        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = height // patch_size
        self.interpolation_scale = interpolation_scale
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim, (self.height, self.width), base_size=(self.height, self.width), interpolation_scale=self.interpolation_scale
        )
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=False)
    
    def get_pos_embed(self, height: int, width: int, device, dtype):
        # If shape matches buffer, reuse it; otherwise recompute like in forward()
        if (self.height, self.width) == (height, width):
            pe = self.pos_embed  # (1, N, D)
        else:
            pe_np = get_2d_sincos_pos_embed(
                embed_dim=self.pos_embed.shape[-1],
                grid_size=(height, width),
                base_size=(height, width),
                interpolation_scale=self.interpolation_scale,
            )
            pe = torch.from_numpy(pe_np).float().unsqueeze(0)
        return pe.to(device=device, dtype=dtype)  # (1, N, D)

    def forward(self, latent):
        height, width = latent.shape[-2] // self.patch_size, latent.shape[-1] // self.patch_size

        latent = self.proj(latent)
        if self.flatten:
            latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC
        if self.layer_norm:
            latent = self.norm(latent)

        # Interpolate positional embeddings if needed.
        # (For PixArt-Alpha: https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L162C151-L162C160)
        if self.height != height or self.width != width:
            pos_embed = get_2d_sincos_pos_embed(
                embed_dim=self.pos_embed.shape[-1],
                grid_size=(height, width),
                base_size=(height, width),
                interpolation_scale=self.interpolation_scale,
            )
            pos_embed = torch.from_numpy(pos_embed)
            pos_embed = pos_embed.float().unsqueeze(0).to(latent.device)
        else:
            pos_embed = self.pos_embed

        return (latent + pos_embed).to(latent.dtype)


class Attention(nn.Module):

    def __init__(
        self,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        out_bias: bool = True,
    ):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.use_bias = bias
        self.dropout = dropout
        self.heads = heads

        self.to_q = nn.Linear(self.inner_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(self.inner_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(self.inner_dim, self.inner_dim, bias=bias)

        self.to_out = nn.ModuleList([
            nn.Linear(self.inner_dim, self.inner_dim, bias=out_bias),
            nn.Dropout(dropout)
        ])
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ):

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = hidden_states.shape[0] if encoder_hidden_states is None else encoder_hidden_states.shape[0]

        query = self.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states


class DiTBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            out_bias=attention_out_bias,
        )

        self.norm2 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)

        self.attn2 = Attention(
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            out_bias=attention_out_bias,
        )  

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        pixel_hidden_states: Optional[torch.FloatTensor] = None,
    ):
        batch_size = hidden_states.shape[0]

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        norm_hidden_states = norm_hidden_states.squeeze(1)
        hidden_states_len = norm_hidden_states.shape[1]
        attn_output = self.attn1(
            torch.cat([pixel_hidden_states, norm_hidden_states], dim=1),
        )[:, -hidden_states_len:]
        attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        norm_hidden_states = hidden_states

        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )
        hidden_states = attn_output + hidden_states

        norm_hidden_states = self.norm2(hidden_states)
        save_norm_hidden_states = norm_hidden_states.clone()
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states)

        ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states, save_norm_hidden_states
    

class D3D_DiT(nn.Module):

    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 72,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "gelu-approximate",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        semantic_channels: int = None,
        pixel_channels: int = None,
        interpolation_scale: float = 1.0,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        if isinstance(sample_size, int):
            sample_size = (sample_size, sample_size)
        self.height = sample_size[0]
        self.width = sample_size[1]

        self.patch_size = patch_size
        interpolation_scale = (
            interpolation_scale if interpolation_scale is not None else max(min(self.config.sample_size) // 32, 1)
        )
        self.pos_embed = PatchEmbed(
            height=sample_size[0],
            width=sample_size[1],
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            interpolation_scale=interpolation_scale,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for d in range(num_layers)
            ]
        )

        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * self.out_channels)

        self.adaln_single = AdaLayerNormClassEmb(inner_dim, semantic_channels)

        self.semantic_projection = PixArtAlphaTextProjection(in_features=semantic_channels, hidden_size=inner_dim)
        self.pixel_projection = PixArtAlphaTextProjection(in_features=pixel_channels, hidden_size=inner_dim)

        self.gradient_checkpointing = gradient_checkpointing

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        pixel_hidden_states: Optional[torch.Tensor] = None,
        name: str = "output",
    ):
        # extract_feature_from_layer = num_blocks//2

        height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
        hidden_states = self.pos_embed(hidden_states)

        pos_tokens = self.pos_embed.get_pos_embed(height, width, hidden_states.device, hidden_states.dtype)  # (1,N,D)

        timestep_id = timestep[0].ceil().item()

        timestep, embedded_timestep = self.adaln_single(
            timestep, class_embedding=encoder_hidden_states[:, 0], hidden_dtype=hidden_states.dtype
        )

        batch_size = hidden_states.shape[0]
        encoder_hidden_states = self.semantic_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])
        
        pixel_hidden_states = self.pixel_projection(pixel_hidden_states)
        pixel_hidden_states = pixel_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        repa = None
        print("timestep_id", timestep_id)

        block_id = 0

        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep,
                    pixel_hidden_states,
                )
            else:
                hidden_states, save_norm_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                    pixel_hidden_states=pixel_hidden_states,
                )
                print("hidden_states", hidden_states.shape)
                if timestep_id % 5 == 0 and block_id % 4 == 0:
                    f = save_norm_hidden_states.detach()                         # (B, N, D)
                    # f = f - pos_tokens                                           # remove pos bias (broadcast over batch)
                    U, S, Vh = torch.linalg.svd(pos_tokens[0], full_matrices=False)  # Vh: (D,D)
                    r = 16
                    B_pe = Vh[:r].T  # (D, r) top PE directions

                    # then for each f: (B,N,D)
                    f = f - torch.matmul(torch.matmul(f, B_pe), B_pe.T)  # remove PE subspace
                    f = f - f.mean(dim=-1, keepdim=True)                         # per-token centering
                    f = f / (f.std(dim=-1, keepdim=True) + 1e-6)                 # per-token whitening
                    f = torch.nn.functional.normalize(f, dim=-1)                 # L2 across channels
                    f = f.to(torch.float16).cpu()                                # compact
                    os.makedirs(f"/sensei-fs-3/users/niladrid/work/Direct3D/repa/{name}", exist_ok=True)
                    torch.save(f, f"/sensei-fs-3/users/niladrid/work/Direct3D/repa/{name}/{block_id}_{timestep_id}.pt")            
            block_id += 1

        shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.squeeze(1)

        hidden_states = hidden_states.reshape(
            shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
        )

        return output

