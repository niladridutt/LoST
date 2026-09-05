# Modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/vae.py

import trimesh
import itertools
import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat
from skimage import measure
from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from direct3d.utils.triplane import sample_from_planes, generate_planes
from diffusers.models.autoencoders.vae import UNetMidBlock2D
from diffusers.models.unets.unet_2d_blocks import UpDecoderBlock2D


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: torch.Tensor, deterministic: bool=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self):
        x = self.mean + self.std * torch.randn_like(self.mean)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.mean(torch.pow(self.mean, 2)
                                        + self.var - 1.0 - self.logvar,
                                        dim=[1, 2, 3])
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=(1, 2, 3)):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi) 
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean
    

class FourierEmbedder(nn.Module):

    def __init__(self,
                 num_freqs: int = 6,
                 input_dim: int = 3):

        super().__init__()
        freq = 2.0 ** torch.arange(num_freqs)
        self.register_buffer("freq", freq, persistent=False)
        self.num_freqs = num_freqs
        self.out_dim = input_dim * (num_freqs * 2 + 1)
        # self.out_dim = input_dim * (num_freqs * 2 + 3)

    def forward(self, x: torch.Tensor):
        embed = (x[..., None].contiguous() * self.freq).view(*x.shape[:-1], -1)
        return torch.cat((x, embed.sin(), embed.cos()), dim=-1)


class OccDecoder(nn.Module):

    def __init__(self, 
                 n_features: int,
                 hidden_dim: int = 64, 
                 num_layers: int = 4, 
                 activation: nn.Module = nn.ReLU,
                 final_activation: str = None):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(3 * n_features, hidden_dim),
            activation(),
            *itertools.chain(*[[
                nn.Linear(hidden_dim, hidden_dim),
                activation(),
            ] for _ in range(num_layers - 2)]),
            nn.Linear(hidden_dim, 1),
        )
        self.final_activation = final_activation

    def forward(self, sampled_features):

        x = rearrange(sampled_features, "N_b N_t N_s C -> N_b N_s (N_t C)")
        x = self.net(x)

        if self.final_activation is None:
            pass
        elif self.final_activation == 'tanh':
            x = torch.tanh(x)
        elif self.final_activation == 'sigmoid':
            x = torch.sigmoid(x)
        else:
            raise ValueError(f"Unknown final activation: {self.final_activation}")

        return x[..., 0]


class Attention(nn.Module):

    def __init__(self, 
                 dim: int, 
                 heads: int = 8, 
                 dim_head: int = 64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, inner_dim)
    
    def forward(self, 
                hidden_states: torch.Tensor, 
                encoder_hidden_states: Optional[torch.Tensor] = None,
    ):
        batch_size = hidden_states.shape[0]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)
        query = self.to_q(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        hidden_states = self.to_out(hidden_states)

        return hidden_states


class TransformerBlock(nn.Module):

    def __init__(self, 
                 num_attention_heads: int,
                 attention_head_dim: int,
                 cross_attention: bool = False):
        super().__init__()
        inner_dim = attention_head_dim * num_attention_heads
        self.norm1 = nn.LayerNorm(inner_dim)
        if cross_attention:
            self.norm1_c = nn.LayerNorm(inner_dim)
        else:
            self.norm1_c = None
        self.attn = Attention(inner_dim, num_attention_heads, attention_head_dim)
        self.norm2 = nn.LayerNorm(inner_dim)
        self.mlp = nn.Sequential(
            nn.Linear(inner_dim, 4 * inner_dim),
            nn.GELU(),
            nn.Linear(4 * inner_dim, inner_dim),
        )

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        if self.norm1_c is not None:
            x = self.attn(self.norm1(x), self.norm1_c(y)) + x
        else:
            x = self.attn(self.norm1(x)) + x
        x = x + self.mlp(self.norm2(x))
        return x
    

class PointEncoder(nn.Module):

    def __init__(self,
                 num_latents: int,
                 in_channels: int,
                 num_attention_heads: int,
                 attention_head_dim: int,
                 num_layers: int,
                 gradient_checkpointing: bool = False):

        super().__init__()

        self.gradient_checkpointing = gradient_checkpointing
        self.num_latents = num_latents
        inner_dim = attention_head_dim * num_attention_heads

        self.learnable_token = nn.Parameter(torch.randn((num_latents, inner_dim)) * 0.01)

        self.proj_in = nn.Linear(in_channels, inner_dim)
        self.cross_attn = TransformerBlock(num_attention_heads, attention_head_dim, cross_attention=True)

        self.self_attn = nn.ModuleList([
            TransformerBlock(num_attention_heads, attention_head_dim) for _ in range(num_layers)
        ])

        self.norm_out = nn.LayerNorm(inner_dim)

    def forward(self, pc):

        bs = pc.shape[0]
        pc = self.proj_in(pc)

        learnable_token = repeat(self.learnable_token, "m c -> b m c", b=bs)

        if self.training and self.gradient_checkpointing:
            latents = torch.utils.checkpoint.checkpoint(self.cross_attn, learnable_token, pc)
            for block in self.self_attn:
                latents = torch.utils.checkpoint.checkpoint(block, latents)
        else:
            latents = self.cross_attn(learnable_token, pc)
            for block in self.self_attn:
                latents = block(latents)

        latents = self.norm_out(latents)

        return latents
    

class TriplaneDecoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        norm_type: str = "group", 
        mid_block_add_attention=True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.up_blocks = nn.ModuleList([])

        temb_channels = in_channels if norm_type == "spatial" else None

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default" if norm_type == "group" else norm_type,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=temb_channels,
            add_attention=mid_block_add_attention,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1
            up_block = UpDecoderBlock2D(
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                resnet_time_scale_shift=norm_type,
                temb_channels=temb_channels,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if norm_type == "group":
            self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        else:
            raise ValueError(f"Unsupported norm type: {norm_type}")
        
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, sample: torch.Tensor):
        r"""The forward method of the `Decoder` class."""

        sample = self.conv_in(sample)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        if self.training and self.gradient_checkpointing:
            # middle
            sample = torch.utils.checkpoint.checkpoint(
                self.mid_block, sample
            )
            sample = sample.to(upscale_dtype)

            # up
            for up_block in self.up_blocks:
                sample = torch.utils.checkpoint.checkpoint(up_block, sample)
        else:
            # middle
            sample = self.mid_block(sample)
            sample = sample.to(upscale_dtype)

            # up
            for up_block in self.up_blocks:
                sample = up_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample
    

class D3D_VAE(nn.Module):
    def __init__(self,
                 triplane_res: int,
                 latent_dim: int = 0,
                 triplane_dim: int = 32,
                 num_freqs: int = 8,
                 num_attention_heads: int = 12,
                 attention_head_dim: int = 64,
                 num_encoder_layers: int = 8,
                 num_geodecoder_layers: int = 5,
                 final_activation: str = None,
                 block_out_channels=[128, 256, 512, 512],
                 mid_block_add_attention=True,
                 gradient_checkpointing: bool = False,
                 latents_scale: float = 1.0,
                 latents_shift: float = 0.0):

        super().__init__()

        self.gradient_checkpointing = gradient_checkpointing

        self.triplane_res = triplane_res
        self.num_latents = triplane_res ** 2 * 3
        self.latent_shape = (latent_dim, triplane_res, 3 * triplane_res)
        self.latents_scale = latents_scale
        self.latents_shift = latents_shift

        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs)

        inner_dim = attention_head_dim * num_attention_heads
        self.encoder = PointEncoder(
            num_latents=self.num_latents,
            in_channels=self.fourier_embedder.out_dim + 3,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_encoder_layers,
            gradient_checkpointing=gradient_checkpointing,
        )

        self.latent_dim = latent_dim

        self.pre_latent = nn.Conv2d(inner_dim, 2 * latent_dim, 1)
        self.post_latent = nn.Conv2d(latent_dim, inner_dim, 1)

        self.decoder = TriplaneDecoder(
            in_channels=inner_dim,
            out_channels=triplane_dim,
            block_out_channels=block_out_channels,
            mid_block_add_attention=mid_block_add_attention,
            gradient_checkpointing=gradient_checkpointing,
        )

        self.plane_axes = generate_planes()
        self.occ_decoder = OccDecoder(
            n_features=triplane_dim,
            num_layers=num_geodecoder_layers,
            final_activation=final_activation,
        )
    
    def rollout(self, triplane):
        triplane = rearrange(triplane, "N_b (N_t C) H_t W_t -> N_b C H_t (N_t W_t)", N_t=3)
        return triplane
    
    def unrollout(self, triplane):
        triplane = rearrange(triplane, "N_b C H_t (N_t W_t) -> N_b N_t C H_t W_t", N_t=3)
        return triplane
    
    def encode(self,
               pc: torch.FloatTensor,
               feats: Optional[torch.FloatTensor] = None):
        print("x shape before fourier:", pc.shape)
        x = self.fourier_embedder(pc)
        print("x shape after fourier:", x.shape)
        if feats is not None:
            x = torch.cat((x, feats), dim=-1)
        x = self.encoder(x)
        x = rearrange(x, "N_b (N_t H_t W_t) C -> N_b (N_t C) H_t W_t", 
                             N_t=3, H_t=self.triplane_res, W_t=self.triplane_res)
        x = self.rollout(x)
        moments = self.pre_latent(x)

        posterior = DiagonalGaussianDistribution(moments)
        latents = posterior.sample()

        return latents, posterior
    
    def decode(self, z, unrollout=False):
        z = self.post_latent(z)
        dec = self.decoder(z)
        if unrollout:
            dec = self.unrollout(dec)
        return dec

    def decode_mesh(self,
                    latents,
                    bounds: Union[Tuple[float], List[float], float] = 1.0,
                    voxel_resolution: int = 512,
                    mc_threshold: float = 0.0):
        triplane = self.decode(latents, unrollout=True)
        mesh = self.triplane2mesh(triplane, 
                                     bounds=bounds, 
                                     voxel_resolution=voxel_resolution, 
                                     mc_threshold=mc_threshold)
        return mesh

    def triplane2mesh(self,
                    latents: torch.FloatTensor,
                    bounds: Union[Tuple[float], List[float], float] = 1.0,
                    voxel_resolution: int = 512,
                    mc_threshold: float = 0.0,
                    chunk_size: int = 50000):

        batch_size = len(latents)

        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
        bbox_min, bbox_max = np.array(bounds[0:3]), np.array(bounds[3:6])
        bbox_length = bbox_max - bbox_min

        x = torch.linspace(bbox_min[0], bbox_max[0], steps=int(voxel_resolution) + 1)
        y = torch.linspace(bbox_min[1], bbox_max[1], steps=int(voxel_resolution) + 1)
        z = torch.linspace(bbox_min[2], bbox_max[2], steps=int(voxel_resolution) + 1)
        xs, ys, zs = torch.meshgrid(x, y, z, indexing='ij')
        xyz = torch.stack((xs, ys, zs), dim=-1)
        xyz = xyz.reshape(-1, 3)
        grid_size = [int(voxel_resolution) + 1, int(voxel_resolution) + 1, int(voxel_resolution) + 1]

        logits_total = []
        for start in tqdm(range(0, xyz.shape[0], chunk_size), desc="Triplane Sampling:"):
            positions = xyz[start:start + chunk_size].to(latents.device)
            positions = repeat(positions, "p d -> b p d", b=batch_size)

            triplane_features = sample_from_planes(self.plane_axes.to(latents.device), 
                                                   latents, positions, 
                                                   box_warp=2.0)
            logits = self.occ_decoder(triplane_features)
            logits_total.append(logits)
        
        logits_total = torch.cat(logits_total, dim=1).view(
            (batch_size, grid_size[0], grid_size[1], grid_size[2])).cpu().numpy()

        meshes = []
        for i in range(batch_size):
            vertices, faces, _, _ = measure.marching_cubes(
                logits_total[i],
                mc_threshold,
                method="lewiner"
            )
            vertices = vertices / grid_size * bbox_length + bbox_min
            faces = faces[:, ::-1]
            meshes.append(trimesh.Trimesh(vertices, faces))
        return meshes

