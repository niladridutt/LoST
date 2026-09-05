import math
import torch
import torch.nn as nn

from src.stage1.diffusion import create_diffusion
from src.stage1.transport import create_transport, Sampler


class DiffLoss(nn.Module):
    """Diffusion Loss"""
    def __init__(self, target_channels, z_channels, depth, width, num_sampling_steps, predict_xstart=False, use_si=False, cond_method="adaln"):
        super(DiffLoss, self).__init__()
        self.in_channels = target_channels
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels * 2 if not use_si else target_channels,  # for vlb loss
            z_channels=z_channels,
            num_res_blocks=depth,
            cond_method=cond_method,
        )
        self.use_si = use_si
        if not use_si:
            self.train_diffusion = create_diffusion(timestep_respacing="", noise_schedule="cosine", predict_xstart=predict_xstart)
            self.gen_diffusion = create_diffusion(timestep_respacing=num_sampling_steps, noise_schedule="cosine", predict_xstart=predict_xstart)
        else:
            self.transport = create_transport()
            self.sampler = Sampler(self.transport)

    def forward(self, target, z, mask=None):
        model_kwargs = dict(c=z)
        if not self.use_si:
            t = torch.randint(0, self.train_diffusion.num_timesteps, (target.shape[0],), device=target.device)
            loss_dict = self.train_diffusion.training_losses(self.net, target, t, model_kwargs)
        else:
            loss_dict = self.transport.training_losses(self.net, target, model_kwargs)
        loss = loss_dict["loss"]
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        return loss.mean()

    def sample(self, z, temperature=1.0, cfg=1.0):
        # diffusion loss sampling
        device = z.device
        if not cfg == 1.0:
            noise = torch.randn(z.shape[0] // 2, self.in_channels, device=device)
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(c=z, cfg_scale=cfg)
            sample_fn = self.net.forward_with_cfg
        else:
            noise = torch.randn(z.shape[0], self.in_channels, device=device)
            model_kwargs = dict(c=z)
            sample_fn = self.net.forward

        if not self.use_si:
            sampled_token_latent = self.gen_diffusion.p_sample_loop(
                sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=False,
                temperature=temperature, device=device
            )
        else:
            sde_sample_fn = self.sampler.sample_sde(diffusion_form="sigma", temperature=temperature)
            sampled_token_latent = sde_sample_fn(noise, sample_fn, **model_kwargs)[-1]
        if cfg != 1.0:
            sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)
        return sampled_token_latent


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block with AdaLN for timestep and optional concatenation for condition.
    """
    def __init__(
        self,
        channels,
        cond_method="adaln",
    ):
        super().__init__()
        self.channels = channels
        self.cond_method = cond_method

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )
        
        # Input dimension depends on conditioning method
        mlp_in_dim = channels * 2 if cond_method == "concat" else channels
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

    def forward(self, x, t, c=None):
        # Apply timestep embedding via AdaLN
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        
        # Concatenate condition if using concat method
        if self.cond_method == "concat" and c is not None:
            h = torch.cat([h, c], dim=-1)
        
        h = self.mlp(h)
        x = x + gate_mlp * h
        return x


class FinalLayer(nn.Module):
    """
    Final layer with AdaLN for timestep and optional concatenation for condition.
    """
    def __init__(self, model_channels, out_channels, cond_method="adaln"):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.cond_method = cond_method
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )
        
        # Output dimension depends on conditioning method
        linear_in_dim = model_channels * 2 if cond_method == "concat" else model_channels
        self.linear = nn.Linear(linear_in_dim, out_channels, bias=True)

    def forward(self, x, t, c=None):
        # Apply timestep embedding via AdaLN
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        
        # Concatenate condition if using concat method
        if self.cond_method == "concat" and c is not None:
            x = torch.cat([x, c], dim=-1)
            
        return self.linear(x)


class SimpleMLPAdaLN(nn.Module):
    """
    MLP for Diffusion Loss with AdaLN for timestep and optional concatenation for condition.
    """
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        cond_method="adaln"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.cond_method = cond_method

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)

        # Create residual blocks
        res_blocks = [ResBlock(model_channels, cond_method) for _ in range(num_res_blocks)]
        self.res_blocks = nn.ModuleList(res_blocks)
        
        self.final_layer = FinalLayer(model_channels, out_channels, cond_method=cond_method)
        self.initialize_weights()

    def initialize_weights(self):
        # Basic initialization for all linear layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        for i, block in enumerate(self.res_blocks):
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x)
        t_emb = self.time_embed(t)
        c_emb = self.cond_embed(c)

        if self.cond_method == "adaln":
            t_combined, c_for_concat = t_emb + c_emb, None
        else:  # concat
            t_combined, c_for_concat = t_emb, c_emb

        for block in self.res_blocks:
            x = block(x, t_combined, c_for_concat)
        return self.final_layer(x, t_combined, c_for_concat)

    def forward_with_cfg(self, x, t, c, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
        