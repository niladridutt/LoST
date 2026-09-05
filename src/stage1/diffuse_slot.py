import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import AutoencoderKL
from src.stage1 import vision_transformer
from src.stage1.diffusion import create_diffusion
from src.stage1.diffusion_transfomer import DiT
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from src.stage1.pretrain import load_predictor_model


class DiT_with_autoenc_cond(DiT):
    def __init__(
        self,
        *args,
        num_autoenc=32,
        autoenc_dim=4,
        use_repa=False,
        z_dim=768,
        encoder_depth=8,
        projector_dim=2048,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.autoenc_dim = autoenc_dim
        self.hidden_size = kwargs["hidden_size"]
        self.null_cond = nn.Parameter(torch.zeros(1, num_autoenc, autoenc_dim))
        torch.nn.init.normal_(self.null_cond, std=.02)
        self.autoenc_cond_embedder = nn.Linear(autoenc_dim, self.hidden_size)
        self.y_embedder = nn.Identity()
        self.cond_drop_prob = 0.1
        
        self.use_repa = False
        self._repa_hook = None
        self.encoder_depth = encoder_depth
        if self.use_repa:
            self.projector = build_mlp(self.hidden_size, projector_dim, z_dim)

    def embed_cond(self, autoenc_cond, drop_mask=None):
        # autoenc_cond: (N, K, D)
        batch_size = autoenc_cond.shape[0]
        if drop_mask is None:
            # randomly drop all conditions, for classifier-free guidance
            if self.training:
                drop_ids = (
                    torch.rand(batch_size, 1, 1, device=autoenc_cond.device)
                    < self.cond_drop_prob
                )
                autoenc_cond_drop = torch.where(drop_ids, self.null_cond, autoenc_cond)
            else:
                autoenc_cond_drop = autoenc_cond
        else:
            autoenc_cond_drop = torch.where(drop_mask[:, :, None], autoenc_cond, self.null_cond)
        return self.autoenc_cond_embedder(autoenc_cond_drop)

    def forward(self, x, t, autoenc_cond, drop_mask=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        autoenc_cond: (N, K, D) tensor of autoencoder conditions (slots)
        """
        x = (
            self.x_embedder(x) + self.pos_embed
        )  
        c = self.t_embedder(t)  
        autoenc = self.embed_cond(autoenc_cond, drop_mask)
        num_tokens = x.shape[1]
        x = torch.cat((x, autoenc), dim=1)
        for i, block in enumerate(self.blocks):
            x = block(x, c)  # (N, T, D)

        x = x[:, :num_tokens]
        x = self.final_layer(x, c) 
        x = self.unpatchify(x)  
        return x

    def forward_with_cfg(self, x, t, autoenc_cond, drop_mask, y=None, cfg_scale=1.0):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, autoenc_cond, drop_mask)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

#################################################################################
#                                   DiT Configs                                  #
#################################################################################


def DiT_with_autoenc_cond_XL_2(**kwargs):
    return DiT_with_autoenc_cond(
        depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs
    )


def DiT_with_autoenc_cond_XL_4(**kwargs):
    return DiT_with_autoenc_cond(
        depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs
    )


def DiT_with_autoenc_cond_XL_8(**kwargs):
    return DiT_with_autoenc_cond(
        depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs
    )


def DiT_with_autoenc_cond_L_2(**kwargs):
    return DiT_with_autoenc_cond(
        depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs
    )


def DiT_with_autoenc_cond_L_4(**kwargs):
    return DiT_with_autoenc_cond(
        depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs
    )


def DiT_with_autoenc_cond_L_8(**kwargs):
    return DiT_with_autoenc_cond(
        depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs
    )


def DiT_with_autoenc_cond_B_2(**kwargs):
    return DiT_with_autoenc_cond(
        depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs
    )


def DiT_with_autoenc_cond_B_4(**kwargs):
    return DiT_with_autoenc_cond(
        depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs
    )


def DiT_with_autoenc_cond_B_8(**kwargs):
    return DiT_with_autoenc_cond(
        depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs
    )


def DiT_with_autoenc_cond_S_2(**kwargs):
    return DiT_with_autoenc_cond(
        depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs
    )


def DiT_with_autoenc_cond_S_4(**kwargs):
    return DiT_with_autoenc_cond(
        depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs
    )


def DiT_with_autoenc_cond_S_8(**kwargs):
    return DiT_with_autoenc_cond(
        depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs
    )


DiT_with_autoenc_cond_models = {
    "DiT-XL-2": DiT_with_autoenc_cond_XL_2,
    "DiT-XL-4": DiT_with_autoenc_cond_XL_4,
    "DiT-XL-8": DiT_with_autoenc_cond_XL_8,
    "DiT-L-2": DiT_with_autoenc_cond_L_2,
    "DiT-L-4": DiT_with_autoenc_cond_L_4,
    "DiT-L-8": DiT_with_autoenc_cond_L_8,
    "DiT-B-2": DiT_with_autoenc_cond_B_2,
    "DiT-B-4": DiT_with_autoenc_cond_B_4,
    "DiT-B-8": DiT_with_autoenc_cond_B_8,
    "DiT-S-2": DiT_with_autoenc_cond_S_2,
    "DiT-S-4": DiT_with_autoenc_cond_S_4,
    "DiT-S-8": DiT_with_autoenc_cond_S_8,
}

class NestedSampler(nn.Module):
    def __init__(
        self,
        num_slots,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.register_buffer("arange", torch.arange(num_slots))
        max_exp = torch.floor(torch.log2(torch.tensor(float(num_slots - 1))))
        powers = 2**torch.arange(0, max_exp + 1, dtype=torch.long)
        num_slots_tensor = torch.tensor([self.num_slots], dtype=torch.long)
        final_samples = torch.cat((powers, num_slots_tensor))
        self.register_buffer("valid_samples", final_samples)

    def powers_of_2_sample(self, num):
        num_options = len(self.valid_samples)
        sample_indices = torch.randint(0, num_options, (num,))
        return self.valid_samples[sample_indices.to(self.valid_samples.device)]

    def sample(self, num):
        samples = self.powers_of_2_sample(num)
        return samples

    def forward(self, batch_size, device, inference_with_n_slots=-1):
        if self.training:
            b = self.sample(batch_size).to(device)
        else:
            if inference_with_n_slots != -1:
                b = torch.full((batch_size,), inference_with_n_slots, device=device)
            else:
                b = torch.full((batch_size,), self.num_slots, device=device)
        b = torch.clamp(b, max=self.num_slots)

        slot_mask = self.arange[None, :] < b[:, None]  # (batch_size, num_slots)
        return slot_mask, b

class DiffuseSlot(nn.Module):
    def __init__(
        self,
        encoder="vit_base_patch16",
        drop_path_rate=0.1,
        enc_img_size=(32,96),
        enc_causal=True,
        num_slots=16,
        slot_dim=256,
        norm_slots=False,
        enable_nest=False,
        enable_nest_after=-1,
        vae=None,
        dit_model="DiT-B-4",
        num_sampling_steps="ddim25",
        use_repa=False,
        repa_encoder_depth=4,
        repa_loss_weight=1.0,
        **kwargs,
    ):
        super().__init__()

        self.use_repa = use_repa
        self.repa_loss_weight = 1
        if use_repa:
            self.repa = load_predictor_model()

        self.diffusion = create_diffusion(timestep_respacing="")
        self.gen_diffusion = create_diffusion(timestep_respacing=num_sampling_steps)
        self.dit_input_size = enc_img_size #enc_img_size // 8 if not "mar" in vae else enc_img_size // 16
        self.dit_in_channels = 16 #4 if not "mar" in vae else 16
        self.dit = DiT_with_autoenc_cond_models[dit_model](
            input_size=self.dit_input_size,
            in_channels=self.dit_in_channels,
            num_autoenc=num_slots,
            autoenc_dim=slot_dim,
            use_repa=use_repa,
            encoder_depth=repa_encoder_depth,
            z_dim=768,
        )
        self.enc_img_size = enc_img_size
        self.enc_causal = enc_causal
        encoder_fn = vision_transformer.__dict__[encoder]

        self.encoder = encoder_fn(
            img_size=enc_img_size,
            num_slots=num_slots,
            drop_path_rate=drop_path_rate,
        )
        self.num_slots = num_slots
        self.norm_slots = norm_slots
        self.num_channels = self.encoder.num_features
        
        self.encoder2slot = nn.Linear(self.num_channels, slot_dim)
        self.nested_sampler = NestedSampler(num_slots)
        self.enable_nest = enable_nest
        self.enable_nest_after = enable_nest_after


    @torch.no_grad()
    def vae_encode(self, x):
        x = x * 2 - 1
        x = self.vae.encode(x)
        if hasattr(x, 'latent_dist'):
            x = x.latent_dist
        return x.sample().mul_(self.scaling_factor)

    @torch.no_grad()
    def vae_decode(self, z):
        z = self.vae.decode(z / self.scaling_factor)
        if hasattr(z, 'sample'):
            z = z.sample
        return (z + 1) / 2

    def encode_slots(self, x):
        slots = self.encoder(x, is_causal=self.enc_causal)
        slots = self.encoder2slot(slots)
        if self.norm_slots:
            slots_std = torch.std(slots, dim=-1, keepdim=True)
            slots_mean = torch.mean(slots, dim=-1, keepdim=True)
            slots = (slots - slots_mean) / slots_std
        return slots, None
    
    def forward_with_latents(
        self,
        x_vae,
        slots,
        z,
        sample=False,
        epoch=None,
        inference_with_n_slots=-1, 
        cfg=1.0,
        first_slot=None,
    ):
        losses = {}
        batch_size = x_vae.shape[0]
        device = x_vae.device
        
        if (
            epoch is not None
            and epoch >= self.enable_nest_after
            and self.enable_nest_after != -1
        ):
            self.enable_nest = True

        t = torch.randint(0, 1000, (x_vae.shape[0],), device=device)

        if self.enable_nest or inference_with_n_slots != -1:
            drop_mask, num_active_slots = self.nested_sampler(
                batch_size, device, 
                inference_with_n_slots=inference_with_n_slots, 
            )
        else:
            drop_mask, num_active_slots = None, torch.tensor(self.num_slots, device=device).expand(batch_size)

        num_active_slots = torch.tensor(self.num_slots, device=device).expand(batch_size)
            
        if sample:
            return self.sample(slots, drop_mask=drop_mask, cfg=cfg)

        model_kwargs = dict(autoenc_cond=slots, drop_mask=drop_mask)
        loss_dict = self.diffusion.training_losses(self.dit, x_vae, t, model_kwargs)
        diff_loss = loss_dict["loss"].mean()
        losses["diff_loss"] = diff_loss
        

        if self.use_repa:
            pred = loss_dict["pred_xstart"]

            with torch.no_grad():
                gt_S_grid, gt_global_features = self.repa(x_vae)

            pred_S_grid, pred_global_features = self.repa(pred)
            pred_global_features = pred_global_features.detach()

            spatial_similarity_scores = F.cosine_similarity(pred_S_grid, gt_S_grid.detach(), dim=1) # 32, 16, 16
            losses_spatial_per_element = 1.0 - spatial_similarity_scores

            loss_sem_spatial = losses_spatial_per_element.mean()

            total_semantic_loss = loss_sem_spatial

            losses["repa_loss"] = total_semantic_loss * self.repa_loss_weight

        return losses
        

    def forward(
        self, 
        x,
        sample=False,
        epoch=None,
        inference_with_n_slots=-1,
        cfg=1.0,
    ):
        latents, image, _ = x
        z = None
        slots, first_slot = self.encode_slots(latents)
        return self.forward_with_latents(latents, slots, z, sample, epoch, inference_with_n_slots, cfg, first_slot)


    @torch.no_grad()
    def encode_slots_only(self, x):
        slots, _ = self.encode_slots(x)
        return slots

    @torch.no_grad()
    def decode_from_slots(self, slots, cfg=1.0, inference_with_n_slots=-1):
        batch_size = slots.shape[0]
        device = slots.device
        drop_mask, num_active_slots = self.nested_sampler(
            batch_size, device, 
            inference_with_n_slots=inference_with_n_slots, 
        )
        return self.sample(slots, drop_mask=drop_mask, cfg=cfg)


    @torch.no_grad()
    def sample(self, slots, drop_mask=None, cfg=1.0):
        print("Sampling with cfg scale:", cfg)
        batch_size = slots.shape[0]
        device = slots.device
        height, width = self.dit_input_size
        z = torch.randn(batch_size, self.dit_in_channels, height, width, device=device)
        if cfg != 1.0:
            z = torch.cat([z, z], 0)
            null_slots = self.dit.null_cond.expand(batch_size, -1, -1)
            slots = torch.cat([slots, null_slots], 0)
            if drop_mask is not None:
                null_cond_mask = torch.ones_like(drop_mask)
                drop_mask = torch.cat([drop_mask, null_cond_mask], 0)
            model_kwargs = dict(autoenc_cond=slots, drop_mask=drop_mask, cfg_scale=cfg)
            sample_fn = self.dit.forward_with_cfg
        else:
            model_kwargs = dict(autoenc_cond=slots, drop_mask=drop_mask)
            sample_fn = self.dit.forward
        samples = self.gen_diffusion.p_sample_loop(
            sample_fn,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=device,
        )
        if cfg != 1.0:
            samples, _ = samples.chunk(2, dim=0)  
        return samples

    def train(self, mode=True):
        """Override train() to keep certain components in eval mode"""
        super().train(mode)
        # self.vae.eval()
        return self


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )



def interpolate_features(features_to_interpolate, source_grid_size, target_grid_size):
    """
    Interpolate patch features from a source grid size to a target grid size.
    
    Args:
        features_to_interpolate: tensor of shape (B, T1, D)
        source_grid_size: tuple (H1, W1) of the source grid
        target_grid_size: tuple (H2, W2) of the target grid

    Returns:
        tensor of shape (B, T2, D) where T2 = H2 * W2
    """
    B, T1, D = features_to_interpolate.shape
    H1, W1 = source_grid_size
    H2, W2 = target_grid_size
    
    assert T1 == H1 * W1, "Source sequence length does not match source grid size."

    x = features_to_interpolate.reshape(B, H1, W1, D).permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(H2, W2), mode='bicubic', align_corners=False)
    return x.permute(0, 2, 3, 1).reshape(B, H2 * W2, D)
