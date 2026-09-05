# Modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py

import os
from tqdm import tqdm
from PIL import Image
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
from typing import Union, List, Optional

import torch
from direct3d.utils import instantiate_from_config, preprocess
from diffusers.utils.torch_utils import randn_tensor


class Direct3dPipeline(object):

    def __init__(self, 
                 vae, 
                 dit,
                 semantic_encoder,
                 pixel_encoder,
                 scheduler):
        self.vae = vae
        self.dit = dit
        self.semantic_encoder = semantic_encoder
        self.pixel_encoder = pixel_encoder
        self.scheduler = scheduler
    
    def to(self, device):
        self.device = torch.device(device)
        self.vae.to(device)
        self.dit.to(device)
        self.semantic_encoder.to(device)
        self.pixel_encoder.to(device)

    @classmethod
    def from_pretrained(cls, 
                        pipeline_path):
        
        if os.path.isdir(pipeline_path):
            config_path = os.path.join(pipeline_path, 'config.yaml')
            model_path = os.path.join(pipeline_path, 'model.ckpt')
        else:
            config_path = hf_hub_download(repo_id=pipeline_path, filename="config.yaml", repo_type="model")
            model_path = hf_hub_download(repo_id=pipeline_path, filename="model.ckpt", repo_type="model")
        
        cfg = OmegaConf.load(config_path)
        state_dict = torch.load(model_path, map_location='cpu')

        vae = instantiate_from_config(cfg.vae)
        vae.load_state_dict(state_dict["vae"], strict=True)
        dit = instantiate_from_config(cfg.dit)
        dit.load_state_dict(state_dict["dit"], strict=True)

        semantic_encoder = instantiate_from_config(cfg.semantic_encoder)
        pixel_encoder = instantiate_from_config(cfg.pixel_encoder)

        scheduler = instantiate_from_config(cfg.scheduler)

        return cls(
            vae=vae, 
            dit=dit, 
            semantic_encoder=semantic_encoder, 
            pixel_encoder=pixel_encoder, 
            scheduler=scheduler)

    def prepare_image(self, image: Union[str, List[str], Image.Image, List[Image.Image]], rmbg: bool = True):
        if not isinstance(image, list):
            image = [image]
        if isinstance(image[0], str):
            image = [Image.open(img) for img in image]
        image = [preprocess(img, rmbg=rmbg) for img in image]
        image = torch.stack([img for img in image]).to(self.device)
        return image
    
    def encode_image(self, image: torch.Tensor, do_classifier_free_guidance: bool = True):
        semantic_cond = self.semantic_encoder(image)
        pixel_cond = self.pixel_encoder(image)
        if do_classifier_free_guidance:
            semantic_uncond = torch.zeros_like(semantic_cond)
            pixel_uncond = torch.zeros_like(pixel_cond)
            semantic_cond = torch.cat([semantic_uncond, semantic_cond], dim=0)
            pixel_cond = torch.cat([pixel_uncond, pixel_cond], dim=0)

        return semantic_cond, pixel_cond
    
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator):
        shape = (
            batch_size,
            num_channels_latents,
            height,
            width,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents
    
    @torch.no_grad()
    def __call__(
        self,
        image: Union[str, List[str], Image.Image, List[Image.Image]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        generator: Optional[torch.Generator] = None,
        mc_threshold: float = -2.0,
        remove_background: bool = True,
        extract_mesh: bool = True,
        # name: str = "output",
        ):

        batch_size = len(image) if isinstance(image, list) else 1
        do_classifier_free_guidance = guidance_scale > 0

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        image = self.prepare_image(image, remove_background)
        semantic_cond, pixel_cond = self.encode_image(image, do_classifier_free_guidance)
        latents = self.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=self.vae.latent_shape[0],
            height=self.vae.latent_shape[1],
            width=self.vae.latent_shape[2],
            dtype=image.dtype,
            device=self.device,
            generator=generator,
        )

        extra_step_kwargs = {
            "generator": generator
        }

        for i, t in enumerate(tqdm(timesteps, desc="Diffusion Sampling:")):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # print("latent_model_input", latent_model_input.shape)

            t = t.expand(latent_model_input.shape[0])

            # print("t", t.shape)

            noise_pred = self.dit(
                hidden_states=latent_model_input,
                timestep=t,
                encoder_hidden_states=semantic_cond,
                pixel_hidden_states=pixel_cond,
                # name=name
            )

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

        # torch.save(latents, f"/sensei-fs-3/users/niladrid/work/Direct3D/latents/{name}.pt")
        # print(self.vae.latents_scale, self.vae.latents_shift)
        latents_to_save = latents.clone()
        latents = 1. / self.vae.latents_scale * latents + self.vae.latents_shift
        meshes = None
        if extract_mesh:
            meshes = self.vae.decode_mesh(latents, mc_threshold=mc_threshold)
        outputs = {"meshes": meshes, "latents": latents_to_save}
        return outputs
    
    @torch.no_grad()
    def decode(
        self,
        latents,
        mc_threshold: float = -2.0,
    ):

        batch_size = latents.shape[0]
        latents = 1. / self.vae.latents_scale * latents + self.vae.latents_shift
        meshes = self.vae.decode_mesh(latents, mc_threshold=mc_threshold)
        outputs = {"meshes": meshes}
        return outputs
        