#!/usr/bin/env python3
"""Generate LoST slot sequences from local images with the autoregressive model."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.stage2.generate import generate
from src.stage2.gpt import GPT_models
from src.utils.datasets import InferenceDataset


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

CONFIG_PATH = "configs/autoregressive_l.yaml"
CKPT_PATH = "./models/ar_step108000/custom_checkpoint_1.pkl"

IMAGE_DIR = "/mnt/localssd/test_flux_images"
OUTPUT_FILE = "./predictions/autoregressive.pt"

DEVICE: Optional[str] = None  # Defaults to cuda when available, otherwise cpu.
BATCH_SIZE = 16
MAX_NEW_TOKENS = 128
CFG_SCALE = 6.0
TEMPERATURE = 1.0


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def resolve_device() -> torch.device:
    device_name = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        print(f"Using device: {device}")
    return device


def setup_model(config_path: str, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    print("Loading autoregressive model...")
    cfg = OmegaConf.load(config_path)
    params = cfg.trainer.params
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    ckpt = {key.replace("._orig_mod", ""): value for key, value in ckpt.items()}
    model = GPT_models[params.gpt_model.target](**params.gpt_model.params)
    msg = model.load_state_dict(ckpt, strict=False)
    if msg:
        print(f"Model loading messages: {msg}")

    model = model.to(device)
    model.eval()
    print("Autoregressive model loaded.")
    return model


def main() -> None:
    if BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE must be at least 1.")

    image_dir = Path(IMAGE_DIR)
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    device = resolve_device()
    model = setup_model(CONFIG_PATH, CKPT_PATH, device)

    dataset = InferenceDataset(str(image_dir))
    if len(dataset) == 0:
        print(f"No PNG images found in {image_dir}")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle=False)
    num_batches = (len(dataset) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Generating slots for {len(dataset)} images in {num_batches} batch(es).")

    generated_batches = []
    ids = []
    with torch.no_grad():
        for batch_index, (images, sha256) in enumerate(dataloader, start=1):
            print(f"[batch {batch_index}/{num_batches}] size={images.shape[0]}")
            result = generate(
                model,
                images.to(device),
                max_new_tokens=MAX_NEW_TOKENS,
                cfg_scale=CFG_SCALE,
                temperature=TEMPERATURE,
            )
            generated_batches.append(result.cpu())
            ids.extend(list(sha256))

            if device.type == "cuda":
                torch.cuda.empty_cache()

    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "reconstructed_latents": torch.cat(generated_batches, dim=0),
        "sha256": ids,
    }
    torch.save(output_data, output_path)
    print(f"Saved {len(ids)} generated samples to {output_path}")


if __name__ == "__main__":
    main()
