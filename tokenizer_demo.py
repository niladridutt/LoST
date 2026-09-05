#!/usr/bin/env python3
"""Run LoST slot-token reconstructions for local Direct3D latent files."""

from __future__ import annotations

import glob
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
from omegaconf import OmegaConf

from src.stage1.diffuse_slot import DiffuseSlot


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

CONFIG_PATH = "configs/tokenizer_l.yaml"
CKPT_PATH = "./models/step20000/custom_checkpoint_1.pkl"
MODEL_NAME = "lost_tokenizer"

LATENT_DIR = "/mnt/localssd/direct3d_latents/val"
LATENT_GLOB = "*.pt"
OUTPUT_DIR = "./predictions"

DEVICE: Optional[str] = None  # Defaults to cuda when available, otherwise cpu.
BATCH_SIZE = 64
MAX_SAMPLES: Optional[int] = None

TOKEN_COUNTS = [1, 4, 16, 512]
CFG_SCALE = 3.0
LATENT_KEY: Optional[str] = None  # Set if latent files are dicts instead of tensors.
LATENT_SCALE = 1.0


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


def setup_model(config_path: str, ckpt_path: str, device: torch.device) -> DiffuseSlot:
    print("Loading LoST tokenizer...")
    cfg = OmegaConf.load(config_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    ckpt = {key.replace("._orig_mod", ""): value for key, value in ckpt.items()}
    model = DiffuseSlot(**cfg["trainer"]["params"]["model"]["params"])
    msg = model.load_state_dict(ckpt, strict=False)
    if msg:
        print(f"Model loading messages: {msg}")

    model = model.to(device)
    model.eval()
    model.enable_nest = True
    print("LoST tokenizer loaded.")
    return model


def find_latent_paths() -> List[Path]:
    paths = sorted(Path(path) for path in glob.glob(str(Path(LATENT_DIR) / LATENT_GLOB)))
    if MAX_SAMPLES is not None:
        paths = paths[:MAX_SAMPLES]
    return paths


def load_latent(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if LATENT_KEY is not None:
        payload = payload[LATENT_KEY]
    if not torch.is_tensor(payload):
        raise TypeError(f"{path} did not load to a tensor. Set LATENT_KEY if needed.")
    return payload.squeeze()


def iter_batches(paths: List[Path], batch_size: int) -> Iterable[List[Path]]:
    for start in range(0, len(paths), batch_size):
        yield paths[start:start + batch_size]


def load_batch(paths: List[Path]) -> Tuple[torch.Tensor, List[str]]:
    latents = []
    ids = []
    for path in paths:
        try:
            latents.append(load_latent(path))
            ids.append(path.stem)
        except Exception as exc:
            print(f"[warn] Could not load {path}: {exc}")

    if not latents:
        raise ValueError("No valid latents in batch.")

    return torch.stack(latents, dim=0), ids


def save_prediction(
    output_dir: Path,
    num_tokens: int,
    batch_start: int,
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    ids: List[str],
) -> Path:
    output_data = {
        "reconstructed_latents": reconstructed.cpu(),
        "original_latents": original.cpu(),
        "sha256": ids,
    }
    save_path = output_dir / f"{MODEL_NAME}_{num_tokens}_batch_{batch_start}.pt"
    torch.save(output_data, save_path)
    return save_path


def main() -> None:
    if BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE must be at least 1.")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = find_latent_paths()
    if not paths:
        print(f"No latent files found in {LATENT_DIR}")
        return

    device = resolve_device()
    model = setup_model(CONFIG_PATH, CKPT_PATH, device)

    num_batches = (len(paths) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Processing {len(paths)} latents in {num_batches} batch(es).")

    for batch_index, batch_paths in enumerate(iter_batches(paths, BATCH_SIZE), start=1):
        batch_start = (batch_index - 1) * BATCH_SIZE
        print(f"\n[batch {batch_index}/{num_batches}] Loading {len(batch_paths)} latents")
        try:
            original_latents, ids = load_batch(batch_paths)
        except Exception as exc:
            print(f"[warn] Skipping batch: {exc}")
            continue

        original_latents_cpu = original_latents
        model_latents = (original_latents * LATENT_SCALE).to(device)
        print(f"[batch {batch_index}/{num_batches}] shape={tuple(model_latents.shape)}")

        for num_tokens in TOKEN_COUNTS:
            start_time = time.time()
            with torch.no_grad():
                reconstructed = model(
                    (model_latents, None, None),
                    sample=True,
                    cfg=CFG_SCALE,
                    inference_with_n_slots=num_tokens,
                )

            mse = ((model_latents - reconstructed) ** 2).mean()
            save_path = save_prediction(
                output_dir,
                num_tokens,
                batch_start,
                reconstructed,
                original_latents_cpu,
                ids,
            )
            print(
                f"[tokens={num_tokens}] mse={mse.item():.6f} "
                f"saved={save_path} time={time.time() - start_time:.2f}s"
            )

            del reconstructed
            if device.type == "cuda":
                torch.cuda.empty_cache()

        del model_latents

    print("\nDone.")


if __name__ == "__main__":
    main()
