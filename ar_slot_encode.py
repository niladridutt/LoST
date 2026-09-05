#!/usr/bin/env python3
"""Encode Direct3D latents into LoST slots using local files only."""

from __future__ import annotations

import glob
import time
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import torch
from omegaconf import OmegaConf

from src.stage1.diffuse_slot import DiffuseSlot


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

CONFIG_PATH = "configs/tokenizer_l.yaml"
CKPT_PATH = (
    "./models/step20000/custom_checkpoint_1.pkl"
)

LATENT_DIR = "/mnt/localssd/direct3d_latents/val"
LATENT_GLOB = "*.pt"
OUTPUT_DIR = "/mnt/localssd/encoded_slots/val"
DONE_FILE: Optional[str] = None  # Defaults to OUTPUT_DIR/done_slots.txt.

DEVICE: Optional[str] = None  # Defaults to cuda when available, otherwise cpu.
BATCH_SIZE = 256
LATENT_KEY: Optional[str] = None  # Set this if inputs are dicts instead of tensors.

SKIP_EXISTING = True


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


def resolve_done_file(output_dir: Path) -> Path:
    return Path(DONE_FILE) if DONE_FILE is not None else output_dir / "done_slots.txt"


def load_done_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()

    with path.open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def append_done_ids(path: Path, ids: Iterable[str]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for item_id in ids:
            handle.write(f"{item_id}\n")


def setup_model(config_path: str, ckpt_path: str, device: torch.device) -> DiffuseSlot:
    print("Loading slot encoder...")
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
    print("Slot encoder loaded.")
    return model


def latent_id(path: Path) -> str:
    return path.stem


def output_path(output_dir: Path, path: Path) -> Path:
    return output_dir / f"{latent_id(path)}.pt"


def load_latent(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if LATENT_KEY is not None:
        payload = payload[LATENT_KEY]
    if not torch.is_tensor(payload):
        raise TypeError(f"{path} did not load to a tensor. Set LATENT_KEY if needed.")
    return payload.squeeze()


def iter_batches(items: List[Path], batch_size: int) -> Iterable[List[Path]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def find_latent_paths(latent_dir: Path, output_dir: Path, done_ids: Set[str]) -> List[Path]:
    latent_paths = sorted(Path(path) for path in glob.glob(str(latent_dir / LATENT_GLOB)))
    if not SKIP_EXISTING:
        return [path for path in latent_paths if latent_id(path) not in done_ids]

    return [
        path
        for path in latent_paths
        if latent_id(path) not in done_ids and not output_path(output_dir, path).exists()
    ]


def load_batch(paths: List[Path]) -> Tuple[torch.Tensor, List[str]]:
    latents = []
    ids = []
    for path in paths:
        try:
            latents.append(load_latent(path))
            ids.append(latent_id(path))
        except Exception as exc:
            print(f"[warn] Could not load {path}: {exc}")

    if not latents:
        raise ValueError("No valid latents in batch.")

    return torch.stack(latents, dim=0), ids


def main() -> None:
    if BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE must be at least 1.")

    latent_dir = Path(LATENT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    done_file = resolve_done_file(output_dir)
    done_file.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device()
    model = setup_model(CONFIG_PATH, CKPT_PATH, device)

    done_ids = load_done_ids(done_file)
    latent_paths = find_latent_paths(latent_dir, output_dir, done_ids)
    if not latent_paths:
        print("No new latents to encode.")
        return

    print(f"Found {len(latent_paths)} latents to encode from {latent_dir}")
    num_batches = (len(latent_paths) + BATCH_SIZE - 1) // BATCH_SIZE

    total_saved = 0
    batches = iter_batches(latent_paths, BATCH_SIZE)
    for batch_index, batch_paths in enumerate(batches, start=1):
        print(f"\n[batch {batch_index}/{num_batches}] Loading {len(batch_paths)} latents")
        try:
            latent_batch, ids = load_batch(batch_paths)
        except Exception as exc:
            print(f"[warn] Skipping batch: {exc}")
            continue

        latent_batch = latent_batch.to(device)
        print(
            f"[batch {batch_index}/{num_batches}] "
            f"Encoding shape={tuple(latent_batch.shape)}"
        )
        start_time = time.time()

        try:
            with torch.no_grad():
                slots = model.encode_slots_only(latent_batch)
        except Exception as exc:
            print(f"[error] Encoding failed for batch {batch_index}: {exc}")
            continue
        finally:
            del latent_batch

        successful_ids = []
        for item_id, slot in zip(ids, slots):
            try:
                torch.save(slot.detach().cpu(), output_dir / f"{item_id}.pt")
                successful_ids.append(item_id)
            except Exception as exc:
                print(f"[warn] Could not save {item_id}: {exc}")

        append_done_ids(done_file, successful_ids)
        total_saved += len(successful_ids)
        print(
            f"[batch {batch_index}/{num_batches}] Saved {len(successful_ids)} slots "
            f"in {time.time() - start_time:.2f}s"
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\nDone. Saved {total_saved} slot files to {output_dir}")


if __name__ == "__main__":
    main()
