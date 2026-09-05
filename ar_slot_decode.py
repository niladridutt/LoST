#!/usr/bin/env python3
"""Decode LoST slot tensors back into Direct3D latents using local files only."""

from __future__ import annotations

import glob
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.stage1.diffuse_slot import DiffuseSlot


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

CONFIG_PATH = "configs/tokenizer_l.yaml"
CKPT_PATH = "./models/step20000/custom_checkpoint_1.pkl"

# Can be a single .pt file, a quoted glob, or a directory containing .pt files.
INPUT_SLOTS = "/mnt/localssd/encoded_slots/val"
OUTPUT_FILE = "/mnt/localssd/decoded_latents/val_decoded.pt"

SLOTS_KEY = "reconstructed_latents"
ID_KEY = "sha256"

DEVICE: Optional[str] = None  # Defaults to cuda when available, otherwise cpu.
BATCH_SIZE = 32
CFG_SCALE = 3.0

NUM_TOKENS: Optional[int] = None  # Set to an int to decode only the first N slots.
MAX_SAMPLES: Optional[int] = None
TARGET_SLOT_COUNT = 512


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
    print("Loading slot decoder...")
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
    print("Slot decoder loaded.")
    return model


def expand_inputs(input_slots: str) -> List[Path]:
    input_path = Path(input_slots).expanduser()
    if input_path.is_dir():
        return sorted(input_path.glob("*.pt"))

    matches = sorted(Path(path) for path in glob.glob(str(input_path)))
    if matches:
        return matches

    if input_path.exists():
        return [input_path]

    raise FileNotFoundError(f"No slot files found for {input_slots}")


def as_slot_batch(tensor: torch.Tensor, path: Path) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim == 3:
        return tensor
    raise ValueError(f"{path} has unsupported slot tensor shape {tuple(tensor.shape)}")


def load_slot_inputs(paths: List[Path]) -> Tuple[torch.Tensor, List[str], Dict[str, List]]:
    slot_batches = []
    ids: List[str] = []
    extras: Dict[str, List] = {}

    for path in paths:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            if SLOTS_KEY not in payload:
                raise KeyError(f"{path} is missing key {SLOTS_KEY!r}")

            slots = as_slot_batch(payload[SLOTS_KEY], path)
            slot_batches.append(slots)

            file_ids = payload.get(ID_KEY)
            if file_ids is None:
                file_ids = [f"{path.stem}_{index:06d}" for index in range(slots.shape[0])]
            elif isinstance(file_ids, str):
                file_ids = [file_ids]
            ids.extend(list(file_ids))

            for key, value in payload.items():
                if key in {SLOTS_KEY, ID_KEY}:
                    continue
                if isinstance(value, list):
                    extras.setdefault(key, []).extend(value)
        elif torch.is_tensor(payload):
            slots = as_slot_batch(payload, path)
            slot_batches.append(slots)
            if slots.shape[0] == 1:
                ids.append(path.stem)
            else:
                ids.extend(f"{path.stem}_{index:06d}" for index in range(slots.shape[0]))
        else:
            raise TypeError(f"{path} did not contain a tensor or dict payload.")

    slots = torch.cat(slot_batches, dim=0)
    if len(ids) != slots.shape[0]:
        raise ValueError(f"Loaded {slots.shape[0]} slot tensors but {len(ids)} ids.")

    return slots, ids, extras


def prepare_slots(slots: torch.Tensor) -> Tuple[torch.Tensor, int]:
    if NUM_TOKENS is not None:
        slots = slots[:, :NUM_TOKENS, :]

    inference_slot_count = slots.shape[1]
    if inference_slot_count > TARGET_SLOT_COUNT:
        raise ValueError(
            f"Input has {inference_slot_count} slots, but TARGET_SLOT_COUNT is "
            f"{TARGET_SLOT_COUNT}."
        )

    if inference_slot_count < TARGET_SLOT_COUNT:
        padding = TARGET_SLOT_COUNT - inference_slot_count
        slots = F.pad(slots, (0, 0, 0, padding), "constant", 0)

    return slots, inference_slot_count


def iter_tensor_batches(tensor: torch.Tensor, batch_size: int) -> Iterable[torch.Tensor]:
    for start in range(0, tensor.shape[0], batch_size):
        yield tensor[start:start + batch_size]


def decode_slots(
    model: DiffuseSlot,
    slots: torch.Tensor,
    inference_slot_count: int,
    device: torch.device,
) -> torch.Tensor:
    decoded_batches = []
    num_batches = (slots.shape[0] + BATCH_SIZE - 1) // BATCH_SIZE

    with torch.no_grad():
        batches = iter_tensor_batches(slots, BATCH_SIZE)
        for batch_index, batch_slots in enumerate(batches, start=1):
            print(
                f"[batch {batch_index}/{num_batches}] "
                f"Decoding shape={tuple(batch_slots.shape)}"
            )
            batch_slots = batch_slots.to(device)
            decoded = model.decode_from_slots(
                batch_slots,
                cfg=CFG_SCALE,
                inference_with_n_slots=inference_slot_count,
            )
            decoded_batches.append(decoded.cpu())

            del batch_slots
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return torch.cat(decoded_batches, dim=0)


def main() -> None:
    if BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE must be at least 1.")

    paths = expand_inputs(INPUT_SLOTS)
    if not paths:
        print("No slot files to decode.")
        return

    print(f"Loading {len(paths)} slot file(s).")
    slots, ids, extras = load_slot_inputs(paths)
    if MAX_SAMPLES is not None:
        slots = slots[:MAX_SAMPLES]
        ids = ids[:MAX_SAMPLES]
        extras = {key: value[:MAX_SAMPLES] for key, value in extras.items()}

    slots, inference_slot_count = prepare_slots(slots)
    print(
        f"Loaded slots shape={tuple(slots.shape)}; "
        f"decoding with {inference_slot_count} active slots."
    )

    device = resolve_device()
    model = setup_model(CONFIG_PATH, CKPT_PATH, device)

    start_time = time.time()
    decoded = decode_slots(model, slots, inference_slot_count, device)
    print(f"Decoded {decoded.shape[0]} samples in {time.time() - start_time:.2f}s")

    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_payload = {
        "reconstructed_latents": decoded,
        ID_KEY: ids,
        **extras,
    }
    torch.save(output_payload, output_path)
    print(f"Saved decoded latents to {output_path}")


if __name__ == "__main__":
    main()
