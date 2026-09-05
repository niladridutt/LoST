#!/usr/bin/env python3
"""Generate FLUX images from a local prompt text file on one GPU."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import torch
from diffusers import FluxPipeline


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

TEMPLATE = (
    "3D render of a {prompt}, isometric view, trending on ArtStation, "
    "Unreal Engine 5, V-Ray render, clean model, white background, "
    "hyperrealistic, high detail."
)

PROMPTS_FILE = "prompts.txt"
LOCAL_OUT_DIR = "/mnt/localssd/flux_images"
MANIFEST_JSONL = "./manifest.jsonl"
DONE_LINES_FILE = "./done_lines.txt"

MODEL_ID = "black-forest-labs/FLUX.1-dev"
DEVICE: Optional[str] = None  # Defaults to cuda when available, otherwise cpu.
BATCH_SIZE = 8
MAX_PROMPTS: Optional[int] = None

HEIGHT = 512
WIDTH = 512
GUIDANCE = 3.5
STEPS = 50
MAX_SEQ_LEN = 512


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def clean_prompt_line(line: str) -> Optional[str]:
    prompt = line.strip()
    if not prompt:
        return None

    prompt = re.sub(r"^\s*[0-9][0-9_]*\.\s*", "", prompt)
    return prompt or None


def load_done_lines(path: Path) -> Set[int]:
    if not path.exists():
        return set()

    done = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                done.add(int(line.strip()))
            except ValueError:
                continue
    return done


def load_prompt_items(path: Path, done_lines: Set[int]) -> List[Tuple[int, str]]:
    items: List[Tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for zero_index, raw in enumerate(handle):
            line_no = zero_index + 1
            prompt = clean_prompt_line(raw)
            if prompt is None or line_no in done_lines:
                continue
            items.append((line_no, prompt))

    if MAX_PROMPTS is not None:
        items = items[:MAX_PROMPTS]
    return items


def iter_batches(items: List[Tuple[int, str]], batch_size: int) -> Iterable[List[Tuple[int, str]]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def make_unique_id(prompt: str) -> str:
    raw = f"{prompt}-{time.time_ns()}-{os.getpid()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def append_done_line(path: Path, line_no: int) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{line_no}\n")


def append_manifest(path: Path, records: List[dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def resolve_device() -> torch.device:
    device_name = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        print(f"Using device: {device}")
    return device


def model_dtype(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def main() -> None:
    if BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE must be at least 1.")

    prompts_file = Path(PROMPTS_FILE)
    output_dir = Path(LOCAL_OUT_DIR)
    manifest_path = Path(MANIFEST_JSONL)
    done_path = Path(DONE_LINES_FILE)

    if not prompts_file.exists():
        raise FileNotFoundError(f"Missing prompts file: {prompts_file}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    done_path.parent.mkdir(parents=True, exist_ok=True)

    done_lines = load_done_lines(done_path)
    prompt_items = load_prompt_items(prompts_file, done_lines)
    if not prompt_items:
        print("No remaining prompts to process.")
        return

    device = resolve_device()
    print(f"Loading pipeline: {MODEL_ID}")
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=model_dtype(device))
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    num_batches = (len(prompt_items) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Generating {len(prompt_items)} prompts in {num_batches} batch(es).")

    for batch_index, batch_items in enumerate(iter_batches(prompt_items, BATCH_SIZE), start=1):
        line_numbers = [item[0] for item in batch_items]
        prompts = [item[1] for item in batch_items]
        full_prompts = [TEMPLATE.format(prompt=prompt) for prompt in prompts]

        print(f"[batch {batch_index}/{num_batches}] lines={line_numbers[0]}-{line_numbers[-1]}")
        try:
            images = pipe(
                full_prompts,
                height=HEIGHT,
                width=WIDTH,
                guidance_scale=GUIDANCE,
                num_inference_steps=STEPS,
                max_sequence_length=MAX_SEQ_LEN,
            ).images
        except Exception as exc:
            print(f"[error] Batch starting at line {line_numbers[0]} failed: {exc}")
            continue

        records = []
        for line_no, prompt, image in zip(line_numbers, prompts, images):
            uid = make_unique_id(prompt)
            filename = f"{uid}.png"
            local_path = output_dir / filename
            image.save(local_path)

            append_done_line(done_path, line_no)
            records.append(
                {
                    "line": line_no,
                    "id": uid,
                    "prompt": prompt,
                    "path": str(local_path),
                }
            )

        append_manifest(manifest_path, records)
        print(f"[batch {batch_index}/{num_batches}] saved {len(records)} image(s)")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Done. Images -> {output_dir}; manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
