#!/usr/bin/env python3
"""Generate FLUX images from a local prompt text file across all visible GPUs."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
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
MAX_GPUS: Optional[int] = None
BATCH_SIZE = 32
WRITER_BUFFER_SIZE = BATCH_SIZE * 2
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


# -----------------------------------------------------------------------------
# Writer
# -----------------------------------------------------------------------------

def writer_proc(write_q: mp.Queue, manifest_path: str, done_file: str) -> None:
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(done_file).parent.mkdir(parents=True, exist_ok=True)

    total_manifest = 0
    total_done = 0
    with open(manifest_path, "a", encoding="utf-8") as manifest_handle:
        with open(done_file, "a", encoding="utf-8") as done_handle:
            while True:
                message = write_q.get()
                if message is None:
                    break

                message_type, payload = message
                if message_type == "STOP":
                    break
                if message_type == "done_line":
                    done_handle.write(f"{int(payload)}\n")
                    done_handle.flush()
                    total_done += 1
                    continue
                if message_type == "manifest":
                    for record in payload:
                        manifest_handle.write(json.dumps(record) + "\n")
                    manifest_handle.flush()
                    total_manifest += len(payload)
                    print(
                        f"[writer] +{len(payload)} manifest records "
                        f"(total {total_manifest}); done={total_done}"
                    )

    print(f"[writer] stopped. manifest={total_manifest}; done={total_done}")


# -----------------------------------------------------------------------------
# Worker
# -----------------------------------------------------------------------------

def worker(rank: int, task_q: mp.Queue, write_q: mp.Queue) -> None:
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"[GPU {rank}] Loading pipeline on {device}")
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    output_dir = Path(LOCAL_OUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    buffer = []

    while True:
        batch_items = task_q.get()
        if batch_items is None:
            break

        line_numbers = [item[0] for item in batch_items]
        prompts = [item[1] for item in batch_items]
        full_prompts = [TEMPLATE.format(prompt=prompt) for prompt in prompts]

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
            print(f"[GPU {rank}] batch starting at line {line_numbers[0]} failed: {exc}")
            continue

        for line_no, prompt, image in zip(line_numbers, prompts, images):
            uid = make_unique_id(prompt)
            filename = f"{uid}.png"
            local_path = output_dir / filename
            image.save(local_path)

            write_q.put(("done_line", line_no))
            buffer.append(
                {
                    "line": line_no,
                    "id": uid,
                    "prompt": prompt,
                    "path": str(local_path),
                }
            )

            if len(buffer) >= WRITER_BUFFER_SIZE:
                write_q.put(("manifest", buffer))
                print(f"[GPU {rank}] sent {len(buffer)} records to writer")
                buffer = []

        torch.cuda.empty_cache()

    if buffer:
        write_q.put(("manifest", buffer))
        print(f"[GPU {rank}] final send {len(buffer)} records to writer")

    print(f"[GPU {rank}] done")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

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
    done_lines = load_done_lines(done_path)
    remaining_items = load_prompt_items(prompts_file, done_lines)
    if not remaining_items:
        print("No remaining prompts to process.")
        return

    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        raise RuntimeError("No CUDA devices available.")

    num_workers = available_gpus if MAX_GPUS is None else min(available_gpus, MAX_GPUS)
    if num_workers < 1:
        raise ValueError("MAX_GPUS must allow at least one worker.")

    print(f"Remaining prompts: {len(remaining_items)}")
    print(f"Using {num_workers} GPU worker(s), batch size {BATCH_SIZE}")
    print(f"Output -> {output_dir}")
    print(f"Manifest -> {manifest_path}")

    ctx = mp.get_context("spawn")
    task_q = ctx.Queue(maxsize=num_workers * 2)
    write_q = ctx.Queue()

    writer = ctx.Process(
        target=writer_proc,
        args=(write_q, str(manifest_path), str(done_path)),
        daemon=False,
    )
    writer.start()

    workers = []
    for rank in range(num_workers):
        proc = ctx.Process(target=worker, args=(rank, task_q, write_q), daemon=False)
        proc.start()
        workers.append(proc)

    try:
        for batch in iter_batches(remaining_items, BATCH_SIZE):
            task_q.put(batch)

        for _ in workers:
            task_q.put(None)

        for proc in workers:
            proc.join()
    except KeyboardInterrupt:
        print("KeyboardInterrupt: cleaning up.")
    finally:
        write_q.put(("STOP", None))
        writer.join()

    print("All workers finished.")


if __name__ == "__main__":
    main()
