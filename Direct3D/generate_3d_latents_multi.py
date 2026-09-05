#!/usr/bin/env python3
"""Generate Direct3D latents and optional meshes from local images on multiple GPUs."""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import torch
from direct3d.pipeline import Direct3dPipeline


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MANIFEST_IN = "./manifest.jsonl"
LOCAL_IMG_DIR = "/mnt/localssd/flux_images"
LOCAL_LATENTS_DIR = "/mnt/localssd/direct3d/latents"
LOCAL_MESHES_DIR = "/mnt/localssd/direct3d/meshes"

MANIFEST_3D_JSONL = "./manifest_3d.jsonl"
DONE_IDS_FILE = "./done_3d.txt"

D3D_MODEL_ID = "DreamTechAI/Direct3D"
MAX_GPUS: Optional[int] = None
BATCH_SIZE = 32
WRITER_BUFFER_SIZE = BATCH_SIZE * 2
MAX_ITEMS: Optional[int] = None

REMOVE_BACKGROUND = False
MC_THRESHOLD = -1.0
GUIDANCE_SCALE = 4.0
NUM_STEPS = 50
EXTRACT_MESH = True


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_done_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()

    with path.open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def image_path_from_record(record: dict) -> Optional[Path]:
    for key in ("image_path_local", "image_path", "path"):
        value = record.get(key)
        if value:
            return Path(value)

    uid = record.get("id")
    if uid:
        return Path(LOCAL_IMG_DIR) / f"{uid}.png"
    return None


def latent_path(uid: str) -> Path:
    return Path(LOCAL_LATENTS_DIR) / f"{uid}.pt"


def mesh_path(uid: str) -> Path:
    return Path(LOCAL_MESHES_DIR) / f"{uid}.glb"


def load_manifest_items(manifest_path: Path, done_ids: Set[str]) -> List[Tuple[str, str]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    items: List[Tuple[str, str]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            uid = record.get("id")
            image_path = image_path_from_record(record)
            if not uid or image_path is None:
                continue
            if uid in done_ids:
                continue
            if latent_path(uid).exists():
                continue

            items.append((uid, str(image_path)))

    if MAX_ITEMS is not None:
        items = items[:MAX_ITEMS]
    return items


def iter_batches(items: List[Tuple[str, str]], batch_size: int) -> Iterable[List[Tuple[str, str]]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def save_outputs(uid: str, image_path: str, output: dict, index: int) -> dict:
    latents = output["latents"][index]
    out_latent_path = latent_path(uid)
    out_latent_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(latents.detach().cpu(), out_latent_path)

    out_mesh_path = None
    if EXTRACT_MESH:
        out_mesh_path = mesh_path(uid)
        out_mesh_path.parent.mkdir(parents=True, exist_ok=True)
        output["meshes"][index].export(out_mesh_path)

    return {
        "id": uid,
        "image_path_local": image_path,
        "latent_path_local": str(out_latent_path),
        "mesh_path_local": str(out_mesh_path) if out_mesh_path is not None else None,
    }


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
                if message_type == "done_id":
                    done_handle.write(f"{payload}\n")
                    done_handle.flush()
                    total_done += 1
                    continue
                if message_type == "manifest3d":
                    for record in payload:
                        manifest_handle.write(json.dumps(record) + "\n")
                    manifest_handle.flush()
                    total_manifest += len(payload)
                    print(
                        f"[writer] +{len(payload)} 3D records "
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
    torch.set_grad_enabled(False)

    print(f"[GPU {rank}] Loading Direct3D on {device}")
    pipeline = Direct3dPipeline.from_pretrained(D3D_MODEL_ID)
    pipeline.to(device)

    buffer = []
    while True:
        batch_items = task_q.get()
        if batch_items is None:
            break

        valid_items = [(uid, path) for uid, path in batch_items if Path(path).exists()]
        missing = len(batch_items) - len(valid_items)
        if missing:
            print(f"[GPU {rank}] missing {missing} image(s)")
        if not valid_items:
            continue

        uids = [uid for uid, _ in valid_items]
        image_paths = [path for _, path in valid_items]

        try:
            output = pipeline(
                image_paths,
                remove_background=REMOVE_BACKGROUND,
                mc_threshold=MC_THRESHOLD,
                guidance_scale=GUIDANCE_SCALE,
                num_inference_steps=NUM_STEPS,
                extract_mesh=EXTRACT_MESH,
            )
        except Exception as exc:
            print(f"[GPU {rank}] batch starting with id={uids[0]} failed: {exc}")
            continue

        for index, (uid, image_path) in enumerate(valid_items):
            try:
                buffer.append(save_outputs(uid, image_path, output, index))
                write_q.put(("done_id", uid))
            except Exception as exc:
                print(f"[GPU {rank}] could not save outputs for {uid}: {exc}")

            if len(buffer) >= WRITER_BUFFER_SIZE:
                write_q.put(("manifest3d", buffer))
                print(f"[GPU {rank}] sent {len(buffer)} records to writer")
                buffer = []

        torch.cuda.empty_cache()

    if buffer:
        write_q.put(("manifest3d", buffer))
        print(f"[GPU {rank}] final send {len(buffer)} records to writer")

    print(f"[GPU {rank}] done")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    if BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE must be at least 1.")

    Path(LOCAL_LATENTS_DIR).mkdir(parents=True, exist_ok=True)
    if EXTRACT_MESH:
        Path(LOCAL_MESHES_DIR).mkdir(parents=True, exist_ok=True)

    manifest_path = Path(MANIFEST_IN)
    manifest_3d_path = Path(MANIFEST_3D_JSONL)
    done_path = Path(DONE_IDS_FILE)

    done_ids = load_done_ids(done_path)
    remaining_items = load_manifest_items(manifest_path, done_ids)
    if not remaining_items:
        print("No remaining images to encode.")
        return

    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        raise RuntimeError("No CUDA devices available.")

    num_workers = available_gpus if MAX_GPUS is None else min(available_gpus, MAX_GPUS)
    if num_workers < 1:
        raise ValueError("MAX_GPUS must allow at least one worker.")

    print(f"Remaining images: {len(remaining_items)}")
    print(f"Using {num_workers} GPU worker(s), batch size {BATCH_SIZE}")
    print(f"Latents -> {LOCAL_LATENTS_DIR}")
    print(f"Meshes  -> {LOCAL_MESHES_DIR if EXTRACT_MESH else 'disabled'}")
    print(f"3D manifest -> {manifest_3d_path}")

    ctx = mp.get_context("spawn")
    task_q = ctx.Queue(maxsize=num_workers * 2)
    write_q = ctx.Queue()

    writer = ctx.Process(
        target=writer_proc,
        args=(write_q, str(manifest_3d_path), str(done_path)),
        daemon=False,
    )
    writer.start()

    workers = []
    for rank in range(num_workers):
        process = ctx.Process(target=worker, args=(rank, task_q, write_q), daemon=False)
        process.start()
        workers.append(process)

    try:
        for batch in iter_batches(remaining_items, BATCH_SIZE):
            task_q.put(batch)

        for _ in workers:
            task_q.put(None)

        for process in workers:
            process.join()
    except KeyboardInterrupt:
        print("KeyboardInterrupt: cleaning up.")
    finally:
        write_q.put(("STOP", None))
        writer.join()

    print("All workers finished.")


if __name__ == "__main__":
    main()
