#!/usr/bin/env python3
"""Generate Direct3D latents and optional meshes from local generated images."""

from __future__ import annotations

import json
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
DEVICE: Optional[str] = None  # Defaults to cuda when available, otherwise cpu.
BATCH_SIZE = 16
MAX_ITEMS: Optional[int] = None

REMOVE_BACKGROUND = False
MC_THRESHOLD = -1.0
GUIDANCE_SCALE = 4.0
NUM_STEPS = 50
EXTRACT_MESH = True


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

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


def load_manifest_items(manifest_path: Path, done_ids: Set[str]) -> List[Tuple[str, Path]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    items: List[Tuple[str, Path]] = []
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

            items.append((uid, image_path))

    if MAX_ITEMS is not None:
        items = items[:MAX_ITEMS]
    return items


def latent_path(uid: str) -> Path:
    return Path(LOCAL_LATENTS_DIR) / f"{uid}.pt"


def mesh_path(uid: str) -> Path:
    return Path(LOCAL_MESHES_DIR) / f"{uid}.glb"


def iter_batches(items: List[Tuple[str, Path]], batch_size: int) -> Iterable[List[Tuple[str, Path]]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def append_done_id(path: Path, uid: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{uid}\n")


def append_manifest(path: Path, records: List[dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def save_outputs(uid: str, image_path: Path, output: dict, index: int) -> dict:
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
        "image_path_local": str(image_path),
        "latent_path_local": str(out_latent_path),
        "mesh_path_local": str(out_mesh_path) if out_mesh_path is not None else None,
    }


def main() -> None:
    if BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE must be at least 1.")

    Path(LOCAL_LATENTS_DIR).mkdir(parents=True, exist_ok=True)
    if EXTRACT_MESH:
        Path(LOCAL_MESHES_DIR).mkdir(parents=True, exist_ok=True)

    manifest_path = Path(MANIFEST_IN)
    manifest_3d_path = Path(MANIFEST_3D_JSONL)
    done_path = Path(DONE_IDS_FILE)
    manifest_3d_path.parent.mkdir(parents=True, exist_ok=True)
    done_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = load_done_ids(done_path)
    items = load_manifest_items(manifest_path, done_ids)
    if not items:
        print("No remaining images to encode.")
        return

    device = resolve_device()
    print(f"Loading Direct3D pipeline: {D3D_MODEL_ID}")
    pipeline = Direct3dPipeline.from_pretrained(D3D_MODEL_ID)
    pipeline.to(device)

    num_batches = (len(items) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Encoding {len(items)} images in {num_batches} batch(es).")

    for batch_index, batch_items in enumerate(iter_batches(items, BATCH_SIZE), start=1):
        valid_items = [(uid, path) for uid, path in batch_items if path.exists()]
        missing = len(batch_items) - len(valid_items)
        if missing:
            print(f"[batch {batch_index}/{num_batches}] missing {missing} image(s)")
        if not valid_items:
            continue

        uids = [uid for uid, _ in valid_items]
        image_paths = [str(path) for _, path in valid_items]
        print(f"[batch {batch_index}/{num_batches}] encoding {len(valid_items)} image(s)")

        try:
            with torch.no_grad():
                output = pipeline(
                    image_paths,
                    remove_background=REMOVE_BACKGROUND,
                    mc_threshold=MC_THRESHOLD,
                    guidance_scale=GUIDANCE_SCALE,
                    num_inference_steps=NUM_STEPS,
                    extract_mesh=EXTRACT_MESH,
                )
        except Exception as exc:
            print(f"[error] Batch starting with id={uids[0]} failed: {exc}")
            continue

        records = []
        for index, (uid, image_path) in enumerate(valid_items):
            try:
                records.append(save_outputs(uid, image_path, output, index))
                append_done_id(done_path, uid)
            except Exception as exc:
                print(f"[warn] Could not save outputs for {uid}: {exc}")

        append_manifest(manifest_3d_path, records)
        print(f"[batch {batch_index}/{num_batches}] saved {len(records)} record(s)")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Done. Latents -> {LOCAL_LATENTS_DIR}; manifest -> {manifest_3d_path}")


if __name__ == "__main__":
    main()
