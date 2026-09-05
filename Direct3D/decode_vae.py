#!/usr/bin/env python3
"""Decode Direct3D latent prediction files into GLB meshes.

The input files are expected to be ``.pt`` dictionaries containing:
  - reconstructed_latents: tensor of predicted Direct3D latents
  - sha256: list of output identifiers

Example:
    python decode_demo_release.py \
        --inputs "/path/to/predictions/2x2_semantic_spatial_new_16*.pt" \
        --output-dir semanticist3d/cvpr_step1x_test_step_20000_resume_16
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Any, Iterable, List, Sequence


DEFAULT_MODEL_ID = "DreamTechAI/Direct3D"
DEFAULT_LATENT_KEY = "reconstructed_latents"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode Direct3D latent .pt files into .glb meshes."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more .pt files or quoted glob patterns.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where decoded .glb meshes will be written.",
    )
    parser.add_argument(
        "--latent-key",
        default=DEFAULT_LATENT_KEY,
        help=f"Latent tensor key to decode. Default: {DEFAULT_LATENT_KEY}",
    )
    parser.add_argument(
        "--id-key",
        default="sha256",
        help="Identifier list key used for output filenames. Default: sha256",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Direct3D model id or local checkpoint path. Default: {DEFAULT_MODEL_ID}",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device used for decoding. Default: cuda:0 when available, else cpu.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of latents to decode per pipeline call. Default: 8",
    )
    parser.add_argument(
        "--mc-threshold",
        type=float,
        default=-1.0,
        help="Marching cubes threshold passed to Direct3D decode. Default: -1.0",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate meshes even when the output .glb already exists.",
    )
    parser.add_argument(
        "--no-fix-normals",
        action="store_true",
        help="Skip trimesh normal repair before export.",
    )
    parser.add_argument(
        "--no-remove-floaters",
        action="store_true",
        help="Skip removal of small disconnected mesh components.",
    )
    parser.add_argument(
        "--remove-degenerate-faces",
        action="store_true",
        help="Run degenerate-face cleanup before export.",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=None,
        help="If set, decimate meshes to this maximum face count.",
    )
    return parser.parse_args()


def expand_inputs(patterns: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        matches = glob.glob(os.path.expanduser(pattern))
        if matches:
            files.extend(Path(match) for match in matches)
        else:
            path = Path(os.path.expanduser(pattern))
            if path.exists():
                files.append(path)

    return sorted(set(files))


def batched_indices(total: int, batch_size: int) -> Iterable[range]:
    for start in range(0, total, batch_size):
        yield range(start, min(start + batch_size, total))


def load_latent_file(
    path: Path, latent_key: str, id_key: str, torch_module: Any
) -> tuple[Any, List[str]]:
    payload = torch_module.load(path, map_location="cpu")
    if latent_key not in payload:
        raise KeyError(f"{path} is missing latent key {latent_key!r}")
    if id_key not in payload:
        raise KeyError(f"{path} is missing id key {id_key!r}")

    latents = payload[latent_key]
    ids = list(payload[id_key])
    if len(latents) != len(ids):
        raise ValueError(
            f"{path} has {len(latents)} latents but {len(ids)} ids under {id_key!r}"
        )

    return latents, ids


def postprocess_mesh(mesh, args: argparse.Namespace):
    from mesh_utils import remove_degenerate_face, remove_floater, reduce_face

    if not args.no_fix_normals:
        mesh.fix_normals()
    if not args.no_remove_floaters:
        mesh = remove_floater(mesh)
    if args.remove_degenerate_faces:
        mesh = remove_degenerate_face(mesh)
    if args.max_faces is not None:
        mesh = reduce_face(mesh, max_facenum=args.max_faces)
    return mesh


def decode_file(
    pipeline: Any,
    latent_file: Path,
    output_dir: Path,
    device: Any,
    args: argparse.Namespace,
    torch_module: Any,
) -> tuple[int, int, int]:
    latents, mesh_ids = load_latent_file(
        latent_file, args.latent_key, args.id_key, torch_module
    )
    written = 0
    skipped = 0
    failed = 0

    print(f"[decode] {latent_file.name}: {len(mesh_ids)} latents")
    for batch in batched_indices(len(mesh_ids), args.batch_size):
        batch_ids = [mesh_ids[i] for i in batch]
        output_paths = [output_dir / f"{mesh_id}.glb" for mesh_id in batch_ids]

        pending = [
            (src_idx, mesh_id, output_path)
            for src_idx, mesh_id, output_path in zip(batch, batch_ids, output_paths)
            if args.overwrite or not output_path.exists()
        ]
        skipped += len(batch_ids) - len(pending)
        if not pending:
            continue

        source_indices = [src_idx for src_idx, _, _ in pending]
        batch_latents = latents[source_indices].to(device)

        try:
            with torch_module.no_grad():
                output = pipeline.decode(batch_latents, mc_threshold=args.mc_threshold)
        except Exception as exc:
            failed += len(pending)
            print(f"[error] failed to decode batch from {latent_file.name}: {exc}")
            continue
        finally:
            del batch_latents

        for mesh, (_, mesh_id, output_path) in zip(output["meshes"], pending):
            try:
                mesh = postprocess_mesh(mesh, args)
                mesh.export(output_path)
                written += 1
            except Exception as exc:
                failed += 1
                print(f"[error] failed to save {mesh_id}: {exc}")

        if torch_module.cuda.is_available() and device.type == "cuda":
            torch_module.cuda.empty_cache()

    return written, skipped, failed


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    import torch
    from direct3d.pipeline import Direct3dPipeline

    latent_files = expand_inputs(args.inputs)
    if not latent_files:
        raise FileNotFoundError(f"No input files matched: {args.inputs}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device_name = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    print(f"[setup] loading {args.model_id} on {device}")
    pipeline = Direct3dPipeline.from_pretrained(args.model_id)
    pipeline.to(device)

    total_written = 0
    total_skipped = 0
    total_failed = 0
    for latent_file in latent_files:
        written, skipped, failed = decode_file(
            pipeline=pipeline,
            latent_file=latent_file,
            output_dir=output_dir,
            device=device,
            args=args,
            torch_module=torch,
        )
        total_written += written
        total_skipped += skipped
        total_failed += failed
        print(
            f"[done] {latent_file.name}: wrote={written} skipped={skipped} failed={failed}"
        )

    print(
        f"[summary] files={len(latent_files)} wrote={total_written} "
        f"skipped={total_skipped} failed={total_failed} output={output_dir}"
    )


if __name__ == "__main__":
    main()
