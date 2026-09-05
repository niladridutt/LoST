import argparse
from pathlib import Path
import os
import numpy as np
import torch
import trimesh
import glob
from direct3d.pipeline import Direct3dPipeline


def save_point_cloud_as_ply(points: np.ndarray, normals: np.ndarray, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    has_normals = normals is not None and normals.shape == points.shape
    header_lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_normals:
        header_lines.extend([
            "property float nx",
            "property float ny",
            "property float nz",
        ])
    header_lines.append("end_header\n")

    with out_path.open("w", encoding="ascii") as fh:
        fh.write("\n".join(header_lines))
        for idx, point in enumerate(points):
            components = [f"{coord:.6f}" for coord in point]
            if has_normals:
                components.extend(f"{val:.6f}" for val in normals[idx])
            fh.write(" ".join(components) + "\n")


def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, force="mesh")
    # if isinstance(mesh, trimesh.Scene):
    #     mesh = trimesh.util.concatenate(tuple(mesh.dump().values()))
    # if not isinstance(mesh, trimesh.Trimesh):
    #     raise ValueError(f"Could not load a triangular mesh from {mesh_path}")
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_faces()
    return mesh


def sample_point_cloud(mesh: trimesh.Trimesh, num_points: int) -> tuple[np.ndarray, np.ndarray]:
    points, face_idx = mesh.sample(num_points, return_index=True)
    normals = mesh.face_normals[face_idx]
    if not np.all(np.isfinite(normals)):
        vertex_normals = mesh.vertex_normals
        normals = vertex_normals[mesh.faces[face_idx][:, 0]]
    return points.astype(np.float32), normals.astype(np.float32)


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    center = (bbox_max + bbox_min) * 0.5
    extent = (bbox_max - bbox_min).max()
    scale = max(extent * 0.5, 1e-6)
    normalized = (points - center) / scale
    return normalized, center, scale


def main() -> None:
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input", type=Path, required=True, help="Source OBJ file.")
    parser.add_argument("--model-id", default="DreamTechAI/Direct3D", help="HF repo or local path for the pipeline.")
    parser.add_argument("--device", default=None, help="torch device string (default: cuda if available else cpu).")
    parser.add_argument("--num-points", type=int, default=81920, help="Surface samples drawn from the input mesh.")
    parser.add_argument("--voxel-resolution", type=int, default=512, help="Marching cubes grid resolution.")
    parser.add_argument("--mc-threshold", type=float, default=0.0, help="Iso-surface threshold for marching cubes.")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    
    config_path = "direct3d/config.yaml"   
    
    
    from huggingface_hub import hf_hub_download
    import os
    from omegaconf import OmegaConf
    from direct3d.utils import instantiate_from_config
    
    cfg = OmegaConf.load(config_path) 
    if os.path.isdir(args.model_id):
        model_path = os.path.join(args.model_id, 'vae.ckpt')
    else:
        model_path = hf_hub_download(repo_id=args.model_id, filename="vae.ckpt", repo_type="model")

    vae = instantiate_from_config(cfg.vae)
    state_dict = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    vae.load_state_dict(state_dict["vae"], strict=True)

    print("VAE weights loaded successfully")
    vae.to(device)
    vae.eval()
    
    files = glob.glob("/mnt/localssd/meshes/*.glb")
    for file_path in files:
        # file_name = file_path.split("/")[-2]
        file_name = os.path.basename(file_path).split(".")[0]
        print(f"Processing {file_name} ...")
        output_path = f"/mnt/localssd/test_step1x_mesh_recon/{file_name}.glb"
        output_path_latents = f"/mnt/localssd/test_step1x_mesh_latents/{file_name}.pt"
        if os.path.exists(output_path):
            print(f"Reconstruction for {file_name} already exists. Skipping...")
            continue
        # output_pc_path = f"./vae_output/{file_name[:-4]}_pc.ply"
        os.makedirs("/".join(output_path.split("/")[:-1]), exist_ok=True)
        os.makedirs("/".join(output_path_latents.split("/")[:-1]), exist_ok=True)
        mesh = load_mesh(file_path)
        points, normals = sample_point_cloud(mesh, args.num_points)
        # save_point_cloud_as_ply(points, normals, output_pc_path)
        points_norm, center, scale = normalize_points(points)

        pc = torch.from_numpy(points_norm).unsqueeze(0).to(device)
        feats = torch.from_numpy(normals).unsqueeze(0).to(device)

        with torch.no_grad():
            latents, posterior = vae.encode(pc, feats=feats)
            latents = posterior.mode()
            torch.save(latents.cpu(), output_path_latents)
            recon_mesh = vae.decode_mesh(
                latents,
                mc_threshold=args.mc_threshold,
                voxel_resolution=args.voxel_resolution,
            )[0]

        recon_mesh.apply_scale(scale)
        recon_mesh.apply_translation(center)
        recon_mesh.export(output_path)
        print(f"Saved reconstruction to {output_path}")


if __name__ == "__main__":
    main()
