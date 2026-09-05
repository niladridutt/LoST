from typing import Callable, List, Optional, Union, Dict, Any
import os
import PIL.Image
import torch
import trimesh
import pymeshlab
import tempfile





def load_mesh(path):
    if path.endswith(".glb"):
        mesh = trimesh.load(path)
    else:
        mesh = pymeshlab.MeshSet()
        mesh.load_new_mesh(path)
    return mesh


def trimesh2pymeshlab(mesh: trimesh.Trimesh):
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as temp_file:
        if isinstance(mesh, trimesh.scene.Scene):
            for idx, obj in enumerate(mesh.geometry.values()):
                if idx == 0:
                    temp_mesh = obj
                else:
                    temp_mesh = temp_mesh + obj
            mesh = temp_mesh
        mesh.export(temp_file.name)
        mesh = pymeshlab.MeshSet()
        mesh.load_new_mesh(temp_file.name)
    return mesh


def pymeshlab2trimesh(mesh: pymeshlab.MeshSet):
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as temp_file:
        mesh.save_current_mesh(temp_file.name)
        mesh = trimesh.load(temp_file.name)
    if isinstance(mesh, trimesh.Scene):
        combined_mesh = trimesh.Trimesh()
        for geom in mesh.geometry.values():
            combined_mesh = trimesh.util.concatenate([combined_mesh, geom])
        mesh = combined_mesh
    return mesh


def import_mesh(mesh):
    mesh_type = type(mesh)
    if isinstance(mesh, str):
        mesh = load_mesh(mesh)
    # elif isinstance(mesh, MeshExtractResult):
    #     mesh = pymeshlab.MeshSet()
    #     mesh_pymeshlab = pymeshlab.Mesh(
    #         vertex_matrix=mesh.verts.cpu().numpy(), face_matrix=mesh.faces.cpu().numpy()
    #     )
    #     mesh.add_mesh(mesh_pymeshlab, "converted_mesh")

    if isinstance(mesh, (trimesh.Trimesh, trimesh.scene.Scene)):
        mesh = trimesh2pymeshlab(mesh)

    return mesh, mesh_type


def remove_floater(mesh):
    mesh, mesh_type = import_mesh(mesh)

    mesh.apply_filter(
        "compute_selection_by_small_disconnected_components_per_face", nbfaceratio=0.001
    )
    mesh.apply_filter("compute_selection_transfer_face_to_vertex", inclusive=False)
    mesh.apply_filter("meshing_remove_selected_vertices_and_faces")

    return pymeshlab2trimesh(mesh)


def remove_degenerate_face(mesh):
    mesh, mesh_type = import_mesh(mesh)

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as temp_file:
        mesh.save_current_mesh(temp_file.name)
        mesh = pymeshlab.MeshSet()
        mesh.load_new_mesh(temp_file.name)

    return pymeshlab2trimesh(mesh)


def reduce_face(mesh, max_facenum=50000):
    mesh, mesh_type = import_mesh(mesh)

    if max_facenum > mesh.current_mesh().face_number():
        return pymeshlab2trimesh(mesh)

    mesh.apply_filter(
        "meshing_decimation_quadric_edge_collapse",
        targetfacenum=max_facenum,
        qualitythr=1.0,
        preserveboundary=True,
        boundaryweight=3,
        preservenormal=True,
        preservetopology=True,
        autoclean=True,
    )

    return pymeshlab2trimesh(mesh)


