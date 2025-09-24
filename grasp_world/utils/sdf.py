"""Signed distance utilities powered by Kaolin."""

from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import torch
from kaolin.io import obj
from kaolin.metrics.trianglemesh import point_to_mesh_distance
from kaolin.ops.mesh import check_sign, index_vertices_by_faces
import trimesh

_DEFAULT_CHUNK_SIZE = 200_000


def _resolve_device(device: torch.device | str | None) -> torch.device:
    """Pick a torch.device, defaulting to CUDA when available."""
    if isinstance(device, torch.device):
        return device
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_mesh_components(mesh_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """Load mesh data needed for SDF queries and move tensors to `device`."""
    mesh = obj.import_mesh(mesh_path, triangulate=True).to_batched()
    verts_cpu = mesh.vertices.to(dtype=torch.float32)
    faces_cpu = mesh.faces.to(dtype=torch.long)
    bounds = (
        verts_cpu.min(dim=1).values.squeeze(0),
        verts_cpu.max(dim=1).values.squeeze(0),
    )
    verts = verts_cpu.to(device)
    faces = faces_cpu.to(device)
    face_vertices = index_vertices_by_faces(verts, faces)
    return {
        "verts": verts,
        "faces": faces,
        "face_vertices": face_vertices,
        "bounds": bounds,
    }


def _signed_distance(
    verts: torch.Tensor,
    faces: torch.Tensor,
    face_vertices: torch.Tensor,
    points: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the signed distance for a batch of query points on `verts` mesh."""
    with torch.no_grad():
        dist2, _, _ = point_to_mesh_distance(points.unsqueeze(0), face_vertices)
        sdf = torch.sqrt(dist2.squeeze(0))
        inside = check_sign(verts, faces, points.unsqueeze(0)).squeeze(0)
        sdf[inside] *= -1.0
    return sdf


def _evaluate_on_points(
    mesh_data: Dict[str, torch.Tensor],
    points: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Chunked signed-distance evaluation to limit memory usage."""
    num_points = points.shape[0]
    sdf = torch.empty(num_points, dtype=mesh_data["verts"].dtype)
    device = mesh_data["verts"].device

    for start in range(0, num_points, chunk_size):
        end = min(start + chunk_size, num_points)
        chunk = points[start:end].to(device)
        sdf_chunk = _signed_distance(
            mesh_data["verts"], mesh_data["faces"], mesh_data["face_vertices"], chunk
        )
        sdf[start:end] = sdf_chunk.detach().cpu()

    return sdf


@lru_cache(maxsize=8)
def _load_trimesh(mesh_path: str) -> trimesh.Trimesh:
    """Load a trimesh mesh with caching for CPU evaluation."""
    return trimesh.load(mesh_path, process=False, force="mesh")


def _signed_distance_trimesh(mesh_path: str, points: torch.Tensor) -> torch.Tensor:
    """Compute signed distance using trimesh for CPU-only environments."""
    mesh = _load_trimesh(mesh_path)
    pts = points.detach().cpu().numpy()
    sdf = mesh.nearest.signed_distance(pts)
    return torch.from_numpy(np.asarray(sdf, dtype=np.float32))


def mesh_sdf(
    mesh_path: str,
    grid_res: int = 64,
    device: torch.device | str | None = "auto",
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = _resolve_device(device)
    mesh_data = _load_mesh_components(mesh_path, device)
    bb_min, bb_max = mesh_data["bounds"]
    grid_axes = [torch.linspace(float(bb_min[i]), float(bb_max[i]), grid_res) for i in range(3)]
    grid = torch.stack(torch.meshgrid(*grid_axes, indexing="ij"), dim=-1)
    flat_points = grid.reshape(-1, 3)
    if device.type == "cpu":
        sdf_flat = _signed_distance_trimesh(mesh_path, flat_points)
    else:
        sdf_flat = _evaluate_on_points(mesh_data, flat_points, chunk_size)
    return flat_points, sdf_flat.view(grid_res, grid_res, grid_res)


def mesh_sdf_from_points(
    mesh_path: str,
    points: torch.Tensor,
    device: torch.device | str | None = "auto",
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> torch.Tensor:
    if points.shape[-1] != 3:
        raise ValueError("Query points must have last dimension of size 3.")

    device = _resolve_device(device)
    if device.type == "cpu":
        sdf_flat = _signed_distance_trimesh(mesh_path, points.reshape(-1, 3))
        return sdf_flat.view(points.shape[:-1])

    mesh_data = _load_mesh_components(mesh_path, device)
    flattened = points.reshape(-1, 3).to(dtype=torch.float32).cpu()
    sdf_flat = _evaluate_on_points(mesh_data, flattened, chunk_size)
    return sdf_flat.view(points.shape[:-1])
