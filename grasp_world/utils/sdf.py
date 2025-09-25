"""Signed-distance utilities and mesh blending helpers."""

from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import torch
from kaolin.io import obj
from kaolin.metrics.trianglemesh import point_to_mesh_distance
from kaolin.ops.mesh import check_sign, index_vertices_by_faces
import trimesh
from skimage import measure

from grasp_world.utils.torch_utils import resolve_device

_DEFAULT_CHUNK_SIZE = 200_000


def load_mesh_components(mesh_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
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


def signed_distance(
    verts: torch.Tensor,
    faces: torch.Tensor,
    face_vertices: torch.Tensor,
    points: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        dist2, _, _ = point_to_mesh_distance(points.unsqueeze(0), face_vertices)
        sdf = torch.sqrt(dist2.squeeze(0))
        inside = check_sign(verts, faces, points.unsqueeze(0)).squeeze(0)
        sdf[inside] *= -1.0
    return sdf


def sample_signed_distance(
    mesh_data: Dict[str, torch.Tensor],
    points: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Sample signed distances for many points using chunked evaluation."""
    num_points = points.shape[0]
    sdf = torch.empty(num_points, dtype=mesh_data["verts"].dtype)
    device = mesh_data["verts"].device

    for start in range(0, num_points, chunk_size):
        end = min(start + chunk_size, num_points)
        chunk = points[start:end].to(device)
        sdf_chunk = signed_distance(
            mesh_data["verts"], mesh_data["faces"], mesh_data["face_vertices"], chunk
        )
        sdf[start:end] = sdf_chunk.detach().cpu()

    return sdf


@lru_cache(maxsize=8)
def load_trimesh(mesh_path: str) -> trimesh.Trimesh:
    return trimesh.load(mesh_path, process=False, force="mesh")


def signed_distance_trimesh(mesh_path: str, points: torch.Tensor) -> torch.Tensor:
    mesh = load_trimesh(mesh_path)
    pts = points.detach().cpu().numpy()
    sdf = mesh.nearest.signed_distance(pts)
    return torch.from_numpy(np.asarray(sdf, dtype=np.float32))


def mesh_sdf(
    mesh_path: str,
    grid_res: int = 64,
    device: torch.device | str | None = "auto",
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = resolve_device(device)
    mesh_data = load_mesh_components(mesh_path, device)
    bb_min, bb_max = mesh_data["bounds"]
    grid_axes = [torch.linspace(float(bb_min[i]), float(bb_max[i]), grid_res) for i in range(3)]
    grid = torch.stack(torch.meshgrid(*grid_axes, indexing="ij"), dim=-1)
    flat_points = grid.reshape(-1, 3)
    if device.type == "cpu":
        sdf_flat = signed_distance_trimesh(mesh_path, flat_points)
    else:
        sdf_flat = sample_signed_distance(mesh_data, flat_points, chunk_size)
    return flat_points, sdf_flat.view(grid_res, grid_res, grid_res)


def mesh_sdf_from_points(
    mesh_path: str,
    points: torch.Tensor,
    device: torch.device | str | None = "auto",
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> torch.Tensor:
    if points.shape[-1] != 3:
        raise ValueError("Query points must have last dimension of size 3.")

    device = resolve_device(device)
    if device.type == "cpu":
        sdf_flat = signed_distance_trimesh(mesh_path, points.reshape(-1, 3))
        return sdf_flat.view(points.shape[:-1])

    mesh_data = load_mesh_components(mesh_path, device)
    flattened = points.reshape(-1, 3).to(dtype=torch.float32).cpu()
    sdf_flat = sample_signed_distance(mesh_data, flattened, chunk_size)
    return sdf_flat.view(points.shape[:-1])


def compute_common_axis(
    mesh_paths: Sequence[str | Path],
    resolution: int,
    margin: float = 0.15,
) -> torch.Tensor:
    """Return a shared coordinate axis covering all meshes with optional margin."""

    mins = []
    maxs = []
    for mesh_path in mesh_paths:
        mesh = obj.import_mesh(str(mesh_path), triangulate=True)
        verts = mesh.vertices
        mins.append(verts.min(dim=0).values)
        maxs.append(verts.max(dim=0).values)

    global_min = torch.stack(mins).min(dim=0).values
    global_max = torch.stack(maxs).max(dim=0).values
    lo = float(global_min.min())
    hi = float(global_max.max())
    span = hi - lo
    lo -= margin * span
    hi += margin * span
    return torch.linspace(lo, hi, resolution)


def sample_mesh_sdfs_on_axis(
    meshes: Mapping[str, str | Path],
    axis: torch.Tensor,
    device: torch.device | str | None = "auto",
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> Dict[str, torch.Tensor]:
    """Sample SDF volumes for multiple meshes on the provided axis-aligned grid."""

    grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
    flat_points = grid.reshape(-1, 3)
    volumes: Dict[str, torch.Tensor] = {}

    for name, mesh_path in meshes.items():
        volume = mesh_sdf_from_points(
            str(Path(mesh_path)), flat_points, device=device, chunk_size=chunk_size
        )
        volumes[name] = volume.view(len(axis), len(axis), len(axis))

    return volumes


def blend_signed_distance_fields(
    A: torch.Tensor,
    B: torch.Tensor,
    weight: float,
    mode: str = "lsb",
    smoothness: float = 4.0,
) -> torch.Tensor:
    if A.shape != B.shape:
        raise ValueError("Input volumes must share the same shape for blending.")

    w = torch.as_tensor(weight, dtype=A.dtype, device=A.device).clamp(0.0, 1.0)

    if mode == "lsb":
        return (1.0 - w) * A + w * B

    if mode == "r":
        k = max(float(smoothness), 1e-5)
        # log-weights to stay stable (avoid log(0))
        logw0 = torch.log(torch.clamp(1.0 - w, min=1e-12))
        logw1 = torch.log(torch.clamp(w, min=1e-12))
        u = logw0 - k * A
        v = logw1 - k * B
        # log( (1-w) e^{-kA} + w e^{-kB} )  computed stably
        return -torch.logaddexp(u, v) / k

    raise ValueError("Unknown blend mode. Use 'lsb' or 'r'.")


def ensure_valid_sdf(volume: np.ndarray | torch.Tensor, level: float = 0.0) -> float:
    """Check if iso-level exists, otherwise shift."""
    vmin, vmax = volume.min(), volume.max()
    if not (vmin <= level <= vmax):
        level = 0.5 * (vmin + vmax)
    return level


def generate_blend_sequence(
    volume_a: torch.Tensor,
    volume_b: torch.Tensor,
    steps: int,
    mode: str = "lsb",
    smoothness: float = 4.0,
) -> Iterable[Tuple[float, torch.Tensor]]:
    """Yield (weight, blended_volume) pairs across the requested number of steps."""

    weights = torch.linspace(0.0, 1.0, steps)
    for weight in weights:
        blended = blend_signed_distance_fields(
            volume_a, volume_b, float(weight), mode=mode, smoothness=smoothness
        )
        yield float(weight), blended


def extract_isosurface_mesh(
    volume: torch.Tensor,
    axis: torch.Tensor,
    level: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run marching cubes on a volume and return world-space vertices and faces."""

    volume_np = volume.detach().cpu().numpy()
    verts, faces, _, _ = measure.marching_cubes(volume_np, level=level)

    axis_np = axis.detach().cpu().numpy()
    idx_coords = np.arange(volume.shape[0], dtype=np.float32)
    verts_world = np.stack(
        [
            np.interp(verts[:, 2], idx_coords, axis_np),
            np.interp(verts[:, 1], idx_coords, axis_np),
            np.interp(verts[:, 0], idx_coords, axis_np),
        ],
        axis=1,
    ).astype(np.float32)

    return verts_world, faces.astype(np.int32)
