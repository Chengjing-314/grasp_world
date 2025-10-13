"""Signed-distance utilities and mesh blending helpers."""

from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple, Literal

import numpy as np
import torch
from kaolin.io import obj
from kaolin.metrics.trianglemesh import point_to_mesh_distance
from kaolin.ops.mesh import check_sign, index_vertices_by_faces
from kaolin.ops.conversions import sdf_to_voxelgrids, voxelgrids_to_trianglemeshes
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
    method: Literal["skimage", "kaolin"] = "kaolin",
    init_res: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run marching cubes or marching tetrahedra and return world-space vertices and faces."""

    if method == "kaolin" and volume.device.type != "cpu":
        verts_idx, faces = voxelgrids_to_trianglemeshes(
            sdf_to_voxelgrids(sdf=volume.unsqueeze(0), init_res=init_res),
            iso_value=level,
        )
        verts_idx = verts_idx.squeeze(0).detach().cpu().numpy()
        faces = faces.squeeze(0).detach().cpu().numpy().astype(np.int32)
    else:
        volume_np = volume.detach().cpu().numpy()
        verts_idx, faces, _, _ = measure.marching_cubes(volume_np, level=level)
        faces = faces.astype(np.int32)

    axis_np = axis.detach().cpu().numpy()
    idx_coords = np.arange(volume.shape[0], dtype=np.float32)

    verts_world = np.stack(
        [
            np.interp(verts_idx[:, 2], idx_coords, axis_np),
            np.interp(verts_idx[:, 1], idx_coords, axis_np),
            np.interp(verts_idx[:, 0], idx_coords, axis_np),
        ],
        axis=1,
    ).astype(np.float32)

    return verts_world, faces


def smoothstep(u: torch.Tensor) -> torch.Tensor:
    """Cubic Hermite smoothstep: 3u^2 - 2u^3, clamped to [0,1]."""
    u = torch.clamp(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def smootherstep(u: torch.Tensor) -> torch.Tensor:
    """Quartic Hermite smootherstep: 6u^5 - 15u^4 + 10u^3, clamped to [0,1]."""
    u = torch.clamp(u, 0.0, 1.0)
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


def multipath_blend(
    volA,
    volB,
    t,
    xi,
    w=0.1,
    gate: Literal["smooth", "smoother"] = "smoother",
):
    if gate == "smooth":
        g = smoothstep((t - xi) / w)
    elif gate == "smoother":
        g = smootherstep((t - xi) / w)
    else:
        raise ValueError(f"Unknown gate: {gate}")
    return (1 - g) * volA + g * volB


def count_inside(phi: torch.Tensor, level: float) -> int:
    return int((phi <= level).sum().item())


def find_isolevel_for_target_volume(
    phi: torch.Tensor, V_target: int, lo=None, hi=None, iters: int = 24
):
    """
    Bisection on isolevel so volume(phi <= level) ~= V_target.
    """
    if lo is None:
        lo = float(phi.min().item())
    if hi is None:
        hi = float(phi.max().item())
    # Monotone: as level increases, inside volume increases.
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        V = count_inside(phi, mid)
        if V < V_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def sequence_refine_by_change(
    volA, volB, xi, w=0.25, gate="smoother", t_list=None, max_frames=32, tol=5e-3
):
    """
    Start with a coarse t_list (e.g., [0,1] or torch.linspace(...)).
    Recursively insert midpoints where the L2 change of phi is large.
    """
    if t_list is None:
        t_list = [0.0, 1.0]
    t_list = sorted([float(t) for t in t_list])

    def blend_at(t):
        return multipath_blend(volA, volB, t, xi, w=w, gate=gate)

    changed = True
    while changed and len(t_list) < max_frames:
        changed = False
        new_list = [t_list[0]]
        prev_phi = blend_at(t_list[0])
        for i in range(1, len(t_list)):
            t0, t1 = t_list[i - 1], t_list[i]
            phi1 = blend_at(t1)
            # field-space change (mean squared)
            diff = torch.mean((phi1 - prev_phi) ** 2).item()
            if diff > tol and len(t_list) < max_frames:
                tm = 0.5 * (t0 + t1)
                new_list.append(tm)  # insert midpoint
                changed = True
            new_list.append(t1)
            prev_phi = phi1
        t_list = sorted(set(new_list))
    return t_list


def target_volume_for_t(volA, volB, xi, t, w=0.25, gate="smoother"):
    """
    Use the *average gate value* to mix volumes of A and B.
    This keeps global size changing smoothly even when topology flips.
    """
    u = (torch.as_tensor(t, dtype=volA.dtype, device=volA.device) - xi) / w
    if gate == "smoother":
        g = smootherstep(u)
    elif gate == "smooth":
        g = smoothstep(u)
    else:
        raise ValueError(f"Unknown gate: {gate}")
    gbar = float(g.mean().item())
    VA = count_inside(volA, 0.0)
    VB = count_inside(volB, 0.0)
    return int(round((1.0 - gbar) * VA + gbar * VB))


def generate_multipath_sequence(volA, volB, xi, steps=8, w=0.1):
    ts = torch.linspace(0, 1, steps)
    for t in ts:
        yield float(t), multipath_blend(volA, volB, float(t), xi, w)


# --- schedules ---
def xi_uniform(shape, value=0.5, device="cpu"):
    return torch.full(shape, value, device=device)


def xi_plane(shape, axis="x", device="cpu"):
    R = shape[0]
    coords = torch.linspace(0, 1, R, device=device)
    if axis == "x":
        return coords.view(-1, 1, 1).expand(R, R, R)
    if axis == "y":
        return coords.view(1, -1, 1).expand(R, R, R)
    if axis == "z":
        return coords.view(1, 1, -1).expand(R, R, R)


def xi_radial(axis, outward=True):
    R = len(axis)
    X, Y, Z = torch.meshgrid(axis, axis, axis, indexing="ij")
    cx = axis[R // 2]
    cy = axis[R // 2]
    cz = axis[R // 2]
    r = torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)
    r = (r - r.min()) / (r.max() - r.min())
    return r if outward else (1 - r)


def gradient_3d(phi, dx=1.0):
    # central differences
    gx = (phi[2:, 1:-1, 1:-1] - phi[:-2, 1:-1, 1:-1]) / (2 * dx)
    gy = (phi[1:-1, 2:, 1:-1] - phi[1:-1, :-2, 1:-1]) / (2 * dx)
    gz = (phi[1:-1, 1:-1, 2:] - phi[1:-1, 1:-1, :-2]) / (2 * dx)
    return gx, gy, gz


def redistance_pde(phi, iterations=50, dt=0.3, dx=1.0):
    """
    PDE-based redistancing in PyTorch.
    Args:
        phi: [R,R,R] tensor, input field
        iterations: number of reinit steps
        dt: pseudo-time step (<=0.5*dx for stability)
    Returns:
        redistanced phi (approx SDF)
    """
    phi = phi.clone()
    sign0 = phi / torch.sqrt(phi * phi + 1e-12)

    for _ in range(iterations):
        gx, gy, gz = gradient_3d(phi, dx)
        gnorm = torch.sqrt(gx * gx + gy * gy + gz * gz + 1e-12)

        # update only interior (because we used slicing for grads)
        update = sign0[1:-1, 1:-1, 1:-1] * (1.0 - gnorm)
        phi[1:-1, 1:-1, 1:-1] += dt * update

    return phi
