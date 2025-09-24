"""Linear signed blending (LSB) demo using Kaolin SDFs with 3D Viser visualization."""

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import viser
from kaolin.io import obj
from skimage import measure

from grasp_world.utils.sdf import mesh_sdf_from_points

MESH_DIR = Path("assets/primitive_shapes")
MESHES: Dict[str, Path] = {
    "sphere": MESH_DIR / "sphere.obj",
    "cube": MESH_DIR / "cube.obj",
}


def compute_axis(resolution: int, margin: float = 0.15) -> torch.Tensor:
    """Build a cubic axis covering all meshes with an optional margin."""
    mins = []
    maxs = []
    for path in MESHES.values():
        mesh = obj.import_mesh(str(path), triangulate=True)
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


def _blend_volumes(
    resolution: int,
    steps: int,
    device: str,
) -> Tuple[np.ndarray, Iterable[Tuple[float, np.ndarray, np.ndarray]]]:

    axis = compute_axis(resolution)
    grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
    flat_points = grid.reshape(-1, 3)

    sphere_sdf = mesh_sdf_from_points(str(MESHES["sphere"]), flat_points, device=device)
    cube_sdf = mesh_sdf_from_points(str(MESHES["cube"]), flat_points, device=device)


    sphere_sdf = sphere_sdf.view(resolution, resolution, resolution)
    cube_sdf = cube_sdf.view(resolution, resolution, resolution)

    axis_np = axis.numpy()
    idx_coords = np.arange(resolution)
    blends = []

    for weight in torch.linspace(0.0, 1.0, steps):
        blended = (1.0 - weight) * sphere_sdf + weight * cube_sdf
        volume = blended.numpy()
        verts, faces, _, _ = measure.marching_cubes(volume, level=0.0)

        verts_world = np.stack(
            [
                np.interp(verts[:, 2], idx_coords, axis_np),
                np.interp(verts[:, 1], idx_coords, axis_np),
                np.interp(verts[:, 0], idx_coords, axis_np),
            ],
            axis=1,
        ).astype(np.float32)

        blends.append((float(weight), verts_world, faces.astype(np.int32)))

    return axis_np, blends


def _lerp_color(weight: float) -> Tuple[int, int, int]:
    c0 = np.array([66, 135, 245], dtype=np.float32)
    c1 = np.array([237, 125, 49], dtype=np.float32)
    color = (1.0 - weight) * c0 + weight * c1
    return tuple(int(channel) for channel in color)


def _quat_from_direction(direction: Tuple[float, float, float]) -> Tuple[float, float, float, float]:
    source = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    target = np.array(direction, dtype=np.float32)
    norm = np.linalg.norm(target)
    if norm < 1e-6:
        return (1.0, 0.0, 0.0, 0.0)
    target = target / norm
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if np.isclose(dot, 1.0):
        return (1.0, 0.0, 0.0, 0.0)
    if np.isclose(dot, -1.0):
        return (0.0, 1.0, 0.0, 0.0)
    axis = np.cross(source, target)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-6:
        return (1.0, 0.0, 0.0, 0.0)
    axis = axis / axis_norm
    angle = np.arccos(dot)
    half = angle / 2.0
    sin_half = np.sin(half)
    return (np.cos(half), axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half)


def visualize_blends(resolution: int = 96, steps: int = 5, device: str = "auto") -> None:
    axis, blends = _blend_volumes(resolution, steps, device)

    server = viser.ViserServer()
    server.scene.world_axes.visible = True
    server.scene.add_grid("/ground", width=4.0, height=4.0, cell_size=0.25)

    # Simple three-point lighting setup for better depth perception.
    server.scene.add_light_directional(
        "/lights/key",
        color=(1.0, 0.95, 0.9),
        intensity=3.0,
        cast_shadow=True,
        wxyz=_quat_from_direction((0.5, -1.0, -0.8)),
    )
    server.scene.add_light_directional(
        "/lights/fill",
        color=(0.6, 0.7, 1.0),
        intensity=1.2,
        wxyz=_quat_from_direction((-0.6, -1.0, -0.2)),
    )
    server.scene.add_light_ambient("/lights/ambient", intensity=0.3)

    span = axis[-1] - axis[0]
    offset_step = span * 1.4 if span > 0 else 1.0

    for idx, (weight, verts, faces) in enumerate(blends):
        name = f"/lsb/step_{idx:02d}"
        color = _lerp_color(weight)
        position = (offset_step * (idx - (len(blends) - 1) / 2.0), 0.0, 0.0)

        server.scene.add_mesh_simple(
            name,
            vertices=verts,
            faces=faces,
            color=color,
            position=position,
            flat_shading=True,
            side="double",
            cast_shadow=True,
            receive_shadow=True,
        )

        server.scene.add_label(
            f"{name}_label",
            text=f"w = {weight:.2f}",
            position=(position[0], 0.0, axis[-1] + 0.2),
        )

    print("Viser server running at http://localhost:8080 — press Ctrl+C to stop.")
    try:
        server.sleep_forever()
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    visualize_blends(resolution=96, steps=5)
