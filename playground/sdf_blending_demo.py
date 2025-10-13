"""Mesh blending demo rendered in Viser."""

import argparse
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import viser

from grasp_world.utils.sdf import (
    compute_common_axis,
    extract_isosurface_mesh,
    sample_mesh_sdfs_on_axis,
    sequence_refine_by_change,
    multipath_blend,
    target_volume_for_t,
    find_isolevel_for_target_volume,
    redistance_pde,
    xi_uniform,
    xi_plane,
    xi_radial,
)

MESH_DIR = Path("assets/objects/primitive_shapes")
MESHES: Dict[str, Path] = {
    "sphere": MESH_DIR / "sphere.obj",
    "cube": MESH_DIR / "cube.obj",
    "cylinder": MESH_DIR / "cylinder.obj",
    "cone": MESH_DIR / "cone.obj",
    "torus": MESH_DIR / "torus.obj",
}


def _lerp_color(weight: float) -> Tuple[int, int, int]:
    """Simple RGB gradient from teal to orange for visualization."""
    c0 = np.array([66, 135, 245], dtype=np.float32)
    c1 = np.array([237, 125, 49], dtype=np.float32)
    color = (1.0 - weight) * c0 + weight * c1
    return tuple(int(channel) for channel in color)


def _quat_from_direction(
    direction: Tuple[float, float, float],
) -> Tuple[float, float, float, float]:
    """Quaternion rotating -Z into the requested direction vector."""
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


def visualize_schedules(src_mesh, tgt_mesh, resolution=64, steps=6, device="auto"):
    axis = compute_common_axis([src_mesh, tgt_mesh], resolution)
    vols = sample_mesh_sdfs_on_axis({"A": src_mesh, "B": tgt_mesh}, axis, device=device)
    A, B = vols["A"], vols["B"]

    schedules = {
        "uniform": xi_uniform(A.shape, 0.5, device=A.device),
        "plane_x": xi_plane(A.shape, "x", device=A.device),
        "radial": xi_radial(axis.to(A.device), outward=True),
    }

    server = viser.ViserServer()
    server.scene.world_axes.visible = True

    span = axis[-1] - axis[0]
    offset_x = span * 1.4
    offset_y = span * 1.4

    for row, (name, xi) in enumerate(schedules.items()):
        t_list = sequence_refine_by_change(
            A, B, xi, w=0.3, gate="smoother", t_list=[0.0, 0.5, 1.0], max_frames=20, tol=4e-3
        )
        for col, t in enumerate(t_list):
            phi = multipath_blend(A, B, t, xi, w=0.3, gate="smoother")
            Vt = target_volume_for_t(A, B, xi, t, w=0.3, gate="smoother")
            iso = find_isolevel_for_target_volume(phi, Vt)
            verts, faces = extract_isosurface_mesh(phi, axis, level=iso, init_res=resolution)
            position = (offset_x * (col - (steps - 1) / 2), row * offset_y, 0.0)
            server.scene.add_mesh_simple(
                f"/{name}/step_{col}",
                vertices=verts,
                faces=faces,
                color=_lerp_color(t),
                position=position,
                flat_shading=True,
                side="double",
            )
            server.scene.add_label(
                f"/{name}/label_{col}",
                text=f"{name}, t={t:.2f}",
                position=(position[0], position[1], axis[-1] + 0.2),
            )

    # Keep the visualization server alive so the viewer stays connected.
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SDF blends in Viser.")
    parser.add_argument("--resolution", type=int, default=96, help="Grid resolution per axis.")
    parser.add_argument("--steps", type=int, default=10, help="Number of blend steps to show.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for SDF sampling (e.g. auto, cpu, cuda:0).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["lsb", "r"],
        default="lsb",
        help="Blend mode: linear signed blend or R-function smooth union.",
    )
    parser.add_argument(
        "--smoothness",
        type=float,
        default=0.1,
        help="Smoothness parameter when --mode=r is selected.",
    )

    args = parser.parse_args()
    visualize_schedules(
        MESHES["sphere"],
        MESHES["torus"],
        resolution=args.resolution,
        steps=args.steps,
        device=args.device,
    )
