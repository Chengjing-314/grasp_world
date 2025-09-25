"""Mesh blending demo rendered in Viser."""

from pathlib import Path
from typing import Dict, Tuple
import torch
from kaolin.io import obj

import argparse
import numpy as np
import viser

from grasp_world.utils.sdf import (
    extract_isosurface_mesh,
    generate_blend_sequence,
    sample_mesh_sdfs_on_axis,
    compute_common_axis,
)

MESH_DIR = Path("assets/primitive_shapes")
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


def visualize_blends(
    src_mesh: Path,
    tgt_mesh: Path,
    resolution: int = 96,
    steps: int = 5,
    device: str = "auto",
    blend_mode: str = "lsb",
    smoothness: float = 4.0,
) -> None:
    axis = compute_common_axis([src_mesh, tgt_mesh], resolution)
    volumes = sample_mesh_sdfs_on_axis({"src": src_mesh, "tgt": tgt_mesh}, axis, device=device)
    blend_sequence = list(
        generate_blend_sequence(
            volumes["src"],
            volumes["tgt"],
            steps,
            mode=blend_mode,
            smoothness=smoothness,
        )
    )

    server = viser.ViserServer()
    server.scene.world_axes.visible = True
    server.scene.add_grid("/ground", width=4.0, height=4.0, cell_size=0.25)

    # Basic lighting rig.
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

    for idx, (weight, volume) in enumerate(blend_sequence):
        verts, faces = extract_isosurface_mesh(volume, axis)
        name = f"/blend/step_{idx:02d}"
        color = _lerp_color(weight)
        position = (offset_step * (idx - (len(blend_sequence) - 1) / 2.0), 0.0, 0.0)

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
    parser = argparse.ArgumentParser(description="Visualize SDF blends in Viser.")
    parser.add_argument("--resolution", type=int, default=96, help="Grid resolution per axis.")
    parser.add_argument("--steps", type=int, default=5, help="Number of blend steps to show.")
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
        default=4.0,
        help="Smoothness parameter when --mode=r is selected.",
    )

    args = parser.parse_args()
    visualize_blends(
        MESHES["sphere"],
        MESHES["torus"],
        resolution=args.resolution,
        steps=args.steps,
        device=args.device,
        blend_mode=args.mode,
        smoothness=args.smoothness,
    )
