"""Flow-matching supervision example with training and visualization (normalized + fixed viewer)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable, Tuple

import torch
import torch.nn.functional as F
import trimesh
import viser

from grasp_world.network.sdf_net import SDFGuide
from grasp_world.network.velocity_net import VelocityNet
from grasp_world.pointnet.pointnet2_encoder import PointNet2Encoder
from grasp_world.pointnet.pointnet2_utils import farthest_point_sample, index_points
from grasp_world.utils.torch_utils import resolve_device

MESH_DIR = Path("assets/objects/primitive_shapes")
MESHES = {
    "sphere": MESH_DIR / "sphere.obj",
    "cube": MESH_DIR / "cube.obj",
    "cylinder": MESH_DIR / "cylinder.obj",
    "cone": MESH_DIR / "cone.obj",
    "torus": MESH_DIR / "torus.obj",
}


# -----------------------------------------------------------------------------
# Point sampling utilities
# -----------------------------------------------------------------------------
def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Center and scale mesh into roughly [-0.5, 0.5]^3 box."""
    mesh.vertices -= mesh.vertices.mean(axis=0)
    scale = max(mesh.extents)
    mesh.vertices /= scale
    return mesh


def sample_surface_points(mesh_path: Path, total_points: int) -> torch.Tensor:
    """Sample a dense point cloud on the mesh surface, normalized to unit scale."""
    mesh = trimesh.load_mesh(mesh_path, process=False)
    mesh = normalize_mesh(mesh)
    pts = mesh.sample(total_points)
    return torch.from_numpy(pts).to(dtype=torch.float32)


def farthest_point_subsample(
    points: torch.Tensor, num_samples: int, device: torch.device
) -> torch.Tensor:
    """Downsample a point cloud using farthest-point sampling."""
    if num_samples >= points.shape[0]:
        return points.to(device)
    pts = points.unsqueeze(0).to(device)  # [1, M, 3]
    idx = farthest_point_sample(pts, num_samples)  # [1, num_samples]
    sampled = index_points(pts, idx)  # [1, num_samples, 3]
    return sampled.squeeze(0)


def smoothstep(t: torch.Tensor) -> torch.Tensor:
    """Cubic Hermite smoothstep: 3t^2 - 2t^3."""
    return t * t * (3.0 - 2.0 * t)


def soft_barycenter(
    target_points: torch.Tensor,
    query_points: torch.Tensor,
    k: int = 8,
    tau: float = 4e-4,
) -> torch.Tensor:
    """Compute a deterministic soft k-NN barycenter for each query point."""
    B, N, _ = query_points.shape
    M = target_points.shape[1]
    dist = torch.cdist(query_points, target_points, p=2)  # [B, N, M]

    if k >= M:
        idx = torch.arange(M, device=query_points.device)[None, None, :].expand(B, N, M)
        dist_k = dist
    else:
        dist_k, idx = torch.topk(dist, k, largest=False, dim=-1)  # [B, N, k]

    expanded_targets = target_points.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, M, 3]
    neighbors = torch.gather(
        expanded_targets, 2, idx.unsqueeze(-1).expand(-1, -1, -1, 3)
    )  # [B, N, k, 3]

    weights = torch.softmax(-dist_k / tau, dim=-1).unsqueeze(-1)  # [B, N, k, 1]
    barycenter = (weights * neighbors).sum(dim=2)  # [B, N, 3]
    return barycenter


def encode_latents(encoder: PointNet2Encoder, points: torch.Tensor) -> torch.Tensor:
    """Encode point clouds (B, N, 3) into latent codes of shape [B, D]."""
    encoder_inputs = points.permute(0, 2, 1)  # [B, 3, N]
    return encoder(encoder_inputs)


def prepare_point_clouds(
    src_mesh: Path,
    tgt_mesh: Path,
    device: torch.device,
    surface_points: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample dense source/target point clouds."""
    src_dense = sample_surface_points(src_mesh, surface_points).to(device).unsqueeze(0)
    tgt_dense = sample_surface_points(tgt_mesh, surface_points).to(device).unsqueeze(0)
    return src_dense, tgt_dense


# -----------------------------------------------------------------------------
# Training helpers
# -----------------------------------------------------------------------------
def training_step(
    sdf_model: SDFGuide,
    velocity_model: VelocityNet,
    src_dense: torch.Tensor,
    tgt_dense: torch.Tensor,
    z_s: torch.Tensor,
    z_t: torch.Tensor,
    samples: int,
    knn: int,
    tau: float,
) -> torch.Tensor:
    """Perform one stochastic flow-matching supervision step."""
    device = src_dense.device
    B, M, _ = src_dense.shape

    perm = torch.randperm(M, device=device)[:samples]
    x0 = src_dense[:, perm, :]  # [B, samples, 3]

    ybar = soft_barycenter(
        target_points=tgt_dense,
        query_points=x0,
        k=knn,
        tau=tau,
    )

    t = torch.rand(B, samples, 1, device=device)
    alpha = smoothstep(t)
    alpha_dot = 6.0 * t * (1.0 - t)
    x_t = (1.0 - alpha) * x0 + alpha * ybar
    u_star = alpha_dot * (ybar - x0)

    phi_t = sdf_model(x_t, t, z_s, z_t)
    v = velocity_model(x_t, t, phi_t, z_s, z_t)
    loss = F.mse_loss(v, u_star)
    return loss


def build_pointcloud(
    points: torch.Tensor,
    color: Tuple[float, float, float],
    offset: Tuple[float, float, float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return positions and colors tensors for viser point cloud."""
    B, N, _ = points.shape
    positions = points.clone().view(-1, 3)
    positions += torch.tensor(offset, dtype=points.dtype, device=points.device)
    colors = torch.tensor(color, dtype=points.dtype, device=points.device).repeat(B * N, 1)
    return positions, colors


def visualize_pointclouds(
    teacher_points: Iterable[Tuple[float, torch.Tensor]],
    network_points: Iterable[Tuple[float, torch.Tensor]],
) -> None:
    """Render teacher vs network point trajectories in Viser (non-overlapping)."""
    import numpy as np

    server = viser.ViserServer()
    server.scene.world_axes.visible = True

    entries = list(zip(teacher_points, network_points))

    # Estimate span for auto spacing
    all_pts = torch.cat([p for _, p in teacher_points], dim=1)
    span = float((all_pts.max() - all_pts.min()).item())
    offset_step = span * 2.0

    palette = [
        (0.22, 0.49, 0.72),
        (1.00, 0.50, 0.05),
        (0.30, 0.69, 0.29),
        (0.60, 0.31, 0.64),
        (0.89, 0.10, 0.11),
        (0.97, 0.58, 0.77),
        (0.66, 0.66, 0.33),
        (0.49, 0.49, 0.49),
    ]

    for idx, ((t_teacher, pts_teacher), (t_network, pts_network)) in enumerate(entries):
        color = palette[idx % len(palette)]
        offset = ((idx - (len(entries) - 1) / 2) * offset_step, 0.0, 0.0)

        teacher_pos, teacher_color = build_pointcloud(pts_teacher, color, offset)
        network_pos, network_color = build_pointcloud(pts_network, color, offset)
        network_pos[:, 2] += span * 1.5  # vertical separation

        server.scene.add_point_cloud(
            f"/teacher/t_{idx:02d}",
            points=teacher_pos.cpu().numpy(),
            colors=teacher_color.cpu().numpy(),
            point_size=0.01,
        )
        server.scene.add_point_cloud(
            f"/network/t_{idx:02d}",
            points=network_pos.cpu().numpy(),
            colors=network_color.cpu().numpy(),
            point_size=0.01,
        )
        server.scene.add_label(
            f"/labels/t_{idx:02d}",
            text=f"t={t_teacher:.2f} | teacher (lower) vs network (upper)",
            position=(offset[0], offset[1], span * 0.7),
        )

    print("Viser server running; press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


def integrate_velocity(
    velocity_model: VelocityNet,
    sdf_model: SDFGuide,
    x_init: torch.Tensor,
    z_s: torch.Tensor,
    z_t: torch.Tensor,
    steps: int = 50,
    device: torch.device = "cuda",
) -> torch.Tensor:
    """Numerically integrate the learned velocity field over t ∈ [0, 1]."""
    B, N, _ = x_init.shape
    x = x_init.clone().to(device)
    t_values = torch.linspace(0.0, 1.0, steps, device=device)
    dt = 1.0 / (steps - 1)

    for t in t_values:
        t_batch = t.expand(B, N, 1)
        with torch.no_grad():
            phi_t = sdf_model(x, t_batch, z_s, z_t)
            v = velocity_model(x, t_batch, phi_t, z_s, z_t)
        x = x + v * dt  # Euler integration
    return x


def gather_teacher_network_samples(
    sdf_model: SDFGuide,
    velocity_model: VelocityNet,
    src_dense: torch.Tensor,
    tgt_dense: torch.Tensor,
    z_s: torch.Tensor,
    z_t: torch.Tensor,
    num_samples: int,
    knn: int,
    tau: float,
    t_values: torch.Tensor,
) -> Tuple[list[Tuple[float, torch.Tensor]], list[Tuple[float, torch.Tensor]]]:
    """Compute teacher and network point clouds for visualization (true integrated flow)."""
    device = src_dense.device
    B, M, _ = src_dense.shape
    perm = torch.randperm(M, device=device)[:num_samples]
    x0 = src_dense[:, perm, :]  # [B, num_samples, 3]

    ybar = soft_barycenter(tgt_dense, x0, k=knn, tau=tau)
    teacher_pc = []
    for t_val in t_values:
        t = torch.full((B, num_samples, 1), t_val, device=device)
        alpha = smoothstep(t)
        x_t = (1.0 - alpha) * x0 + alpha * ybar
        teacher_pc.append((float(t_val), x_t.detach().cpu()))

    steps = 100
    all_t = torch.linspace(0.0, 1.0, steps, device=device)
    dt = 1.0 / (steps - 1)
    x = x0.clone()
    snapshots = []
    snapshot_indices = [int(t * (steps - 1)) for t in t_values.tolist()]

    for step, t in enumerate(all_t):
        t_batch = t.expand(B, num_samples, 1)
        with torch.no_grad():
            phi_t = sdf_model(x, t_batch, z_s, z_t)
            v = velocity_model(x, t_batch, phi_t, z_s, z_t)
        x = x + v * dt
        if step in snapshot_indices:
            snapshots.append(x.clone().detach().cpu())

    network_pc = [(float(t_val), snap) for t_val, snap in zip(t_values.tolist(), snapshots)]
    return (
        teacher_pc,
        network_pc,
    )  # -----------------------------------------------------------------------------


# Main demo
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train SDFGuide + VelocityNet with soft k-NN flow supervision."
    )
    parser.add_argument("--src", type=str, default="sphere", choices=MESHES.keys())
    parser.add_argument("--tgt", type=str, default="torus", choices=MESHES.keys())
    parser.add_argument(
        "--surface", type=int, default=4096, help="Number of surface samples per mesh."
    )
    parser.add_argument(
        "--samples", type=int, default=1024, help="Number of samples per training step."
    )
    parser.add_argument("--knn", type=int, default=8, help="k value for soft barycenter.")
    parser.add_argument(
        "--tau", type=float, default=4e-4, help="Softmax temperature for k-NN weights."
    )
    parser.add_argument(
        "--train-steps", type=int, default=500, help="Number of training iterations."
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam.")
    parser.add_argument(
        "--vis-steps", type=int, default=5, help="Number of timesteps for visualization."
    )
    parser.add_argument(
        "--vis-samples", type=int, default=512, help="Number of source points to visualize."
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Computation device (auto/cpu/cuda)."
    )
    parser.add_argument("--visualize", action="store_true", help="Launch Viser visualization.")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    src_dense, tgt_dense = prepare_point_clouds(
        MESHES[args.src], MESHES[args.tgt], device, args.surface
    )

    encoder = PointNet2Encoder(device=device, normal_channel=False).to(device)
    with torch.no_grad():
        z_s = encode_latents(encoder, src_dense)
        z_t = encode_latents(encoder, tgt_dense)

    latent_dim = z_s.shape[-1]
    sdf_model = SDFGuide(latent_dim=latent_dim).to(device)
    velocity_model = VelocityNet(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(
        list(sdf_model.parameters()) + list(velocity_model.parameters()), lr=args.lr
    )

    sdf_model.train()
    velocity_model.train()

    for step in range(1, args.train_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = training_step(
            sdf_model,
            velocity_model,
            src_dense,
            tgt_dense,
            z_s,
            z_t,
            samples=min(args.samples, src_dense.shape[1]),
            knn=args.knn,
            tau=args.tau,
        )
        loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == args.train_steps:
            print(f"[train] step={step:04d} | loss={loss.item():.6f}")

    sdf_model.eval()
    velocity_model.eval()

    if not args.visualize:
        return

    t_values = torch.linspace(0.0, 1.0, args.vis_steps, device=device)
    teacher_pc, network_pc = gather_teacher_network_samples(
        sdf_model,
        velocity_model,
        src_dense,
        tgt_dense,
        z_s,
        z_t,
        num_samples=min(args.vis_samples, src_dense.shape[1]),
        knn=args.knn,
        tau=args.tau,
        t_values=t_values,
    )

    visualize_pointclouds(teacher_pc, network_pc)


if __name__ == "__main__":
    main()
