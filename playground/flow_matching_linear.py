"""Very small flow-matching demo with a learnable linear-blend flow network."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from grasp_world.utils.sdf import (
    compute_common_axis,
    extract_isosurface_mesh,
    sample_mesh_sdfs_on_axis,
)

MESH_DIR = Path("assets/objects/primitive_shapes")
MESHES = {
    "sphere": MESH_DIR / "sphere.obj",
    "cube": MESH_DIR / "cube.obj",
    "cylinder": MESH_DIR / "cylinder.obj",
    "cone": MESH_DIR / "cone.obj",
    "torus": MESH_DIR / "torus.obj",
}


class LinearBlendFlowNet(nn.Module):
    """A tiny flow-matching network that learns the interpolation weight w(t).

    The network takes the blend time `t` as input and predicts a scalar weight `w` in (0, 1).
    The intermediate SDF and velocity field are computed as

        phi_t = (1 - w) * phi_src + w * phi_tgt
        velocity = dw/dt * (phi_tgt - phi_src)

    Because w(t) is produced by a differentiable MLP, gradients back-propagate through both
    phi_t and velocity, enabling standard flow-matching training objectives.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        phi_src: torch.Tensor,
        phi_tgt: torch.Tensor,
        t: float | torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if phi_src.shape != phi_tgt.shape:
            raise ValueError("Source and target SDF volumes must have matching shapes.")

        if not torch.is_tensor(t):
            t_tensor = torch.tensor([float(t)], dtype=phi_src.dtype, device=phi_src.device)
        else:
            t_tensor = t.to(dtype=phi_src.dtype, device=phi_src.device).reshape(-1, 1)

        if t_tensor.shape[1] != 1:
            raise ValueError("Blend time tensor must be shape (N, 1) or scalar.")

        t_tensor = torch.clamp(t_tensor, 0.0, 1.0)
        t_tensor.requires_grad_(True)

        weight = torch.sigmoid(self.mlp(t_tensor))  # (N, 1)
        # dw/dt needed for velocity; create_graph allows higher-order gradients during training.
        (dw_dt,) = torch.autograd.grad(
            outputs=weight,
            inputs=t_tensor,
            grad_outputs=torch.ones_like(weight),
            create_graph=True,
        )

        w = weight.reshape(weight.shape[0], 1, 1, 1, 1)
        dw_dt = dw_dt.reshape(dw_dt.shape[0], 1, 1, 1, 1)

        phi_src = phi_src.unsqueeze(0).unsqueeze(0)
        phi_tgt = phi_tgt.unsqueeze(0).unsqueeze(0)
        phi_t = (1.0 - w) * phi_src + w * phi_tgt
        velocity = dw_dt * (phi_tgt - phi_src)

        return phi_t.squeeze(1), velocity.squeeze(1), weight.squeeze(1)

    def integrate(self, phi_src: torch.Tensor, phi_tgt: torch.Tensor, t_values: Iterable[float]):
        """Generate intermediate fields; keeps graph for potential gradient-based objectives."""
        for t in t_values:
            phi_t, velocity, weight = self(phi_src, phi_tgt, t)
            yield float(torch.as_tensor(t)), phi_t[0], velocity[0], weight[0].detach()


def _prepare_volumes(
    src_mesh: Path,
    tgt_mesh: Path,
    resolution: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    axis = compute_common_axis([src_mesh, tgt_mesh], resolution)
    sdf_volumes = sample_mesh_sdfs_on_axis({"src": src_mesh, "tgt": tgt_mesh}, axis, device=device)
    return axis, sdf_volumes["src"], sdf_volumes["tgt"]


def _make_mesh_sequence(
    phi_src: torch.Tensor,
    phi_tgt: torch.Tensor,
    axis: torch.Tensor,
    steps: int,
) -> list[Tuple[float, torch.Tensor, torch.Tensor]]:
    """Use the linear flow to build a list of meshes for inspection or export."""
    model = LinearBlendFlowNet()
    meshes = []
    for t, phi_t, _, _ in model.integrate(phi_src, phi_tgt, torch.linspace(0.0, 1.0, steps)):
        verts, faces = extract_isosurface_mesh(phi_t, axis, level=0.0, init_res=len(axis))
        meshes.append((t, verts, faces))
    return meshes


def visualize_with_viser(
    axis: torch.Tensor,
    phi_src: torch.Tensor,
    phi_tgt: torch.Tensor,
    model: LinearBlendFlowNet,
    t_values: torch.Tensor,
):
    import numpy as np
    import viser

    server = viser.ViserServer()
    server.scene.world_axes.visible = True
    server.scene.add_grid(
        "/grid", width=10.0, height=10.0, position=(0.0, 0.0, float(axis[0]) - 0.1)
    )

    t_values = t_values.to(dtype=phi_src.dtype, device=phi_src.device)

    # Helper to extract mesh as NumPy + bbox size for spacing
    def _mesh_and_span(field: torch.Tensor):
        verts_t, faces_t = extract_isosurface_mesh(field, axis, level=0.0, init_res=len(axis))
        # === critical: convert to numpy ===
        if isinstance(verts_t, torch.Tensor):
            verts = verts_t.detach().cpu().numpy().astype(np.float32)
        else:
            verts = np.asarray(verts_t, dtype=np.float32)
        if isinstance(faces_t, torch.Tensor):
            faces = faces_t.detach().cpu().numpy().astype(np.int32)
        else:
            faces = np.asarray(faces_t, dtype=np.int32)
        # bbox span to drive spacing
        span = float(np.max(verts.max(axis=0) - verts.min(axis=0)))
        return verts, faces, span if span > 1e-6 else 1.0

    # Precompute a representative span for consistent spacing
    with torch.no_grad():
        verts0, faces0, span0 = _mesh_and_span(((1.0 - 0.5) * phi_src + 0.5 * phi_tgt).detach())
    offset_x = span0 * 1.6
    offset_y = span0 * 1.2

    for idx, t_tensor in enumerate(t_values):
        t_scalar = float(t_tensor.item())
        phi_pred, _, weight = model(phi_src, phi_tgt, t_tensor)
        phi_pred = phi_pred[0].detach()
        phi_gt = ((1.0 - t_scalar) * phi_src + t_scalar * phi_tgt).detach()
        weight_scalar = float(weight.detach().item())

        # Parent frame for this column; children are positioned relative to it
        col_name = f"/col_{idx:02d}"
        server.scene.add_frame(
            col_name,
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(offset_x * (idx - (len(t_values) - 1) / 2.0), 0.0, 0.0),
        )

        for row, (label, field, y_off) in enumerate(
            [("Ground Truth", phi_gt, 0.0), ("Network", phi_pred, offset_y)]
        ):
            verts, faces, _ = _mesh_and_span(field)
            node = f"{col_name}/{label.replace(' ', '_')}/mesh"
            server.scene.add_mesh_simple(
                node,
                vertices=verts,
                faces=faces,
                # position is now relative to the column frame → only y shifts here
                position=(0.0, y_off, 0.0),
                flat_shading=True,
                side="double",
            )
            server.scene.add_label(
                f"{col_name}/{label.replace(' ', '_')}/label",
                text=f"{label} | t={t_scalar:.2f} | w={weight_scalar:.3f}",
                position=(0.0, y_off, float(axis[-1]) + 0.2),
            )

    print("Viser server running; press Ctrl+C to exit.")
    import time

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


def main():
    parser = argparse.ArgumentParser(description="Linear flow-matching SDF blending demo.")
    parser.add_argument("--src", type=str, default="sphere", choices=MESHES.keys())
    parser.add_argument("--tgt", type=str, default="torus", choices=MESHES.keys())
    parser.add_argument("--resolution", type=int, default=80)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device string understood by grasp_world.utils.torch_utils.resolve_device.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional directory to save intermediate meshes as OBJ files.",
    )
    parser.add_argument("--train-steps", type=int, default=0, help="Optional supervised steps.")
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for optional training."
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Launch a Viser viewer showing ground-truth vs network blends after training.",
    )
    args = parser.parse_args()

    axis, phi_src, phi_tgt = _prepare_volumes(
        MESHES[args.src], MESHES[args.tgt], args.resolution, args.device
    )
    model = LinearBlendFlowNet()

    if args.train_steps > 0:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        phi_src_exp = phi_src.unsqueeze(0)
        phi_tgt_exp = phi_tgt.unsqueeze(0)
        for step in range(args.train_steps):
            optimizer.zero_grad()
            t_batch = torch.rand(8, 1, dtype=phi_src.dtype, device=phi_src.device)
            phi_pred, vel_pred, weights = model(phi_src, phi_tgt, t_batch)

            # Closed-form linear blend target for supervision.
            phi_gt = (1.0 - t_batch.view(-1, 1, 1, 1)) * phi_src_exp + t_batch.view(
                -1, 1, 1, 1
            ) * phi_tgt_exp
            vel_gt = (phi_tgt_exp - phi_src_exp).expand_as(phi_gt)

            loss_phi = F.mse_loss(phi_pred, phi_gt)
            loss_vel = F.mse_loss(vel_pred, vel_gt)
            loss = loss_phi + loss_vel
            loss.backward()
            optimizer.step()

            if (step + 1) % max(1, args.train_steps // 5) == 0 or step == args.train_steps - 1:
                w_mean = weights.mean().detach().item()
                print(
                    f"[train] step={step + 1:04d} | loss={loss.item():.6f} "
                    f"| loss_phi={loss_phi.item():.6f} | loss_vel={loss_vel.item():.6f} | w_mean={w_mean:.3f}"
                )

    model.eval()
    t_values = torch.linspace(0.0, 1.0, args.steps)
    print(f"Interpolating {args.src} -> {args.tgt} with {len(t_values)} steps.")
    for t, phi, velocity, weight in model.integrate(phi_src, phi_tgt, t_values):
        diff = torch.linalg.norm(velocity).item()
        print(f"t={t:.3f} | velocity L2 = {diff:.5f} | w(t)={weight.item():.3f}")

    if args.save is not None:
        args.save.mkdir(parents=True, exist_ok=True)
        for idx, (t, verts, faces) in enumerate(
            _make_mesh_sequence(phi_src, phi_tgt, axis, args.steps)
        ):
            out_path = args.save / f"blend_{idx:03d}_t{t:.2f}.obj"
            # Faces need to be 1-indexed for OBJ format.
            with open(out_path, "w", encoding="ascii") as fp:
                for v in verts.tolist():
                    fp.write(f"v {v[0]} {v[1]} {v[2]}\n")
                for face in faces.tolist():
                    i, j, k = (f + 1 for f in face)
                    fp.write(f"f {i} {j} {k}\n")
            print(f"Saved {out_path}")

    if args.visualize:
        visualize_with_viser(axis, phi_src, phi_tgt, model, t_values)


if __name__ == "__main__":
    main()
