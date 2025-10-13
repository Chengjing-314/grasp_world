from __future__ import annotations

import math
import torch
import torch.nn as nn


class FourierFeature(nn.Module):
    """
    Positional encoding for coordinates (x or t).
    gamma(x) = [sin(2πBx), cos(2πBx)]
    """

    def __init__(self, in_dim: int = 3, num_frequencies: int = 6, scale: float = 10.0) -> None:
        super().__init__()
        B = torch.randn((in_dim, num_frequencies)) * scale
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_dim]
        x_proj = 2 * math.pi * x @ self.B  # [..., num_freq]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 8,
    ) -> None:
        super().__init__()
        layers = []
        last = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(last, hidden_dim), nn.SiLU()]
            last = hidden_dim
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------------------------------------
# VelocityNet: shared motion field
# -------------------------------------------------------
class VelocityNet(nn.Module):
    """
    Neural velocity field v(x, t, sdf_t | z_s, z_t)
    - Works for both flow-matching and 4Deform-style geometry losses.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 128,
        n_layers: int = 8,
        fourier_x: int = 6,
        fourier_t: int = 4,
        fourier_sdf: int = 4,
        fourier_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.fourier_x = FourierFeature(3, fourier_x, fourier_scale)
        self.fourier_t = FourierFeature(1, fourier_t, fourier_scale)
        self.fourier_sdf = FourierFeature(1, fourier_sdf, fourier_scale)

        # raw features: xyz (3) + time (1) + sdf scalar (1)
        # fourier features: 2 per frequency band for each encoding
        in_dim = (3 + 1 + 1) + 2 * (fourier_x + fourier_t + fourier_sdf) + 2 * latent_dim
        self.mlp = MLP(in_dim, 3, hidden_dim, n_layers)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        sdf_t: torch.Tensor,
        z_s: torch.Tensor,
        z_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Sample coordinates at the current timestep with shape [B, N, 3].
            t: Blend time for each sample, shape [B, N, 1] (or broadcastable to it).
            sdf_t: Signed distance values of the object at the current timestep, shape [B, N, 1].
            z_s: Source latent code from PointNet2Encoder, shape [B, latent_dim].
            z_t: Target latent code from PointNet2Encoder, shape [B, latent_dim].

        returns v: [B, N, 3]
        """
        B, N, _ = x.shape
        # encode
        x_flat = x.reshape(-1, 3)
        t_flat = t.reshape(-1, 1)
        sdf_flat = sdf_t.reshape(-1, 1)

        x_enc = self.fourier_x(x_flat)
        t_enc = self.fourier_t(t_flat)
        sdf_enc = self.fourier_sdf(sdf_flat)

        z_s_expand = z_s[:, None, :].expand(-1, N, -1).reshape(B * N, -1)
        z_t_expand = z_t[:, None, :].expand(-1, N, -1).reshape(B * N, -1)

        inp = torch.cat(
            [x_flat, t_flat, sdf_flat, x_enc, t_enc, sdf_enc, z_s_expand, z_t_expand],
            dim=-1,
        )
        v = self.mlp(inp)
        return v.view(B, N, 3)


# -------------------------------------------------------
# Jacobian utilities (for losses later)
# -------------------------------------------------------
def jacobian(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute dy/dx for y:[B,3] and x:[B,3] -> [B,3,3]."""
    B, D = y.shape
    J = []
    for i in range(D):
        grad = torch.autograd.grad(y[:, i].sum(), x, create_graph=True)[0]
        J.append(grad)
    return torch.stack(J, dim=1)  # [B,3,3]


def divergence(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute ∇·v for v:[B,3]"""
    div = 0.0
    for i in range(3):
        grad = torch.autograd.grad(v[:, i].sum(), x, create_graph=True)[0][:, i]
        div += grad
    return div.unsqueeze(-1)


# Example sanity test
if __name__ == "__main__":
    B, N = 2, 64
    x = torch.randn(B, N, 3, requires_grad=True)
    t = torch.rand(B, N, 1)
    sdf = torch.randn(B, N, 1)
    z_s = torch.randn(B, 256)
    z_t = torch.randn(B, 256)

    model = VelocityNet(latent_dim=256)
    v = model(x, t, sdf, z_s, z_t)
    print("Velocity shape:", v.shape)  # [B,N,3]
