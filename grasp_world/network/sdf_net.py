import torch
import torch.nn as nn
import math


# -------------------------------------------------------
# Fourier feature encoder (same as in VelocityNet)
# -------------------------------------------------------
class FourierFeature(nn.Module):
    def __init__(self, in_dim=3, num_frequencies=6, scale=10.0):
        super().__init__()
        B = torch.randn((in_dim, num_frequencies)) * scale
        self.register_buffer("B", B)

    def forward(self, x):
        x_proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# -------------------------------------------------------
# SDFGuide: time-conditioned implicit field φ(x,t|z_s,z_t)
# -------------------------------------------------------
class SDFGuide(nn.Module):
    """
    Learns a continuous implicit SDF field for shape interpolation.
    Inputs:
        x:    [B,N,3]  spatial samples
        t:    [B,N,1]  time in [0,1]
        z_s:  [B,D]    source latent
        z_t:  [B,D]    target latent
    Returns:
        φ(x,t|z_s,z_t): [B,N,1]
    """

    def __init__(
        self,
        latent_dim=256,
        hidden_dim=128,
        n_layers=8,
        fourier_x=6,
        fourier_t=4,
        fourier_scale=10.0,
    ):
        super().__init__()
        self.fourier_x = FourierFeature(3, fourier_x, fourier_scale)
        self.fourier_t = FourierFeature(1, fourier_t, fourier_scale)

        in_dim = (3 + 1) + 2 * (fourier_x + fourier_t) + 2 * latent_dim
        layers = []
        last = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(last, hidden_dim), nn.SiLU()]
            last = hidden_dim
        layers += [nn.Linear(last, 1)]
        self.mlp = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, t, z_s, z_t):
        B, N, _ = x.shape
        x_flat = x.reshape(-1, 3)
        t_flat = t.reshape(-1, 1)
        x_enc = self.fourier_x(x_flat)
        t_enc = self.fourier_t(t_flat)
        z_s_expand = z_s[:, None, :].expand(-1, N, -1).reshape(B * N, -1)
        z_t_expand = z_t[:, None, :].expand(-1, N, -1).reshape(B * N, -1)

        inp = torch.cat([x_flat, t_flat, x_enc, t_enc, z_s_expand, z_t_expand], dim=-1)
        phi = self.mlp(inp)
        return phi.view(B, N, 1)


# -------------------------------------------------------
# Utility functions for geometric losses
# -------------------------------------------------------


def gradient_phi(phi, x):
    """Compute ∇φ wrt x"""
    grad = torch.autograd.grad(
        phi.sum(), x, create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    return grad


def time_derivative_phi(phi_net, x, t, z_s, z_t):
    """Compute ∂φ/∂t"""
    t.requires_grad_(True)
    phi = phi_net(x, t, z_s, z_t)
    dphi_dt = torch.autograd.grad(
        phi.sum(), t, create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    return dphi_dt


def eikonal_loss(grad_phi):
    """(‖∇φ‖ - 1)²"""
    return ((grad_phi.norm(dim=-1) - 1.0) ** 2).mean()


# Example sanity test
if __name__ == "__main__":
    B, N = 2, 512
    x = torch.randn(B, N, 3, requires_grad=True)
    t = torch.rand(B, N, 1)
    z_s = torch.randn(B, 256)
    z_t = torch.randn(B, 256)

    sdf = SDFGuide(latent_dim=256)
    phi = sdf(x, t, z_s, z_t)
    print("φ shape:", phi.shape)  # [B,N,1]
    grad = gradient_phi(phi, x)
    print("∇φ shape:", grad.shape)
