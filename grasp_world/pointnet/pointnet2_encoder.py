from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnet2 import get_model

_DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[2] / "assets" / "checkpoints" / "pointnet2" / "best_model.pth"
)


class PointNet2Encoder(nn.Module):
    """
    A frozen wrapper around pretrained PointNet++ for global latent extraction.
    """

    def __init__(
        self,
        device: torch.device,
        checkpoint_path: str | Path | None = None,
        normal_channel: bool = False,
        num_class: int = 40,
    ):
        super().__init__()
        self.backbone = get_model(num_class=num_class, normal_channel=normal_channel)

        # load pretrained weights if provided
        if checkpoint_path is None:
            checkpoint_path = _DEFAULT_CHECKPOINT

        checkpoint_path = Path(checkpoint_path)

        if checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location=device, weights_only=False)
            self.backbone.load_state_dict(state, strict=False)
            print(f"[✓] Loaded pretrained PointNet++ from {checkpoint_path}")
        else:
            print(
                f"[!] PointNet++ checkpoint not found at {checkpoint_path}. Starting uninitialized."
            )

        # freeze all parameters
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    @torch.no_grad()
    def forward(self, xyz):
        """
        xyz: [B, C, N] where C=3 or 6 (with normals)
        returns: normalized global latent [B, 1024]
        """
        # run through PointNet++
        _, l3_points = self.backbone(xyz)
        z = l3_points.view(l3_points.size(0), -1)  # [B, 1024]
        z = F.normalize(z, dim=-1)
        return z
