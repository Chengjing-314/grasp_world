from typing import Literal, Optional, Union
import numpy as np
import torch
from jaxtyping import Float, Array


def rotation_from_facing(
    facing: np.ndarray | torch.Tensor,
    up: np.ndarray | torch.Tensor = np.array([0.0, 0.0, 1.0]),
    return_type: Literal["np", "pt"] = "pt",
):
    # Normalize input
    f = facing / np.linalg.norm(facing)
    u = up / np.linalg.norm(up)
    # Make orthogonal
    if abs(np.dot(f, u)) > 0.999:  # handle parallel case
        u = np.array([0.0, 1.0, 0.0])
    right = np.cross(u, f)
    right /= np.linalg.norm(right)
    up = np.cross(f, right)
    R = np.stack([right, up, f], axis=1)
    if return_type == "pt":
        return torch.from_numpy(R)
    return R


def rotation_to_tf(
    R: Optional[Union[Float[np.ndarray, "B 3 3"], Float[torch.Tensor, "B 3 3"]]] = None,
    t: Optional[Union[Float[np.ndarray, "B 3"], Float[torch.Tensor, "B 3"]]] = None,
    return_type: Literal["np", "pt"] = "pt",
) -> Union[
    Float[torch.Tensor, "B 4 4"],
    Float[np.ndarray, "B 4 4"],
]:
    """
    Construct or convert homogeneous transform(s) from rotation and translation.

    Args:
        R: (B, 3, 3) or (3, 3) rotation matrix/matrices, or None.
        t: (B, 3) or (3,) translation vector(s), or None.
        return_type: "pt" for torch.Tensor, "np" for numpy.ndarray.

    Returns:
        (B, 4, 4) homogeneous transform(s) in the same or requested backend.

    Notes:
        - If only R is provided → translation = 0.
        - If only t is provided → rotation = identity.
        - R and t must have the same batch size if both are provided.
        - No broadcasting is performed.
    """
    # Determine backend
    backend = "pt" if (isinstance(R, torch.Tensor) or isinstance(t, torch.Tensor)) else "np"

    def ensure_batch(x, target_dim):
        if x is None:
            return None
        if x.ndim == target_dim - 1:
            x = x[None, ...]
        return x

    R = ensure_batch(R, 3)
    t = ensure_batch(t, 2)

    # Determine batch size
    if R is not None and t is not None:
        assert R.shape[0] == t.shape[0], f"Batch mismatch: R {R.shape}, t {t.shape}"
        B = R.shape[0]
    elif R is not None:
        B = R.shape[0]
    elif t is not None:
        B = t.shape[0]
    else:
        raise ValueError("At least one of R or t must be provided")

    if backend == "pt":
        if R is None:
            R = torch.eye(3, device=t.device, dtype=t.dtype).repeat(B, 1, 1)
        if t is None:
            t = torch.zeros((B, 3), device=R.device, dtype=R.dtype)
        tf = torch.eye(4, device=R.device, dtype=R.dtype).repeat(B, 1, 1)
        tf[:, :3, :3] = R
        tf[:, :3, 3] = t
        if return_type == "np":
            tf = tf.cpu().numpy()
    else:
        if R is None:
            R = np.tile(np.eye(3)[None, :, :], (B, 1, 1))
        if t is None:
            t = np.zeros((B, 3))
        tf = np.tile(np.eye(4)[None, :, :], (B, 1, 1))
        tf[:, :3, :3] = R
        tf[:, :3, 3] = t
        if return_type == "pt":
            tf = torch.from_numpy(tf)

    if tf.shape[0] == 1:
        tf = tf[0]

    return tf
