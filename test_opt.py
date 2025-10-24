import torch
import torch.nn.functional as F
from dataclasses import dataclass
from grasp_world.robo.handmodel import DexHand, HANDS
import viser
import numpy as np


@dataclass
class OptResult:
    q: torch.Tensor
    success: bool
    iters: int
    final_penetration: torch.Tensor  # (B,)
    history: dict


def visualize_before_after(hand: DexHand, q_before: torch.Tensor, q_after: torch.Tensor):
    """
    Visualize before and after self-collision optimization directly in viser.
    Cyan = before, Magenta = after.
    """
    # Compute world spheres
    c_before, r_before = hand.compute_world_spheres(q_before, "np")
    c_after, r_after = hand.compute_world_spheres(q_after, "np")

    # Start viser
    vis = viser.ViserServer()
    vis.scene.reset()

    # Unpack single batch (assuming q is shape [1, n])
    c0, r0 = c_before[0], r_before[0]
    c1, r1 = c_after[0], r_after[0]

    # Plot before (cyan)
    for j in range(len(c0)):
        vis.scene.add_icosphere(
            position=tuple(c0[j]),
            radius=float(r0[j]),
            color=(0.2, 0.7, 1.0),  # cyan
            opacity=0.4,
            name=f"before_{j}",
        )

    # Plot after (magenta)
    for j in range(len(c1)):
        vis.scene.add_icosphere(
            position=tuple(c1[j]),
            radius=float(r1[j]),
            color=(1.0, 0.3, 0.8),  # magenta
            opacity=0.4,
            name=f"after_{j}",
        )

    print("✅ viser visualization started.")
    print("Cyan = before optimization, Magenta = after optimization.")
    vis.sleep_forever()
    return vis


def optimize_self_collision_free(
    hand: DexHand,
    q_init: torch.Tensor,
    *,
    max_iters: int = 500,
    lr: float = 5e-3,
    tol: float = 1e-2,  # stop when max(batch_penetration) < tol
    reg_lambda: float = 1e-1,  # L2 trust-region to stay near q_init
    grad_clip: float | None = 1.0,  # set None to disable
    joint_lower: torch.Tensor | None = None,  # shape (N,) or (B,N)
    joint_upper: torch.Tensor | None = None,  # shape (N,) or (B,N)
    limits_barrier: float = 1e-3,  # soft barrier strength; 0 to disable
    debug_every: int = 0,  # e.g., 50 to print every 50 iters
    device: torch.device | None = None,
) -> OptResult:
    """
    Minimizes self-penetration from q_init with optional limits.
    Works with batched q_init (B,N) or unbatched (N,).
    """
    # Prepare shapes / device
    q0 = q_init.detach()
    if q0.ndim == 1:
        q0 = q0.unsqueeze(0)  # (1,N)
    B, N = q0.shape
    dev = device or q0.device

    q = q0.clone().to(dev).requires_grad_(True)
    opt = torch.optim.Adam([q], lr=lr)

    hist_pen = []
    hist_total = []

    def project_to_limits():
        if joint_lower is None or joint_upper is None:
            return
        lo = joint_lower
        hi = joint_upper
        if lo.ndim == 1:
            lo = lo.unsqueeze(0).expand_as(q)
        if hi.ndim == 1:
            hi = hi.unsqueeze(0).expand_as(q)
        with torch.no_grad():
            q.data.clamp_(lo, hi)

    # Initial projection (optional)
    project_to_limits()

    for it in range(1, max_iters + 1):
        opt.zero_grad(set_to_none=True)

        # Collision loss (returns (B,))
        pen = hand.calculate_self_penetration(q, debug=False)  # (B,)
        pen_mean = pen.mean()

        # Trust-region regularizer to avoid wild moves
        reg = (q - q0).pow(2).mean()

        # Optional: soft barrier near/outside limits (keeps gradients informative)
        barrier = torch.tensor(0.0, device=dev)
        if limits_barrier > 0 and joint_lower is not None and joint_upper is not None:
            lo = joint_lower
            hi = joint_upper
            if lo.ndim == 1:
                lo = lo.unsqueeze(0).expand_as(q)
            if hi.ndim == 1:
                hi = hi.unsqueeze(0).expand_as(q)
            # Penalize violations smoothly; relu gives zero inside range
            barrier_lo = F.softplus(lo - q, beta=50.0)  # sharp-ish
            barrier_hi = F.softplus(q - hi, beta=50.0)
            barrier = (barrier_lo + barrier_hi).mean()

        loss = pen_mean + reg_lambda * reg + limits_barrier * barrier

        loss.backward()

        # Optional gradient clipping for stability
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_([q], grad_clip)

        opt.step()

        # Hard project back into limits post-step
        project_to_limits()

        # Logging
        hist_pen.append(pen.detach().max().item())
        hist_total.append(loss.detach().item())

        if debug_every and it % debug_every == 0:
            print(f"[iter {it:04d}] max_pen={hist_pen[-1]:.3e}  loss={hist_total[-1]:.3e}")

        # Early stop if all in batch are (near) collision-free
        if pen.detach().max().item() < tol:
            if debug_every:
                print(f"Converged at iter {it} with max_pen={pen.detach().max().item():.3e}")
            vis = visualize_before_after(hand, q_init, q)
            return OptResult(
                q=q.detach() if q_init.ndim == 2 else q.detach().squeeze(0),
                success=True,
                iters=it,
                final_penetration=pen.detach(),
                history={"max_penetration": hist_pen, "loss": hist_total},
            )

    # Not fully converged
    final_pen = hand.calculate_self_penetration(q, debug=False).detach()
    return OptResult(
        q=q.detach() if q_init.ndim == 2 else q.detach().squeeze(0),
        success=final_pen.max().item() < tol,
        iters=max_iters,
        final_penetration=final_pen,
        history={"max_penetration": hist_pen, "loss": hist_total},
    )


# if __name__ == "__main__":
#     hand = DexHand(HANDS["shadow_hand"])
#     q_init = torch.zeros((1, len(hand.robot.get_joints())))
#     q_init[:, 6] = -0.35
#     result = optimize_self_collision_free(hand, q_init)
#     print(result)
