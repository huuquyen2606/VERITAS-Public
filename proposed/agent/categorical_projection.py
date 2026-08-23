"""
C51 Categorical Distributional Projection for VERITAS.
"""

from __future__ import annotations

import torch


def compute_categorical_projection(
    next_probs: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    v_min: float = -500.0,
    v_max: float = 200.0,
    num_atoms: int = 51,
) -> torch.Tensor:
    """
    Projects the Bellman-shifted distribution back onto the fixed support.

    Args:
        next_probs: Target probabilities for greedy next-actions of shape (B, num_atoms).
        rewards: Immediate rewards of shape (B,).
        dones: Terminal flags of shape (B,).
        gamma: Discount factor.
        v_min: Minimum support value.
        v_max: Maximum support value.
        num_atoms: Number of distributional atoms.

    Returns:
        Projected target distribution of shape (B, num_atoms).
    """
    B, N = next_probs.shape
    assert N == num_atoms, f"next_probs has {N} atoms, expected {num_atoms}"

    device = next_probs.device
    dtype = next_probs.dtype

    delta_z = (v_max - v_min) / (num_atoms - 1)
    z = torch.linspace(v_min, v_max, num_atoms, device=device, dtype=dtype)

    dones_f = dones.float() if dones.dtype == torch.bool else dones
    Tz = (
        rewards.unsqueeze(1)
        + (1.0 - dones_f.unsqueeze(1))
        * gamma
        * z.unsqueeze(0)
    )

    Tz = Tz.clamp(v_min, v_max)

    bj = (Tz - v_min) / delta_z
    bj = bj.clamp(0, num_atoms - 1)

    l = bj.floor().long()
    u = bj.ceil().long()

    frac = bj - l.float()
    weight_l = next_probs * (1.0 - frac)
    weight_u = next_probs * frac

    target_dist = torch.zeros(B, N, device=device, dtype=dtype)
    target_dist.scatter_add_(1, l, weight_l)
    target_dist.scatter_add_(1, u, weight_u)

    return target_dist
