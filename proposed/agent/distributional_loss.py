"""
Distributional Loss and PER Priority Update for VERITAS.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_distributional_loss(
    log_probs: torch.Tensor,
    target_dist: torch.Tensor,
    actions: torch.Tensor,
    per_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the categorical cross-entropy loss with PER weighting.

    Args:
        log_probs: Log-probabilities over atoms of shape (B, num_actions, num_atoms).
        target_dist: Projected target distribution of shape (B, num_atoms).
        actions: Sampled actions of shape (B,).
        per_weights: Importance-sampling weights of shape (B,).

    Returns:
        mean_loss: Weighted scalar loss.
        td_errors_for_per: Per-sample TD-errors of shape (B,).
    """
    B = log_probs.size(0)

    chosen_log_probs = log_probs[
        torch.arange(B, device=log_probs.device), actions
    ]

    eltwise_loss = -target_dist * chosen_log_probs

    td_errors = eltwise_loss.sum(dim=-1)
    td_errors_for_per = td_errors.detach()

    mean_loss = (td_errors * per_weights).sum() / B

    return mean_loss, td_errors_for_per

