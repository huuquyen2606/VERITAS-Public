"""VERITAS Agent Orchestrator."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from agent.agent_network import MultiBranchDistDuelingDQN
    from agent.categorical_projection import compute_categorical_projection
    from agent.distributional_loss import compute_distributional_loss
except ImportError:
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from agent.agent_network import MultiBranchDistDuelingDQN
    from agent.categorical_projection import compute_categorical_projection
    from agent.distributional_loss import compute_distributional_loss


class VERITASAgent:
    """
    Multi-branch Distributional Dueling Double DQN Agent with NoisyNet.

    Args:
        num_actions: Size of the agent's action space (default 19).
        num_atoms: Number of distributional atoms for C51 (default 51).
        v_min: Minimum support value for the distribution (default -500.0).
        v_max: Maximum support value for the distribution (default 200.0).
        sigma_init: Initial standard deviation for NoisyLinear layers (default 0.5).
        gamma: Discount factor (default 0.99).
        lr: Learning rate for the Adam optimizer (default 1e-4).
        adam_eps: Epsilon parameter for the Adam optimizer (default 1.5e-4).
        grad_clip: Maximum gradient norm for clipping, disabled if 0 (default 10.0).
        device: Hardware device to run the networks on, "cuda" or "cpu".
    """

    def __init__(
        self,
        num_actions: int = 19,
        num_atoms: int = 51,
        v_min: float = -500.0,
        v_max: float = 200.0,
        sigma_init: float = 0.5,
        gamma: float = 0.99,
        lr: float = 1e-4,
        adam_eps: float = 1.5e-4,
        grad_clip: float = 10.0,
        device: str | torch.device = "cpu",
    ) -> None:
        self.gamma = gamma
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms
        self.num_actions = num_actions
        self.grad_clip = grad_clip
        self.device = torch.device(device)

        net_kwargs = dict(
            num_actions=num_actions,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            sigma_init=sigma_init,
        )
        self.policy_net = MultiBranchDistDuelingDQN(**net_kwargs).to(self.device)
        self.target_net = MultiBranchDistDuelingDQN(**net_kwargs).to(self.device)

        self.sync_target_network()
        self.target_net.eval()
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=lr,
            eps=adam_eps,
        )

    @torch.no_grad()
    def select_action(
        self,
        state_flat: np.ndarray | torch.Tensor,
        eval_mode: bool = False,
    ) -> int:
        """
        Selects a discrete action given a flat observation.

        Args:
            state_flat: Flat observation array or tensor with shape (2717,).
            eval_mode: If True, sets network to evaluation mode (NoisyNet disabled).

        Returns:
            Selected integer action index.
        """
        if isinstance(state_flat, np.ndarray):
            state_flat = torch.from_numpy(state_flat).float()
        state_flat = state_flat.to(self.device).unsqueeze(0)

        if eval_mode:
            self.policy_net.eval()
        else:
            self.policy_net.train()

        q = self.policy_net.q_values(state_flat)
        action = q.argmax(dim=-1).item()

        return action

    def train_step(
        self,
        batch_data: Tuple[
            torch.Tensor,   
            torch.Tensor,   
            torch.Tensor,   
            torch.Tensor,   
            torch.Tensor,   
            torch.Tensor,   
        ],
    ) -> Tuple[float, np.ndarray]:
        """
        Executes one gradient update step using a mini-batch of PER data.

        Args:
            batch_data: Tuple containing (states, actions, rewards, next_states, 
                dones, per_weights).

        Returns:
            loss_value: Calculated scalar float loss.
            td_errors: Numpy array of shape (B,) containing the per-sample TD-errors.
        """
        states, actions, rewards, next_states, dones, per_weights = batch_data

        states      = states.to(self.device)
        actions     = actions.to(self.device).long()
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device).float()
        per_weights = per_weights.to(self.device)

        self.policy_net.train()

        with torch.no_grad():
            next_log_probs_policy = self.policy_net.forward_flat(next_states)
            next_probs_policy = next_log_probs_policy.exp()
            next_q_policy = (
                next_probs_policy * self.policy_net.z_values.unsqueeze(0).unsqueeze(0)
            ).sum(dim=-1)
            next_actions = next_q_policy.argmax(dim=-1)

            next_log_probs_target = self.target_net.forward_flat(next_states)
            B = next_states.size(0)
            next_probs = next_log_probs_target[
                torch.arange(B, device=self.device), next_actions
            ].exp()

            target_dist = compute_categorical_projection(
                next_probs=next_probs,
                rewards=rewards,
                dones=dones,
                gamma=self.gamma,
                v_min=self.v_min,
                v_max=self.v_max,
                num_atoms=self.num_atoms,
            )

        current_log_probs = self.policy_net.forward_flat(states)

        loss, td_errors = compute_distributional_loss(
            log_probs=current_log_probs,
            target_dist=target_dist,
            actions=actions,
            per_weights=per_weights,
        )

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()

        return loss.item(), td_errors.cpu().numpy()

    def sync_target_network(self) -> None:
        """Performs a hard copy of the policy network weights into the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())


