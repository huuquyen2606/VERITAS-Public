"""Multi-branch Distributional Dueling DQN Network for VERITAS."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedNoisyLinear(nn.Module):
    """
    Factorized Gaussian NoisyNet layer for exploration.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        sigma_init: Initial value for the standard deviation weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer("eps_in", torch.zeros(in_features))
        self.register_buffer("eps_out", torch.zeros(out_features))

        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.empty(out_features))
        self.sigma_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_bias = nn.Parameter(torch.empty(out_features))

        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(self.mu_weight, -bound, bound)
        nn.init.uniform_(self.mu_bias, -bound, bound)
        nn.init.constant_(self.sigma_weight, sigma_init / math.sqrt(in_features))
        nn.init.constant_(self.sigma_bias, sigma_init / math.sqrt(in_features))

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        """Samples fresh noise vectors."""
        eps_in = self._f(torch.randn(self.in_features, device=self.mu_weight.device))
        eps_out = self._f(torch.randn(self.out_features, device=self.mu_weight.device))
        self.eps_in.copy_(eps_in)
        self.eps_out.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.reset_noise()
            weight = self.mu_weight + self.sigma_weight * self.eps_out.unsqueeze(1) * self.eps_in.unsqueeze(0)
            bias = self.mu_bias + self.sigma_bias * self.eps_out
        else:
            weight = self.mu_weight
            bias = self.mu_bias
        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


def _ln_block(in_dim: int, out_dim: int) -> nn.Sequential:
    """Creates a sequential block: Linear -> LayerNorm -> ReLU."""
    linear_layer = nn.Linear(in_dim, out_dim)
    nn.init.kaiming_normal_(linear_layer.weight, mode='fan_in', nonlinearity='relu')
    nn.init.constant_(linear_layer.bias, 0.0)

    return nn.Sequential(
        linear_layer,
        nn.LayerNorm(out_dim),
        nn.ReLU(inplace=True),
    )


class MultiBranchDistDuelingDQN(nn.Module):
    """
    Multi-branch Distributional Dueling Double DQN architecture.

    Args:
        num_actions: Number of selectable actions.
        num_atoms: Number of distributional atoms.
        v_min: Minimum support value.
        v_max: Maximum support value.
        sigma_init: Initial NoisyLinear standard deviation.
    """

    DIM_STATIC: int = 2381
    DIM_DYN: int = 184
    DIM_CODE: int = 112
    DIM_ENV: int = 40
    DIM_FUSION: int = 256 + 128 + 64 + 32  
    DIM_BACKBONE_OUT: int = 256

    def __init__(
        self,
        num_actions: int = 19,
        num_atoms: int = 51,
        v_min: float = -500.0,
        v_max: float = 200.0,
        sigma_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms

        self.register_buffer(
            "z_values",
            torch.linspace(v_min, v_max, num_atoms),
        )

        self.static_encoder = nn.Sequential(
            _ln_block(self.DIM_STATIC, 1024),
            _ln_block(1024, 256),
        )

        self.dynamic_encoder = nn.Sequential(
            _ln_block(self.DIM_DYN, 256),
            _ln_block(256, 128),
        )

        self.code_encoder = nn.Sequential(
            _ln_block(self.DIM_CODE, 128),
            _ln_block(128, 64),
        )

        self.env_encoder = nn.Sequential(
            _ln_block(self.DIM_ENV, 64),
            _ln_block(64, 32),
        )

        self.backbone = nn.Sequential(
            _ln_block(self.DIM_FUSION, 512),
            _ln_block(512, self.DIM_BACKBONE_OUT),
        )

        self.value_stream = FactorizedNoisyLinear(
            self.DIM_BACKBONE_OUT, num_atoms, sigma_init=sigma_init,
        )

        self.advantage_stream = FactorizedNoisyLinear(
            self.DIM_BACKBONE_OUT, num_actions * num_atoms, sigma_init=sigma_init,
        )

    def forward(
        self,
        x_static: torch.Tensor,
        x_dyn: torch.Tensor,
        x_code: torch.Tensor,
        x_env: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes distributional Q-value log-probabilities.

        Args:
            x_static: Static features (B, 2381).
            x_dyn: Dynamic features (B, 184).
            x_code: Code features (B, 112).
            x_env: Environment features (B, 40).

        Returns:
            log_probs: Log-softmax over atoms of shape (B, num_actions, num_atoms).
        """
        B = x_static.size(0)
        A = self.num_actions
        N = self.num_atoms

        h_static = self.static_encoder(x_static)
        h_dyn = self.dynamic_encoder(x_dyn)
        h_code = self.code_encoder(x_code)
        h_env = self.env_encoder(x_env)

        fused = torch.cat([h_static, h_dyn, h_code, h_env], dim=1)
        backbone_out = self.backbone(fused)

        value = self.value_stream(backbone_out).view(B, 1, N)
        advantage = self.advantage_stream(backbone_out).view(B, A, N)

        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        log_probs = F.log_softmax(q_atoms, dim=-1)

        return log_probs

    def forward_flat(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adapts a flat [B, 2717] vector from the Environment into the four branches.

        Args:
            x: Flat observation tensor of shape (B, 2717).

        Returns:
            log_probs: Log-softmax over atoms of shape (B, num_actions, num_atoms).
        """
        assert x.shape[-1] == 2717, (
            f"Expected input dim 2717, got {x.shape[-1]}"
        )

        x_static = x[:, :2381]
        x_dyn    = x[:, 2381:2565]
        x_code   = x[:, 2565:2677]
        x_env    = x[:, 2677:]

        return self.forward(x_static, x_dyn, x_code, x_env)

    def q_values(self, x_flat: torch.Tensor) -> torch.Tensor:
        """
        Computes expected Q-values from a flat observation tensor.

        Args:
            x_flat: Flat observation of shape (B, 2717).

        Returns:
            q: Expected Q-values of shape (B, num_actions).
        """
        log_probs = self.forward_flat(x_flat)
        probs = log_probs.exp()
        q = (probs * self.z_values.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        return q

    def reset_noise(self) -> None:
        """Re-samples noise in every NoisyLinear layer."""
        for module in self.modules():
            if isinstance(module, FactorizedNoisyLinear):
                module.reset_noise()


