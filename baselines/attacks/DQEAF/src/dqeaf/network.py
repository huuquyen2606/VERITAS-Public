from __future__ import annotations

import torch
from torch import nn


class DQEAFNet(nn.Module):
    """Convolutional Q-network from paper Section III-F."""

    def __init__(self, input_dim: int = 513, action_dim: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, 256, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(64 * input_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected input shape [B, 513], got {tuple(x.shape)}")
        x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = x.flatten(start_dim=1)
        return self.head(x)
