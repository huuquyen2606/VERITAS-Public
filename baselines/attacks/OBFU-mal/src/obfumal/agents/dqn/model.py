from typing import List, Optional

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Optional[List[int]] = None):
        super(QNetwork, self).__init__()

        if hidden_dims is None:
            hidden_dims = [1024, 256]

        layers = []
        prev = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            prev = dim
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
