from __future__ import annotations

import torch
from torch import nn


class FCNRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.05) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)