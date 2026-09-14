"""Residual adapters applied to the frozen VLMapper output."""

import torch
from torch import nn


class PlainResidualAdapter(nn.Module):
    """Add the bottleneck correction directly to the VLMapper output."""

    def __init__(self, dimension=1024, bottleneck_dim=256, dropout=0.1):
        super().__init__()
        self.pre_norm = nn.LayerNorm(dimension)
        self.down = nn.Linear(dimension, bottleneck_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, dimension)

    def forward(self, mapped_features, lengths=None):
        del lengths
        hidden = self.down(self.pre_norm(mapped_features))
        delta = self.up(self.dropout(self.activation(hidden)))
        return mapped_features + delta


class FixedResidualAdapter(PlainResidualAdapter):
    """Use the same small residual scale during training and inference."""

    def __init__(self, dimension=1024, bottleneck_dim=256, dropout=0.1,
                 initial_gate=-4.0):
        super().__init__(dimension, bottleneck_dim, dropout)
        scale = torch.sigmoid(torch.tensor(float(initial_gate)))
        self.register_buffer("residual_scale", scale)

    def forward(self, mapped_features, lengths=None):
        del lengths
        hidden = self.down(self.pre_norm(mapped_features))
        delta = self.up(self.dropout(self.activation(hidden)))
        return mapped_features + self.residual_scale * delta


class GatedResidualAdapter(nn.Module):
    """Learn how strongly the bottleneck correction changes VLMapper output."""

    def __init__(self, dimension=1024, bottleneck_dim=256, dropout=0.1,
                 initial_gate=-4.0):
        super().__init__()
        self.pre_norm = nn.LayerNorm(dimension)
        self.down = nn.Linear(dimension, bottleneck_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, dimension)
        self.gate_logit = nn.Parameter(torch.tensor(float(initial_gate)))

    def forward(self, mapped_features, lengths=None):
        del lengths
        hidden = self.down(self.pre_norm(mapped_features))
        delta = self.up(self.dropout(self.activation(hidden)))
        return mapped_features + torch.sigmoid(self.gate_logit) * delta
