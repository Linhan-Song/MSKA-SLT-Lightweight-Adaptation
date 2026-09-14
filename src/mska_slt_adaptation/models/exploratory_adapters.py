"""Adapter variants retained for the exploratory comparisons in the dissertation."""

import torch
from torch import nn

from .common import masked_temporal_mean, sinusoidal_position_encoding


class ExpansionResidualAdapter(nn.Module):
    """The existing E2 residual mapper, separated from temporal pooling."""

    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, output_dim)
        self.input_norm = nn.LayerNorm(output_dim)
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
        )
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, features, lengths=None):
        del lengths
        x = self.input_norm(self.input_projection(features))
        return self.output_norm(x + self.mlp(x))


class GatedBottleneckAdapter(nn.Module):
    """A parameter-efficient residual adapter with a zero-initialized gate."""

    def __init__(self, input_dim, output_dim, bottleneck_dim=256, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, output_dim)
        self.pre_norm = nn.LayerNorm(output_dim)
        self.down = nn.Linear(output_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, output_dim)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, features, lengths=None):
        del lengths
        base = self.input_projection(features)
        delta = self.up(self.dropout(self.activation(self.down(self.pre_norm(base)))))
        return base + torch.tanh(self.gate) * delta


class TemporalGatedBottleneckAdapter(nn.Module):
    """Bottleneck adapter with lightweight temporal self-attention."""

    def __init__(self, input_dim, output_dim, bottleneck_dim=256, num_heads=4,
                 dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, output_dim)
        self.pre_norm = nn.LayerNorm(output_dim)
        self.down = nn.Linear(output_dim, bottleneck_dim)
        self.temporal_attention = nn.MultiheadAttention(
            bottleneck_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.temporal_norm = nn.LayerNorm(bottleneck_dim)
        self.ffn = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim * 2, bottleneck_dim),
        )
        self.ffn_norm = nn.LayerNorm(bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, features, lengths=None):
        base = self.input_projection(features)
        hidden = self.down(self.pre_norm(base))
        positions = sinusoidal_position_encoding(
            hidden.shape[1], hidden.shape[2], hidden.device, hidden.dtype
        )
        hidden_with_position = hidden + positions.unsqueeze(0)
        key_padding_mask = None
        if lengths is not None:
            key_padding_mask = (
                torch.arange(hidden.shape[1], device=lengths.device).unsqueeze(0)
                >= lengths.unsqueeze(1)
            )
        attended, _ = self.temporal_attention(
            hidden_with_position, hidden_with_position, hidden_with_position,
            key_padding_mask=key_padding_mask, need_weights=False,
        )
        hidden = self.temporal_norm(hidden + self.dropout(attended))
        hidden = self.ffn_norm(hidden + self.ffn(hidden))
        delta = self.up(hidden)
        return base + torch.tanh(self.gate) * delta


class SEGatedResidualAdapter(nn.Module):
    """Official-mapper residual adapter with sample-conditioned SE channel scaling."""

    def __init__(self, dimension=1024, bottleneck_dim=128, se_hidden_dim=64,
                 dropout=0.1, initial_gate=-4.0):
        super().__init__()
        self.pre_norm = nn.LayerNorm(dimension)
        self.down = nn.Linear(dimension, bottleneck_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, dimension)
        self.se_down = nn.Linear(dimension, se_hidden_dim)
        self.se_up = nn.Linear(se_hidden_dim, dimension)
        self.gate_logit = nn.Parameter(torch.tensor(float(initial_gate)))

        # With 2 * sigmoid(0) == 1, the SE branch starts as an identity scale.
        nn.init.zeros_(self.se_up.weight)
        nn.init.zeros_(self.se_up.bias)

    def forward(self, mapped_features, lengths=None):
        normalized = self.pre_norm(mapped_features)
        hidden = self.down(normalized)
        delta = self.up(self.dropout(self.activation(hidden)))
        context = masked_temporal_mean(normalized, lengths)
        channel_logits = self.se_up(self.activation(self.se_down(context)))
        channel_scale = 2.0 * torch.sigmoid(channel_logits).unsqueeze(1)
        return (
            mapped_features
            + torch.sigmoid(self.gate_logit) * channel_scale * delta
        )


class TemporalGatedResidualAdapter(nn.Module):
    """Gated correction with a lightweight local temporal convolution."""

    def __init__(self, dimension=1024, bottleneck_dim=256, dropout=0.1,
                 kernel_size=3, initial_gate=-4.0):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Temporal kernel size must be odd")
        self.pre_norm = nn.LayerNorm(dimension)
        self.down = nn.Linear(dimension, bottleneck_dim)
        self.temporal = nn.Conv1d(
            bottleneck_dim, bottleneck_dim, kernel_size,
            padding=kernel_size // 2, groups=bottleneck_dim,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, dimension)
        self.gate_logit = nn.Parameter(torch.tensor(float(initial_gate)))

    def forward(self, mapped_features, lengths=None):
        hidden = self.down(self.pre_norm(mapped_features))
        hidden = self.temporal(hidden.transpose(1, 2)).transpose(1, 2)
        delta = self.up(self.dropout(self.activation(hidden)))
        if lengths is not None:
            valid = (
                torch.arange(delta.shape[1], device=lengths.device).unsqueeze(0)
                < lengths.unsqueeze(1)
            ).unsqueeze(-1)
            delta = delta * valid.to(delta.dtype)
        return mapped_features + torch.sigmoid(self.gate_logit) * delta
