"""Temporal refinement before VLMapper."""

import torch
from torch import nn

from .common import sinusoidal_position_encoding


class TemporalRefiner(nn.Module):
    """Refine the variable-length Recognition sequence before VLMapper.

    Local, global and local-global attention use the same projections and FFN.
    Only the permitted temporal connections change between the three variants.
    A small initial gate keeps the output close to the frozen Recognition
    features at the beginning of training.
    """

    def __init__(self, feature_dim=512, attention_dim=128, num_heads=4,
                 ffn_dim=256, num_layers=1, dropout=0.1, initial_gate=-4.0,
                 attention_mode="global", local_radius=8, global_stride=4,
                 zero_output_projection=False):
        super().__init__()
        if attention_dim % num_heads != 0:
            raise ValueError("attention_dim must be divisible by num_heads")
        if num_layers < 1:
            raise ValueError("num_layers must be at least one")
        self.attention_mode = str(attention_mode).lower()
        if self.attention_mode not in {"global", "local", "local_global"}:
            raise ValueError(f"Unknown gloss attention mode: {attention_mode}")
        if local_radius < 1 or global_stride < 1:
            raise ValueError("local_radius and global_stride must be positive")
        self.local_radius = int(local_radius)
        self.global_stride = int(global_stride)
        self.input_norm = nn.LayerNorm(feature_dim)
        self.input_projection = nn.Linear(feature_dim, attention_dim)
        # All reported experiments use scaled sinusoidal absolute positions.
        self.position_scale = nn.Parameter(torch.ones(1))
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                "attention_norm": nn.LayerNorm(attention_dim),
                "attention": nn.MultiheadAttention(
                    attention_dim, num_heads, dropout=dropout, batch_first=True
                ),
                "ffn_norm": nn.LayerNorm(attention_dim),
                "ffn": nn.Sequential(
                    nn.Linear(attention_dim, ffn_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ffn_dim, attention_dim),
                ),
            })
            if self.attention_mode == "local_global":
                layer["global_attention"] = nn.MultiheadAttention(
                    attention_dim, num_heads, dropout=dropout, batch_first=True
                )
            self.layers.append(layer)
        self.output_projection = nn.Linear(attention_dim, feature_dim)
        if zero_output_projection:
            nn.init.zeros_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)
        self.dropout = nn.Dropout(dropout)
        self.gate_logit = nn.Parameter(torch.tensor(float(initial_gate)))
        self.last_scaled_delta = None

    def forward(self, features, lengths):
        sequence_length = features.shape[1]
        positions = sinusoidal_position_encoding(
            sequence_length, features.shape[-1], features.device, features.dtype
        )
        positioned = (
            features
            + self.position_scale.to(features.dtype) * positions.unsqueeze(0)
        )
        hidden = self.input_projection(self.input_norm(positioned))
        key_padding_mask = (
            torch.arange(sequence_length, device=lengths.device).unsqueeze(0)
            >= lengths.unsqueeze(1)
        )
        for layer in self.layers:
            normalized = layer["attention_norm"](hidden)
            attention_mask = None
            if self.attention_mode in {"local", "local_global"}:
                positions = torch.arange(sequence_length, device=features.device)
                offsets = positions.unsqueeze(0) - positions.unsqueeze(1)
                outside_window = offsets.abs() > self.local_radius
                attention_mask = outside_window
            attended, _ = layer["attention"](
                normalized, normalized, normalized,
                key_padding_mask=key_padding_mask,
                attn_mask=attention_mask,
                need_weights=False,
            )
            if self.attention_mode == "local_global":
                global_hidden = normalized[:, ::self.global_stride]
                global_lengths = torch.div(
                    lengths + self.global_stride - 1,
                    self.global_stride, rounding_mode="floor",
                )
                global_padding_mask = (
                    torch.arange(global_hidden.shape[1], device=lengths.device).unsqueeze(0)
                    >= global_lengths.unsqueeze(1)
                )
                globally_attended, _ = layer["global_attention"](
                    normalized, global_hidden, global_hidden,
                    key_padding_mask=global_padding_mask, need_weights=False,
                )
                # Equal fixed branch weighting keeps this comparison focused
                # on the attention topology instead of adding another gate.
                attended = attended + 0.5 * globally_attended
            hidden = hidden + self.dropout(attended)
            hidden = hidden + self.dropout(layer["ffn"](layer["ffn_norm"](hidden)))
        delta = self.output_projection(hidden)
        scaled_delta = torch.sigmoid(self.gate_logit).to(features.dtype) * delta
        self.last_scaled_delta = scaled_delta
        refined = features + scaled_delta
        # A padded query can have no valid key inside its local window. Selecting
        # the original value prevents a NaN from leaking out of those positions.
        valid = (~key_padding_mask).unsqueeze(-1)
        return torch.where(valid, refined, features)
