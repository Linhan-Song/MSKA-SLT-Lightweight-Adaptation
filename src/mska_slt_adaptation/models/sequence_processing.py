"""Optional sequence-length processing used in the supplementary experiments."""

import torch
import torch.nn.functional as F
from torch import nn

from .common import sinusoidal_position_encoding


class TemporalCompressor(nn.Module):
    """Apply the sequence treatment selected for a controlled experiment."""

    def __init__(self, compressor_type="average", feature_dim=512, prompt_length=32,
                 num_heads=4, dropout=0.1):
        super().__init__()
        self.compressor_type = compressor_type.lower()
        self.prompt_length = prompt_length
        if self.compressor_type == "query":
            self.queries = nn.Parameter(torch.empty(prompt_length, feature_dim))
            nn.init.normal_(self.queries, std=0.02)
            self.input_norm = nn.LayerNorm(feature_dim)
            self.attention = nn.MultiheadAttention(
                feature_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.attention_norm = nn.LayerNorm(feature_dim)
            self.ffn = nn.Sequential(
                nn.Linear(feature_dim, feature_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim * 2, feature_dim),
            )
            self.output_norm = nn.LayerNorm(feature_dim)
        elif self.compressor_type == "gated_avgmax":
            # Begin close to average pooling and learn a channel-wise mixture
            # with the max-pooled representation.
            self.mix_logit = nn.Parameter(torch.full((feature_dim,), -4.0))
        elif self.compressor_type not in {"none", "average", "capped_average", "max"}:
            raise ValueError(f"Unknown compressor type: {compressor_type}")

    def forward(self, features, lengths):
        if self.compressor_type == "none":
            return features, lengths
        if self.compressor_type == "capped_average":
            compressed = []
            output_lengths = []
            for sample, length in zip(features, lengths):
                valid_length = int(length.item())
                valid = sample[:valid_length]
                if valid_length > self.prompt_length:
                    pooled = F.adaptive_avg_pool1d(
                        valid.transpose(0, 1).unsqueeze(0), self.prompt_length
                    ).squeeze(0).transpose(0, 1)
                    compressed.append(pooled)
                    output_lengths.append(self.prompt_length)
                else:
                    padding = sample.new_zeros(
                        self.prompt_length - valid_length, sample.shape[-1]
                    )
                    compressed.append(torch.cat([valid, padding], dim=0))
                    output_lengths.append(valid_length)
            return (
                torch.stack(compressed, dim=0),
                torch.tensor(output_lengths, dtype=torch.long, device=lengths.device),
            )
        if self.compressor_type in {"average", "max", "gated_avgmax"}:
            compressed = []
            for sample, length in zip(features, lengths):
                valid = sample[:int(length.item())].transpose(0, 1).unsqueeze(0)
                if self.compressor_type == "average":
                    pooled = F.adaptive_avg_pool1d(valid, self.prompt_length)
                elif self.compressor_type == "max":
                    pooled = F.adaptive_max_pool1d(valid, self.prompt_length)
                else:
                    average = F.adaptive_avg_pool1d(valid, self.prompt_length)
                    maximum = F.adaptive_max_pool1d(valid, self.prompt_length)
                    mix = torch.sigmoid(self.mix_logit).view(1, -1, 1)
                    pooled = average + mix * (maximum - average)
                compressed.append(pooled.squeeze(0).transpose(0, 1))
            output = torch.stack(compressed, dim=0)
            output_lengths = torch.full(
                (features.shape[0],), self.prompt_length,
                dtype=torch.long, device=lengths.device,
            )
            return output, output_lengths

        sequence_length = features.shape[1]
        positions = sinusoidal_position_encoding(
            sequence_length, features.shape[-1], features.device, features.dtype
        )
        inputs = self.input_norm(features + positions.unsqueeze(0))
        key_padding_mask = (
            torch.arange(sequence_length, device=lengths.device).unsqueeze(0)
            >= lengths.unsqueeze(1)
        )
        queries = self.queries.unsqueeze(0).expand(features.shape[0], -1, -1)
        attended, _ = self.attention(
            queries, inputs, inputs, key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        output = self.attention_norm(queries + attended)
        output = self.output_norm(output + self.ffn(output))
        output_lengths = torch.full(
            (features.shape[0],), self.prompt_length,
            dtype=torch.long, device=lengths.device,
        )
        return output, output_lengths
