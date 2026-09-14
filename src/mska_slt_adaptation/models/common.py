"""Small tensor utilities shared by the adaptation modules."""

import math

import torch
import torch.nn.functional as F


def adaptive_temporal_pool(features, lengths, prompt_length):
    """Pool each valid (unpadded) feature sequence to a fixed token count."""
    compressed = []
    for sample, length in zip(features, lengths):
        valid = sample[: int(length.item())].transpose(0, 1).unsqueeze(0)
        pooled = F.adaptive_avg_pool1d(valid, prompt_length)
        compressed.append(pooled.squeeze(0).transpose(0, 1))
    return torch.stack(compressed, dim=0)


def sinusoidal_position_encoding(length, dimension, device, dtype):
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dimension, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / dimension)
    )
    encoding = torch.zeros(length, dimension, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term[:encoding[:, 1::2].shape[1]])
    return encoding.to(dtype=dtype)


def masked_temporal_mean(features, lengths):
    """Mean over valid temporal positions for a padded [B, T, C] tensor."""
    if lengths is None:
        return features.mean(dim=1)
    positions = torch.arange(features.shape[1], device=features.device).unsqueeze(0)
    valid = positions < lengths.to(features.device).unsqueeze(1)
    weights = valid.unsqueeze(-1).to(features.dtype)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (features * weights).sum(dim=1) / denominator
