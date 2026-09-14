"""Prompt-based alternatives evaluated during early model development."""

import math

import torch
from torch import nn

from .common import adaptive_temporal_pool, masked_temporal_mean, sinusoidal_position_encoding


class TemporalSoftPromptAdapter(nn.Module):
    def __init__(self, input_dim, output_dim, prompt_length=32, dropout=0.1):
        super().__init__()
        self.prompt_length = prompt_length
        self.input_projection = nn.Linear(input_dim, output_dim)
        self.input_norm = nn.LayerNorm(output_dim)
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
        )
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, features, lengths):
        x = adaptive_temporal_pool(features, lengths, self.prompt_length)
        x = self.input_norm(self.input_projection(x))
        return self.output_norm(x + self.mlp(x))


class ConditionalSoftPrompt(nn.Module):
    """Generate a low-rank per-sample correction to a shared soft prompt."""

    def __init__(self, input_dim, output_dim, prompt_length=16, hidden_dim=128,
                 rank=8, initial_gate=-4.0):
        super().__init__()
        self.prompt_length = int(prompt_length)
        self.rank = int(rank)
        self.input_norm = nn.LayerNorm(input_dim)
        self.context_projection = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.coefficient_projection = nn.Linear(
            hidden_dim, self.prompt_length * self.rank
        )
        self.prompt_basis = nn.Parameter(torch.empty(self.rank, output_dim))
        self.gate_logit = nn.Parameter(torch.tensor(float(initial_gate)))
        nn.init.normal_(self.coefficient_projection.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.coefficient_projection.bias)
        nn.init.normal_(self.prompt_basis, mean=0.0, std=0.02)

    def forward(self, features, lengths, base_prompt):
        context = masked_temporal_mean(features, lengths)
        hidden = self.activation(self.context_projection(self.input_norm(context)))
        coefficients = self.coefficient_projection(hidden).view(
            features.shape[0], self.prompt_length, self.rank
        )
        dynamic_prompt = torch.matmul(coefficients, self.prompt_basis)
        shared_prompt = base_prompt.unsqueeze(0).expand(features.shape[0], -1, -1)
        return shared_prompt + torch.sigmoid(self.gate_logit) * dynamic_prompt


class TemporalCrossAttentionSoftPrompt(nn.Module):
    """Generate per-video prompts by attending over the full MSKA sequence.

    The shared base prompts act as sixteen distinct queries.  Keys and values
    come from the original, uncompressed [T, 512] recognition features, so the
    generator can preserve temporal ordering while the normal visual pathway
    remains untouched.
    """

    def __init__(self, input_dim, output_dim, prompt_length=16,
                 attention_dim=64, num_heads=4, dropout=0.1,
                 initial_gate=-2.0, use_position_encoding=False):
        super().__init__()
        if attention_dim % num_heads != 0:
            raise ValueError("Prompt attention_dim must be divisible by num_heads")
        self.prompt_length = int(prompt_length)
        self.attention_dim = int(attention_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.attention_dim // self.num_heads
        self.use_position_encoding = bool(use_position_encoding)
        self.query_norm = nn.LayerNorm(output_dim)
        self.feature_norm = nn.LayerNorm(input_dim)
        self.query_projection = nn.Linear(output_dim, self.attention_dim)
        self.key_projection = nn.Linear(input_dim, self.attention_dim)
        self.value_projection = nn.Linear(input_dim, self.attention_dim)
        self.context_norm = nn.LayerNorm(self.attention_dim)
        self.output_projection = nn.Linear(self.attention_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate_logit = nn.Parameter(torch.tensor(float(initial_gate)))
        if self.use_position_encoding:
            self.position_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_parameter("position_scale", None)
        self.last_attention = None

        # Begin close to the fixed-prompt control while still allowing useful
        # gradients to reach the temporal attention path from the first step.
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, features, lengths, base_prompt):
        batch_size, sequence_length, _ = features.shape
        shared_prompt = base_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        queries = self.query_projection(self.query_norm(shared_prompt))
        normalized_features = self.feature_norm(features)
        if self.use_position_encoding:
            positions = sinusoidal_position_encoding(
                sequence_length, normalized_features.shape[-1],
                normalized_features.device, normalized_features.dtype,
            )
            if lengths is not None:
                valid = (
                    torch.arange(sequence_length, device=features.device).unsqueeze(0)
                    < lengths.to(features.device).unsqueeze(1)
                ).unsqueeze(-1)
                positions = positions.unsqueeze(0) * valid.to(positions.dtype)
            else:
                positions = positions.unsqueeze(0)
            normalized_features = normalized_features + self.position_scale * positions
        keys = self.key_projection(normalized_features)
        values = self.value_projection(normalized_features)

        def split_heads(tensor):
            return tensor.view(
                batch_size, tensor.shape[1], self.num_heads, self.head_dim
            ).transpose(1, 2)

        queries = split_heads(queries)
        keys = split_heads(keys)
        values = split_heads(values)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )
        if lengths is not None:
            invalid = (
                torch.arange(sequence_length, device=features.device).unsqueeze(0)
                >= lengths.to(features.device).unsqueeze(1)
            )
            scores = scores.masked_fill(invalid[:, None, None, :], -1e4)
        attention_probabilities = torch.softmax(scores, dim=-1)
        self.last_attention = attention_probabilities
        attention = self.dropout(attention_probabilities)
        context = torch.matmul(attention, values)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, self.prompt_length, self.attention_dim
        )
        dynamic_prompt = self.output_projection(self.context_norm(context))
        return shared_prompt + torch.sigmoid(self.gate_logit) * dynamic_prompt
