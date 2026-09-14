"""Q-Former, Houlsby and Soft-Gloss alternatives used in exploratory experiments."""

import torch
import torch.nn.functional as F
from torch import nn

from .common import adaptive_temporal_pool


class QFormerConnector(nn.Module):
    """Lightweight cross-attention connector over full Recognition features.

    In residual mode the frozen official VLMapper tokens provide the queries
    and the module predicts only a gated correction. In replacement mode, a
    shared bank of learned queries produces all visual tokens directly. This
    keeps Q1 and Q2 architecturally matched while isolating whether preserving
    the official cross-modal alignment is beneficial.
    """

    def __init__(self, input_dim=512, output_dim=1024, token_count=32,
                 attention_dim=128, num_heads=4, ffn_dim=256, dropout=0.1,
                 initial_gate=-4.0, replacement=False):
        super().__init__()
        if attention_dim % num_heads != 0:
            raise ValueError("Q-Former attention_dim must be divisible by num_heads")
        self.replacement = bool(replacement)
        self.token_count = int(token_count)
        self.feature_norm = nn.LayerNorm(input_dim)
        self.feature_projection = nn.Linear(input_dim, attention_dim)
        if self.replacement:
            self.query_tokens = nn.Parameter(
                torch.empty(self.token_count, attention_dim)
            )
            nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
            self.query_projection = None
        else:
            self.register_parameter("query_tokens", None)
            self.query_norm = nn.LayerNorm(output_dim)
            self.query_projection = nn.Linear(output_dim, attention_dim)
        self.cross_attention = nn.MultiheadAttention(
            attention_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(attention_dim)
        self.ffn = nn.Sequential(
            nn.Linear(attention_dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, attention_dim),
        )
        self.output_norm = nn.LayerNorm(attention_dim)
        self.output_projection = nn.Linear(attention_dim, output_dim)
        self.gate_logit = None
        if not self.replacement:
            self.gate_logit = nn.Parameter(torch.tensor(float(initial_gate)))
            nn.init.normal_(self.output_projection.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.output_projection.bias)

    def forward(self, recognition_features, recognition_lengths,
                mapped_features=None):
        batch_size, sequence_length, _ = recognition_features.shape
        memory = self.feature_projection(self.feature_norm(recognition_features))
        key_padding_mask = (
            torch.arange(sequence_length, device=recognition_lengths.device).unsqueeze(0)
            >= recognition_lengths.unsqueeze(1)
        )
        if self.replacement:
            queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            if mapped_features is None:
                raise ValueError("Residual Q-Former requires VLMapper tokens")
            queries = self.query_projection(self.query_norm(mapped_features))
        attended, _ = self.cross_attention(
            queries, memory, memory, key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden = self.attention_norm(queries + attended)
        hidden = self.output_norm(hidden + self.ffn(hidden))
        output = self.output_projection(hidden)
        if self.replacement:
            return output
        return mapped_features + torch.sigmoid(self.gate_logit) * output


class HoulsbyEncoderAdapter(nn.Module):
    """Bottleneck residual inserted after one frozen mBART encoder layer."""

    def __init__(self, dimension=1024, bottleneck_dim=64, dropout=0.1,
                 initial_gate=-4.0):
        super().__init__()
        self.pre_norm = nn.LayerNorm(dimension)
        self.down = nn.Linear(dimension, bottleneck_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, dimension)
        self.gate_logit = nn.Parameter(torch.tensor(float(initial_gate)))

    def forward(self, hidden_states):
        delta = self.up(self.dropout(self.activation(
            self.down(self.pre_norm(hidden_states))
        )))
        return hidden_states + torch.sigmoid(self.gate_logit) * delta


class SoftGlossFusionAdapter(nn.Module):
    """Fuse frozen SLR posteriors with the mapped visual representation."""

    def __init__(self, classifier_weight, classifier_bias, aligned_embeddings,
                 output_dim=1024, bottleneck_dim=128, dropout=0.1,
                 initial_gate=-4.0, temperature=1.0):
        super().__init__()
        if classifier_weight.ndim != 2:
            raise ValueError("Gloss classifier weight must have shape [V, D]")
        if aligned_embeddings.ndim != 2:
            raise ValueError("Aligned gloss embeddings must have shape [V, H]")
        if classifier_weight.shape[0] != aligned_embeddings.shape[0]:
            raise ValueError(
                "Recognition vocabulary and aligned embedding rows differ: "
                f"{classifier_weight.shape[0]} vs {aligned_embeddings.shape[0]}"
            )
        if aligned_embeddings.shape[1] != output_dim:
            raise ValueError(
                f"Expected {output_dim}-D gloss embeddings, got "
                f"{aligned_embeddings.shape[1]}"
            )
        if temperature <= 0:
            raise ValueError("Soft-gloss temperature must be positive")
        self.temperature = float(temperature)
        self.register_buffer("classifier_weight", classifier_weight.detach().clone())
        self.register_buffer("classifier_bias", classifier_bias.detach().clone())
        self.register_buffer("aligned_embeddings", aligned_embeddings.detach().clone())
        self.pre_norm = nn.LayerNorm(output_dim)
        self.down = nn.Linear(output_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, output_dim)
        self.gate_logit = nn.Parameter(torch.tensor(float(initial_gate)))
        nn.init.normal_(self.up.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.up.bias)

    def forward(self, recognition_features, recognition_lengths,
                visual_features, target_length):
        # Recognition features and the SLR head are frozen; keeping this branch
        # out of autograd saves memory while the alignment adapter still trains.
        with torch.no_grad():
            logits = F.linear(
                recognition_features.float(),
                self.classifier_weight.float(),
                self.classifier_bias.float(),
            )
            probabilities = torch.softmax(logits / self.temperature, dim=-1)
            soft_gloss = torch.matmul(
                probabilities, self.aligned_embeddings.float()
            )
            soft_gloss = adaptive_temporal_pool(
                soft_gloss, recognition_lengths, int(target_length)
            )
        soft_gloss = soft_gloss.to(visual_features.dtype)
        hidden = self.down(self.pre_norm(soft_gloss))
        delta = self.up(self.dropout(self.activation(hidden)))
        return visual_features + torch.sigmoid(self.gate_logit) * delta
