"""VLMapper 之前的时间序列细化模块。

该文件对应论文的 Local Temporal Refiner 以及 Global、Local-Global 对照。
三种模式共用输入投影、注意力和 FFN，主要区别是允许建立连接的时间位置。
"""

import torch
from torch import nn

from .common import sinusoidal_position_encoding


class TemporalRefiner(nn.Module):
    """细化 Recognition 输出的可变长度序列，但保持 [B,T,512] 形状不变。

    最终配置为 attention_dim=96、num_heads=4、ffn_dim=192、local_radius=4。
    模块先在低维空间计算注意力，再投影回 512 维，并通过可学习 Gate 以残差
    形式加到原始 Recognition 特征上。
    """

    def __init__(self, feature_dim=512, attention_dim=128, num_heads=4,
                 ffn_dim=256, num_layers=1, dropout=0.1, initial_gate=-4.0,
                 attention_mode="global", local_radius=8, global_stride=4,
                 zero_output_projection=False):
        super().__init__()
        if attention_dim % num_heads != 0:
            # MultiheadAttention 要求每个头分到相同数量的特征维度。
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
        # 512→attention_dim 的瓶颈决定时序模块的主要参数量。
        self.input_norm = nn.LayerNorm(feature_dim)
        self.input_projection = nn.Linear(feature_dim, attention_dim)
        # 最终实验统一使用正弦绝对位置编码，position_scale 学习其作用幅度。
        self.position_scale = nn.Parameter(torch.ones(1))
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            # 各层都采用 Pre-Norm。local 与 global 共用这里的主注意力参数，
            # Local-Global 才额外创建一个稀疏全局分支。
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
        # sigmoid(-4)≈0.018，使新增模块在训练开始时接近恒等映射。
        self.gate_logit = nn.Parameter(torch.tensor(float(initial_gate)))
        self.last_scaled_delta = None

    def forward(self, features, lengths):
        """根据实际长度细化序列；padding 位置最终保持输入值不变。"""
        sequence_length = features.shape[1]
        positions = sinusoidal_position_encoding(
            sequence_length, features.shape[-1], features.device, features.dtype
        )
        positioned = (
            features
            + self.position_scale.to(features.dtype) * positions.unsqueeze(0)
        )
        hidden = self.input_projection(self.input_norm(positioned))
        # True 表示该位置是 padding，所有注意力分支都不能把它作为 key/value。
        key_padding_mask = (
            torch.arange(sequence_length, device=lengths.device).unsqueeze(0)
            >= lengths.unsqueeze(1)
        )
        for layer in self.layers:
            normalized = layer["attention_norm"](hidden)
            attention_mask = None
            if self.attention_mode in {"local", "local_global"}:
                positions = torch.arange(sequence_length, device=features.device)
                # offsets[i,j] 表示 query i 与 key j 的相对距离。绝对值大于
                # local_radius 的连接被遮蔽，因此 r=4 时最多查看前后各 4 步。
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
                # 全局分支每隔 global_stride 取一个 key/value，在不恢复完整
                # 二次复杂度的情况下给局部注意力补充远距离信息。
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
                # 固定采用 0.5 权重，不额外引入可学习 Gate，使对照重点落在
                # 时间连接方式，而不是新增的缩放参数。
                attended = attended + 0.5 * globally_attended
            hidden = hidden + self.dropout(attended)
            hidden = hidden + self.dropout(layer["ffn"](layer["ffn_norm"](hidden)))
        delta = self.output_projection(hidden)
        gate = torch.sigmoid(self.gate_logit).to(features.dtype)
        scaled_delta = gate * delta
        self.last_scaled_delta = scaled_delta
        refined = features + scaled_delta
        # padding query 的局部窗口中可能没有有效 key。这里直接保留原输入，
        # 既保证 padding 不被修改，也防止 NaN 传播到后续 VLMapper。
        valid = (~key_padding_mask).unsqueeze(-1)
        return torch.where(valid, refined, features)
