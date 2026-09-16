"""作用在冻结 VLMapper 输出上的残差适配器。

本文件对应论文中的接口适配实验。三个类使用相同的瓶颈结构，区别只在于
残差修正量是否缩放、缩放系数是否可学习，因此可以把性能差异归因于 Gate。
张量形状始终是 [batch, time, 1024]，不会改变序列长度。
"""

import torch
from torch import nn


class PlainResidualAdapter(nn.Module):
    """不缩放残差，直接把瓶颈网络产生的修正量加到原表示上。"""

    def __init__(self, dimension=1024, bottleneck_dim=256, dropout=0.1):
        super().__init__()
        # 先归一化再降维，避免直接在 1024 维空间学习完整变换。
        self.pre_norm = nn.LayerNorm(dimension)
        self.down = nn.Linear(dimension, bottleneck_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, dimension)

    def forward(self, mapped_features, lengths=None):
        # Adapter 不改变时间长度，因此这里不需要 lengths；保留参数是为了让
        # 所有接口模块拥有一致的调用方式。
        del lengths
        hidden = self.down(self.pre_norm(mapped_features))
        delta = self.up(self.dropout(self.activation(hidden)))
        # Plain 版本没有保护性缩放，实验中用它观察全量残差是否过强。
        return mapped_features + delta


class FixedResidualAdapter(PlainResidualAdapter):
    """使用固定的小残差系数，作为可学习 Gate 的容量匹配对照。"""

    def __init__(self, dimension=1024, bottleneck_dim=256, dropout=0.1,
                 initial_gate=-4.0):
        super().__init__(dimension, bottleneck_dim, dropout)
        # initial_gate=-4 时 sigmoid 约为 0.018。register_buffer 会让该值
        # 随检查点保存和设备迁移，但不会被优化器更新。
        scale = torch.sigmoid(torch.tensor(float(initial_gate)))
        self.register_buffer("residual_scale", scale)

    def forward(self, mapped_features, lengths=None):
        del lengths
        hidden = self.down(self.pre_norm(mapped_features))
        delta = self.up(self.dropout(self.activation(hidden)))
        return mapped_features + self.residual_scale * delta


class GatedResidualAdapter(nn.Module):
    """论文最终采用的接口适配器，由模型学习残差修正的实际幅度。"""

    def __init__(self, dimension=1024, bottleneck_dim=256, dropout=0.1,
                 initial_gate=-4.0):
        super().__init__()
        self.pre_norm = nn.LayerNorm(dimension)
        self.down = nn.Linear(dimension, bottleneck_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, dimension)
        # 存储 logit 而不是直接存储 0~1 的 Gate，可避免训练后系数越界。
        # 负初始化让模型从接近原 VLMapper 的状态开始，降低破坏预训练表示的风险。
        self.gate_logit = nn.Parameter(torch.tensor(float(initial_gate)))

    def forward(self, mapped_features, lengths=None):
        del lengths
        hidden = self.down(self.pre_norm(mapped_features))
        delta = self.up(self.dropout(self.activation(hidden)))
        gate = torch.sigmoid(self.gate_logit)
        return mapped_features + gate * delta
