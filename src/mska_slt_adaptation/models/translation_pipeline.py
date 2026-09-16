"""把冻结的 MSKA 组件与论文中的轻量模块连接起来。

最终前向顺序为：缓存的 Recognition 特征 -> Temporal Refiner ->
冻结的 VLMapper -> Residual Adapter -> 冻结的 mBART。
"""

from pathlib import Path

import torch
from torch import nn

from .alternative_connectors import HoulsbyEncoderAdapter, QFormerConnector
from .residual_adapters import (
    FixedResidualAdapter,
    GatedResidualAdapter,
    PlainResidualAdapter,
)
from .sequence_processing import TemporalCompressor
from .temporal_refiner import TemporalRefiner


def apply_lora(model, task_type, rank, alpha, dropout, target_modules):
    """给单独保留的 Qwen 探索模型添加 LoRA。"""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        task_type=task_type,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )
    return get_peft_model(model, config)


class MSKATranslationModel(nn.Module):
    """组合冻结主干、接口 Adapter 和可选 Temporal Refiner。"""

    def __init__(
        self,
        translation_cfg,
        pretrained_component_dir,
        prompt_length=32,
        compressor_type="none",
        compressor_heads=4,
        adapter_type="none",
        bottleneck_dim=128,
        initial_gate=-4.0,
        train_vlmapper=False,
        dropout=0.1,
        use_temporal_refiner=False,
        temporal_dim=96,
        temporal_heads=4,
        temporal_ffn_dim=192,
        temporal_layers=1,
        temporal_initial_gate=-4.0,
        temporal_mode="local",
        local_radius=4,
        global_stride=4,
        temporal_zero_output_projection=False,
        qformer_attention_dim=128,
        qformer_heads=4,
        qformer_ffn_dim=256,
        houlsby_encoder_layers=0,
        houlsby_bottleneck_dim=64,
        houlsby_initial_gate=-4.0,
    ):
        super().__init__()
        from mska_backbone.translation import TranslationNetwork
        from mska_backbone.vl_mapper import VLMapper

        self.adapter_type = str(adapter_type).lower()
        self.temporal_refiner = None
        self.interface_adapter = None
        self.qformer_connector = None
        self.houlsby_adapters = nn.ModuleList()
        self._houlsby_hook_handles = []

        # 从公开检查点拆出的文件中恢复 mBART 和 VLMapper。
        component_dir = Path(pretrained_component_dir)
        self.translation = TranslationNetwork(cfg=translation_cfg)
        translation_state = torch.load(
            component_dir / "translation.pth", map_location="cpu"
        )
        self.translation.load_state_dict(translation_state, strict=True)
        for parameter in self.translation.parameters():
            parameter.requires_grad = False

        output_dim = self.translation.input_dim
        self.mapper = VLMapper(
            cfg={"type": "projection"}, in_features=512, out_features=output_dim
        )
        mapper_state = torch.load(
            component_dir / "vl_mapper.pth", map_location="cpu"
        )
        self.mapper.load_state_dict(mapper_state, strict=True)
        for parameter in self.mapper.parameters():
            parameter.requires_grad = bool(train_vlmapper)

        # compressor_type="none" 表示保留 Recognition 的可变长度序列。
        self.compressor = TemporalCompressor(
            compressor_type, 512, prompt_length, compressor_heads, dropout
        )

        if self.adapter_type == "plain_residual":
            self.interface_adapter = PlainResidualAdapter(
                output_dim, bottleneck_dim, dropout
            )
        elif self.adapter_type == "fixed_residual":
            self.interface_adapter = FixedResidualAdapter(
                output_dim, bottleneck_dim, dropout, initial_gate
            )
        elif self.adapter_type == "gated_residual":
            self.interface_adapter = GatedResidualAdapter(
                output_dim, bottleneck_dim, dropout, initial_gate
            )
        elif self.adapter_type == "qformer_residual":
            self.qformer_connector = QFormerConnector(
                input_dim=512,
                output_dim=output_dim,
                token_count=prompt_length,
                attention_dim=qformer_attention_dim,
                num_heads=qformer_heads,
                ffn_dim=qformer_ffn_dim,
                dropout=dropout,
                initial_gate=initial_gate,
                replacement=False,
            )
        elif self.adapter_type != "none":
            raise ValueError(f"Unsupported interface adapter: {self.adapter_type}")

        if use_temporal_refiner:
            self.temporal_refiner = TemporalRefiner(
                feature_dim=512,
                attention_dim=temporal_dim,
                num_heads=temporal_heads,
                ffn_dim=temporal_ffn_dim,
                num_layers=temporal_layers,
                dropout=dropout,
                initial_gate=temporal_initial_gate,
                attention_mode=temporal_mode,
                local_radius=local_radius,
                global_stride=global_stride,
                zero_output_projection=temporal_zero_output_projection,
            )

        self._attach_houlsby_adapters(
            int(houlsby_encoder_layers),
            output_dim,
            int(houlsby_bottleneck_dim),
            dropout,
            houlsby_initial_gate,
        )

    def _attach_houlsby_adapters(
        self, layer_count, dimension, bottleneck_dim, dropout, initial_gate
    ):
        """把保留的 Houlsby 对照模块挂到 mBART 编码器末端。"""
        if layer_count == 0:
            return
        encoder_layers = self.translation.model.model.encoder.layers
        if layer_count > len(encoder_layers):
            raise ValueError("houlsby_encoder_layers exceeds mBART depth")
        for layer_index in range(len(encoder_layers) - layer_count, len(encoder_layers)):
            adapter = HoulsbyEncoderAdapter(
                dimension=dimension,
                bottleneck_dim=bottleneck_dim,
                dropout=dropout,
                initial_gate=initial_gate,
            )
            self.houlsby_adapters.append(adapter)
            handle = encoder_layers[layer_index].register_forward_hook(
                self._make_houlsby_hook(adapter)
            )
            self._houlsby_hook_handles.append(handle)

    @staticmethod
    def _make_houlsby_hook(adapter):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                return (adapter(output[0]),) + output[1:]
            return adapter(output)

        return hook

    @property
    def text_tokenizer(self):
        return self.translation.text_tokenizer

    def _map(self, features, lengths):
        """把 [B,T,512] Recognition 特征映射为 mBART 输入表示。"""
        original_features = features
        original_lengths = lengths

        if self.temporal_refiner is not None:
            features = self.temporal_refiner(features, lengths)

        compressed, compressed_lengths = self.compressor(features, lengths)
        mapped = self.mapper({"gloss_feature": compressed})

        if self.adapter_type == "qformer_residual":
            mapped = self.qformer_connector(
                original_features, original_lengths, mapped
            )
        elif self.interface_adapter is not None:
            mapped = self.interface_adapter(mapped, compressed_lengths)

        return mapped, compressed_lengths

    def train(self, mode=True):
        """切换训练状态，同时让冻结组件保持在评估模式。"""
        super().train(mode)
        self.translation.eval()
        if not any(parameter.requires_grad for parameter in self.mapper.parameters()):
            self.mapper.eval()
        if self.interface_adapter is not None and not any(
            parameter.requires_grad for parameter in self.interface_adapter.parameters()
        ):
            self.interface_adapter.eval()
        if self.houlsby_adapters:
            self.houlsby_adapters.train(mode)
        return self

    def forward(self, features, lengths, texts):
        """计算教师强制训练使用的翻译损失。"""
        mapped, mapped_lengths = self._map(features, lengths)
        tokenized = self.text_tokenizer(texts)
        outputs = self.translation(
            input_feature=mapped, input_lengths=mapped_lengths, **tokenized
        )
        return outputs["translation_loss"]

    @torch.no_grad()
    def generate_text(self, features, lengths, generation_cfg):
        """使用冻结的 mBART 解码器生成译文。"""
        mapped, mapped_lengths = self._map(features, lengths)
        transformer_inputs = self.translation.prepare_feature_inputs(
            mapped, mapped_lengths
        )
        outputs = self.translation.generate(**transformer_inputs, **generation_cfg)
        return outputs["decoded_sequences"]


def trainable_state_dict(model):
    """只保存当前允许更新的任务相关参数。"""
    trainable = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in trainable
    }


def count_parameters(model):
    """返回模型总参数量和当前可训练参数量。"""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable
