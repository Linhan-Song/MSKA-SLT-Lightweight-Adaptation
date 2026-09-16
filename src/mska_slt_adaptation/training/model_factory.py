"""把 YAML 实验配置转换成可以训练的模型对象。

这样设计后，超参数扫描只需复制并修改 YAML，不需要在 Python 文件中反复
改数字，也能保留每一次实验的完整配置。
"""

from pathlib import Path

import yaml

from ..models.qwen_prompt_model import QwenSoftPromptModel
from ..models.translation_pipeline import MSKATranslationModel


def build_model(config):
    """读取统一配置结构，创建最终模型或保留的探索性模型。"""
    experiment = config["experiment"].lower()
    model_config = config["model"]

    if experiment == "qwen_prompt":
        # Qwen 路径仅用于探索性实验，与论文最终的冻结 mBART 路径分开。
        return QwenSoftPromptModel(
            model_name=model_config["pretrained_model_name_or_path"],
            prompt_length=model_config.get("prompt_length", 32),
            lora_rank=model_config.get("lora_rank", 8),
            lora_alpha=model_config.get("lora_alpha", 16),
            lora_dropout=model_config.get("lora_dropout", 0.05),
            instruction=model_config.get("instruction"),
            load_in_4bit=model_config.get("load_in_4bit", True),
            compute_dtype=model_config.get("compute_dtype", "bfloat16"),
        )

    if experiment != "frozen_mbart":
        raise ValueError(f"Unknown experiment: {experiment}")

    # 原 MSKA 配置仍提供 mBART、词表和数据集相关参数。
    with Path(config["mska_config"]).open("r", encoding="utf-8") as handle:
        mska_config = yaml.safe_load(handle)

    interface = model_config.get("interface_adapter", {})
    temporal = model_config.get("temporal_refiner", {})
    sequence = model_config.get("sequence_processing", {})
    exploratory = model_config.get("exploratory", {})

    # YAML 使用便于阅读的名称，模型内部沿用简短的 compressor 类型。
    sequence_mode = sequence.get("mode", "variable_length")
    sequence_aliases = {
        "variable_length": "none",
        "average_pool": "average",
        "max_pool": "max",
        "capped_average_pool": "capped_average",
        "query_pool": "query",
    }
    compressor_type = sequence_aliases.get(sequence_mode, sequence_mode)

    # 下面的映射集中展示了论文符号和实际代码参数的对应关系，例如
    # radius→local_radius、dimension→temporal_dim、FFN→temporal_ffn_dim。
    return MSKATranslationModel(
        translation_cfg=mska_config["model"]["TranslationNetwork"],
        pretrained_component_dir=model_config["pretrained_component_dir"],
        prompt_length=sequence.get("target_length", 32),
        compressor_type=compressor_type,
        compressor_heads=sequence.get("attention_heads", 4),
        adapter_type=interface.get("type", "none"),
        bottleneck_dim=interface.get("bottleneck_dim", 128),
        initial_gate=interface.get("initial_gate", -4.0),
        dropout=model_config.get("dropout", 0.1),
        train_vlmapper=model_config.get("train_vlmapper", False),
        # 第一阶段 enabled=false；第二阶段和时间结构对比 enabled=true。
        use_temporal_refiner=temporal.get("enabled", False),
        temporal_mode=temporal.get("mode", "local"),
        local_radius=temporal.get("radius", 4),
        global_stride=temporal.get("global_stride", 4),
        temporal_dim=temporal.get("dimension", 96),
        temporal_heads=temporal.get("attention_heads", 4),
        temporal_ffn_dim=temporal.get("ffn_dimension", 192),
        temporal_layers=temporal.get("layers", 1),
        temporal_initial_gate=temporal.get("initial_gate", -4.0),
        temporal_zero_output_projection=temporal.get("zero_output_projection", False),
        qformer_attention_dim=exploratory.get("qformer_attention_dim", 128),
        qformer_heads=exploratory.get("qformer_heads", 4),
        qformer_ffn_dim=exploratory.get("qformer_ffn_dim", 256),
        houlsby_encoder_layers=exploratory.get("houlsby_encoder_layers", 0),
        houlsby_bottleneck_dim=exploratory.get("houlsby_bottleneck_dim", 64),
        houlsby_initial_gate=exploratory.get("houlsby_initial_gate", -4.0),
    )
