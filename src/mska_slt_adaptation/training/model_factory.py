"""Create a translation model from the YAML experiment configuration."""

from pathlib import Path

import yaml

from ..models.qwen_prompt_model import QwenSoftPromptModel
from ..models.translation_pipeline import MSKATranslationModel


def build_model(config):
    experiment = config["experiment"].lower()
    model_config = config["model"]

    if experiment == "qwen_prompt":
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

    with Path(config["mska_config"]).open("r", encoding="utf-8") as handle:
        mska_config = yaml.safe_load(handle)

    interface = model_config.get("interface_adapter", {})
    temporal = model_config.get("temporal_refiner", {})
    sequence = model_config.get("sequence_processing", {})
    exploratory = model_config.get("exploratory", {})

    sequence_mode = sequence.get("mode", "variable_length")
    sequence_aliases = {
        "variable_length": "none",
        "average_pool": "average",
        "max_pool": "max",
        "capped_average_pool": "capped_average",
        "query_pool": "query",
    }
    compressor_type = sequence_aliases.get(sequence_mode, sequence_mode)

    return MSKATranslationModel(
        translation_cfg=mska_config["model"]["TranslationNetwork"],
        experiment=experiment,
        prompt_length=sequence.get("target_length", 32),
        use_lora=False,
        compressor_type=compressor_type,
        compressor_heads=sequence.get("attention_heads", 4),
        adapter_type=interface.get("type", "none"),
        bottleneck_dim=interface.get("bottleneck_dim", 128),
        initial_gate=interface.get("initial_gate", -4.0),
        dropout=model_config.get("dropout", 0.1),
        freeze_gloss_embedding=True,
        pretrained_component_dir=model_config["pretrained_component_dir"],
        train_vlmapper=model_config.get("train_vlmapper", False),
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
        temporal_identity_weight=temporal.get("identity_loss_weight", 0.0),
        temporal_kernel_size=exploratory.get("temporal_kernel_size", 3),
        se_hidden_dim=exploratory.get("se_hidden_dim", 64),
        soft_prompt_length=exploratory.get("soft_prompt_length", 0),
        conditional_prompt=exploratory.get("conditional_prompt", False),
        conditional_prompt_hidden_dim=exploratory.get("prompt_hidden_dimension", 128),
        conditional_prompt_rank=exploratory.get("prompt_rank", 8),
        conditional_prompt_initial_gate=exploratory.get("prompt_initial_gate", -4.0),
        conditional_prompt_position_encoding=exploratory.get("prompt_position_encoding", False),
        prompt_diversity_weight=exploratory.get("prompt_diversity_weight", 0.0),
        soft_gloss_fusion=exploratory.get("soft_gloss_fusion", False),
        soft_gloss_bottleneck_dim=exploratory.get("soft_gloss_bottleneck_dim", 128),
        soft_gloss_initial_gate=exploratory.get("soft_gloss_initial_gate", -4.0),
        soft_gloss_temperature=exploratory.get("soft_gloss_temperature", 1.0),
        recognition_gloss2id_path=exploratory.get("recognition_gloss2id_path"),
        translation_gloss2id_path=exploratory.get("translation_gloss2id_path"),
        qformer_attention_dim=exploratory.get("qformer_attention_dim", 128),
        qformer_heads=exploratory.get("qformer_heads", 4),
        qformer_ffn_dim=exploratory.get("qformer_ffn_dim", 256),
        houlsby_encoder_layers=exploratory.get("houlsby_encoder_layers", 0),
        houlsby_bottleneck_dim=exploratory.get("houlsby_bottleneck_dim", 64),
        houlsby_initial_gate=exploratory.get("houlsby_initial_gate", -4.0),
    )
