"""End-to-end translation pathway used by the adaptation experiments."""

import pickle
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .alternative_connectors import HoulsbyEncoderAdapter, QFormerConnector, SoftGlossFusionAdapter
from .exploratory_adapters import (
    ExpansionResidualAdapter,
    GatedBottleneckAdapter,
    SEGatedResidualAdapter,
    TemporalGatedBottleneckAdapter,
    TemporalGatedResidualAdapter,
)
from .prompt_modules import ConditionalSoftPrompt, TemporalCrossAttentionSoftPrompt
from .residual_adapters import FixedResidualAdapter, GatedResidualAdapter, PlainResidualAdapter
from .sequence_processing import TemporalCompressor
from .temporal_refiner import TemporalRefiner


def apply_lora(model, task_type, rank, alpha, dropout, target_modules):
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
    """Translation pipeline built from frozen MSKA components and lightweight modules."""

    def __init__(self, translation_cfg, experiment="e1", prompt_length=32,
                 lora_rank=8, lora_alpha=16, lora_dropout=0.05,
                 use_lora=True, compressor_type=None, adapter_type=None,
                 bottleneck_dim=256, adapter_heads=4, compressor_heads=4,
                 dropout=0.1, freeze_gloss_embedding=False,
                 pretrained_component_dir=None, train_vlmapper=False,
                 initial_gate=-4.0, temporal_kernel_size=3,
                 soft_prompt_length=0, se_hidden_dim=64,
                 conditional_prompt=False, conditional_prompt_hidden_dim=128,
                 conditional_prompt_rank=8, conditional_prompt_initial_gate=-4.0,
                 soft_gloss_fusion=False, soft_gloss_bottleneck_dim=128,
                 soft_gloss_initial_gate=-4.0, soft_gloss_temperature=1.0,
                 recognition_gloss2id_path=None,
                 translation_gloss2id_path=None,
                 conditional_prompt_position_encoding=False,
                 prompt_diversity_weight=0.0,
                 use_temporal_refiner=False,
                 temporal_dim=128, temporal_heads=4,
                 temporal_ffn_dim=256, temporal_layers=1,
                 temporal_initial_gate=-4.0,
                 temporal_mode="global", local_radius=8,
                 global_stride=4,
                 temporal_zero_output_projection=False,
                 temporal_identity_weight=0.0,
                 qformer_attention_dim=128, qformer_heads=4,
                 qformer_ffn_dim=256, houlsby_encoder_layers=0,
                 houlsby_bottleneck_dim=64,
                 houlsby_initial_gate=-4.0):
        super().__init__()
        from mska_backbone.translation import TranslationNetwork
        from mska_backbone.vl_mapper import VLMapper

        self.experiment = experiment.lower()
        self.prompt_length = prompt_length
        self.soft_prompt_length = int(soft_prompt_length)
        self.use_lora = bool(use_lora)
        self.prompt_diversity_weight = float(prompt_diversity_weight)
        self.last_prompt_diversity_loss = None
        self.temporal_identity_weight = float(
            temporal_identity_weight
        )
        self.last_temporal_identity_loss = None
        self.last_mapped_features = None
        self.temporal_refiner = None
        self.qformer_connector = None
        self.houlsby_adapters = nn.ModuleList()
        self._houlsby_hook_handles = []
        if pretrained_component_dir and self.use_lora:
            raise ValueError("Frozen-component experiments do not use LoRA")
        self.translation = TranslationNetwork(cfg=translation_cfg)
        for parameter in self.translation.model.parameters():
            parameter.requires_grad = False
        if self.use_lora:
            from peft import TaskType
            self.translation.model = apply_lora(
                self.translation.model, TaskType.SEQ_2_SEQ_LM, lora_rank,
                lora_alpha, lora_dropout, ["q_proj", "v_proj"],
            )
        if freeze_gloss_embedding or not self.use_lora:
            self.translation.gloss_embedding.weight.requires_grad = False
        output_dim = self.translation.input_dim
        if self.soft_prompt_length > 0:
            self.soft_prompt = nn.Parameter(
                torch.empty(self.soft_prompt_length, output_dim)
            )
            nn.init.normal_(self.soft_prompt, mean=0.0, std=0.02)
        else:
            self.register_parameter("soft_prompt", None)
        self.conditional_prompt = None
        self.conditional_prompt_type = str(
            conditional_prompt if isinstance(conditional_prompt, str) else "mean"
        ).lower()
        if conditional_prompt:
            if self.soft_prompt is None:
                raise ValueError("Conditional Prompt requires soft_prompt_length > 0")
            if self.conditional_prompt_type in {"true", "mean", "global_mean"}:
                self.conditional_prompt_type = "mean"
                self.conditional_prompt = ConditionalSoftPrompt(
                    input_dim=512,
                    output_dim=output_dim,
                    prompt_length=self.soft_prompt_length,
                    hidden_dim=conditional_prompt_hidden_dim,
                    rank=conditional_prompt_rank,
                    initial_gate=conditional_prompt_initial_gate,
                )
            elif self.conditional_prompt_type in {"cross_attention", "temporal_cross_attention"}:
                self.conditional_prompt_type = "cross_attention"
                self.conditional_prompt = TemporalCrossAttentionSoftPrompt(
                    input_dim=512,
                    output_dim=output_dim,
                    prompt_length=self.soft_prompt_length,
                    attention_dim=conditional_prompt_hidden_dim,
                    num_heads=conditional_prompt_rank,
                    dropout=dropout,
                    initial_gate=conditional_prompt_initial_gate,
                    use_position_encoding=conditional_prompt_position_encoding,
                )
            else:
                raise ValueError(
                    f"Unknown conditional_prompt type: {self.conditional_prompt_type}"
                )
        if compressor_type is None:
            compressor_type = "none" if self.experiment == "e1" else "average"
        if adapter_type is None:
            adapter_type = "expansion" if self.experiment == "e2" else "vlmapper"
        self.compressor_type = compressor_type.lower()
        adapter_aliases = {"vlmapper": "none"}
        self.adapter_type = adapter_aliases.get(adapter_type.lower(), adapter_type.lower())
        self.compressor = TemporalCompressor(
            self.compressor_type, 512, prompt_length, compressor_heads, dropout
        )
        self.soft_gloss_adapter = None
        self.pretrained_component_mode = bool(pretrained_component_dir)
        if self.pretrained_component_mode:
            component_dir = Path(pretrained_component_dir)
            translation_state = torch.load(
                component_dir / "translation.pth", map_location="cpu"
            )
            self.translation.load_state_dict(translation_state, strict=True)
            self.mapper = VLMapper(
                cfg={"type": "projection"}, in_features=512, out_features=output_dim
            )
            mapper_state = torch.load(component_dir / "vl_mapper.pth", map_location="cpu")
            self.mapper.load_state_dict(mapper_state, strict=True)
            for parameter in self.translation.parameters():
                parameter.requires_grad = False
            for parameter in self.mapper.parameters():
                parameter.requires_grad = bool(train_vlmapper)
            if self.adapter_type == "none":
                self.interface_adapter = None
            elif self.adapter_type == "plain_residual":
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
            elif self.adapter_type == "se_gated_residual":
                self.interface_adapter = SEGatedResidualAdapter(
                    output_dim, bottleneck_dim, se_hidden_dim, dropout, initial_gate
                )
            elif self.adapter_type == "temporal_gated_residual":
                self.interface_adapter = TemporalGatedResidualAdapter(
                    output_dim, bottleneck_dim, dropout,
                    temporal_kernel_size, initial_gate,
                )
            elif self.adapter_type in {
                "qformer_residual", "qformer_replacement"
            }:
                self.interface_adapter = None
                self.qformer_connector = QFormerConnector(
                    input_dim=512, output_dim=output_dim,
                    token_count=prompt_length,
                    attention_dim=qformer_attention_dim,
                    num_heads=qformer_heads, ffn_dim=qformer_ffn_dim,
                    dropout=dropout, initial_gate=initial_gate,
                    replacement=(
                        self.adapter_type == "qformer_replacement"
                    ),
                )
            else:
                raise ValueError(
                    f"Unsupported interface adapter: {self.adapter_type}"
                )
            if soft_gloss_fusion:
                if self.compressor_type != "average":
                    raise ValueError(
                        "The controlled soft-gloss experiment requires "
                        "compressor_type=average"
                    )
                if not recognition_gloss2id_path or not translation_gloss2id_path:
                    raise ValueError(
                        "Soft-gloss fusion requires both gloss2id paths"
                    )
                with Path(recognition_gloss2id_path).open("rb") as handle:
                    recognition_gloss2id = pickle.load(handle)
                with Path(translation_gloss2id_path).open("rb") as handle:
                    translation_gloss2id = pickle.load(handle)
                recognition_state = torch.load(
                    component_dir / "recognition.pth", map_location="cpu"
                )
                classifier_weight = recognition_state[
                    "fuse_visual_head.gloss_output_layer.weight"
                ]
                classifier_bias = recognition_state[
                    "fuse_visual_head.gloss_output_layer.bias"
                ]
                vocabulary_size = classifier_weight.shape[0]
                if max(recognition_gloss2id.values()) >= vocabulary_size:
                    raise ValueError(
                        "Recognition gloss ids exceed the frozen classifier size"
                    )
                aligned_embeddings = torch.zeros(
                    vocabulary_size, output_dim,
                    dtype=self.translation.gloss_embedding.weight.dtype,
                )
                matched = 0
                for gloss, recognition_id in recognition_gloss2id.items():
                    # CTC blank has no lexical content; unmatched rows also
                    # remain zero instead of injecting an arbitrary token.
                    if gloss == "<si>" or gloss not in translation_gloss2id:
                        continue
                    translation_id = translation_gloss2id[gloss]
                    aligned_embeddings[recognition_id] = (
                        self.translation.gloss_embedding.weight[
                            translation_id
                        ].detach().cpu()
                    )
                    matched += 1
                print(
                    "Soft-gloss vocabulary alignment: "
                    f"matched={matched}/{len(recognition_gloss2id)}, "
                    "blank=<si> mapped to zero",
                    flush=True,
                )
                self.soft_gloss_adapter = SoftGlossFusionAdapter(
                    classifier_weight=classifier_weight,
                    classifier_bias=classifier_bias,
                    aligned_embeddings=aligned_embeddings,
                    output_dim=output_dim,
                    bottleneck_dim=soft_gloss_bottleneck_dim,
                    dropout=dropout,
                    initial_gate=soft_gloss_initial_gate,
                    temperature=soft_gloss_temperature,
                )
        elif self.adapter_type == "vlmapper":
            self.mapper = VLMapper(
                cfg={"type": "projection"}, in_features=512, out_features=output_dim
            )
        elif self.adapter_type == "expansion":
            self.mapper = ExpansionResidualAdapter(512, output_dim, dropout)
        elif self.adapter_type == "gated_bottleneck":
            self.mapper = GatedBottleneckAdapter(
                512, output_dim, bottleneck_dim, dropout
            )
        elif self.adapter_type == "temporal_gated":
            self.mapper = TemporalGatedBottleneckAdapter(
                512, output_dim, bottleneck_dim, adapter_heads, dropout
            )
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")
        # Construction follows the optimisation order used in the dissertation:
        # the interface adapter is initialised before the temporal refiner.
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
        if int(houlsby_encoder_layers) > 0:
            encoder_layers = self.translation.model.model.encoder.layers
            layer_count = int(houlsby_encoder_layers)
            if layer_count > len(encoder_layers):
                raise ValueError("houlsby_encoder_layers exceeds mBART depth")
            for layer_index in range(len(encoder_layers) - layer_count, len(encoder_layers)):
                adapter = HoulsbyEncoderAdapter(
                    dimension=output_dim,
                    bottleneck_dim=houlsby_bottleneck_dim,
                    dropout=dropout,
                    initial_gate=houlsby_initial_gate,
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
        recognition_features = features
        recognition_lengths = lengths
        if self.temporal_refiner is not None:
            features = self.temporal_refiner(features, lengths)
        compressed, compressed_lengths = self.compressor(features, lengths)
        if self.pretrained_component_mode:
            if self.adapter_type == "qformer_replacement":
                mapped = self.qformer_connector(
                    recognition_features, recognition_lengths
                )
            else:
                mapped = self.mapper({"gloss_feature": compressed})
            if self.adapter_type == "qformer_residual":
                mapped = self.qformer_connector(
                    recognition_features, recognition_lengths, mapped
                )
            elif self.interface_adapter is not None:
                mapped = self.interface_adapter(mapped, compressed_lengths)
        elif self.adapter_type == "vlmapper":
            mapped = self.mapper({"gloss_feature": compressed})
        else:
            mapped = self.mapper(compressed, compressed_lengths)
        if self.soft_gloss_adapter is not None:
            mapped = self.soft_gloss_adapter(
                features, lengths, mapped, mapped.shape[1]
            )
        if self.soft_prompt is not None:
            if self.conditional_prompt is None:
                prompt = self.soft_prompt.unsqueeze(0).expand(mapped.shape[0], -1, -1)
            else:
                if self.conditional_prompt_type == "cross_attention":
                    prompt = self.conditional_prompt(features, lengths, self.soft_prompt)
                else:
                    prompt = self.conditional_prompt(
                        compressed, compressed_lengths, self.soft_prompt
                    )
            mapped = torch.cat([prompt.to(mapped.dtype), mapped], dim=1)
            compressed_lengths = compressed_lengths + self.soft_prompt_length
        return mapped, compressed_lengths

    def train(self, mode=True):
        super().train(mode)
        if not self.use_lora:
            self.translation.model.eval()
        if len(self.houlsby_adapters) > 0:
            self.houlsby_adapters.train(mode)
        if self.pretrained_component_mode and not any(
            parameter.requires_grad for parameter in self.mapper.parameters()
        ):
            self.mapper.eval()
        if self.pretrained_component_mode and self.interface_adapter is not None and not any(
            parameter.requires_grad for parameter in self.interface_adapter.parameters()
        ):
            self.interface_adapter.eval()
        return self

    def forward(self, features, lengths, texts):
        mapped, mapped_lengths = self._map(features, lengths)
        self.last_mapped_features = mapped
        tokenized = self.text_tokenizer(texts)
        outputs = self.translation(
            input_feature=mapped, input_lengths=mapped_lengths, **tokenized
        )
        loss = outputs["translation_loss"]
        self.last_temporal_identity_loss = None
        if (
            self.training
            and self.temporal_identity_weight > 0
            and self.temporal_refiner is not None
            and self.temporal_refiner.last_scaled_delta is not None
        ):
            sequence_length = features.shape[1]
            valid = (
                torch.arange(sequence_length, device=lengths.device).unsqueeze(0)
                < lengths.unsqueeze(1)
            ).unsqueeze(-1)
            scaled_delta = self.temporal_refiner.last_scaled_delta
            numerator = (scaled_delta.float().pow(2) * valid).sum()
            denominator = (features.float().pow(2) * valid).sum().clamp_min(1e-8)
            identity_loss = numerator / denominator
            self.last_temporal_identity_loss = identity_loss.detach()
            loss = loss + self.temporal_identity_weight * identity_loss
        self.last_prompt_diversity_loss = None
        if (
            self.training
            and self.prompt_diversity_weight > 0
            and self.conditional_prompt is not None
            and getattr(self.conditional_prompt, "last_attention", None) is not None
        ):
            # Average heads, L2-normalize each prompt's temporal attention,
            # and penalize only off-diagonal prompt similarity.
            attention = self.conditional_prompt.last_attention.mean(dim=1)
            attention = F.normalize(attention, p=2, dim=-1)
            similarity = torch.matmul(attention, attention.transpose(-2, -1))
            identity = torch.eye(
                similarity.shape[-1], device=similarity.device,
                dtype=similarity.dtype,
            ).unsqueeze(0)
            diversity_loss = ((similarity - identity) ** 2).mean()
            self.last_prompt_diversity_loss = diversity_loss.detach()
            loss = loss + self.prompt_diversity_weight * diversity_loss
        return loss

    @torch.no_grad()
    def generate_text(self, features, lengths, generation_cfg):
        mapped, mapped_lengths = self._map(features, lengths)
        transformer_inputs = self.translation.prepare_feature_inputs(mapped, mapped_lengths)
        outputs = self.translation.generate(**transformer_inputs, **generation_cfg)
        return outputs["decoded_sequences"]


def trainable_state_dict(model):
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in trainable
    }


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable
