import json
import tempfile
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mska_slt_adaptation.data import CachedFeatureDataset, collate_cached_features
from mska_slt_adaptation.models.alternative_connectors import (
    HoulsbyEncoderAdapter,
    QFormerConnector,
    SoftGlossFusionAdapter,
)
from mska_slt_adaptation.models.exploratory_adapters import SEGatedResidualAdapter
from mska_slt_adaptation.models.prompt_modules import (
    ConditionalSoftPrompt,
    TemporalCrossAttentionSoftPrompt,
    TemporalSoftPromptAdapter,
)
from mska_slt_adaptation.models.residual_adapters import (
    FixedResidualAdapter,
    GatedResidualAdapter,
    PlainResidualAdapter,
)
from mska_slt_adaptation.models.sequence_processing import TemporalCompressor
from mska_slt_adaptation.models.temporal_refiner import TemporalRefiner


def test_residual_gate_ablation_adapters_are_capacity_matched():
    features = torch.randn(2, 5, 16)
    ungated = PlainResidualAdapter(16, 4, dropout=0.0)
    fixed = FixedResidualAdapter(16, 4, dropout=0.0, initial_gate=-4.0)
    gated = GatedResidualAdapter(16, 4, dropout=0.0, initial_gate=-4.0)

    fixed.load_state_dict(ungated.state_dict(), strict=False)
    gated.load_state_dict(ungated.state_dict(), strict=False)
    ungated_delta = ungated(features) - features
    fixed_delta = fixed(features) - features
    gated_delta = gated(features) - features
    expected_scale = torch.sigmoid(torch.tensor(-4.0))

    assert torch.allclose(fixed_delta, expected_scale * ungated_delta, atol=1e-6)
    assert torch.allclose(gated_delta, expected_scale * ungated_delta, atol=1e-6)
    assert sum(p.numel() for p in gated.parameters()) == (
        sum(p.numel() for p in ungated.parameters()) + 1
    )


def test_capped_average_only_compresses_long_sequences():
    compressor = TemporalCompressor(
        compressor_type="capped_average", feature_dim=2, prompt_length=5,
        dropout=0.0,
    )
    features = torch.zeros(2, 7, 2)
    features[0, :3] = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    features[1] = torch.arange(14, dtype=torch.float32).reshape(7, 2)
    output, output_lengths = compressor(features, torch.tensor([3, 7]))

    assert output.shape == (2, 5, 2)
    assert output_lengths.tolist() == [3, 5]
    assert torch.equal(output[0, :3], features[0, :3])
    assert torch.count_nonzero(output[0, 3:]) == 0
    assert torch.isfinite(output[1]).all()


def test_adapter_shape_and_gradient():
    adapter = TemporalSoftPromptAdapter(512, 1024, prompt_length=32, dropout=0.0)
    features = torch.randn(2, 17, 512, requires_grad=True)
    lengths = torch.tensor([17, 9])
    output = adapter(features, lengths)
    assert output.shape == (2, 32, 1024)
    output.mean().backward()
    assert features.grad is not None


def test_temporal_refiner_mask_gate_and_gradient():
    refiner = TemporalRefiner(
        feature_dim=8, attention_dim=4, num_heads=2, ffn_dim=8,
        num_layers=1, dropout=0.0, initial_gate=-4.0,
    )
    features = torch.randn(2, 6, 8, requires_grad=True)
    lengths = torch.tensor([6, 3])
    changed_padding = features.detach().clone()
    changed_padding[1, 3:] = 1000.0
    output = refiner(features, lengths)
    changed_output = refiner(changed_padding, lengths)

    assert output.shape == features.shape
    assert torch.allclose(output[1, :3], changed_output[1, :3], atol=1e-5)
    assert torch.equal(output[1, 3:], features[1, 3:])
    output.mean().backward()
    assert features.grad is not None
    assert refiner.gate_logit.grad is not None
    assert refiner.layers[0]["attention"].in_proj_weight.grad is not None


def test_local_and_local_global_temporal_attention_modes():
    features = torch.randn(2, 9, 8, requires_grad=True)
    lengths = torch.tensor([9, 5])
    for mode in ("local", "local_global"):
        refiner = TemporalRefiner(
            feature_dim=8, attention_dim=4, num_heads=2, ffn_dim=8,
            num_layers=1, dropout=0.0, initial_gate=-4.0,
            attention_mode=mode, local_radius=2, global_stride=3,
        )
        changed = features.detach().clone()
        changed[1, 5:] = 1000.0
        output = refiner(features, lengths)
        changed_output = refiner(changed, lengths)
        assert output.shape == features.shape
        assert torch.allclose(output[1, :5], changed_output[1, :5], atol=1e-5)
        assert torch.equal(output[1, 5:], features[1, 5:])
        output.mean().backward(retain_graph=True)
        assert refiner.gate_logit.grad is not None
        if mode == "local_global":
            assert refiner.layers[0]["global_attention"].in_proj_weight.grad is not None


def test_zero_output_projection_starts_as_identity():
    refiner = TemporalRefiner(
        feature_dim=8, attention_dim=4, num_heads=2, ffn_dim=8,
        num_layers=1, dropout=0.0, initial_gate=-4.0,
        attention_mode="local", local_radius=2,
        zero_output_projection=True,
    )
    features = torch.randn(2, 7, 8, requires_grad=True)
    lengths = torch.tensor([7, 4])
    output = refiner(features, lengths)
    assert torch.equal(output, features)
    assert refiner.position_scale.shape == (1,)
    assert torch.count_nonzero(refiner.output_projection.weight) == 0
    output.sum().backward()
    assert refiner.output_projection.weight.grad is not None


def test_se_adapter_shape_gradient_and_identity_scale():
    adapter = SEGatedResidualAdapter(
        dimension=16, bottleneck_dim=4, se_hidden_dim=2,
        dropout=0.0, initial_gate=-4.0,
    )
    features = torch.randn(2, 7, 16, requires_grad=True)
    lengths = torch.tensor([7, 4])
    normalized = adapter.pre_norm(features)
    context = torch.stack([normalized[0].mean(0), normalized[1, :4].mean(0)])
    scale = 2.0 * torch.sigmoid(
        adapter.se_up(adapter.activation(adapter.se_down(context)))
    )
    assert torch.allclose(scale, torch.ones_like(scale))
    output = adapter(features, lengths)
    assert output.shape == features.shape
    output.mean().backward()
    assert features.grad is not None


def test_qformer_residual_and_replacement_connectors():
    recognition = torch.randn(2, 11, 8)
    lengths = torch.tensor([11, 6])
    mapped_features = torch.randn(2, 4, 16)
    residual = QFormerConnector(
        input_dim=8, output_dim=16, token_count=4,
        attention_dim=8, num_heads=2, ffn_dim=12, dropout=0.0,
        replacement=False,
    )
    residual_output = residual(recognition, lengths, mapped_features)
    assert residual_output.shape == mapped_features.shape
    residual_output.mean().backward()
    assert residual.gate_logit.grad is not None
    replacement = QFormerConnector(
        input_dim=8, output_dim=16, token_count=4,
        attention_dim=8, num_heads=2, ffn_dim=12, dropout=0.0,
        replacement=True,
    )
    replacement_output = replacement(recognition, lengths)
    assert replacement_output.shape == mapped_features.shape
    replacement_output.mean().backward()
    assert replacement.query_tokens.grad is not None


def test_houlsby_adapter_shape_and_gradient():
    adapter = HoulsbyEncoderAdapter(
        dimension=16, bottleneck_dim=4, dropout=0.0, initial_gate=-4.0
    )
    hidden = torch.randn(2, 5, 16)
    output = adapter(hidden)
    assert output.shape == hidden.shape
    output.mean().backward()
    assert adapter.down.weight.grad is not None
    assert adapter.gate_logit.grad is not None


def test_conditional_prompt_shape_and_gradient():
    generator = ConditionalSoftPrompt(
        input_dim=8, output_dim=16, prompt_length=3,
        hidden_dim=4, rank=2, initial_gate=-4.0,
    )
    features = torch.randn(2, 5, 8, requires_grad=True)
    lengths = torch.tensor([5, 3])
    base_prompt = torch.randn(3, 16)
    prompt = generator(features, lengths, base_prompt)
    assert prompt.shape == (2, 3, 16)
    prompt.mean().backward()
    assert features.grad is not None


def test_soft_gloss_fusion_shape_gradient_and_frozen_buffers():
    vocabulary_size = 6
    classifier_weight = torch.randn(vocabulary_size, 8)
    classifier_bias = torch.randn(vocabulary_size)
    aligned_embeddings = torch.randn(vocabulary_size, 16)
    aligned_embeddings[0].zero_()
    fusion = SoftGlossFusionAdapter(
        classifier_weight=classifier_weight,
        classifier_bias=classifier_bias,
        aligned_embeddings=aligned_embeddings,
        output_dim=16,
        bottleneck_dim=4,
        dropout=0.0,
        initial_gate=-4.0,
    )
    recognition = torch.randn(2, 7, 8)
    visual = torch.randn(2, 5, 16)
    output = fusion(recognition, torch.tensor([7, 4]), visual, target_length=5)
    assert output.shape == visual.shape
    output.mean().backward()
    assert fusion.down.weight.grad is not None
    assert fusion.up.weight.grad is not None
    assert fusion.gate_logit.grad is not None
    parameter_names = {name for name, _ in fusion.named_parameters()}
    assert "classifier_weight" not in parameter_names
    assert "aligned_embeddings" not in parameter_names


def test_position_aware_temporal_prompt_attention():
    generator = TemporalCrossAttentionSoftPrompt(
        input_dim=8, output_dim=16, prompt_length=3,
        attention_dim=8, num_heads=2, dropout=0.0,
        initial_gate=-2.0, use_position_encoding=True,
    )
    features = torch.randn(2, 6, 8)
    prompt = generator(features, torch.tensor([6, 4]), torch.randn(3, 16))
    assert prompt.shape == (2, 3, 16)
    assert generator.last_attention.shape == (2, 2, 3, 6)
    assert torch.all(generator.last_attention[1, :, :, 4:] < 1e-6)
    prompt.mean().backward()
    assert generator.position_scale.grad is not None


def test_cached_dataset_and_collate(tmp_path):
    sample_dir = tmp_path / "train"
    sample_dir.mkdir()
    records = []
    for index, length in enumerate([4, 7]):
        sample_path = sample_dir / f"{index:08d}.pt"
        torch.save({
            "name": f"sample-{index}",
            "text": f"text {index}",
            "gloss": f"gloss {index}",
            "length": length,
            "feature": torch.randn(length, 512).half(),
        }, sample_path)
        relative_path = (
            f"train\\{sample_path.name}" if index == 1
            else f"train/{sample_path.name}"
        )
        records.append({"name": f"sample-{index}", "feature_path": relative_path})
    manifest = tmp_path / "train.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    dataset = CachedFeatureDataset(manifest)
    batch = collate_cached_features([dataset[0], dataset[1]])
    assert batch["features"].shape == (2, 7, 512)
    assert batch["lengths"].tolist() == [4, 7]


if __name__ == "__main__":
    test_residual_gate_ablation_adapters_are_capacity_matched()
    test_capped_average_only_compresses_long_sequences()
    test_adapter_shape_and_gradient()
    test_temporal_refiner_mask_gate_and_gradient()
    test_local_and_local_global_temporal_attention_modes()
    test_zero_output_projection_starts_as_identity()
    test_se_adapter_shape_gradient_and_identity_scale()
    test_qformer_residual_and_replacement_connectors()
    test_houlsby_adapter_shape_and_gradient()
    test_conditional_prompt_shape_and_gradient()
    test_soft_gloss_fusion_shape_gradient_and_frozen_buffers()
    test_position_aware_temporal_prompt_attention()
    with tempfile.TemporaryDirectory() as directory:
        test_cached_dataset_and_collate(Path(directory))
    print("component smoke tests passed")
