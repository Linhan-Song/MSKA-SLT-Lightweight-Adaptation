"""Qwen prompt experiment retained for completeness."""

import math

import torch
from torch import nn

from .prompt_modules import TemporalSoftPromptAdapter
from .translation_pipeline import apply_lora


class QwenSoftPromptModel(nn.Module):
    """E3: temporal soft prompts injected into a 4-bit Qwen causal LM."""

    def __init__(self, model_name, prompt_length=32, lora_rank=8, lora_alpha=16,
                 lora_dropout=0.05, instruction=None, load_in_4bit=True,
                 compute_dtype="bfloat16"):
        super().__init__()
        from peft import TaskType, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = getattr(torch, compute_dtype)
        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": {"": torch.cuda.current_device()},
        }
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        base = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        base.config.use_cache = False
        if load_in_4bit:
            base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
        else:
            base.gradient_checkpointing_enable()
            if hasattr(base, "enable_input_require_grads"):
                base.enable_input_require_grads()
        self.language_model = apply_lora(
            base, TaskType.CAUSAL_LM, lora_rank, lora_alpha,
            lora_dropout, ["q_proj", "v_proj"],
        )
        hidden_size = self.language_model.config.hidden_size
        model_device = self.language_model.get_input_embeddings().weight.device
        self.adapter = TemporalSoftPromptAdapter(512, hidden_size, prompt_length).to(model_device)
        self.prompt_scale = nn.Parameter(
            torch.tensor(1.0 / math.sqrt(hidden_size), dtype=torch.float32, device=model_device)
        )
        self.instruction = instruction or (
            "Übersetze die folgende Darstellung der Deutschen Gebärdensprache in einen natürlichen "
            "deutschen Satz. Gib ausschließlich den deutschen Satz aus, ohne Erklärung:\n"
        )

    def _prefix_embeddings(self, features, lengths):
        device = features.device
        visual = self.adapter(features, lengths) * self.prompt_scale
        prefix_ids = self.tokenizer(
            [self.instruction] * features.shape[0], add_special_tokens=True,
            padding=True, return_tensors="pt",
        )["input_ids"].to(device)
        prefix = self.language_model.get_input_embeddings()(prefix_ids)
        visual = visual.to(dtype=prefix.dtype)
        return torch.cat([prefix, visual], dim=1)

    def _training_batch(self, features, lengths, texts):
        device = features.device
        prefixes = self._prefix_embeddings(features, lengths)
        target_rows = self.tokenizer(texts, add_special_tokens=False)["input_ids"]
        eos = self.tokenizer.eos_token_id
        embed = self.language_model.get_input_embeddings()
        rows, labels = [], []
        for index, target in enumerate(target_rows):
            target = target + [eos]
            target_ids = torch.tensor(target, dtype=torch.long, device=device)
            target_embeds = embed(target_ids)
            row = torch.cat([prefixes[index], target_embeds], dim=0)
            label = torch.full((row.shape[0],), -100, dtype=torch.long, device=device)
            label[-len(target):] = target_ids
            rows.append(row)
            labels.append(label)
        max_length = max(row.shape[0] for row in rows)
        hidden = rows[0].shape[-1]
        inputs = torch.zeros(len(rows), max_length, hidden, dtype=rows[0].dtype, device=device)
        attention = torch.zeros(len(rows), max_length, dtype=torch.long, device=device)
        padded_labels = torch.full((len(rows), max_length), -100, dtype=torch.long, device=device)
        for index, (row, label) in enumerate(zip(rows, labels)):
            inputs[index, :row.shape[0]] = row
            attention[index, :row.shape[0]] = 1
            padded_labels[index, :label.shape[0]] = label
        return inputs, attention, padded_labels

    def forward(self, features, lengths, texts):
        inputs, attention, labels = self._training_batch(features, lengths, texts)
        outputs = self.language_model(
            inputs_embeds=inputs, attention_mask=attention, labels=labels,
            use_cache=False, return_dict=True,
        )
        return outputs.loss

    @torch.no_grad()
    def generate_text(self, features, lengths, generation_cfg):
        prefixes = self._prefix_embeddings(features, lengths)
        attention = torch.ones(prefixes.shape[:2], dtype=torch.long, device=features.device)
        outputs = self.language_model.generate(
            inputs_embeds=prefixes,
            attention_mask=attention,
            max_new_tokens=generation_cfg.get("max_new_tokens", 100),
            num_beams=generation_cfg.get("num_beams", 3),
            length_penalty=generation_cfg.get("length_penalty", 1.0),
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
