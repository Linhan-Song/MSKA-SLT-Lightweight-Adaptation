"""Train or evaluate one cached-feature experiment."""

import argparse
import copy
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ..data import CachedFeatureDataset, collate_cached_features
from ..models.translation_pipeline import count_parameters
from .checkpoints import (
    apply_parameter_freezing,
    load_checkpoint,
    load_initial_weights,
    save_checkpoint,
)
from .evaluation import evaluate, write_predictions
from .model_factory import build_model


STOP_REQUESTED = False


def request_stop(signum, _frame):
    global STOP_REQUESTED
    print(f"Received signal {signum}; checkpointing at the next safe boundary.", flush=True)
    STOP_REQUESTED = True


def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate an MSKA-SLT adaptation experiment.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--interface-adapter", default="")
    parser.add_argument("--sequence-processing", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prompt-length", type=int, default=None)
    parser.add_argument("--bottleneck-dim", type=int, default=None)
    parser.add_argument("--initial-gate", type=float, default=None)
    return parser.parse_args()


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        torch.use_deterministic_algorithms(True, warn_only=True)
        print(
            "Deterministic mode enabled: cuDNN deterministic, TF32 off, "
            "math SDPA forced",
            flush=True,
        )


def make_loader(manifest, batch_size, workers, shuffle, seed):
    dataset = CachedFeatureDataset(manifest)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=True, collate_fn=collate_cached_features, generator=generator,
    )


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root / "src"))
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if args.interface_adapter:
        cfg.setdefault("model", {}).setdefault("interface_adapter", {})["type"] = args.interface_adapter
    if args.sequence_processing:
        cfg.setdefault("model", {}).setdefault("sequence_processing", {})["mode"] = args.sequence_processing
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.prompt_length is not None:
        cfg.setdefault("model", {}).setdefault(
            "sequence_processing", {}
        )["target_length"] = args.prompt_length
    if args.bottleneck_dim is not None:
        cfg.setdefault("model", {}).setdefault("interface_adapter", {})["bottleneck_dim"] = args.bottleneck_dim
    if args.initial_gate is not None:
        cfg.setdefault("model", {}).setdefault("interface_adapter", {})["initial_gate"] = args.initial_gate
    deterministic = bool(cfg.get("training", {}).get("deterministic", False))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_everything(cfg.get("seed", 0), deterministic=deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Training requires an NVIDIA GPU.")

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(cfg)
    initial_checkpoints = cfg.get("model", {}).get("initial_checkpoints", [])
    if isinstance(initial_checkpoints, str):
        initial_checkpoints = [initial_checkpoints]
    for initial_checkpoint in initial_checkpoints:
        load_initial_weights(initial_checkpoint, model)
    apply_parameter_freezing(
        model, cfg.get("model", {}).get("freeze_parameter_prefixes", [])
    )
    if cfg["experiment"].lower() != "e3":
        model.to(device)
    total, trainable = count_parameters(model)
    print(f"Parameters: total={total:,}, trainable={trainable:,} ({100 * trainable / total:.3f}%)")
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    print("Trainable parameter tensors:", flush=True)
    for name in trainable_names:
        print(f"  {name}", flush=True)
    if not trainable_names:
        raise RuntimeError("No trainable parameters were selected.")
    if not cfg.get("model", {}).get("use_lora", True):
        forbidden = [
            name for name in trainable_names
            if name.startswith("translation.model") or "gloss_embedding" in name or "lora_" in name
        ]
        if forbidden:
            raise RuntimeError(f"Frozen-mBART run has unexpected trainable parameters: {forbidden}")

    adapter_parameters, prompt_parameters, lora_parameters = [], [], []
    named_adapter_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_" in name:
            lora_parameters.append(parameter)
        elif name == "soft_prompt" or name.startswith("conditional_prompt."):
            prompt_parameters.append(parameter)
        else:
            adapter_parameters.append(parameter)
            named_adapter_parameters.append((name, parameter))
    parameter_groups = []
    temporal_learning_rate = cfg["training"].get("temporal_learning_rate")
    if adapter_parameters and temporal_learning_rate is not None:
        group_specs = {
            "temporal_decay": [[], float(temporal_learning_rate), True],
            "temporal_no_decay": [[], float(temporal_learning_rate), False],
            "interface_decay": [
                [], float(cfg["training"].get("adapter_learning_rate", 1e-5)), True
            ],
            "interface_no_decay": [
                [], float(cfg["training"].get("adapter_learning_rate", 1e-5)), False
            ],
            "interface_gate": [
                [], float(cfg["training"].get("adapter_gate_learning_rate", 5e-6)), False
            ],
            "other_decay": [
                [], float(cfg["training"].get("adapter_learning_rate", 1e-5)), True
            ],
            "other_no_decay": [
                [], float(cfg["training"].get("adapter_learning_rate", 1e-5)), False
            ],
        }
        for name, parameter in named_adapter_parameters:
            no_decay = (
                name.endswith(".bias")
                or "norm" in name.lower()
                or "gate_logit" in name
                or "position_scale" in name
            )
            if name == "interface_adapter.gate_logit":
                group_name = "interface_gate"
            elif name.startswith("temporal_refiner."):
                group_name = "temporal_no_decay" if no_decay else "temporal_decay"
            elif name.startswith("interface_adapter."):
                group_name = "interface_no_decay" if no_decay else "interface_decay"
            else:
                group_name = "other_no_decay" if no_decay else "other_decay"
            group_specs[group_name][0].append(parameter)
        weight_decay = float(cfg["training"].get("weight_decay", 0.01))
        for group_name, (parameters, learning_rate, use_decay) in group_specs.items():
            if parameters:
                parameter_groups.append({
                    "params": parameters,
                    "lr": learning_rate,
                    "weight_decay": weight_decay if use_decay else 0.0,
                    "group_name": group_name,
                })
                print(
                    f"Optimizer group {group_name}: tensors={len(parameters)} "
                    f"lr={learning_rate:g} weight_decay="
                    f"{weight_decay if use_decay else 0.0:g}",
                    flush=True,
                )
    elif adapter_parameters:
        parameter_groups.append({
            "params": adapter_parameters,
            "lr": float(cfg["training"].get("adapter_learning_rate", 1e-3)),
            "weight_decay": float(cfg["training"].get("weight_decay", 0.01)),
        })
    if lora_parameters:
        parameter_groups.append({
            "params": lora_parameters,
            "lr": float(cfg["training"].get("lora_learning_rate", 2e-4)),
            "weight_decay": float(cfg["training"].get("weight_decay", 0.01)),
        })
    if prompt_parameters:
        parameter_groups.append({
            "params": prompt_parameters,
            "lr": float(cfg["training"].get("prompt_learning_rate", 5e-4)),
            "weight_decay": float(cfg["training"].get("prompt_weight_decay", 0.0)),
        })
    optimizer = torch.optim.AdamW(
        parameter_groups,
    )
    epochs = int(cfg["training"].get("epochs", 3))
    warmup_ratio = float(cfg["training"].get("warmup_ratio", 0.0))
    step_scheduler = warmup_ratio > 0
    if step_scheduler:
        accumulation_for_schedule = int(
            cfg["training"].get("gradient_accumulation_steps", 8)
        )
        train_examples = len(CachedFeatureDataset(cfg["data"]["train_manifest"]))
        optimizer_steps_per_epoch = max(1, train_examples // accumulation_for_schedule)
        total_optimizer_steps = max(1, optimizer_steps_per_epoch * epochs)
        warmup_steps = max(1, int(total_optimizer_steps * warmup_ratio))

        def learning_rate_factor(step):
            if step < warmup_steps:
                return max(1.0 / warmup_steps, float(step + 1) / warmup_steps)
            progress = min(
                1.0,
                float(step - warmup_steps)
                / max(1, total_optimizer_steps - warmup_steps),
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, learning_rate_factor
        )
        print(
            f"Step scheduler: total_steps={total_optimizer_steps} "
            f"warmup_steps={warmup_steps}",
            flush=True,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )

    teacher_interface_adapter = None
    distillation_weight = float(
        cfg["training"].get("feature_distillation_weight", 0.0)
    )
    if distillation_weight > 0:
        if getattr(model, "interface_adapter", None) is None:
            raise ValueError("Interface-feature distillation requires interface_adapter")
        teacher_interface_adapter = copy.deepcopy(model.interface_adapter).to(device).eval()
        for parameter in teacher_interface_adapter.parameters():
            parameter.requires_grad = False
        print(
            f"Enabled frozen-Interface-feature distillation with weight={distillation_weight:g}",
            flush=True,
        )
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(cfg["training"].get("use_grad_scaler", True))
    )

    state = {
        "epoch": 0,
        "next_batch": 0,
        "global_step": 0,
        "optimizer_steps": 0,
        "elapsed_seconds": 0.0,
        "best_bleu4": -1.0,
        "epochs_without_improvement": 0,
    }
    resume_path = args.resume or cfg["training"].get("resume", "")
    if resume_path:
        state.update(load_checkpoint(resume_path, model, optimizer, scheduler))
        print(f"Resumed from {resume_path}: {state}")

    batch_size = int(cfg["training"].get("batch_size", 1))
    workers = int(cfg["training"].get("num_workers", 2))
    dev_loader = make_loader(cfg["data"]["dev_manifest"], batch_size, workers, False, cfg.get("seed", 0))
    test_loader = make_loader(cfg["data"]["test_manifest"], batch_size, workers, False, cfg.get("seed", 0))
    generation_cfg = cfg.get("generation", {"num_beams": 3, "max_new_tokens": 100})
    autocast_enabled = bool(cfg["training"].get("use_autocast", True))
    autocast_dtype = getattr(
        torch,
        cfg["training"].get(
            "autocast_dtype", cfg.get("model", {}).get("compute_dtype", "float16")
        ),
    )

    if args.eval_only:
        dev_scores, dev_refs, dev_hyps = evaluate(
            model, dev_loader, device, generation_cfg, args.max_eval_batches,
            autocast_enabled, autocast_dtype,
        )
        test_scores, test_refs, test_hyps = evaluate(
            model, test_loader, device, generation_cfg, args.max_eval_batches,
            autocast_enabled, autocast_dtype,
        )
        print(json.dumps({"dev": dev_scores, "test": test_scores}, ensure_ascii=False, indent=2))
        with (output_dir / "eval_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump({"dev": dev_scores, "test": test_scores}, handle, ensure_ascii=False, indent=2)
        write_predictions(output_dir / "dev_predictions.jsonl", dev_refs, dev_hyps)
        write_predictions(output_dir / "test_predictions.jsonl", test_refs, test_hyps)
        return

    accumulation = int(cfg["training"].get("gradient_accumulation_steps", 8))
    checkpoint_every = int(cfg["training"].get("checkpoint_every_steps", 1000))
    max_runtime_seconds = float(cfg.get("runtime", {}).get("max_gpu_hours", 52)) * 3600
    run_start = time.time()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(int(state["epoch"]), epochs):
        train_loader = make_loader(
            cfg["data"]["train_manifest"], batch_size, workers, True,
            cfg.get("seed", 0) + epoch,
        )
        freeze_adapter = epoch < int(
            cfg["training"].get("adapter_freeze_epochs", 0)
        )
        model.train()
        if freeze_adapter and getattr(model, "interface_adapter", None) is not None:
            model.interface_adapter.eval()
            print(f"Epoch {epoch + 1}: residual adapter frozen", flush=True)
        start_batch = int(state["next_batch"]) if epoch == int(state["epoch"]) else 0
        for batch_index, batch in enumerate(train_loader):
            if batch_index < start_batch:
                continue
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            features = batch["features"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(
                enabled=autocast_enabled, dtype=autocast_dtype
            ):
                unscaled_loss = model(features, lengths, batch["texts"])
                if teacher_interface_adapter is not None:
                    with torch.no_grad():
                        teacher_compressed, teacher_lengths = model.compressor(
                            features, lengths
                        )
                        teacher_base = model.mapper({
                            "gloss_feature": teacher_compressed
                        })
                        teacher_mapped = teacher_interface_adapter(
                            teacher_base, teacher_lengths
                        )
                    student_mapped = model.last_mapped_features
                    distillation_loss = (
                        (student_mapped.float() - teacher_mapped.float()).pow(2).mean()
                        / teacher_mapped.float().pow(2).mean().clamp_min(1e-8)
                    )
                    unscaled_loss = (
                        unscaled_loss
                        + distillation_weight * distillation_loss
                    )
                if not torch.isfinite(unscaled_loss):
                    raise FloatingPointError(
                        f"Non-finite loss at epoch={epoch} batch={batch_index}: {unscaled_loss.item()}"
                    )
                loss = unscaled_loss / accumulation
            scaler.scale(loss).backward()
            state["global_step"] += 1
            if state["global_step"] % accumulation == 0:
                scaler.unscale_(optimizer)
                if freeze_adapter and getattr(model, "interface_adapter", None) is not None:
                    for parameter in model.interface_adapter.parameters():
                        parameter.grad = None
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
                )
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_stepped = (
                    not scaler.is_enabled()
                    or scaler.get_scale() >= previous_scale
                )
                optimizer.zero_grad(set_to_none=True)
                if optimizer_stepped:
                    state["optimizer_steps"] += 1
                    if step_scheduler:
                        scheduler.step()
            state["epoch"] = epoch
            state["next_batch"] = batch_index + 1
            elapsed = state["elapsed_seconds"] + (time.time() - run_start)
            if batch_index % 100 == 0:
                print(
                    f"epoch={epoch + 1}/{epochs} batch={batch_index}/{len(train_loader)} "
                    f"loss={float(loss.detach().cpu()) * accumulation:.4f} "
                    f"elapsed_hours={elapsed / 3600:.2f}", flush=True,
                )
            should_checkpoint = state["global_step"] % checkpoint_every == 0
            should_stop = STOP_REQUESTED or elapsed >= max_runtime_seconds
            if should_checkpoint or should_stop:
                snapshot = dict(state)
                snapshot["elapsed_seconds"] = elapsed
                save_checkpoint(output_dir / "checkpoint.pth", model, optimizer, scheduler, snapshot, cfg)
            if should_stop:
                print("Stopped safely because of a signal or runtime limit.", flush=True)
                return

        if not step_scheduler:
            scheduler.step()
        state["epoch"] = epoch + 1
        state["next_batch"] = 0
        state["elapsed_seconds"] += time.time() - run_start
        run_start = time.time()
        dev_scores, dev_refs, dev_hyps = evaluate(
            model, dev_loader, device, generation_cfg, args.max_eval_batches,
            autocast_enabled, autocast_dtype,
        )
        improved = dev_scores["bleu4"] > state["best_bleu4"]
        if improved:
            state["best_bleu4"] = dev_scores["bleu4"]
            state["epochs_without_improvement"] = 0
            save_checkpoint(output_dir / "best_checkpoint.pth", model, optimizer, scheduler, state, cfg)
            write_predictions(output_dir / "best_dev_predictions.jsonl", dev_refs, dev_hyps)
        else:
            state["epochs_without_improvement"] += 1
        save_checkpoint(output_dir / "checkpoint.pth", model, optimizer, scheduler, state, cfg)
        with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"epoch": epoch + 1, "dev": dev_scores}, ensure_ascii=False) + "\n")
        print(f"DEV epoch {epoch + 1}: {json.dumps(dev_scores, ensure_ascii=False)}")
        patience = int(cfg["training"].get("early_stopping_patience", 0))
        if patience and state["epochs_without_improvement"] >= patience:
            print(
                f"Early stopping after {state['epochs_without_improvement']} "
                "epochs without dev BLEU-4 improvement.",
                flush=True,
            )
            break

    load_checkpoint(output_dir / "best_checkpoint.pth", model)
    dev_scores, dev_refs, dev_hyps = evaluate(
        model, dev_loader, device, generation_cfg, args.max_eval_batches,
        autocast_enabled, autocast_dtype,
    )
    run_test = bool(cfg.get("evaluation", {}).get("run_test_after_training", True))
    test_scores, test_refs, test_hyps = None, [], []
    if run_test:
        test_scores, test_refs, test_hyps = evaluate(
            model, test_loader, device, generation_cfg, args.max_eval_batches,
            autocast_enabled, autocast_dtype,
        )
    model_config = cfg.get("model", {})
    exploratory_config = model_config.get("exploratory", {})
    final = {
        "experiment": cfg["experiment"],
        "sequence_processing": model_config.get("sequence_processing", {}).get("mode"),
        "interface_adapter": model_config.get("interface_adapter", {}).get("type"),
        "soft_prompt_length": exploratory_config.get("soft_prompt_length", 0),
        "conditional_prompt": exploratory_config.get("conditional_prompt", False),
        "soft_gloss_fusion": exploratory_config.get("soft_gloss_fusion", False),
        "soft_gloss_gate": (
            float(torch.sigmoid(model.soft_gloss_adapter.gate_logit).detach().cpu())
            if getattr(model, "soft_gloss_adapter", None) is not None else None
        ),
        "prompt_gate": (
            float(torch.sigmoid(model.conditional_prompt.gate_logit).detach().cpu())
            if getattr(model, "conditional_prompt", None) is not None else None
        ),
        "prompt_position_encoding": exploratory_config.get(
            "prompt_position_encoding", False
        ),
        "prompt_diversity_weight": exploratory_config.get(
            "prompt_diversity_weight", 0.0
        ),
        "temporal_refiner": model_config.get("temporal_refiner", {}),
        "temporal_gate": (
            float(torch.sigmoid(model.temporal_refiner.gate_logit).detach().cpu())
            if getattr(model, "temporal_refiner", None) is not None else None
        ),
        "dev": dev_scores,
        "test": test_scores,
        "parameters": {"total": total, "trainable": trainable},
        "elapsed_seconds": state["elapsed_seconds"],
    }
    with (output_dir / "final_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(final, handle, ensure_ascii=False, indent=2)
    write_predictions(output_dir / "dev_predictions.jsonl", dev_refs, dev_hyps)
    if run_test:
        write_predictions(output_dir / "test_predictions.jsonl", test_refs, test_hyps)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
