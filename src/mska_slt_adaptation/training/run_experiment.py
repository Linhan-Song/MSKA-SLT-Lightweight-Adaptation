"""使用缓存 Recognition 特征训练或评估一个实验配置。

主实验不重复运行 Recognition 网络。本文件负责两阶段中当前阶段的优化：
第一阶段只训练 Gated Residual Adapter；第二阶段先加载并冻结同 seed 的
Adapter 检查点，再只训练 Local Temporal Refiner。
"""

import argparse
import json
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
    """接收中断信号后，在下一个安全位置保存检查点再退出。"""
    global STOP_REQUESTED
    print(f"Received signal {signum}; checkpointing at the next safe boundary.", flush=True)
    STOP_REQUESTED = True


def parse_args():
    """读取命令行参数；通常只需传入 --config。"""
    parser = argparse.ArgumentParser(description="Train or evaluate an MSKA-SLT adaptation experiment.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    return parser.parse_args()


def seed_everything(seed, deterministic=False):
    """同步设置 Python、NumPy、CPU 和所有 CUDA 设备的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # 完全确定性更利于复查，但通常会降低训练速度。
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
    """根据特征清单创建 DataLoader，并单独控制数据打乱的随机种子。"""
    dataset = CachedFeatureDataset(manifest)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=True, collate_fn=collate_cached_features, generator=generator,
    )


def main():
    """执行模型构建、训练、开发集选择和最终评估。"""
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root / "src"))
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    # YAML 保存一次实验的完整结构和训练设置。
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
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
    # 第二阶段的 final_model 配置在这里加载同一 seed 的第一阶段 Adapter。
    # load_initial_weights 只加载权重，不继承第一阶段的优化器和训练进度。
    initial_checkpoints = cfg.get("model", {}).get("initial_checkpoints", [])
    if isinstance(initial_checkpoints, str):
        initial_checkpoints = [initial_checkpoints]
    for initial_checkpoint in initial_checkpoints:
        load_initial_weights(initial_checkpoint, model)
    # 第二阶段的前缀为 interface_adapter，因此最终只有 temporal_refiner
    # 保持 requires_grad=True；梯度仍可以穿过冻结模块回传到 Refiner。
    apply_parameter_freezing(
        model, cfg.get("model", {}).get("freeze_parameter_prefixes", [])
    )
    model.to(device)
    total, trainable = count_parameters(model)
    print(f"Parameters: total={total:,}, trainable={trainable:,} ({100 * trainable / total:.3f}%)")
    # 启动训练前打印所有可训练参数，便于从日志检查冻结策略是否正确。
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    print("Trainable parameter tensors:", flush=True)
    for name in trainable_names:
        print(f"  {name}", flush=True)
    if not trainable_names:
        raise RuntimeError("No trainable parameters were selected.")
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(cfg["training"].get("adapter_learning_rate", 1e-3)),
        weight_decay=float(cfg["training"].get("weight_decay", 0.01)),
    )
    epochs = int(cfg["training"].get("epochs", 3))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(cfg["training"].get("use_grad_scaler", True))
    )

    # next_batch 使运行被时间限制或信号中断后可以从 epoch 中间继续。
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
        # 评估模式不创建训练 batch，只生成开发集和测试集结果。
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

    # batch_size=1 时通过梯度累积得到更大的有效 batch。
    accumulation = int(cfg["training"].get("gradient_accumulation_steps", 8))
    checkpoint_every = int(cfg["training"].get("checkpoint_every_steps", 1000))
    max_runtime_seconds = float(cfg.get("runtime", {}).get("max_gpu_hours", 52)) * 3600
    run_start = time.time()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(int(state["epoch"]), epochs):
        # 每个 epoch 改变 DataLoader seed，但同一实验重复运行仍可复现。
        train_loader = make_loader(
            cfg["data"]["train_manifest"], batch_size, workers, True,
            cfg.get("seed", 0) + epoch,
        )
        model.train()
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
                if not torch.isfinite(unscaled_loss):
                    raise FloatingPointError(
                        f"Non-finite loss at epoch={epoch} batch={batch_index}: {unscaled_loss.item()}"
                    )
                # 先除以累积步数，使累计后的梯度尺度接近一次大 batch 更新。
                loss = unscaled_loss / accumulation
            scaler.scale(loss).backward()
            state["global_step"] += 1
            if state["global_step"] % accumulation == 0:
                # unscale 后再裁剪，否则裁剪阈值会受到 GradScaler 比例影响。
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
                )
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # 溢出时 GradScaler 会跳过参数更新；此时学习率调度器也不应前进。
                optimizer_stepped = (
                    not scaler.is_enabled()
                    or scaler.get_scale() >= previous_scale
                )
                optimizer.zero_grad(set_to_none=True)
                if optimizer_stepped:
                    state["optimizer_steps"] += 1
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
            # 达到运行时间上限或收到 Ctrl+C 时，先保存可恢复检查点。
            should_stop = STOP_REQUESTED or elapsed >= max_runtime_seconds
            if should_checkpoint or should_stop:
                snapshot = dict(state)
                snapshot["elapsed_seconds"] = elapsed
                save_checkpoint(output_dir / "checkpoint.pth", model, optimizer, scheduler, snapshot, cfg)
            if should_stop:
                print("Stopped safely because of a signal or runtime limit.", flush=True)
                return

        scheduler.step()
        state["epoch"] = epoch + 1
        state["next_batch"] = 0
        state["elapsed_seconds"] += time.time() - run_start
        run_start = time.time()
        dev_scores, dev_refs, dev_hyps = evaluate(
            model, dev_loader, device, generation_cfg, args.max_eval_batches,
            autocast_enabled, autocast_dtype,
        )
        # 论文以开发集 BLEU-4 选择最佳 epoch，而不是使用测试集挑模型。
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
        # 连续若干 epoch 没有提升则提前结束，减少无效计算。
        patience = int(cfg["training"].get("early_stopping_patience", 0))
        if patience and state["epochs_without_improvement"] >= patience:
            print(
                f"Early stopping after {state['epochs_without_improvement']} "
                "epochs without dev BLEU-4 improvement.",
                flush=True,
            )
            break

    # 训练结束后重新加载最佳开发集检查点，避免用最后一个 epoch 报告结果。
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
    # final_metrics.json 保存结构设置、参数量、Gate 和最终指标，后续可直接
    # 汇总成论文表格，不必从控制台日志手工抄写。
    final = {
        "experiment": cfg["experiment"],
        "sequence_processing": model_config.get("sequence_processing", {}).get("mode"),
        "interface_adapter": model_config.get("interface_adapter", {}).get("type"),
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
