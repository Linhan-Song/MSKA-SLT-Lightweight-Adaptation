"""Run/resume the preregistered Local Temporal Refiner development grid."""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from itertools import product
from pathlib import Path


RADII = (2, 4, 8)
DIMENSIONS = (80, 96, 112)
HEADS = (2, 4, 8)
UPSTREAM = {
    seed: f"outputs/local_nopool_gated128_seed{seed}/best_checkpoint.pth"
    for seed in range(3)
}
STAGE1_PARAMETERS = {80: 135_618, 96: 174_722, 112: 217_922}
EXISTING = {
    (4, 80, 4, 160): "outputs/local_nopool_gated128_local_sweep_r4_d80_h4_f160_seed0",
    (4, 96, 2, 192): "outputs/local_nopool_gated128_local_candidate_r4_d96_h2_f192_seed0",
    (4, 96, 4, 192): "outputs/local_nopool_gated128_local_sweep_r4_d96_h4_f192_seed0",
    (8, 96, 2, 192): "outputs/local_nopool_gated128_local_candidate_r8_d96_h2_f192_seed0",
    (8, 96, 4, 192): "outputs/local_nopool_gated128_local_candidate_r8_d96_h4_f192_seed0",
}
SUMMARY_FIELDS = [
    "radius", "dimension", "heads", "ffn", "seed", "source", "bleu1",
    "bleu2", "bleu3", "bleu4", "rouge", "loss",
    "trainable_parameters", "elapsed_seconds", "run_directory",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--python", type=Path,
        default=Path(r"C:/Users/18730/.conda/envs/mska/python.exe"),
    )
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def queue_log(root, message):
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {message}"
    print(line, flush=True)
    with (root / "queue.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def make_config(seed, radius, dimension, heads, ffn, output_dir):
    return {
        "experiment": "frozen_mbart",
        "seed": seed,
        "mska_config": "configs/phoenix-2014t_s2t.yaml",
        "output_dir": str(output_dir),
        "data": {
            "train_manifest": "data/Phoenix-2014T/mska_feature_cache_official_slt/train.jsonl",
            "dev_manifest": "data/Phoenix-2014T/mska_feature_cache_official_slt/dev.jsonl",
            "test_manifest": "data/Phoenix-2014T/mska_feature_cache_official_slt/test.jsonl",
        },
        "model": {
            "use_lora": False,
            "freeze_gloss_embedding": True,
            "official_component_dir": "pretrained_models/Phoenix-2014T_SLT/components",
            "train_official_mapper": False,
            "compressor_type": "none",
            "gloss_temporal_attention": True,
            "gloss_attention_mode": "local",
            "gloss_attention_local_radius": radius,
            "gloss_attention_dim": dimension,
            "gloss_attention_heads": heads,
            "gloss_attention_ffn_dim": ffn,
            "gloss_attention_layers": 1,
            "gloss_attention_initial_gate": -4.0,
            "adapter_type": "official_gated_residual",
            "bottleneck_dim": 128,
            "initial_gate": -4.0,
            "dropout": 0.1,
            "initial_checkpoints": [UPSTREAM[seed]],
            "freeze_parameter_prefixes": ["residual_adapter"],
        },
        "training": {
            "epochs": 20,
            "early_stopping_patience": 4,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "adapter_learning_rate": 2.0e-4,
            "weight_decay": 0.01,
            "num_workers": 0,
            "checkpoint_every_steps": 1000,
        },
        "generation": {
            "num_beams": 5, "max_length": 100, "length_penalty": 1.0,
        },
        "evaluation": {"run_test_after_training": False},
        "budget": {"max_gpu_hours": 12},
    }


def validate_metrics(path, dimension, ffn):
    metrics = read_json(path)
    if metrics.get("test") is not None:
        raise RuntimeError(f"Test evaluation detected: {path}")
    if int(metrics["dev"]["samples"]) != 519:
        raise RuntimeError(f"Invalid development sample count: {path}")
    for field in ("bleu1", "bleu2", "bleu3", "bleu4", "rouge", "loss"):
        if not math.isfinite(float(metrics["dev"][field])):
            raise RuntimeError(f"Non-finite {field}: {path}")
    if ffn == 2 * dimension:
        actual = int(metrics["parameters"]["trainable"])
        if actual != STAGE1_PARAMETERS[dimension]:
            raise RuntimeError(f"Trainable parameters {actual} do not match grid: {path}")
    return metrics


def validate_reused_config(experiment_root, run_dir, radius, dimension, heads, ffn):
    metrics = read_json(run_dir / "final_metrics.json")
    config = None
    checkpoint = run_dir / "best_checkpoint.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    # Saved checkpoints hold the exact training config; load only this trusted local file.
    import torch
    config = torch.load(checkpoint, map_location="cpu")["config"]
    model = config["model"]
    expected = {
        "compressor_type": "none",
        "gloss_attention_mode": "local",
        "gloss_attention_local_radius": radius,
        "gloss_attention_dim": dimension,
        "gloss_attention_heads": heads,
        "gloss_attention_ffn_dim": ffn,
        "gloss_attention_layers": 1,
        "gloss_attention_initial_gate": -4.0,
        "initial_checkpoints": [UPSTREAM[0]],
        "freeze_parameter_prefixes": ["residual_adapter"],
    }
    for key, value in expected.items():
        if model.get(key) != value:
            raise RuntimeError(f"Reused configuration mismatch for {key}: {run_dir}")
    if config.get("seed") != 0:
        raise RuntimeError(f"Reused configuration is not seed 0: {run_dir}")
    if config.get("evaluation", {}).get("run_test_after_training") is not False:
        raise RuntimeError(f"Reused configuration permits test evaluation: {run_dir}")
    return metrics


def to_row(radius, dimension, heads, ffn, seed, source, run_dir, metrics):
    dev = metrics["dev"]
    return {
        "radius": radius,
        "dimension": dimension,
        "heads": heads,
        "ffn": ffn,
        "seed": seed,
        "source": source,
        "bleu1": dev["bleu1"],
        "bleu2": dev["bleu2"],
        "bleu3": dev["bleu3"],
        "bleu4": dev["bleu4"],
        "rouge": dev["rouge"],
        "loss": dev["loss"],
        "trainable_parameters": metrics["parameters"]["trainable"],
        "elapsed_seconds": metrics.get("elapsed_seconds", 0.0),
        "run_directory": str(run_dir),
    }


def run_one(args, seed, radius, dimension, heads, ffn, name):
    run_dir = (args.output_root / "runs" / name).resolve()
    config_path = (args.output_root / "configs" / f"{name}.json").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        config_path,
        make_config(seed, radius, dimension, heads, ffn, run_dir),
    )
    final_path = run_dir / "final_metrics.json"
    if final_path.exists():
        return run_dir, validate_metrics(final_path, dimension, ffn)
    command = [
        str(args.python), "-u", "-m", "budget_experiments.train_budget",
        "--config", str(config_path), "--output-dir", str(run_dir),
    ]
    checkpoint = run_dir / "checkpoint.pth"
    if checkpoint.exists():
        command.extend(["--resume", str(checkpoint)])
        queue_log(args.output_root, f"RESUME {name}")
    else:
        queue_log(args.output_root, f"START {name}")
    with (run_dir / "stdout.log").open("a", encoding="utf-8") as stdout, (
        run_dir / "stderr.log"
    ).open("a", encoding="utf-8") as stderr:
        completed = subprocess.run(
            command,
            cwd=args.experiment_root,
            stdout=stdout,
            stderr=stderr,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        )
    if completed.returncode or not final_path.exists():
        raise RuntimeError(f"{name} failed with exit code {completed.returncode}")
    metrics = validate_metrics(final_path, dimension, ffn)
    queue_log(args.output_root, f"COMPLETE {name} dev_BLEU4={metrics['dev']['bleu4']:.6f}")
    return run_dir, metrics


def select_best(rows):
    return min(
        rows,
        key=lambda row: (
            -float(row["bleu4"]),
            int(row["trainable_parameters"]),
            float(row["loss"]),
        ),
    )


def main():
    args = parse_args()
    args.experiment_root = args.experiment_root.resolve()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "configs").mkdir(exist_ok=True)
    (args.output_root / "runs").mkdir(exist_ok=True)
    if not args.python.exists():
        raise FileNotFoundError(args.python)
    for upstream in UPSTREAM.values():
        if not (args.experiment_root / upstream).exists():
            raise FileNotFoundError(args.experiment_root / upstream)
    write_json(args.output_root / "search_plan.json", {
        "radii": RADII,
        "dimensions": DIMENSIONS,
        "heads": HEADS,
        "stage1_ffn": "2 * dimension",
        "stage2_ffn_multipliers": [1, 2, 4],
        "selection": "max dev BLEU-4; tie fewer parameters; tie lower dev loss",
        "stage1_seed": 0,
        "test_evaluation": False,
        "additional_seeds": "deferred",
    })
    if args.plan_only:
        queue_log(args.output_root, "PLAN VALIDATED stage1=27 new=22 stage2_new=2")
        return
    queue_log(args.output_root, "QUEUE START stage1=27 test_evaluation=false")

    stage1 = []
    for radius, dimension, heads in product(RADII, DIMENSIONS, HEADS):
        ffn = 2 * dimension
        key = (radius, dimension, heads, ffn)
        if key in EXISTING:
            run_dir = args.experiment_root / EXISTING[key]
            metrics = validate_reused_config(
                args.experiment_root, run_dir, radius, dimension, heads, ffn
            )
            metrics = validate_metrics(run_dir / "final_metrics.json", dimension, ffn)
            source = "reused"
        else:
            name = f"local_grid_r{radius}_d{dimension}_h{heads}_f{ffn}_seed0"
            run_dir, metrics = run_one(
                args, 0, radius, dimension, heads, ffn, name
            )
            source = "new"
        stage1.append(to_row(
            radius, dimension, heads, ffn, 0, source, run_dir, metrics
        ))
        write_csv(args.output_root / "stage1_summary.csv", stage1)

    stage1_best = select_best(stage1)
    write_json(args.output_root / "stage1_selection.json", stage1_best)
    queue_log(args.output_root, f"STAGE1 SELECTED {stage1_best}")

    radius = int(stage1_best["radius"])
    dimension = int(stage1_best["dimension"])
    heads = int(stage1_best["heads"])
    stage2 = []
    for multiplier in (1, 2, 4):
        ffn = multiplier * dimension
        if multiplier == 2:
            row = dict(stage1_best)
            row["source"] = "reused_stage1"
        else:
            name = f"local_ffn_r{radius}_d{dimension}_h{heads}_f{ffn}_seed0"
            run_dir, metrics = run_one(
                args, 0, radius, dimension, heads, ffn, name
            )
            row = to_row(radius, dimension, heads, ffn, 0, "new", run_dir, metrics)
        stage2.append(row)
        write_csv(args.output_root / "stage2_ffn_summary.csv", stage2)

    selected = select_best(stage2)
    write_json(args.output_root / "selected_configuration.json", selected)
    queue_log(args.output_root, f"FINAL CONFIG SELECTED {selected}")
    final = {
        "configuration": {
            key: selected[key] for key in ("radius", "dimension", "heads", "ffn")
        },
        "seed": 0,
        "dev": {
            key: selected[key]
            for key in ("bleu1", "bleu2", "bleu3", "bleu4", "rouge", "loss")
        },
        "additional_seeds": "deferred",
        "test_evaluated": False,
    }
    write_json(args.output_root / "final_summary.json", final)
    queue_log(args.output_root, f"QUEUE COMPLETE {final}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"GRID SEARCH FAILED: {error}", file=sys.stderr, flush=True)
        raise
