"""Audit completed test evaluations and build publication-ready result tables.

The script reads the immutable prediction files from the original experiment
workspace.  It never trains a model or changes a checkpoint.  All systems are
evaluated on the same 642 PHOENIX14T test references, and paired bootstrap
intervals resample complete sentence pairs.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


SYSTEMS = {
    "Original checkpoint": {
        "parameters": 0,
        "runs": ["outputs/official_b1_frozen_eval"],
        "checkpoints": ["pretrained_models/Phoenix-2014T_SLT/best.pth"],
    },
    "Full VLMapper": {
        "parameters": 1_574_912,
        "runs": [
            f"outputs/baseline_test_eval_seeds012_20260830/full_vlmapper_seed{seed}"
            for seed in range(3)
        ],
        "checkpoints": [
            "outputs/official_a1_vlmapper_finetune/best_checkpoint.pth",
            "outputs/next_official_a1_vlmapper_seed1/best_checkpoint.pth",
            "outputs/next_official_a1_vlmapper_seed2/best_checkpoint.pth",
        ],
    },
    "Gated Residual-128": {
        "parameters": 265_345,
        "runs": [
            f"outputs/baseline_test_eval_seeds012_20260830/nopool_gated128_seed{seed}"
            for seed in range(3)
        ],
        "checkpoints": [
            f"outputs/local_nopool_gated128_seed{seed}/best_checkpoint.pth"
            for seed in range(3)
        ],
    },
    "Gated Residual-128 + Local d96": {
        "parameters": 440_067,
        "runs": [
            f"outputs/final_local_r4_d96_h4_f192_seed{seed}_test_eval"
            for seed in range(3)
        ],
        "checkpoints": [
            "outputs/local_nopool_gated128_local_sweep_r4_d96_h4_f192_seed0/best_checkpoint.pth",
            "outputs/local_nopool_gated128_local_best_r4_d96_h4_f192_seed1/best_checkpoint.pth",
            "outputs/local_nopool_gated128_local_best_r4_d96_h4_f192_seed2/best_checkpoint.pth",
        ],
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260928)
    return parser.parse_args()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sentence_bleu_statistics(sacrebleu, references, hypotheses):
    """Return additive corpus-BLEU statistics for fast paired resampling."""
    statistics = np.zeros((len(references), 10), dtype=np.int64)
    for row, (reference, hypothesis) in enumerate(zip(references, hypotheses)):
        output = hypothesis.rstrip()
        ref_ngrams, _, closest_len = sacrebleu.ref_stats(output, [reference.rstrip()])
        system_ngrams = sacrebleu.extract_ngrams(output)
        correct = [0, 0, 0, 0]
        total = [0, 0, 0, 0]
        for ngram, count in system_ngrams.items():
            order = len(ngram.split()) - 1
            correct[order] += min(count, ref_ngrams.get(ngram, 0))
            total[order] += count
        statistics[row] = correct + total + [len(output.split()), closest_len]
    return statistics


def bleu4_from_statistics(sacrebleu, statistics, sampled):
    totals = statistics[sampled].sum(axis=0)
    return sacrebleu.compute_bleu(
        totals[:4].tolist(), totals[4:8].tolist(), int(totals[8]), int(totals[9]),
        smooth_method="floor", smooth_value=0.0, use_effective_order=True,
    ).scores[3]


def main():
    args = parse_args()
    root = args.experiment_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root))
    from metrics import bleu, rouge
    import sacrebleu

    manifest_root = root / "data/Phoenix-2014T/mska_feature_cache_official_slt"
    manifests = {
        split: read_jsonl(manifest_root / f"{split}.jsonl")
        for split in ("train", "dev", "test")
    }
    expected_counts = {"train": 7096, "dev": 519, "test": 642}
    manifest_counts = {split: len(rows) for split, rows in manifests.items()}
    if manifest_counts != expected_counts:
        raise RuntimeError(f"Unexpected manifest counts: {manifest_counts}")
    manifest_ids = {
        split: {row["name"].split("/", 1)[-1] for row in rows}
        for split, rows in manifests.items()
    }
    manifest_overlaps = {
        "train_dev": len(manifest_ids["train"] & manifest_ids["dev"]),
        "train_test": len(manifest_ids["train"] & manifest_ids["test"]),
        "dev_test": len(manifest_ids["dev"] & manifest_ids["test"]),
    }
    if any(manifest_overlaps.values()):
        raise RuntimeError(f"Data split overlap detected: {manifest_overlaps}")

    loaded = {}
    canonical_references = None
    audit_runs = []
    for system, specification in SYSTEMS.items():
        loaded[system] = []
        for seed, relative in enumerate(specification["runs"]):
            run_dir = root / relative
            metrics_path = run_dir / "eval_metrics.json"
            predictions_path = run_dir / "test_predictions.jsonl"
            checkpoint_path = root / specification["checkpoints"][seed]
            metrics = read_json(metrics_path)["test"]
            predictions = read_jsonl(predictions_path)
            references = [row["reference"] for row in predictions]
            hypotheses = [row["hypothesis"] for row in predictions]
            if canonical_references is None:
                canonical_references = references
            if references != canonical_references:
                raise RuntimeError(f"Reference mismatch: {predictions_path}")
            if len(references) != 642 or metrics["samples"] != 642:
                raise RuntimeError(f"Expected 642 test samples: {run_dir}")
            recomputed = bleu(references, hypotheses)
            recomputed["rouge"] = rouge(references, hypotheses)
            error = max(abs(float(recomputed[key]) - float(metrics[key])) for key in recomputed)
            if error > 1e-10:
                raise RuntimeError(f"Saved metric mismatch ({error}): {run_dir}")
            loaded[system].append({
                "seed": seed,
                "metrics": metrics,
                "references": references,
                "hypotheses": hypotheses,
                "bleu_statistics": sentence_bleu_statistics(
                    sacrebleu, references, hypotheses
                ),
            })
            statistic_score = bleu4_from_statistics(
                sacrebleu, loaded[system][-1]["bleu_statistics"], np.arange(642)
            )
            if abs(statistic_score - float(metrics["bleu4"])) > 1e-10:
                raise RuntimeError(f"BLEU sufficient-statistic mismatch: {run_dir}")
            audit_runs.append({
                "system": system,
                "seed": seed if len(specification["runs"]) > 1 else None,
                "directory": relative,
                "checkpoint": specification["checkpoints"][seed],
                "checkpoint_sha256": sha256(checkpoint_path),
                "samples": len(references),
                "metrics_sha256": sha256(metrics_path),
                "predictions_sha256": sha256(predictions_path),
                "max_recomputation_error": error,
            })

    metric_names = ["bleu1", "bleu2", "bleu3", "bleu4", "rouge", "loss"]
    principal_rows = []
    for system, runs in loaded.items():
        row = {
            "condition": system,
            "seeds": len(runs),
            "task_specific_parameters": SYSTEMS[system]["parameters"],
        }
        for metric in metric_names:
            values = np.array([run["metrics"][metric] for run in runs], dtype=float)
            row[f"{metric}_mean"] = f"{values.mean():.4f}"
            row[f"{metric}_std"] = "" if len(values) == 1 else f"{values.std(ddof=1):.4f}"
        principal_rows.append(row)
    principal_fields = ["condition", "seeds", "task_specific_parameters"] + [
        field for metric in metric_names for field in (f"{metric}_mean", f"{metric}_std")
    ]
    write_csv(args.output_dir / "test_principal_results.csv", principal_rows, principal_fields)

    multiseed_rows = []
    for seed in range(3):
        full = loaded["Full VLMapper"][seed]["metrics"]["bleu4"]
        gated = loaded["Gated Residual-128"][seed]["metrics"]["bleu4"]
        local = loaded["Gated Residual-128 + Local d96"][seed]["metrics"]["bleu4"]
        multiseed_rows.append({
            "seed": seed,
            "full_vlmapper": f"{full:.4f}",
            "gated_residual_128": f"{gated:.4f}",
            "local_r4_d96_h4_ffn192": f"{local:.4f}",
            "gated_minus_full": f"{gated - full:.4f}",
            "local_minus_gated": f"{local - gated:.4f}",
        })
    means = {
        "full_vlmapper": np.mean([run["metrics"]["bleu4"] for run in loaded["Full VLMapper"]]),
        "gated_residual_128": np.mean([run["metrics"]["bleu4"] for run in loaded["Gated Residual-128"]]),
        "local_r4_d96_h4_ffn192": np.mean([run["metrics"]["bleu4"] for run in loaded["Gated Residual-128 + Local d96"]]),
    }
    multiseed_rows.append({
        "seed": "mean",
        **{key: f"{value:.4f}" for key, value in means.items()},
        "gated_minus_full": f"{means['gated_residual_128'] - means['full_vlmapper']:.4f}",
        "local_minus_gated": f"{means['local_r4_d96_h4_ffn192'] - means['gated_residual_128']:.4f}",
    })
    write_csv(
        args.output_dir / "test_multiseed_bleu4.csv",
        multiseed_rows,
        ["seed", "full_vlmapper", "gated_residual_128", "local_r4_d96_h4_ffn192", "gated_minus_full", "local_minus_gated"],
    )

    rng = np.random.default_rng(args.bootstrap_seed)
    bootstrap_rows = []
    comparisons = [
        ("Gated Residual-128", "Full VLMapper"),
        ("Gated Residual-128 + Local d96", "Gated Residual-128"),
    ]
    for candidate_name, baseline_name in comparisons:
        for seed in range(3):
            candidate = loaded[candidate_name][seed]
            baseline = loaded[baseline_name][seed]
            observed = candidate["metrics"]["bleu4"] - baseline["metrics"]["bleu4"]
            differences = np.empty(args.bootstrap_samples, dtype=float)
            for index in range(args.bootstrap_samples):
                sampled = rng.integers(0, 642, size=642)
                differences[index] = (
                    bleu4_from_statistics(sacrebleu, candidate["bleu_statistics"], sampled)
                    - bleu4_from_statistics(sacrebleu, baseline["bleu_statistics"], sampled)
                )
            probability_positive = float(np.mean(differences > 0))
            two_sided_p = min(1.0, 2 * min(probability_positive, 1 - probability_positive))
            bootstrap_rows.append({
                "candidate": candidate_name,
                "baseline": baseline_name,
                "seed": seed,
                "observed_bleu4_difference": f"{observed:.4f}",
                "ci95_low": f"{np.quantile(differences, 0.025):.4f}",
                "ci95_high": f"{np.quantile(differences, 0.975):.4f}",
                "probability_positive": f"{probability_positive:.4f}",
                "two_sided_p": f"{two_sided_p:.4f}",
                "bootstrap_samples": args.bootstrap_samples,
            })
            print(f"bootstrap {candidate_name} vs {baseline_name} seed={seed} done", flush=True)
    write_csv(
        args.output_dir / "test_paired_bootstrap.csv",
        bootstrap_rows,
        ["candidate", "baseline", "seed", "observed_bleu4_difference", "ci95_low", "ci95_high", "probability_positive", "two_sided_p", "bootstrap_samples"],
    )

    audit = {
        "test_samples": 642,
        "manifest_counts": manifest_counts,
        "manifest_id_overlaps": manifest_overlaps,
        "all_references_identical": True,
        "metric_recomputation_tolerance": 1e-10,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_samples": args.bootstrap_samples,
        "runs": audit_runs,
    }
    (args.output_dir / "test_evaluation_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
