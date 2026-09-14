import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Cache frozen MSKA visual features.")
    parser.add_argument("--config", default="configs/base/phoenix-2014t_s2t.yaml")
    parser.add_argument(
        "--output-dir", default="data/Phoenix-2014T/recognition_features"
    )
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--recognition-state", default="",
        help="Optional Recognition state dictionary extracted from the official SLT checkpoint.",
    )
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(path, records):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root / "src"))

    from mska_backbone.datasets import S2T_Dataset
    from mska_backbone.recognition import Recognition
    from mska_backbone.tokenizers import GlossTokenizer_S2G

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    device = torch.device(args.device)
    recognition_cfg = config["model"]["RecognitionNetwork"]
    model_args = SimpleNamespace(device=args.device)
    recognition = Recognition(cfg=recognition_cfg, args=model_args).to(device).eval()
    if args.recognition_state:
        state = torch.load(args.recognition_state, map_location="cpu")
        recognition.load_state_dict(state, strict=True)
        print(f"Loaded official Recognition state: {args.recognition_state}")
    for parameter in recognition.parameters():
        parameter.requires_grad = False

    tokenizer = GlossTokenizer_S2G(config["gloss"])
    split_paths = {
        "train": config["data"]["train_label_path"],
        "dev": config["data"]["dev_label_path"],
        "test": config["data"]["test_label_path"],
    }
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    metadata = {
        "source_config": str(Path(args.config).resolve()),
        "recognition_checkpoint": str(
            Path(args.recognition_state or recognition_cfg["pretrained_path"]).resolve()
        ),
        "recognition_checkpoint_sha256": file_sha256(
            args.recognition_state or recognition_cfg["pretrained_path"]
        ),
        "feature_key": recognition_cfg.get("gloss_feature_ensemble", "gloss_feature"),
        "dtype": "float16",
        "deterministic_preprocessing": True,
    }
    with (output_root / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    for split in args.splits:
        if split not in split_paths:
            raise ValueError(f"Unknown split: {split}")
        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_root / f"{split}.jsonl"
        dataset = S2T_Dataset(
            path=split_paths[split], tokenizer=tokenizer, config=config,
            args=model_args, phase="val", training_refurbish=True,
        )
        loader = DataLoader(
            dataset, batch_size=args.batch_size, num_workers=args.num_workers,
            collate_fn=dataset.collate_fn, shuffle=False, pin_memory=True,
        )
        records = []
        completed = 0
        with torch.inference_mode():
            for batch in loader:
                batch_count = len(batch["name"])
                expected_paths = [split_dir / f"{completed + row:08d}.pt" for row in range(batch_count)]
                if not args.overwrite and all(path.exists() for path in expected_paths):
                    for row, feature_path in enumerate(expected_paths):
                        records.append({
                            "name": batch["name"][row],
                            "feature_path": str(feature_path.relative_to(output_root)),
                        })
                        completed += 1
                    if completed % 250 == 0:
                        write_manifest(manifest_path, records)
                        print(f"[{split}] reused {completed}/{len(dataset)}")
                    if args.max_samples and completed >= args.max_samples:
                        break
                    continue
                outputs = recognition(batch)
                features = outputs["gloss_feature"].detach().cpu()
                lengths = outputs["input_lengths"].cpu().tolist()
                for row in range(features.shape[0]):
                    if args.max_samples and completed >= args.max_samples:
                        break
                    feature_path = split_dir / f"{completed:08d}.pt"
                    if args.overwrite or not feature_path.exists():
                        length = int(lengths[row])
                        with feature_path.open("wb") as feature_handle:
                            torch.save({
                                "name": batch["name"][row],
                                "text": batch["text"][row],
                                "gloss": batch["gloss"][row],
                                "length": length,
                                "feature": features[row, :length].to(torch.float16).contiguous(),
                            }, feature_handle)
                    records.append({
                        "name": batch["name"][row],
                        "feature_path": str(feature_path.relative_to(output_root)),
                    })
                    completed += 1
                if args.max_samples and completed >= args.max_samples:
                    break
                if completed % 250 == 0:
                    write_manifest(manifest_path, records)
                    print(f"[{split}] cached {completed}/{len(dataset)}")
        write_manifest(manifest_path, records)
        print(f"[{split}] wrote {completed} samples to {manifest_path}")


if __name__ == "__main__":
    main()
