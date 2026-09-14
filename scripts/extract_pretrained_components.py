"""Split the official end-to-end MSKA checkpoint into reusable frozen components."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


PREFIXES = {
    "recognition": "recognition_network.",
    "translation": "translation_network.",
    "vl_mapper": "vl_mapper.",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default="pretrained_models/Phoenix-2014T_SLT/best.pth"
    )
    parser.add_argument(
        "--output-dir", default="pretrained_models/Phoenix-2014T_SLT/components"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, dict):
        raise TypeError("Official checkpoint does not contain a model state dictionary")

    counts = {}
    for name, prefix in PREFIXES.items():
        component = {
            key[len(prefix):]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if not component:
            raise RuntimeError(f"No keys found for official component {name!r}")
        output_path = output_dir / f"{name}.pth"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {output_path}")
        torch.save(component, output_path)
        counts[name] = len(component)
        print(f"Wrote {len(component)} tensors to {output_path}")

    metadata = {
        "source_checkpoint": str(checkpoint_path.resolve()),
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "source_epoch": checkpoint.get("epoch"),
        "component_tensor_counts": counts,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
