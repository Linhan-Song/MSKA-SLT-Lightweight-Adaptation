import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class CachedFeatureDataset(Dataset):
    """Read fused Recognition features without loading them all into memory."""

    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Missing feature manifest: {self.manifest_path}")
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        if not self.records:
            raise ValueError(f"Feature manifest is empty: {self.manifest_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        # A cache created on Windows may later be used on Linux. Normalising the
        # separator here keeps one manifest usable in both environments.
        feature_path = Path(str(record["feature_path"]).replace("\\", "/"))
        if not feature_path.is_absolute():
            feature_path = self.manifest_path.parent / feature_path
        with feature_path.open("rb") as feature_handle:
            sample = torch.load(feature_handle, map_location="cpu")
        feature = sample["feature"].float()
        if feature.ndim != 2:
            raise ValueError(f"Expected [T,D] feature, got {tuple(feature.shape)} from {feature_path}")
        return {
            "name": sample["name"],
            "text": sample["text"],
            "gloss": sample.get("gloss", ""),
            "feature": feature,
            "length": int(sample.get("length", feature.shape[0])),
        }


def collate_cached_features(batch):
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    max_length = int(lengths.max().item())
    feature_dim = batch[0]["feature"].shape[-1]
    features = torch.zeros(len(batch), max_length, feature_dim, dtype=torch.float32)
    for index, item in enumerate(batch):
        valid = min(item["length"], item["feature"].shape[0])
        features[index, :valid] = item["feature"][:valid]
    return {
        "features": features,
        "lengths": lengths,
        "texts": [item["text"] for item in batch],
        "names": [item["name"] for item in batch],
        "glosses": [item["gloss"] for item in batch],
    }
