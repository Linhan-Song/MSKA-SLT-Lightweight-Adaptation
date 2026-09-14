"""Data loading for cached MSKA Recognition features."""

from .cached_features import CachedFeatureDataset, collate_cached_features

__all__ = ["CachedFeatureDataset", "collate_cached_features"]
