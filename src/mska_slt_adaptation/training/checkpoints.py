"""Checkpoint loading, saving and parameter-freezing helpers."""

import os
import time
from pathlib import Path

import torch

from ..models.translation_pipeline import trainable_state_dict


def save_checkpoint(path, model, optimizer, scheduler, state, cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "trainable_model": trainable_state_dict(model),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "state": state,
        "config": cfg,
    }
    with temporary.open("wb") as checkpoint_handle:
        torch.save(payload, checkpoint_handle)
    # On Windows, antivirus/indexing can briefly hold the destination open and
    # make an otherwise atomic replacement fail with WinError 5. Retry the
    # replacement without rewriting the checkpoint payload.
    for attempt in range(6):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.25 * (2 ** attempt))
    print(f"Saved checkpoint: {path}", flush=True)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    with Path(path).open("rb") as checkpoint_handle:
        checkpoint = torch.load(checkpoint_handle, map_location="cpu")
    result = model.load_state_dict(checkpoint["trainable_model"], strict=False)
    unexpected = [key for key in result.unexpected_keys if "num_batches_tracked" not in key]
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint.get("state", {})


def load_initial_weights(path, model):
    """Warm-start model weights without restoring optimizer or training state."""
    with Path(path).open("rb") as checkpoint_handle:
        checkpoint = torch.load(checkpoint_handle, map_location="cpu")
    state_dict = checkpoint.get("trainable_model", checkpoint)
    result = model.load_state_dict(state_dict, strict=False)
    unexpected = [key for key in result.unexpected_keys if "num_batches_tracked" not in key]
    if unexpected:
        raise RuntimeError(f"Unexpected warm-start checkpoint keys: {unexpected}")
    loaded = sorted(set(state_dict).difference(result.unexpected_keys))
    if not loaded:
        raise RuntimeError(f"Warm-start checkpoint loaded no parameters: {path}")
    print(f"Warm-started {len(loaded)} tensors from {path}", flush=True)


def apply_parameter_freezing(model, prefixes):
    """Freeze exact parameter names or complete dotted-name subtrees."""
    prefixes = [str(prefix).rstrip(".") for prefix in prefixes]
    matched = {prefix: [] for prefix in prefixes}
    for name, parameter in model.named_parameters():
        for prefix in prefixes:
            if name == prefix or name.startswith(prefix + "."):
                parameter.requires_grad = False
                matched[prefix].append(name)
                break
    missing = [prefix for prefix, names in matched.items() if not names]
    if missing:
        raise RuntimeError(f"Freeze prefixes matched no parameters: {missing}")
    for prefix, names in matched.items():
        print(f"Froze {len(names)} tensors under {prefix}", flush=True)
