"""Translation decoding and metric evaluation."""

import json
import time
from pathlib import Path

import numpy as np
import torch


def evaluate(model, loader, device, generation_cfg, max_batches=0,
              autocast_enabled=True, autocast_dtype=torch.float16):
    from mska_backbone.metrics import bleu, rouge

    model.eval()
    references, hypotheses = [], []
    losses = []
    evaluation_started = time.time()
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        features = batch["features"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(
            enabled=device.type == "cuda" and autocast_enabled,
            dtype=autocast_dtype,
        ):
            loss = model(features, lengths, batch["texts"])
        predicted = model.generate_text(features, lengths, generation_cfg)
        losses.append(float(loss.detach().cpu()))
        references.extend(batch["texts"])
        hypotheses.extend(predicted)
        if batch_index % 100 == 0:
            elapsed_minutes = (time.time() - evaluation_started) / 60
            print(
                f"eval batch={batch_index}/{len(loader)} "
                f"elapsed_minutes={elapsed_minutes:.1f}",
                flush=True,
            )
    scores = bleu(references=references, hypotheses=hypotheses)
    scores["rouge"] = rouge(references=references, hypotheses=hypotheses)
    scores["loss"] = float(np.mean(losses)) if losses else float("nan")
    scores["samples"] = len(references)
    return scores, references, hypotheses


def write_predictions(path, references, hypotheses):
    with Path(path).open("w", encoding="utf-8") as handle:
        for reference, hypothesis in zip(references, hypotheses):
            handle.write(json.dumps({"reference": reference, "hypothesis": hypothesis}, ensure_ascii=False) + "\n")
