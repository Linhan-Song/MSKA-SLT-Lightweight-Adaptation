# Held-out PHOENIX14T test evaluation

## Protocol

All model selection was completed on the 519-example development set before
these results were examined. The frozen saved checkpoints were then evaluated
on the same 642-example test set using beam size 5, maximum generation length
100 and length penalty 1.0. Full VLMapper, Gated Residual-128 and the final
Gated Residual + Local model use matched seeds 0, 1 and 2. The unchanged
Original checkpoint has no random training seed and is evaluated once.

The audit verified that the train, development and test manifests contain
7,096, 519 and 642 distinct sample IDs with no overlap. References in every
prediction file match exactly, every evaluation contains 642 examples, and
recomputing BLEU-1 through BLEU-4 and ROUGE-L from the saved sentence-level
predictions reproduces the stored scores with zero numerical error. Checkpoint,
metric and prediction hashes are recorded in `test_evaluation_audit.json`.

## Principal results

| Condition | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | Loss | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original checkpoint | 54.2206 | 42.0191 | 34.1764 | 28.7529 | 53.0698 | 22.6190 | 0 |
| Full VLMapper | 53.6770 ± 0.1192 | 41.6126 ± 0.1907 | 33.9396 ± 0.1907 | 28.5577 ± 0.1796 | 52.5237 ± 0.1177 | 20.2226 ± 0.0174 | 1,574,912 |
| Gated Residual-128 | 53.8994 ± 0.2915 | 41.9585 ± 0.2855 | 34.1880 ± 0.2356 | 28.7811 ± 0.2089 | 52.7368 ± 0.1910 | 20.3137 ± 0.0234 | 265,345 |
| Gated Residual-128 + Local d96 | 53.8147 ± 0.1277 | 41.8795 ± 0.0518 | 34.1665 ± 0.0319 | 28.7808 ± 0.0219 | 52.9213 ± 0.1162 | 20.2568 ± 0.0350 | 440,067 |

Trained-condition values are the three-seed mean ± sample standard deviation.
The loss and generation metrics need not rank systems identically: loss is
teacher-forced token cross-entropy, whereas BLEU and ROUGE are computed from
beam-search outputs.

## Seed-wise BLEU-4

| Seed | Full VLMapper | Gated Residual-128 | Gated + Local | Gated − Full | Local − Gated |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 28.3956 | 29.0152 | 28.7713 | +0.6196 | -0.2439 |
| 1 | 28.5267 | 28.7146 | 28.8058 | +0.1879 | +0.0912 |
| 2 | 28.7507 | 28.6135 | 28.7651 | -0.1372 | +0.1516 |
| Mean | 28.5577 | 28.7811 | 28.7808 | +0.2234 | -0.0003 |

Gated Residual-128 matches or slightly exceeds the other trained conditions in
the three-seed mean while using 83.15% fewer trainable parameters than Full
VLMapper. Its main supported claim is therefore parameter efficiency rather
than a statistically established quality improvement.

The Local Temporal Refiner's development gains of +0.2419, +0.1088 and +0.2731
BLEU-4 do not transfer consistently to the held-out test set. Test changes are
-0.2439, +0.0912 and +0.1516, producing an essentially unchanged mean. This
result should be reported directly; the held-out evidence does not support a
claim that Local refinement improves translation quality.

## Paired bootstrap analysis

The analysis uses 5,000 paired resamples of the 642 test sentences and seed
20260928. Each resample uses identical sentence indices for both systems.

| Comparison | Seed | BLEU-4 difference | 95% interval | Two-sided p |
| --- | ---: | ---: | ---: | ---: |
| Gated − Full | 0 | +0.6196 | [-0.0536, 1.3180] | 0.0692 |
| Gated − Full | 1 | +0.1879 | [-0.5086, 0.9112] | 0.5976 |
| Gated − Full | 2 | -0.1372 | [-0.8810, 0.6152] | 0.7192 |
| Local − Gated | 0 | -0.2439 | [-0.7671, 0.2534] | 0.3620 |
| Local − Gated | 1 | +0.0912 | [-0.4389, 0.6409] | 0.7496 |
| Local − Gated | 2 | +0.1516 | [-0.2715, 0.5697] | 0.4852 |

Every interval includes zero. None of these seed-wise test differences meets a
two-sided 0.05 significance threshold.

## Files

- `test_principal_results.csv`: all test metrics, means and standard deviations.
- `test_multiseed_bleu4.csv`: matched seed-wise BLEU-4 and differences.
- `test_paired_bootstrap.csv`: confidence intervals and p-values.
- `test_evaluation_audit.json`: source paths and SHA-256 hashes.
- `scripts/summarize_test_results.py`: deterministic audit and analysis script.
