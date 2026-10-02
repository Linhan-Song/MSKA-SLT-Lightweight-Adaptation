# Local Temporal Refiner grid-search results

This report records the preregistered seed-0 development-set search in
`experiments/local_grid_search/PLAN.md`. The PHOENIX14T test set was not
evaluated, and seeds 1 and 2 remain deferred.

## Selected configuration

The selected Local Temporal Refiner uses radius 4, attention dimension 96,
8 attention heads and FFN dimension 192. It has 174,722 trainable parameters.

| Metric | Development result |
| --- | ---: |
| BLEU-1 | 53.6682 |
| BLEU-2 | 41.4286 |
| BLEU-3 | 33.7189 |
| BLEU-4 | **28.3853** |
| ROUGE-L | 52.9712 |
| Loss | 22.6504 |

The prior matching configuration with 4 heads reached 28.3093 development
BLEU-4. Increasing the head count to 8 improved the seed-0 result by 0.0761
BLEU-4 without changing the trainable parameter count.

## Stage 1: structural grid

All 27 combinations of radius {2, 4, 8}, dimension {80, 96, 112} and heads
{2, 4, 8} were evaluated with FFN dimension fixed to twice the attention
dimension. Twenty-two runs were newly trained and five exact matches were
reused after validating their saved configurations and metrics.

| Rank | Radius | Dimension | Heads | FFN | Dev BLEU-4 | Source |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 4 | 96 | 8 | 192 | **28.3853** | new |
| 2 | 4 | 96 | 4 | 192 | 28.3093 | reused |
| 3 | 2 | 112 | 4 | 224 | 28.2573 | new |
| 4 | 4 | 96 | 2 | 192 | 28.2296 | reused |
| 5 | 4 | 112 | 8 | 224 | 28.2283 | new |

Mean development BLEU-4 was highest at radius 4 (28.22) and dimension 96
(28.21). Head-count means were close, so the 8-head choice is supported by
the selected interaction rather than a uniform head-count advantage.

## Stage 2: FFN width

The Stage-1 winning radius, dimension and head count were fixed while the FFN
dimension was varied.

| FFN dimension | Trainable parameters | Dev BLEU-4 | Dev ROUGE-L | Source |
| ---: | ---: | ---: | ---: | --- |
| 96 | 156,194 | 28.1087 | 52.6385 | new |
| 192 | 174,722 | **28.3853** | 52.9712 | reused from Stage 1 |
| 384 | 211,778 | 28.2612 | **53.1694** | new |

FFN dimension 192 remained the selection because development BLEU-4 was the
preregistered primary metric. The wider FFN had higher ROUGE-L but lower
BLEU-4 and more trainable parameters.

## Validation

- Stage 1 contains 27 unique configurations and exactly the preregistered
  Cartesian product.
- All entries use seed 0; the source counts are 22 new and 5 reused.
- Stage 2 contains exactly FFN dimensions 96, 192 and 384 for radius 4,
  dimension 96 and 8 heads.
- Every new result contains 519 development samples, finite metrics and a null
  test result. The final summary also records `test_evaluated: false`.
- `final_summary.json` and `selected_configuration.json` agree on the selected
  configuration and metrics.

The complete tables are in `results/local_grid_search_stage1.csv` and
`results/local_grid_search_ffn.csv`. These are seed-0 development results;
multi-seed confirmation is required before treating the 0.0761 BLEU-4 gain as
stable.
