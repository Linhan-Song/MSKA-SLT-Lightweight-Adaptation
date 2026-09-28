# Local Temporal Refiner grid-search plan

This follow-up search is specified before launching any new runs. It is a
development-set study. No new configuration may be selected using PHOENIX14T
test results.

## Fixed conditions

- Frozen Recognition network, VLMapper, mBART and matched Gated Residual-128
  checkpoint.
- Variable-length Recognition features; no temporal pooling.
- One Local Temporal Refiner layer, dropout 0.1 and initial gate logit -4.0.
- AdamW learning rate 2e-4, weight decay 0.01, batch size 1 and gradient
  accumulation 8.
- Up to 20 epochs with development BLEU-4 early stopping patience 4.
- Beam size 5, maximum generation length 100 and length penalty 1.0.
- Stage-1 search seed: 0. Test evaluation disabled.

## Stage 1: structural grid

The full Cartesian product contains 27 configurations:

- local radius: {2, 4, 8}
- attention dimension: {80, 96, 112}
- attention heads: {2, 4, 8}
- FFN dimension: twice the attention dimension

These values refine the region identified by the earlier one-factor study:
dimension 96 was strongest, dimension 80 was the nearest smaller candidate,
dimension 128 declined, radius 4 was strongest in the available comparisons,
and the interaction between radius, dimension and head count remained
incomplete. All selected dimensions are divisible by every head count.

## Stage 2: FFN grid

For the best Stage-1 radius/dimension/head tuple, compare FFN dimensions
{d, 2d, 4d} with all other settings fixed. The 2d run is reused from Stage 1.

## Selection and validation

Select the highest seed-0 development BLEU-4. Exact ties are resolved by lower
trainable parameter count and then lower development loss. This grid-search run
stops after selecting the seed-0 configuration. Seeds 1 and 2 are deliberately
deferred until the complete hyperparameter table has been reviewed; they must
not be used to reselect the configuration.

Because the project test set was inspected before this follow-up study, any
later test evaluation must be described as post-hoc confirmation rather than a
pristine held-out evaluation. A second data set is required for a fully
independent publication claim.

## Reuse policy

An existing run may be reused only when radius, dimension, heads, FFN,
initialisation, optimiser, seed, upstream checkpoint and decoding settings
match exactly. Reused results are validated from their saved config and final
metrics and are identified as reused in the search summary.
