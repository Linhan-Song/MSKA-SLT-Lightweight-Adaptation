# MSKA-SLT Lightweight Adaptation

This repository contains the implementation used for the dissertation
*Keypoint-Based Sign Language Translation with Temporal Attention and Gated
Residual Adaptation*. The work extends MSKA-SLT with two small trainable
modules while keeping the Recognition network, the original VLMapper and
mBART frozen in the main experiments.

The final forward path is:

```text
Recognition features
    -> Local Temporal Refiner
    -> frozen VLMapper
    -> Gated Residual Adapter
    -> frozen mBART
```

Training is carried out in two stages. Stage 1 trains the Gated Residual
Adapter. Stage 2 loads the adapter checkpoint with the same random seed,
freezes it, and trains the Local Temporal Refiner. The final configuration
uses a bottleneck width of 128 for the interface adapter and
`radius=4, dimension=96, heads=4, FFN=192` for the temporal refiner.

## Repository layout

```text
configs/
  base/                    Original PHOENIX14T MSKA-SLT configuration
  interface_adapter/       VLMapper and residual-adapter comparisons
  sequence_processing/     Supplementary fixed-length experiments
  temporal_comparison/     Global, Local and Local-Global attention
  local_capacity/          Internal dimension and FFN comparison
  final_model/             Matched two-stage runs for seeds 0, 1 and 2
  exploratory/             Joint training and alternative connectors

src/mska_backbone/         MSKA components needed by this project
src/mska_slt_adaptation/
  data/                    Cached Recognition feature loader
  models/
    residual_adapters.py   Plain, fixed-scale and learnable-gate adapters
    temporal_refiner.py    Global, Local and Local-Global temporal attention
    sequence_processing.py Variable-length and fixed-length processing
    translation_pipeline.py Complete path from cached features to mBART
    alternative_connectors.py
    exploratory_adapters.py
    prompt_modules.py
  training/
    model_factory.py       Builds a model from a YAML configuration
    checkpoints.py         Checkpoint loading, warm-start and freezing
    evaluation.py          Decoding and metric calculation
    run_experiment.py      Training and evaluation loop

scripts/                   Small command-line entry points
tests/                     Component tests
results/                   Tables reproduced from the dissertation
```

The project-specific implementation is kept in `mska_slt_adaptation` rather
than mixed into the original backbone. In particular, the two modules used by
the final model can be read independently in
`models/residual_adapters.py` and `models/temporal_refiner.py`.

## Environment

The reported experiments used Python 3.10.13, PyTorch 2.0.1 with CUDA 11.8
and Transformers 4.28. Install the CUDA build of PyTorch suitable for the
machine first, then install the remaining packages and the local source tree:

```bash
pip install -r requirements.txt
pip install -e .
```

The feature-caching script also needs TensorFlow for the CTC decoder used by
the original MSKA Recognition network:

```bash
pip install -r requirements-recognition.txt
```

Install `requirements-exploratory.txt` only when running the Qwen or LoRA
experiments retained for completeness.

## Data and pretrained components

PHOENIX14T and pretrained model weights are not distributed in this
repository. Follow the data and checkpoint instructions in the original
[MSKA repository](https://github.com/sutwangyan/MSKA), then use the following
layout:

```text
data/Phoenix-2014T/
pretrained_models/Phoenix-2014T_SLT/best.pth
```

Split the official SLT checkpoint into the frozen components used by the
cached-feature experiments:

```bash
python scripts/extract_pretrained_components.py
```

Cache the fused Recognition features once:

```bash
python scripts/cache_recognition_features.py \
  --config configs/base/phoenix-2014t_s2t.yaml \
  --output-dir data/Phoenix-2014T/recognition_features \
  --recognition-state pretrained_models/Phoenix-2014T_SLT/components/recognition.pth
```

The cache keeps the variable-length sequence produced by the Recognition
network. Padding is applied only when samples are assembled into a batch.

## Reproducing the final two-stage model

Train the first-stage interface adapter:

```bash
python scripts/train_experiment.py \
  --config configs/interface_adapter/gated_residual_b128_seed0.yaml
```

The seed-0 second-stage configuration points to the checkpoint produced by
that command:

```bash
python scripts/train_experiment.py \
  --config configs/final_model/local_r4_d96_h4_ffn192_seed0.yaml
```

Seeds 1 and 2 use the files with the corresponding seed number. This preserves
the checkpoint matching used in the dissertation: the Local run for seed `k`
always starts from the Gated Residual checkpoint for seed `k`.

Evaluate a completed run with:

```bash
python scripts/evaluate_checkpoint.py \
  --config configs/final_model/local_r4_d96_h4_ffn192_seed0.yaml \
  --resume outputs/final_two_stage_seed0/best_checkpoint.pth
```

## Configuration groups

- `interface_adapter` contains the bottleneck-width comparison and the Plain,
  Fixed and Gated residual variants.
- `temporal_comparison` keeps `dimension=96`, `heads=4` and `FFN=192` fixed
  while changing the temporal connection pattern.
- `local_capacity` fixes Local attention at `radius=4` and `heads=4`, then
  changes the internal dimension together with `FFN=2d`.
- `final_model` contains the selected Local configuration for the three matched
  seeds.
- `sequence_processing` is supplementary. Fixed-length average pooling is not
  part of the final model.

## Attribution

The backbone code in `src/mska_backbone` is derived from the public MSKA
implementation by Mo Guan, Yan Wang, Guangkun Ma, Jiarui Liu and Mingzu Sun.
The original project and paper should be cited when this code is used. The
lightweight adaptation modules, cached-feature training pipeline and experiment
configurations in `mska_slt_adaptation` are the additions made for this
dissertation.
