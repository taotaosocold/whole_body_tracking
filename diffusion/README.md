# CASBOT conditional motion generation

The DDPM and Flow Matching paths share the same conditional dataset and model. The current 10 Hz input uses two historical frames of terrain/proprioception and outputs ten future motion frames. Each historical proprioceptive token contains 25 joint positions and the repeated desired velocity command `[vx, vy, wz]`; historical measured root velocity is intentionally omitted. Each 33×21 (693-D) terrain scan is encoded by the CNN into a 48-D feature before cross-attention.

Run commands from the `whole_body_tracking` repository root in the `smp` environment.

## Prepare data

```bash
conda activate smp
python diffusion/scripts/height_map_to_npz.py \
  --input-dir diffusion/source/data \
  --output-dir diffusion/source/datasets
```

This preprocessing is shared by both methods and only needs to be repeated when the source motions or feature format change.
Format v7 removes historical measured root velocity from proprioception and expands the terrain CNN feature to 48 dimensions, so old processed NPZ files and old checkpoints are intentionally incompatible.

## Train DDPM

```bash
python diffusion/scripts/train_ddpm.py \
  --data-dir diffusion/source/datasets \
  --name casbot_ddpm \
  --batch-size 1024 \
  --no-use-wandb
```

Checkpoints are written below `logs/ddpm/`.

## Train Flow Matching

```bash
python diffusion/scripts/train_flow_matching.py \
  --data-dir diffusion/source/datasets \
  --name casbot_flow_matching \
  --batch-size 1024 \
  --no-use-wandb
```

Checkpoints are written below `logs/flow_matching/`. The default sampler is 10-step Euler; use `--sampler heun` for Heun sampling.

## Visualize and evaluate

The visualizer and evaluator detect DDPM versus Flow Matching from the checkpoint automatically.

```bash
python diffusion/scripts/casbot_generate_viz_terrain.py \
  --ckpt-path logs/flow_matching/casbot_flow_matching/<run>/pretrained.pt \
  --data-dir diffusion/source/datasets

python diffusion/scripts/evaluate_first_future_frame.py \
  logs/flow_matching/casbot_flow_matching/<run>/pretrained.pt \
  --data-dir diffusion/source/datasets
```

## Benchmark sampling

```bash
python diffusion/scripts/benchmark_ddpm.py \
  logs/ddpm/casbot_ddpm/<run>/pretrained.pt --batch-size 2048

python diffusion/scripts/benchmark_flow_matching.py \
  logs/flow_matching/casbot_flow_matching/<run>/pretrained.pt --batch-size 2048
```

## Export ONNX

```bash
python diffusion/source/utils/export.py \
  logs/flow_matching/casbot_flow_matching/<run>/pretrained.pt
```

The exported ONNX metadata records the generation method and its sampling parameters.
