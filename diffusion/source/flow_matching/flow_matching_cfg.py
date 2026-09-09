"""Conditional rectified-flow training configuration."""

from __future__ import annotations

from dataclasses import dataclass

from source.utils import detect_device


@dataclass
class FlowMatchingCfg:
  # Data
  data_dir: str = "diffusion/source/datasets"
  norm_stats_file: str = ""
  terrain_norm_stats_file: str = ""
  proprio_norm_stats_file: str = ""
  train_split: float = 0.9

  # Shared conditional motion model
  d_model: int = 256
  nhead: int = 4
  num_layers: int = 2
  dropout: float = 0.0
  terrain_height: int = 21
  terrain_width: int = 33
  terrain_feature_dim: int = 48
  proprio_dim: int = 28

  # Rectified flow.  Model time input is scaled before sinusoidal embedding;
  # ODE integration itself always uses normalized t in [0, 1].
  num_noise_samples: int = 10
  time_embedding_scale: float = 1000.0
  sampling_steps: int = 10
  sampler: str = "euler"

  # EMA
  use_ema: bool = False
  ema_decay: float = 0.9999

  # Training
  batch_size: int = 1024
  num_epochs: int = 2000
  lr: float = 3e-4
  weight_decay: float = 1e-4
  max_grad_norm: float = 1.0

  # Logging
  name: str = "casbot_flow_matching"
  log_interval: int = 10
  save_interval: int = 100
  log_dir: str = "logs/flow_matching"
  wandb_project: str = "smp"
  use_wandb: bool = True

  device: str = ""
  seed: int = 42

  def __post_init__(self) -> None:
    if self.sampler not in ("euler", "heun"):
      raise ValueError(f"sampler must be 'euler' or 'heun', got {self.sampler!r}")
    if self.sampling_steps <= 0:
      raise ValueError("sampling_steps must be positive")
    if not self.device:
      self.device = detect_device()
