"""Export a diffusion denoiser checkpoint to ONNX.

The exported model represents one epsilon-prediction (denoising) call.  The
DDPM/DDIM scheduler remains responsible for invoking the model at each
sampling timestep.

Example:
  python diffusion/source/utils/export.py \
    logs/pretrain/casbot_diffusion/20260813_233015/pretrained.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
import torch.nn as nn


DIFFUSION_ROOT = Path(__file__).resolve().parents[2]
if str(DIFFUSION_ROOT) not in sys.path:
  sys.path.insert(0, str(DIFFUSION_ROOT))

from source.pretrain.model import DiffusionDenoiser  # noqa: E402


class _ConditionalDenoiser(nn.Module):
  """Give ONNX a forward with four required tensor inputs."""

  def __init__(self, model: DiffusionDenoiser) -> None:
    super().__init__()
    self.model = model

  def forward(
    self,
    x_t: torch.Tensor,
    timestep: torch.Tensor,
    terrain: torch.Tensor,
    proprio: torch.Tensor,
  ) -> torch.Tensor:
    return self.model(x_t, timestep, terrain=terrain, proprio=proprio)


def _build_model(checkpoint: dict[str, Any], use_ema: bool) -> DiffusionDenoiser:
  cfg = checkpoint["cfg"]
  terrain_dim = cfg.get("terrain_dim")
  if terrain_dim is None:
    raise ValueError(
      "This exporter currently expects a terrain-conditioned checkpoint."
    )

  model = DiffusionDenoiser(
    feature_dim=int(cfg["feature_dim"]),
    window_size=int(cfg["window_size"]),
    d_model=int(cfg.get("d_model", 256)),
    nhead=int(cfg.get("nhead", 4)),
    num_layers=int(cfg.get("num_layers", 2)),
    dropout=float(cfg.get("dropout", 0.0)),
    head_dim=cfg.get("head_dim"),
    terrain_dim=int(terrain_dim),
    terrain_height=int(cfg.get("terrain_height", 21)),
    terrain_width=int(cfg.get("terrain_width", 33)),
    terrain_feature_dim=int(cfg.get("terrain_feature_dim", 23)),
    proprio_dim=int(cfg.get("proprio_dim", 31)),
  )
  state_key = "model_ema" if use_ema else "model"
  if state_key not in checkpoint:
    raise KeyError(f"Checkpoint does not contain {state_key!r}")
  model.load_state_dict(checkpoint[state_key], strict=True)
  return model.eval()


def _export_onnx(
  model: nn.Module,
  output_path: Path,
  cfg: dict[str, Any],
  opset: int,
) -> tuple[torch.Tensor, ...]:
  batch = 1
  window = int(cfg["window_size"])
  feature_dim = int(cfg["feature_dim"])
  terrain_dim = int(cfg["terrain_dim"])
  proprio_dim = int(cfg.get("proprio_dim", 31))
  history = int(cfg.get("history_size", 4))
  inputs = (
    torch.randn(batch, window, feature_dim, dtype=torch.float32),
    torch.zeros(batch, dtype=torch.int64),
    torch.randn(batch, history, terrain_dim, dtype=torch.float32),
    torch.randn(batch, history, proprio_dim, dtype=torch.float32),
  )
  with torch.inference_mode():
    torch.onnx.export(
      model,
      inputs,
      str(output_path),
      input_names=["x_t", "timestep", "terrain", "proprio"],
      output_names=["noise"],
      opset_version=opset,
      do_constant_folding=True,
      external_data=False,
    )
  return inputs


def _validate_coordinate_semantics(cfg: dict[str, Any]) -> tuple[str, str]:
  root_body = cfg.get("root_body")
  terrain_layout = cfg.get("terrain_layout")
  if root_body != "waist_yaw_link" or terrain_layout != "root_z_minus_terrain_z":
    raise ValueError(
      "Checkpoint lacks the required format-v3 coordinate semantics: "
      f"root_body={root_body!r}, terrain_layout={terrain_layout!r}. "
      "Rebuild the dataset and retrain; do not re-export an old checkpoint."
    )
  return str(root_body), str(terrain_layout)


def _embed_metadata(
  onnx_path: Path,
  checkpoint: dict[str, Any],
  cfg: dict[str, Any],
) -> None:
  """Embed sampling configuration and normalization arrays in the ONNX file."""
  root_body, terrain_layout = _validate_coordinate_semantics(cfg)
  model = onnx.load(str(onnx_path), load_external_data=True)
  metadata = {
    "diffusion.format_version": "3",
    "diffusion.num_timesteps": str(int(cfg.get("num_timesteps", 50))),
    "diffusion.window_size": str(int(cfg["window_size"])),
    "diffusion.feature_dim": str(int(cfg["feature_dim"])),
    "diffusion.terrain_dim": str(int(cfg["terrain_dim"])),
    "diffusion.proprio_dim": str(int(cfg.get("proprio_dim", 31))),
    "diffusion.history_size": str(int(cfg.get("history_size", 4))),
    "diffusion.future_size": str(int(cfg.get("future_size", cfg["window_size"]))),
    "diffusion.proprio_layout": "joint_pos,target_heading_rot6d",
    "diffusion.root_body": str(root_body),
    "diffusion.terrain_layout": str(terrain_layout),
  }
  for key in (
    "q_low", "q_high", "t_q_low", "t_q_high", "p_q_low", "p_q_high",
  ):
    if key not in checkpoint:
      raise KeyError(f"Checkpoint does not contain required normalization data {key!r}")
    values = np.asarray(checkpoint[key], dtype=np.float32).tolist()
    metadata[f"diffusion.{key}"] = json.dumps(values, separators=(",", ":"))

  del model.metadata_props[:]
  for key, value in metadata.items():
    prop = model.metadata_props.add()
    prop.key = key
    prop.value = value
  onnx.save_model(model, str(onnx_path), save_as_external_data=False)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("checkpoint", type=Path, help="Path to pretrained.pt")
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=None,
    help="Default: <checkpoint directory>/export",
  )
  parser.add_argument("--name", default=None, help="ONNX stem (default: checkpoint stem)")
  parser.add_argument("--opset", type=int, default=18)
  parser.add_argument("--use-ema", action="store_true")
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  checkpoint_path = args.checkpoint.expanduser().resolve()
  if not checkpoint_path.is_file():
    raise FileNotFoundError(checkpoint_path)

  output_dir = (
    args.output_dir.expanduser().resolve()
    if args.output_dir is not None
    else checkpoint_path.parent / "export"
  )
  output_dir.mkdir(parents=True, exist_ok=True)
  stem = args.name or checkpoint_path.stem
  onnx_path = output_dir / f"{stem}.onnx"

  checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
  cfg = checkpoint["cfg"]
  _validate_coordinate_semantics(cfg)
  model = _ConditionalDenoiser(_build_model(checkpoint, args.use_ema)).eval()

  print(f"[INFO] Exporting ONNX: {onnx_path}")
  example_inputs = _export_onnx(model, onnx_path, cfg, args.opset)
  with torch.inference_mode():
    expected = model(*example_inputs)
  if not torch.isfinite(expected).all():
    raise RuntimeError("PyTorch model produced non-finite output during export check")

  _embed_metadata(onnx_path, checkpoint, cfg)
  onnx.checker.check_model(onnx.load(str(onnx_path)))
  print(f"[OK] ONNX model: {onnx_path}")
  print("[OK] Sampling config and normalization stats embedded in ONNX metadata")
  print("[NOTE] This model is one denoising step; the sampler calls it repeatedly.")


if __name__ == "__main__":
  main()
