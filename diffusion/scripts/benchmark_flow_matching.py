"""Benchmark batched rectified-flow sampling without launching IsaacLab."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch


DIFFUSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIFFUSION_ROOT))

from source.common.model import DiffusionDenoiser  # noqa: E402
from source.flow_matching.sampler import sample_flow  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--sampler", choices=("euler", "heun"), default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    if not hasattr(np, "_core"):
        sys.modules.setdefault("numpy._core", np.core)
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
        sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = checkpoint["cfg"]
    if cfg.get("generative_method") != "flow_matching":
        raise ValueError("benchmark_flow_matching.py requires a Flow Matching checkpoint")
    model = DiffusionDenoiser(
        feature_dim=int(cfg["feature_dim"]), window_size=int(cfg["window_size"]),
        d_model=int(cfg.get("d_model", 256)), nhead=int(cfg.get("nhead", 4)),
        num_layers=int(cfg.get("num_layers", 2)), dropout=float(cfg.get("dropout", 0.0)),
        head_dim=cfg.get("head_dim"), terrain_dim=int(cfg["terrain_dim"]),
        terrain_height=int(cfg.get("terrain_height", 21)),
        terrain_width=int(cfg.get("terrain_width", 33)),
        terrain_feature_dim=int(cfg.get("terrain_feature_dim", 23)),
        proprio_dim=int(cfg.get("proprio_dim", 31)),
    ).to(device)
    model.load_state_dict(checkpoint.get("model_ema", checkpoint["model"]), strict=True)
    model.eval().requires_grad_(False)

    batch = args.batch_size
    future = int(cfg.get("future_size", cfg["window_size"]))
    history = int(cfg.get("history_size", 4))
    terrain = torch.randn((batch, history, int(cfg["terrain_dim"])), device=device)
    proprio = torch.randn((batch, history, int(cfg.get("proprio_dim", 31))), device=device)
    steps = args.sampling_steps or int(cfg.get("sampling_steps", 10))
    method = args.sampler or str(cfg.get("sampler", "euler"))

    timings: list[float] = []
    peak_memory = 0
    with torch.inference_mode():
        for run in range(args.runs + 1):
            torch.cuda.reset_peak_memory_stats(device)
            noise = torch.randn((batch, future, int(cfg["feature_dim"])), device=device)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            sample_flow(model, noise, terrain, proprio, steps, method,
                        float(cfg.get("time_embedding_scale", 1000.0)))
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device))
            if run > 0:
                timings.append(elapsed)
                print(f"run={run} sample_ms={elapsed * 1000.0:.3f} "
                      f"env_samples_per_s={batch / elapsed:.1f}")
    mean = sum(timings) / len(timings)
    print(f"mean_sample_ms={mean * 1000.0:.3f} mean_env_samples_per_s={batch / mean:.1f} "
          f"peak_allocated_mib={peak_memory / 2**20:.1f} steps={steps} sampler={method}")


if __name__ == "__main__":
    main()
