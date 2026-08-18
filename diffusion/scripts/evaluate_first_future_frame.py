"""Evaluate the first generated future frame against its paired ground truth."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


DIFFUSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIFFUSION_ROOT))

from source.common.model import DiffusionDenoiser  # noqa: E402
from source.ddpm.scheduler import DDPMScheduler  # noqa: E402
from source.flow_matching.sampler import sample_flow  # noqa: E402


def _numpy_pickle_compatibility() -> None:
    if not hasattr(np, "_core"):
        sys.modules.setdefault("numpy._core", np.core)
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
        sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


def _mean_l2(error: torch.Tensor) -> float:
    return torch.linalg.vector_norm(error, dim=-1).mean().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path, default=DIFFUSION_ROOT / "source/datasets")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    _numpy_pickle_compatibility()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = checkpoint["cfg"]
    model = DiffusionDenoiser(
        feature_dim=int(cfg["feature_dim"]),
        window_size=int(cfg["window_size"]),
        d_model=int(cfg.get("d_model", 256)),
        nhead=int(cfg.get("nhead", 4)),
        num_layers=int(cfg.get("num_layers", 2)),
        dropout=float(cfg.get("dropout", 0.0)),
        head_dim=cfg.get("head_dim"),
        terrain_dim=int(cfg["terrain_dim"]),
        terrain_height=int(cfg.get("terrain_height", 21)),
        terrain_width=int(cfg.get("terrain_width", 33)),
        terrain_feature_dim=int(cfg.get("terrain_feature_dim", 23)),
        proprio_dim=int(cfg.get("proprio_dim", 31)),
    ).to(device)
    model.load_state_dict(checkpoint.get("model_ema", checkpoint["model"]), strict=True)
    model.eval().requires_grad_(False)
    method = str(cfg.get("generative_method", "ddpm"))
    scheduler = (
        DDPMScheduler(int(cfg.get("num_timesteps", 50))).to(device)
        if method == "ddpm" else None
    )
    if method not in ("ddpm", "flow_matching"):
        raise ValueError(f"Unknown checkpoint generative_method={method!r}")

    motions, terrains, proprios = [], [], []
    for path in sorted(args.data_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            motions.append(data["motion_windows"].astype(np.float32))
            terrains.append(data["terrain"].astype(np.float32))
            proprios.append(data["proprio"].astype(np.float32))
    if not motions:
        raise FileNotFoundError(f"No conditional NPZ files found in {args.data_dir}")
    motion = np.concatenate(motions)
    terrain = np.concatenate(terrains)
    proprio = np.concatenate(proprios)

    generator = np.random.default_rng(args.seed)
    count = min(args.samples, len(motion))
    indices = generator.choice(len(motion), size=count, replace=False)
    truth = torch.from_numpy(motion[indices]).to(device)
    terrain = torch.from_numpy(terrain[indices]).to(device)
    proprio = torch.from_numpy(proprio[indices]).to(device)

    def stat(name: str) -> torch.Tensor:
        return torch.as_tensor(checkpoint[name], dtype=torch.float32, device=device)

    q_low, q_high = stat("q_low"), stat("q_high")
    t_low, t_high = stat("t_q_low"), stat("t_q_high")
    p_low, p_high = stat("p_q_low"), stat("p_q_high")
    terrain = 2.0 * (terrain - t_low) / (t_high - t_low) - 1.0
    proprio_norm = 2.0 * (proprio - p_low) / (p_high - p_low) - 1.0

    # CasBot feature layout: root pos3, rot6d6, joint pos25, joint vel25,
    # five end effectors (15), root linear velocity3, root angular velocity3.
    groups = {
        "root_pos_l2_m": (slice(0, 3), "l2"),
        "root_rot6d_l2": (slice(3, 9), "l2"),
        "joint_pos_mae_rad": (slice(9, 34), "mae"),
        "joint_vel_mae_rad_s": (slice(34, 59), "mae"),
        "ee_pos_l2_m": (slice(59, 74), "ee_l2"),
        "root_lin_vel_l2_m_s": (slice(74, 77), "l2"),
        "root_ang_vel_l2_rad_s": (slice(77, 80), "l2"),
    }
    totals = {name: [] for name in groups}

    torch.manual_seed(args.seed)
    with torch.inference_mode():
        for _ in range(args.repeats):
            generated = torch.randn((count, int(cfg["window_size"]), int(cfg["feature_dim"])), device=device)
            if method == "ddpm":
                assert scheduler is not None
                for step in reversed(range(scheduler.num_timesteps)):
                    timestep = torch.full((count,), step, dtype=torch.long, device=device)
                    noise = model(generated, timestep, terrain=terrain, proprio=proprio_norm)
                    generated = scheduler.step(noise, generated, step)
            else:
                generated = sample_flow(
                    model, generated, terrain, proprio_norm,
                    int(cfg.get("sampling_steps", 10)), str(cfg.get("sampler", "euler")),
                    float(cfg.get("time_embedding_scale", 1000.0)),
                )
            generated = (generated + 1.0) * 0.5 * (q_high - q_low) + q_low
            error = generated[:, 0] - truth[:, 0]
            for name, (feature_slice, metric) in groups.items():
                part = error[:, feature_slice]
                if metric == "mae":
                    value = part.abs().mean().item()
                elif metric == "ee_l2":
                    value = torch.linalg.vector_norm(part.reshape(count, 5, 3), dim=-1).mean().item()
                else:
                    value = _mean_l2(part)
                totals[name].append(value)

    print(f"method={method} paired_samples={count} stochastic_repeats={args.repeats}")
    for name, values in totals.items():
        print(f"{name}: mean={np.mean(values):.6f} repeat_std={np.std(values):.6f}")

    # A directly observable continuity baseline from the four-frame condition:
    # true future-frame-1 joint angles versus the final historical joint angles.
    true_joint_step = (truth[:, 0, 9:34] - proprio[:, -1, :25]).abs().mean().item()
    print(f"true_history4_to_future1_joint_pos_mae_rad: {true_joint_step:.6f}")


if __name__ == "__main__":
    main()
