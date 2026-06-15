#!/usr/bin/env python3
"""
Bayesian optimization of reward weights via Optuna.

Runs multiple short training trials with different reward weight combinations,
using Optuna's TPE sampler to efficiently search for the best weights.

Usage:
    python scripts/optuna_reward_search.py \
        --task Tracking-Flat-CASBOT-No-Disturbance-v0 \
        --motion_file /path/to/motion.npz \
        --max_iterations 3000 \
        --n_trials 20 \
        --seeds_per_trial 2 \
        --headless
"""

import argparse
import copy
import importlib.metadata as metadata
import os
import sys
import time
from datetime import datetime

import numpy as np
import optuna
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Optuna reward weight search for whole-body tracking.")
parser.add_argument("--task", type=str, required=True, help="Gym task ID, e.g. Tracking-Flat-CASBOT-No-Disturbance-v0")
parser.add_argument("--motion_file", type=str, required=True, help="Path to motion .npz file.")
parser.add_argument("--max_iterations", type=int, default=2000, help="RL iterations per trial.")
parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials.")
parser.add_argument("--seeds_per_trial", type=int, default=2, help="Seeds per trial (median taken).")
parser.add_argument("--num_envs", type=int, default=None, help="Override number of environments.")
parser.add_argument("--base_seed", type=int, default=42, help="Base random seed.")
parser.add_argument("--output_dir", type=str, default="logs/optuna_reward_search", help="Output directory for logs.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import whole_body_tracking.tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner

# ---- Resolve config instances from gym registry ----
# load_cfg_from_registry returns INSTANCES (not classes), so we deepcopy per trial.
_base_env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
_base_agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

# ---- Reward weight search space ----
# Anchor position/orientation: global tracking
SEARCH_SPACE = {
    "motion_global_anchor_pos": (0.1, 2.0),
    "motion_global_anchor_ori": (0.1, 2.0),
    # Body position/orientation: relative tracking
    "motion_body_pos": (0.5, 5.0),
    "motion_body_ori": (0.5, 5.0),
    # Body linear/angular velocity
    "motion_body_lin_vel": (0.5, 5.0),
    "motion_body_ang_vel": (0.5, 5.0),
    # Action rate regularization
    "action_rate_l2": (1e-3, 0.5),
    # Joint limit penalty
    "joint_limit": (1.0, 30.0),
}


def sample_reward_weights(trial: optuna.Trial) -> dict:
    """Sample reward weights from the search space."""
    weights = {}
    for name, (low, high) in SEARCH_SPACE.items():
        log_scale = name in ("action_rate_l2",)
        if log_scale:
            weights[name] = trial.suggest_float(name, low, high, log=True)
        else:
            weights[name] = trial.suggest_float(name, low, high)
    return weights


def apply_weights(env_cfg: ManagerBasedRLEnvCfg, weights: dict):
    """Apply sampled weights to the env config's reward terms."""
    rewards = env_cfg.rewards
    rewards.motion_global_anchor_pos.weight = weights["motion_global_anchor_pos"]
    rewards.motion_global_anchor_ori.weight = weights["motion_global_anchor_ori"]
    rewards.motion_body_pos.weight = weights["motion_body_pos"]
    rewards.motion_body_ori.weight = weights["motion_body_ori"]
    rewards.motion_body_lin_vel.weight = weights["motion_body_lin_vel"]
    rewards.motion_body_ang_vel.weight = weights["motion_body_ang_vel"]
    rewards.action_rate_l2.weight = -abs(weights["action_rate_l2"])
    rewards.joint_limit.weight = -abs(weights["joint_limit"])


def run_training(
    env_cfg: ManagerBasedRLEnvCfg,
    agent_cfg: dict,
    task_name: str,
    trial_dir: str,
    seed: int,
) -> float:
    """Run a single training and return the median episode length (last 100 episodes).

    Uses episode length as the signal — it's independent of reward weight scale,
    unlike episode return which varies with the weights being searched.
    Longer episodes = better tracking = better reward weights.
    """
    env_cfg = copy.deepcopy(env_cfg)
    env_cfg.seed = seed
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.commands.motion.motion_file = os.path.abspath(args_cli.motion_file)

    agent_cfg = copy.deepcopy(agent_cfg)
    agent_cfg["seed"] = seed
    agent_cfg["max_iterations"] = args_cli.max_iterations

    env = gym.make(task_name, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    runner = MotionOnPolicyRunner(env, agent_cfg, log_dir=trial_dir, device=env_cfg.sim.device)
    runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)

    # Use episode LENGTH (not reward) — independent of reward weight scale.
    # A policy with good reward weights survives longer before hitting termination.
    lenbuffer = getattr(runner.logger, "lenbuffer", None)
    if lenbuffer is None or len(lenbuffer) == 0:
        perf = 0.0
    else:
        perf = float(np.median(list(lenbuffer)))

    env.close()
    torch.cuda.empty_cache()
    return perf


def objective(trial: optuna.Trial) -> float:
    """Optuna objective: sample weights, run N seeds, return median performance."""
    weights = sample_reward_weights(trial)

    # Create base configs (deepcopy so each trial gets fresh instances)
    env_cfg = copy.deepcopy(_base_env_cfg)
    agent_cfg = copy.deepcopy(_base_agent_cfg)
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    agent_cfg_dict = agent_cfg.to_dict()

    apply_weights(env_cfg, weights)

    perf_per_seed = []
    for s in range(args_cli.seeds_per_trial):
        seed = args_cli.base_seed + trial.number * args_cli.seeds_per_trial + s
        trial_dir = os.path.join(
            args_cli.output_dir,
            f"trial_{trial.number:03d}",
            f"seed_{seed}",
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )
        os.makedirs(trial_dir, exist_ok=True)

        print(f"\n[Trial {trial.number} | Seed {seed}] Weights: ", end="")
        for k, v in weights.items():
            print(f"{k}={v:.4f}", end=" ")
        print()

        t0 = time.time()
        perf = run_training(env_cfg, agent_cfg_dict, args_cli.task, trial_dir, seed)
        elapsed = time.time() - t0
        perf_per_seed.append(perf)
        print(f"  Seed {seed}: return={perf:.4f}  ({elapsed:.0f}s)")

        # Early stop: if first seed barely survives, skip remaining seeds
        if s == 0 and perf < 2.0:
            print(f"  Aborting trial {trial.number} (ep_len={perf:.2f})")
            break

    median_perf = float(np.median(perf_per_seed))
    print(f"  Trial {trial.number} median return: {median_perf:.4f}")
    return median_perf


def main():
    os.makedirs(args_cli.output_dir, exist_ok=True)

    # Use a fresh sampler and study (no resume for simplicity)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args_cli.base_seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )

    print(f"[INFO] Starting Optuna reward weight search")
    print(f"  Task: {args_cli.task}")
    print(f"  Motion: {args_cli.motion_file}")
    print(f"  Trials: {args_cli.n_trials}")
    print(f"  Seeds per trial: {args_cli.seeds_per_trial}")
    print(f"  Max iterations per run: {args_cli.max_iterations}")
    print(f"  Output: {args_cli.output_dir}")

    study.optimize(objective, n_trials=args_cli.n_trials)

    print("\n" + "=" * 60)
    print("Best trial:")
    print(f"  Value: {study.best_value:.4f}")
    print(f"  Params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v:.4f}")

    # Save results
    import json

    results = {
        "best_value": float(study.best_value),
        "best_params": {k: float(v) for k, v in study.best_params.items()},
        "task": args_cli.task,
        "motion_file": args_cli.motion_file,
        "max_iterations": args_cli.max_iterations,
        "n_trials": args_cli.n_trials,
        "seeds_per_trial": args_cli.seeds_per_trial,
    }
    results_path = os.path.join(args_cli.output_dir, "best_weights.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    simulation_app.close()


if __name__ == "__main__":
    main()
