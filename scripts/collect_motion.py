"""Script to collect filtered motion by running a trained policy.

.. code-block:: bash

    python scripts/collect_motion.py \\
        --task=Tracking-Flat-CASBOT-Wo-State-Estimation-No-Disturbance-v0 \\
        --motion_file=/path/to/motion.npz \\
        --load_run=2026-05-18_14-41-40 \\
        --checkpoint=model_68500.pt \\
        --output=filtered_motion.npz
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os as _os
import sys

# allow importing cli_args from the sibling rsl_rl directory
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "rsl_rl"))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Collect filtered motion by running a trained policy.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument("--output", type=str, default="filtered_motion.npz", help="Output npz file path.")
parser.add_argument("--max_steps", type=int, default=None, help="Max steps to collect (default: entire motion).")
# append RSL-RL cli arguments (adds --load_run, --checkpoint, etc.)
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
import importlib.metadata as metadata
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import whole_body_tracking.tasks  # noqa: F401


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Collect filtered motion with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set motion file
    if args_cli.motion_file is not None:
        env_cfg.commands.motion.motion_file = args_cli.motion_file

    # --- determine motion length to set episode length ---
    motion_data = np.load(args_cli.motion_file)
    motion_fps = motion_data["fps"].item()
    motion_duration = motion_data["joint_pos"].shape[0] / motion_fps
    env_cfg.episode_length_s = motion_duration + 1.0
    print(f"[INFO] Motion duration: {motion_duration:.1f}s, episode_length_s set to {env_cfg.episode_length_s:.1f}s")

    # --- disable all failure-based terminations, keep only time_out ---
    # The robot starts from frame-0 pose which we write manually; any stale
    # body_pos_relative_w from the random-bin reset will cause spurious failures.
    for key in list(env_cfg.terminations.__dict__.keys()):
        if key != "time_out":
            setattr(env_cfg.terminations, key, None)

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # resolve checkpoint
    log_root_path = _os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = _os.path.abspath(log_root_path)
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # load policy
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # get the robot articulation
    robot = env.unwrapped.scene["robot"]

    motion_cmd = env.unwrapped.command_manager.get_term("motion")

    # Prevent future auto-resampling.
    motion_cmd.time_left[:] = 1e9

    # Reset to frame 0: _resample during env.reset() already assigned a random bin.
    motion_cmd.time_steps[:] = 0

    # Write frame-0 state into the simulator so robot pose matches frame 0.
    all_env_ids = torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device)
    joint_pos = motion_cmd.motion.joint_pos[0:1]
    joint_vel = motion_cmd.motion.joint_vel[0:1]
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=all_env_ids)
    pos_w = motion_cmd.motion.body_pos_w[0:1]
    quat_w = motion_cmd.motion.body_quat_w[0:1]
    lin_vel_w = motion_cmd.motion.body_lin_vel_w[0:1]
    ang_vel_w = motion_cmd.motion.body_ang_vel_w[0:1]
    root_state = torch.cat([pos_w[:, 0], quat_w[:, 0], lin_vel_w[:, 0], ang_vel_w[:, 0]], dim=-1)
    robot.write_root_state_to_sim(root_state, env_ids=all_env_ids)

    # determine max steps and fps
    decimation = env_cfg.decimation
    env_step_dt = env_cfg.sim.dt * decimation
    motion_time_steps = motion_cmd.motion.time_step_total
    max_steps = args_cli.max_steps if args_cli.max_steps is not None else motion_time_steps
    fps = 1.0 / env_step_dt
    print(f"[INFO] Motion frames: {motion_time_steps}, collecting {max_steps} steps at {fps:.1f} fps")

    # buffers
    joint_pos_list = []
    joint_vel_list = []
    body_pos_w_list = []
    body_quat_w_list = []
    body_lin_vel_w_list = []
    body_ang_vel_w_list = []

    obs = env.get_observations()

    for step in range(max_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

        if dones[0]:
            print(f"[WARN] Episode terminated at step {step}. Stopping collection.")
            break

        jp = robot.data.joint_pos[0].cpu().numpy()
        jv = robot.data.joint_vel[0].cpu().numpy()
        bp = robot.data.body_pos_w[0].cpu().numpy()
        bq = robot.data.body_quat_w[0].cpu().numpy()
        blv = robot.data.body_lin_vel_w[0].cpu().numpy()
        bav = robot.data.body_ang_vel_w[0].cpu().numpy()

        joint_pos_list.append(jp)
        joint_vel_list.append(jv)
        body_pos_w_list.append(bp)
        body_quat_w_list.append(bq)
        body_lin_vel_w_list.append(blv)
        body_ang_vel_w_list.append(bav)

        if step % 100 == 0:
            print(f"[INFO] Step {step}/{max_steps}")

    joint_pos = np.stack(joint_pos_list, axis=0)
    joint_vel = np.stack(joint_vel_list, axis=0)
    body_pos_w = np.stack(body_pos_w_list, axis=0)
    body_quat_w = np.stack(body_quat_w_list, axis=0)
    body_lin_vel_w = np.stack(body_lin_vel_w_list, axis=0)
    body_ang_vel_w = np.stack(body_ang_vel_w_list, axis=0)

    output_path = args_cli.output
    np.savez(
        output_path,
        fps=np.array([fps]),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
    )
    print(f"[INFO] Collected motion saved to: {output_path}")
    print(f"  fps: {fps}")
    print(f"  joint_pos: {joint_pos.shape}")
    print(f"  joint_vel: {joint_vel.shape}")
    print(f"  body_pos_w: {body_pos_w.shape}")
    print(f"  body_quat_w: {body_quat_w.shape}")
    print(f"  body_lin_vel_w: {body_lin_vel_w.shape}")
    print(f"  body_ang_vel_w: {body_ang_vel_w.shape}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
