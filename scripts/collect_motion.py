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
parser.add_argument(
    "--collect_height_map",
    action="store_true",
    help="Save the noise-free height-scanner XYZ map for every collected frame.",
)
parser.add_argument(
    "--center_xy",
    action="store_true",
    help=(
        "For parkour, load only the selected motion/STL in a 1x1 terrain at "
        "world XY=(0, 0), without changing the motion's authored coordinates."
    ),
)
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


def _select_parkour_motion(command_cfg, motion_file: str | None) -> tuple[int, str]:
    """Resolve one motion from a multi-motion parkour command configuration."""
    configured_files = list(command_cfg.motion_files)
    if motion_file is None:
        if len(configured_files) != 1:
            raise ValueError(
                "This parkour task contains multiple motions. Pass --motion_file with one of: "
                + ", ".join(configured_files)
            )
        motion_index = 0
    else:
        requested_name = _os.path.basename(motion_file)
        requested_stem = _os.path.splitext(requested_name)[0]
        matches = [
            index
            for index, configured_name in enumerate(configured_files)
            if requested_name == _os.path.basename(configured_name)
            or requested_stem == _os.path.splitext(_os.path.basename(configured_name))[0]
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Parkour motion {motion_file!r} is not uniquely configured. Available motions: "
                + ", ".join(configured_files)
            )
        motion_index = matches[0]

    motion_path = _os.path.join(command_cfg.motion_folder, configured_files[motion_index])
    return motion_index, motion_path


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Collect filtered motion with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # A tracking task owns one motion_file, while a parkour task owns a configured
    # motion/terrain list.  For parkour, --motion_file selects one existing pair.
    is_parkour = hasattr(env_cfg.commands.motion, "motion_files")
    parkour_motion_index = None
    if is_parkour:
        parkour_motion_index, selected_motion_path = _select_parkour_motion(
            env_cfg.commands.motion, args_cli.motion_file
        )

        if args_cli.center_xy:
            # Build only the selected motion/terrain pair. A 1x1 terrain grid is
            # centered by TerrainGenerator at the world origin, so no column or
            # row layout offset is added to the authored motion/STL coordinates.
            selected_motion_name = env_cfg.commands.motion.motion_files[
                parkour_motion_index
            ]
            selected_terrain_key = f"motion_{parkour_motion_index:03d}"
            terrain_generator = env_cfg.scene.terrain.terrain_generator
            selected_terrain_cfg = terrain_generator.sub_terrains[
                selected_terrain_key
            ]
            selected_terrain_cfg.proportion = 1.0
            terrain_generator.sub_terrains = {
                "selected_motion": selected_terrain_cfg
            }
            terrain_generator.num_rows = 1
            terrain_generator.num_cols = 1

            selected_motion_stem = _os.path.splitext(
                _os.path.basename(selected_motion_name)
            )[0]
            env_cfg.commands.motion.motion_files = [selected_motion_name]
            env_cfg.commands.motion.motion_terrain_columns = {
                selected_motion_stem: 0
            }
            parkour_motion_index = 0
            env_cfg.scene.num_envs = 1
            print(
                "[INFO] --center_xy: using one environment with only the "
                f"selected motion and terrain: {selected_motion_name}"
            )
    else:
        if args_cli.motion_file is None:
            selected_motion_path = env_cfg.commands.motion.motion_file
        else:
            selected_motion_path = args_cli.motion_file
            env_cfg.commands.motion.motion_file = selected_motion_path

    # --- determine motion length to set episode length ---
    with np.load(selected_motion_path) as motion_data:
        motion_fps = float(np.asarray(motion_data["fps"]).reshape(-1)[0])
        selected_motion_steps = motion_data["joint_pos"].shape[0]
    motion_duration = selected_motion_steps / motion_fps
    env_cfg.episode_length_s = motion_duration + 1.0
    print(f"[INFO] Selected motion: {selected_motion_path}")
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
    if is_parkour:
        # Pin every collection environment to the selected motion and one copy
        # of its statically bound terrain.
        motion_cmd.active_motion_idx[:] = parkour_motion_index
        motion_cmd.motion_origins[:] = motion_cmd._terrain_origins_by_motion[
            parkour_motion_index, 0
        ]
        if args_cli.center_xy:
            terrain_origin = motion_cmd.motion_origins[0]
            if not torch.allclose(
                terrain_origin[:2], torch.zeros_like(terrain_origin[:2]), atol=1.0e-6
            ):
                raise RuntimeError(
                    "The single centered terrain was expected at world XY=(0, 0), "
                    f"but its origin is {terrain_origin.tolist()}."
                )
            print(
                "[INFO] Single terrain world origin: "
                f"({terrain_origin[0].item():.6f}, "
                f"{terrain_origin[1].item():.6f}, "
                f"{terrain_origin[2].item():.6f})"
            )

    # Write frame-0 state into the simulator so robot pose matches frame 0.
    all_env_ids = torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device)
    if is_parkour:
        first_frame = int(motion_cmd.motion.motion_start[parkour_motion_index].item())
        root_origins = motion_cmd.motion_origins
    else:
        first_frame = 0
        root_origins = env.unwrapped.scene.env_origins
    joint_pos = motion_cmd.motion.joint_pos[first_frame : first_frame + 1].expand(
        env.unwrapped.num_envs, -1
    )
    joint_vel = motion_cmd.motion.joint_vel[first_frame : first_frame + 1].expand(
        env.unwrapped.num_envs, -1
    )
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=all_env_ids)
    pos_w = motion_cmd.motion.body_pos_w[first_frame, 0].unsqueeze(0) + root_origins
    quat_w = motion_cmd.motion.body_quat_w[first_frame, 0].unsqueeze(0).expand(
        env.unwrapped.num_envs, -1
    )
    lin_vel_w = motion_cmd.motion.body_lin_vel_w[first_frame, 0].unsqueeze(0).expand(
        env.unwrapped.num_envs, -1
    )
    ang_vel_w = motion_cmd.motion.body_ang_vel_w[first_frame, 0].unsqueeze(0).expand(
        env.unwrapped.num_envs, -1
    )
    root_state = torch.cat([pos_w, quat_w, lin_vel_w, ang_vel_w], dim=-1)
    robot.write_root_state_to_sim(root_state, env_ids=all_env_ids)
    if is_parkour:
        authored_root = motion_cmd.motion.body_pos_w[first_frame, 0]
        print(
            "[INFO] Frame-0 root initialization: "
            f"authored=({authored_root[0].item():.6f}, "
            f"{authored_root[1].item():.6f}, {authored_root[2].item():.6f}), "
            f"terrain_origin=({root_origins[0, 0].item():.6f}, "
            f"{root_origins[0, 1].item():.6f}, {root_origins[0, 2].item():.6f}), "
            f"sim_root=({pos_w[0, 0].item():.6f}, "
            f"{pos_w[0, 1].item():.6f}, {pos_w[0, 2].item():.6f})"
        )

    # determine max steps and fps
    decimation = env_cfg.decimation
    env_step_dt = env_cfg.sim.dt * decimation
    motion_time_steps = selected_motion_steps
    # ParkourMotionCommand resamples immediately after its final frame. Since
    # collection happens after env.step(), stop one step earlier to avoid
    # saving the first frame of a different motion/terrain at the end.
    default_steps = max(1, motion_time_steps - 1) if is_parkour else motion_time_steps
    max_steps = args_cli.max_steps if args_cli.max_steps is not None else default_steps
    fps = 1.0 / env_step_dt
    print(f"[INFO] Motion frames: {motion_time_steps}, collecting {max_steps} steps at {fps:.1f} fps")

    # buffers
    joint_pos_list = []
    joint_vel_list = []
    body_pos_w_list = []
    body_quat_w_list = []
    body_lin_vel_w_list = []
    body_ang_vel_w_list = []
    elevation_map_xyz_list = []

    if args_cli.collect_height_map:
        if "height_scanner" not in env.unwrapped.scene.sensors:
            raise ValueError(
                "--collect_height_map requires a task with a 'height_scanner' sensor "
                "(for example Tracking-Parkour-CASBOT-v0)."
            )
        from isaaclab.managers import SceneEntityCfg
        from whole_body_tracking.tasks.locomotion.velocity.mdp import elevation_map_xyz
        from isaaclab.utils.math import quat_apply, yaw_quat

        height_sensor = env.unwrapped.scene.sensors["height_scanner"]
        height_sensor_cfg = SceneEntityCfg("height_scanner")
        height_map_shape = tuple(height_sensor.cfg.pattern_cfg.size)
        height_map_resolution = float(height_sensor.cfg.pattern_cfg.resolution)
        print(
            f"[INFO] Collecting height maps: {height_sensor.data.ray_hits_w.shape[1]} rays, "
            f"size={height_map_shape}, resolution={height_map_resolution} m"
        )

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
        if args_cli.collect_height_map:
            height_xyz = elevation_map_xyz(
                env.unwrapped, height_sensor_cfg, noise=False
            )[0].reshape(-1, 3)
            if args_cli.center_xy:
                # Convert the yaw-aligned sensor-frame samples back to world
                # coordinates. The authored robot/STL world coordinates are
                # preserved, so no additional XY centering shift is applied.
                sensor_yaw = yaw_quat(height_sensor.data.quat_w[0:1]).expand(
                    height_xyz.shape[0], -1
                )
                height_xyz = quat_apply(sensor_yaw, height_xyz)
                height_xyz += height_sensor.data.pos_w[0]
            elevation_map_xyz_list.append(height_xyz.cpu().numpy())

        if step % 100 == 0:
            print(f"[INFO] Step {step}/{max_steps}")

    joint_pos = np.stack(joint_pos_list, axis=0)
    joint_vel = np.stack(joint_vel_list, axis=0)
    body_pos_w = np.stack(body_pos_w_list, axis=0)
    body_quat_w = np.stack(body_quat_w_list, axis=0)
    body_lin_vel_w = np.stack(body_lin_vel_w_list, axis=0)
    body_ang_vel_w = np.stack(body_ang_vel_w_list, axis=0)

    output_path = args_cli.output
    output_data = dict(
        fps=np.array([fps]),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
    )
    if args_cli.collect_height_map:
        elevation_map_xyz_data = np.stack(elevation_map_xyz_list, axis=0)
        output_data.update(
            elevation_map_xyz=elevation_map_xyz_data,
            height_map_grid_size=np.asarray(height_map_shape, dtype=np.float32),
            height_map_resolution=np.asarray([height_map_resolution], dtype=np.float32),
            height_map_frame=np.asarray(
                "world" if args_cli.center_xy else "yaw_aligned_sensor"
            ),
        )
    np.savez(output_path, **output_data)
    print(f"[INFO] Collected motion saved to: {output_path}")
    print(f"  fps: {fps}")
    print(f"  joint_pos: {joint_pos.shape}")
    print(f"  joint_vel: {joint_vel.shape}")
    print(f"  body_pos_w: {body_pos_w.shape}")
    print(f"  body_quat_w: {body_quat_w.shape}")
    print(f"  body_lin_vel_w: {body_lin_vel_w.shape}")
    print(f"  body_ang_vel_w: {body_ang_vel_w.shape}")
    if args_cli.collect_height_map:
        print(f"  elevation_map_xyz: {output_data['elevation_map_xyz'].shape}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
