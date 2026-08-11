"""Parkour randomization terms matching InstinctLab perceptive shadowing."""

from typing import Literal

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster


def randomize_default_joint_pos(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    offset_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"] = "add",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize calibrated default joint positions and update the action offset."""
    asset: Articulation = env.scene[asset_cfg.name]
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    joint_ids = (
        slice(None)
        if asset_cfg.joint_ids == slice(None)
        else torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)
    )
    pos = _randomize_prop_by_op(
        asset.data.default_joint_pos.clone(),
        offset_distribution_params,
        env_ids,
        joint_ids,
        operation=operation,
        distribution=distribution,
    )[env_ids][:, joint_ids]
    write_env_ids = env_ids[:, None] if env_ids != slice(None) and joint_ids != slice(None) else env_ids
    asset.data.default_joint_pos[write_env_ids, joint_ids] = pos
    env.action_manager.get_term("joint_pos")._offset[write_env_ids, joint_ids] = pos


def randomize_ray_offsets(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    offset_pose_ranges: dict[str, tuple[float, float]],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize ray origins and directions to model sensor installation error."""
    del distribution
    sensor: RayCaster = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=sensor.device)

    ray_starts = sensor.ray_starts[env_ids]
    ray_directions = sensor.ray_directions[env_ids]
    ranges = torch.tensor(
        [offset_pose_ranges.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]],
        device=ray_starts.device,
    )
    samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=ray_starts.device
    )
    samples = samples[:, None, :].repeat(1, sensor.num_rays, 1)
    ray_starts += samples[..., :3]
    rotations = math_utils.quat_from_euler_xyz(
        samples[..., 3].flatten(), samples[..., 4].flatten(), samples[..., 5].flatten()
    ).reshape(*ray_directions.shape[:-1], 4)
    ray_directions = math_utils.quat_apply(rotations, ray_directions)
    sensor.ray_starts[env_ids] = ray_starts
    sensor.ray_directions[env_ids] = ray_directions
