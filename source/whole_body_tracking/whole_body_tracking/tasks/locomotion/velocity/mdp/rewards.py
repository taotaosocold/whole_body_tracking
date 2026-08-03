"""Custom locomotion reward functions not available in isaaclab/isaaclab_tasks MDP."""

from __future__ import annotations

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame."""
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def feet_air_time_positive_biped(
    env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    Rewards the agent for taking steps up to a specified threshold while keeping
    one foot in the air at a time.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def air_time_variance_penalty(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize differences in air/contact time between the two feet."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


def feet_stumble(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Detect feet hitting vertical surfaces."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    return torch.any(forces_xy > 4 * forces_z, dim=1).float()


def feet_too_near(env, threshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize feet that are closer than the requested threshold."""
    asset = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def joint_coordination_rel(
    env,
    asset_cfg: SceneEntityCfg,
    coord_joints: list[list[str]],
    coord_signs: list[list[float]] | None = None,
) -> torch.Tensor:
    """Penalize deviation from coordinated relative motion of specified joint pairs."""
    asset = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_coord_joints_cache") or env.joint_coord_joints_cache is None:
        env.joint_coord_joints_cache = [
            [asset.find_joints(joint_name)[0] for joint_name in joint_pair]
            for joint_pair in coord_joints
        ]

    if coord_signs is None:
        coord_signs = [[1.0, 1.0]] * len(coord_joints)

    penalty = torch.zeros(env.num_envs, device=env.device)
    for joint_indices, signs in zip(env.joint_coord_joints_cache, coord_signs):
        joint_1 = (
            asset.data.joint_pos[:, joint_indices[0][0]]
            - asset.data.default_joint_pos[:, joint_indices[0][0]]
        )
        joint_2 = (
            asset.data.joint_pos[:, joint_indices[1][0]]
            - asset.data.default_joint_pos[:, joint_indices[1][0]]
        )
        penalty += torch.square(signs[0] * joint_1 - signs[1] * joint_2)
    return penalty / len(coord_joints)


def upward(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize deviation from upright orientation."""
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward
