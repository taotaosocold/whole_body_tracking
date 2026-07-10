from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from whole_body_tracking.tasks.adaptivemimic.mdp.commands import MotionCommand
from whole_body_tracking.tasks.adaptivemimic.mdp.rewards import _get_body_indexes


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_rotate_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_rotate_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_anchor_ori_probabilistic(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float, prob: float = 0.005
) -> torch.Tensor:
    """Bernoulli probabilistic termination for anchor orientation (Stubborn paper).

    Instead of hard-terminating when the orientation error exceeds the threshold,
    each step beyond the threshold terminates with probability *prob* (default 0.005).
    This gives the policy ~200 expected extra steps to recover from falls.
    The orientation error is computed in the yaw-aligned frame, ignoring heading.
    """
    from isaaclab.utils.math import matrix_from_quat, quat_error_magnitude, quat_mul

    command: MotionCommand = env.command_manager.get_term(command_name)

    # De-yaw both quaternions
    def _de_yaw(q):
        R = matrix_from_quat(q)
        yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
        half_neg = -yaw / 2.0
        qi = torch.stack(
            [torch.cos(half_neg), torch.zeros_like(half_neg), torch.zeros_like(half_neg), torch.sin(half_neg)],
            dim=-1,
        )
        return quat_mul(qi, q)

    error = quat_error_magnitude(
        _de_yaw(command.anchor_quat_w), _de_yaw(command.robot_anchor_quat_w)
    )
    over_threshold = error > threshold
    rand = torch.rand(env.num_envs, device=env.device)
    return over_threshold & (rand < prob)


def bad_anchor_pos_z_only_probabilistic(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, prob: float = 0.005
) -> torch.Tensor:
    """Bernoulli probabilistic termination for anchor height (Stubborn paper).

    Same probabilistic scheme as bad_anchor_ori_probabilistic, applied to the
    vertical position error of the anchor body.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    over_threshold = torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold
    rand = torch.rand(env.num_envs, device=env.device)
    return over_threshold & (rand < prob)


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_global_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)
