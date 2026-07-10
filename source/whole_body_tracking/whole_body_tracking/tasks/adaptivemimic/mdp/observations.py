from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, quat_mul, subtract_frame_transforms

from whole_body_tracking.tasks.adaptivemimic.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def _de_yaw(quat_w: torch.Tensor) -> torch.Tensor:
    """Remove the yaw (world-Z rotation) component from a world-frame quaternion.

    The returned quaternion contains only the roll and pitch components,
    preserving gravity-alignment information (Stubborn paper, Sec. III-B).
    """
    R = matrix_from_quat(quat_w)
    yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    half_neg_yaw = -yaw / 2.0
    q_yaw_inv = torch.stack(
        [torch.cos(half_neg_yaw), torch.zeros_like(half_neg_yaw), torch.zeros_like(half_neg_yaw), torch.sin(half_neg_yaw)],
        dim=-1,
    )
    return quat_mul(q_yaw_inv, quat_w)


def motion_anchor_ori_yaw_aligned_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Relative anchor orientation in the yaw-aligned frame (Stubborn paper).

    Both the robot and motion anchor orientations are de-yawed before computing
    the relative orientation, so the result encodes only pitch/roll differences.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot_deyaw = _de_yaw(command.robot_anchor_quat_w)
    anchor_deyaw = _de_yaw(command.anchor_quat_w)
    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w, robot_deyaw,
        command.anchor_pos_w, anchor_deyaw,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)
