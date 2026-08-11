"""Parkour reward terms matching InstinctLab perceptive shadowing."""

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def applied_torque_limits_by_ratio(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    limit_ratio: float = 0.8,
) -> torch.Tensor:
    """Penalize squared torque exceeding a ratio of the simulated effort limit."""
    asset: Articulation = env.scene[asset_cfg.name]
    effort_limits = asset.data.joint_effort_limits[:, asset_cfg.joint_ids]
    applied_torque = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids])
    excess = (applied_torque - effort_limits * limit_ratio).clip(min=0.0)
    return torch.sum(torch.square(excess), dim=-1)


def ankle_self_collision(
    env,
    sensor_names: tuple[str, ...],
    threshold: float = 1.0,
) -> torch.Tensor:
    """Count ankles contacting robot links, excluding terrain contacts."""
    violations = torch.zeros(env.num_envs, device=env.device)
    for sensor_name in sensor_names:
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        forces = sensor.data.force_matrix_w_history
        if forces is None:
            raise RuntimeError(
                f"Contact sensor '{sensor_name}' needs filter_prim_paths_expr "
                "to report ankle self-collision forces."
            )
        # [env, history, sensor_body, filtered_body, xyz] -> [env]
        max_force = torch.linalg.vector_norm(forces, dim=-1).amax(dim=(1, 2, 3))
        violations += (max_force > threshold).float()
    return violations
