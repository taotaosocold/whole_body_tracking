"""Parkour reward terms matching InstinctLab perceptive shadowing."""

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply


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


def foot_edge_support_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg,
    foot_sample_points: tuple[tuple[float, float, float], ...] = (
        (-0.085, -0.045, -0.067),
        (-0.085, 0.045, -0.067),
        (0.155, -0.045, -0.067),
        (0.155, 0.045, -0.067),
    ),
    contact_threshold: float = 5.0,
    support_tolerance: float = 0.03,
    max_horizontal_query_distance: float = 0.06,
) -> torch.Tensor:
    """Penalize a contacting foot whose sole corners are not equally supported.

    Four inset sole points are transformed from each ankle-roll frame to world
    coordinates.  Their terrain heights are obtained from the nearest rays in
    the existing waist-mounted height scan.  A fully supported sole has nearly
    equal corner-to-terrain gaps; a foot spanning a stair edge has one or more
    corners over a lower tread and therefore a large gap spread.

    The term is gated by measured foot contact, so swing feet are not penalized.
    It returns the squared excess gap spread summed over the selected feet.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    height_sensor: RayCaster = env.scene.sensors[height_sensor_cfg.name]

    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
    num_envs, num_feet, _ = foot_pos_w.shape

    local_points = torch.tensor(
        foot_sample_points,
        device=foot_pos_w.device,
        dtype=foot_pos_w.dtype,
    )
    num_points = local_points.shape[0]
    local_points = local_points.view(1, 1, num_points, 3).expand(
        num_envs, num_feet, num_points, 3
    )
    corner_quat = foot_quat_w[:, :, None, :].expand(
        num_envs, num_feet, num_points, 4
    )
    corner_pos_w = foot_pos_w[:, :, None, :] + quat_apply(
        corner_quat.reshape(-1, 4), local_points.reshape(-1, 3)
    ).reshape(num_envs, num_feet, num_points, 3)

    # Query the closest existing height ray for each sole point. Looping over
    # eight points avoids allocating [num_envs, feet, points, 693] at once.
    ray_hits_w = height_sensor.data.ray_hits_w
    finite_rays = torch.isfinite(ray_hits_w).all(dim=-1)
    corner_terrain_z = torch.empty(
        (num_envs, num_feet, num_points),
        device=foot_pos_w.device,
        dtype=foot_pos_w.dtype,
    )
    corner_query_valid = torch.zeros(
        (num_envs, num_feet, num_points),
        device=foot_pos_w.device,
        dtype=torch.bool,
    )
    env_indices = torch.arange(num_envs, device=foot_pos_w.device)
    for foot_index in range(num_feet):
        for point_index in range(num_points):
            delta_xy = (
                ray_hits_w[..., :2]
                - corner_pos_w[:, foot_index, point_index, :2].unsqueeze(1)
            )
            distance_sq = torch.sum(torch.square(delta_xy), dim=-1)
            # Invalid/missed rays are excluded from nearest-neighbour selection;
            # they are not assigned an invented terrain height.
            distance_sq = torch.where(
                finite_rays & torch.isfinite(distance_sq),
                distance_sq,
                torch.full_like(distance_sq, torch.inf),
            )
            nearest_ray = torch.argmin(distance_sq, dim=1)
            nearest_distance_sq = distance_sq[env_indices, nearest_ray]
            nearest_height = ray_hits_w[env_indices, nearest_ray, 2]
            query_valid = (
                torch.isfinite(nearest_height)
                & torch.isfinite(nearest_distance_sq)
                & (nearest_distance_sq <= max_horizontal_query_distance**2)
            )
            corner_query_valid[:, foot_index, point_index] = query_valid
            # This fallback is used only to keep intermediate arithmetic finite.
            # query_valid below masks the complete foot out of the reward, so the
            # fallback value never represents terrain and never creates reward.
            corner_terrain_z[:, foot_index, point_index] = torch.where(
                query_valid,
                nearest_height,
                corner_pos_w[:, foot_index, point_index, 2],
            )

    sole_gaps = corner_pos_w[..., 2] - corner_terrain_z
    gap_spread = sole_gaps.amax(dim=-1) - sole_gaps.amin(dim=-1)
    unsupported_excess = torch.clamp(gap_spread - support_tolerance, min=0.0)
    complete_terrain_query = corner_query_valid.all(dim=-1)

    contact_forces = contact_sensor.data.net_forces_w_history
    max_contact_force = torch.linalg.vector_norm(contact_forces, dim=-1).amax(dim=1)
    in_contact = max_contact_force[:, contact_sensor_cfg.body_ids] > contact_threshold
    active = in_contact & complete_terrain_query
    penalty = torch.where(
        active,
        torch.square(unsupported_excess),
        torch.zeros_like(unsupported_excess),
    )
    penalty = torch.sum(penalty, dim=1)
    if not torch.isfinite(penalty).all():
        raise RuntimeError(
            "foot_edge_support_penalty produced a non-finite value despite "
            "terrain-query validity masking"
        )
    return penalty
