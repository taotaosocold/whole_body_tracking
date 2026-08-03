"""Custom observations for terrain locomotion."""

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat


def elevation_map_xyz(env, sensor_cfg: SceneEntityCfg, noise: bool = False) -> torch.Tensor:
    """Return AME-style XYZ terrain points in the yaw-aligned sensor frame."""

    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    relative_pos_w = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
    sensor_quat = sensor.data.quat_w
    num_envs, num_rays, _ = relative_pos_w.shape

    if getattr(sensor.cfg, "ray_alignment", "base") == "yaw":
        sensor_quat = yaw_quat(sensor_quat)

    sensor_quat = sensor_quat.unsqueeze(1).expand(num_envs, num_rays, 4).reshape(-1, 4).float()
    sensor_coords = quat_apply_inverse(sensor_quat, relative_pos_w.reshape(-1, 3))
    sensor_coords = sensor_coords.reshape(num_envs, num_rays, 3)
    sensor_coords = torch.nan_to_num(sensor_coords)

    if noise:
        if (
            getattr(env, "_elevation_map_offset", None) is None
            or env._elevation_map_offset.shape != (num_envs, 1)
        ):
            env._elevation_map_offset = torch.zeros(
                (num_envs, 1), device=sensor_coords.device, dtype=sensor_coords.dtype
            )

        # AME resamples one shared height bias per environment on reset.
        if hasattr(env, "reset_buf"):
            reset_env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if len(reset_env_ids) > 0:
                env._elevation_map_offset[reset_env_ids] = (
                    torch.rand(
                        (len(reset_env_ids), 1),
                        device=sensor_coords.device,
                        dtype=sensor_coords.dtype,
                    )
                    * 0.1
                    - 0.05
                )

        # Only height is corrupted: independent 3 cm Gaussian noise plus the
        # per-environment shared bias. X/Y coordinates remain exact.
        sensor_coords[..., 2] += torch.randn_like(sensor_coords[..., 2]) * 0.03
        sensor_coords[..., 2] += env._elevation_map_offset

    sensor_coords[..., 2] = torch.clamp(sensor_coords[..., 2], min=-1.2, max=0.0)
    return sensor_coords.reshape(num_envs, num_rays * 3)
