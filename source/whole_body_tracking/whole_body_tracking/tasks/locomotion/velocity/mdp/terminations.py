"""Termination functions specific to velocity locomotion."""

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg


def root_height_below_env_origin_minimum(
    env, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when root height relative to the environment origin is too low."""

    asset: RigidObject = env.scene[asset_cfg.name]
    terrain_base_height = torch.clamp(env.scene.env_origins[:, 2], max=0.0)
    return asset.data.root_pos_w[:, 2] - terrain_base_height < minimum_height
