"""CNN + global-average-pooling encoder for parkour height maps."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg


@configclass
class RslRlParkourCnnModelCfg(RslRlMLPModelCfg):
    """Configuration for :class:`ParkourCnnModel`."""

    class_name: str = "whole_body_tracking.tasks.parkour.cnn_encoder:ParkourCnnModel"
    map_scan_dim: tuple[int, int, int] = (33, 21, 3)
    terrain_feature_dim: int = 32


class ParkourCnnModel(MLPModel):
    """Encode the XYZ elevation map with a 3-layer CNN and GAP."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (1024, 512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        map_scan_dim: tuple[int, int, int] | list[int] = (33, 21, 3),
        terrain_feature_dim: int = 32,
    ) -> None:
        self.map_scan_dim = tuple(map_scan_dim)
        self.L, self.W, self.coord_dim = self.map_scan_dim
        self.map_scan_size = self.L * self.W * self.coord_dim
        self.terrain_feature_dim = terrain_feature_dim

        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )

        self.proprio_dim = self.obs_dim - self.map_scan_size
        if self.proprio_dim <= 0:
            raise ValueError(
                f"Terrain scan ({self.map_scan_size}) does not fit {obs_set} observation "
                f"dimension ({self.obs_dim})."
            )

        self.map_cnn = nn.Sequential(
            nn.Conv2d(self.coord_dim, 16, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, self.terrain_feature_dim, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        print(
            f"ParkourCnnModel[{obs_set}]: map={self.L}x{self.W}x{self.coord_dim}, "
            f"proprio={self.proprio_dim}, terrain_feature={self.terrain_feature_dim}"
        )

    def _get_latent_dim(self) -> int:
        return self.obs_dim - self.map_scan_size + self.terrain_feature_dim

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        del masks, hidden_state
        flat_obs = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        flat_obs = self.obs_normalizer(flat_obs)

        proprio = flat_obs[:, :-self.map_scan_size]
        terrain = flat_obs[:, -self.map_scan_size :].reshape(
            -1, self.W, self.L, self.coord_dim
        )
        terrain = terrain.permute(0, 3, 1, 2)
        terrain_feature = self.map_cnn(terrain).flatten(1)
        return torch.cat((proprio, terrain_feature), dim=-1)
