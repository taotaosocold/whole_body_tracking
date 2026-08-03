"""CNN and proprioception-conditioned cross-attention model for terrain locomotion."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg


@configclass
class RslRlTerrainEncoderModelCfg(RslRlMLPModelCfg):
    """Configuration for :class:`TerrainEncoderModel`."""

    class_name: str = (
        "whole_body_tracking.tasks.locomotion.velocity.terrain_encoder:TerrainEncoderModel"
    )
    map_scan_dim: tuple[int, int, int] = (33, 21, 3)
    mha_dim: int = 64
    num_heads: int = 16
    cnn_downsample: bool = True


class TerrainEncoderModel(MLPModel):
    """Encode an XYZ terrain map using CNN features queried by proprioception.

    The final ``L * W * coord_dim`` values of the selected observation groups
    must contain the flattened terrain scan.  With the default 33 x 21 x 3
    input, the stride-two CNN produces 17 x 11 = 187 terrain tokens.  A single
    proprioceptive query attends over these tokens and the resulting 64-D
    terrain feature is concatenated with the original proprioception before the
    actor or critic MLP head.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        map_scan_dim: tuple[int, int, int] | list[int] = (33, 21, 3),
        mha_dim: int = 64,
        num_heads: int = 16,
        cnn_downsample: bool = True,
        cnns: nn.ModuleDict | None = None,
    ) -> None:
        self.map_scan_dim = tuple(map_scan_dim)
        self.L, self.W, self.coord_dim = self.map_scan_dim
        self.map_scan_size = self.L * self.W * self.coord_dim
        self.mha_dim = mha_dim
        self.num_heads = num_heads
        self.cnn_downsample = cnn_downsample

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
                f"Terrain scan ({self.map_scan_size}) does not fit {obs_set} observation dimension "
                f"({self.obs_dim}). Check map_scan_dim and the height_scan observation."
            )

        # PPO injects the actor's ModuleDict into the critic configuration when
        # share_cnn_encoders=True.  This matches AME's shared CNN and MHA while
        # retaining separate actor/critic proprioception projections.
        if cnns is None:
            stride = 2 if self.cnn_downsample else 1
            cnns = nn.ModuleDict(
                {
                    "map_cnn": nn.Sequential(
                        nn.Conv2d(3, 16, kernel_size=5, stride=stride, padding=2),
                        nn.ReLU(),
                        nn.BatchNorm2d(16),
                        nn.Conv2d(16, self.mha_dim, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(self.mha_dim),
                    ),
                    "mha": nn.MultiheadAttention(
                        embed_dim=self.mha_dim,
                        num_heads=self.num_heads,
                        batch_first=True,
                    ),
                }
            )
        self.cnns = cnns
        self.proprio_embedding = nn.Linear(self.proprio_dim, self.mha_dim)

        print(
            f"TerrainEncoderModel[{obs_set}]: map={self.L}x{self.W}x{self.coord_dim}, "
            f"proprio={self.proprio_dim}, tokens={self.num_terrain_tokens}, mha={self.mha_dim}"
        )

    @property
    def num_terrain_tokens(self) -> int:
        if not self.cnn_downsample:
            return self.L * self.W
        return (self.L // 2 + 1) * (self.W // 2 + 1)

    def _get_latent_dim(self) -> int:
        # Called by MLPModel.__init__ before the terrain modules are built.
        return self.obs_dim - self.map_scan_size + self.mha_dim

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
        # RayCaster stores the scan flattened with the opposite spatial order;
        # match AME by reshaping to [W, L, XYZ] before Conv2d.
        terrain = flat_obs[:, -self.map_scan_size :].reshape(
            -1, self.W, self.L, self.coord_dim
        )
        terrain = terrain.permute(0, 3, 1, 2)
        local_features = self.cnns["map_cnn"](terrain)
        local_features = local_features.permute(0, 2, 3, 1).flatten(1, 2)
        if local_features.shape[1] != self.num_terrain_tokens:
            raise RuntimeError(
                f"CNN produced {local_features.shape[1]} terrain tokens; "
                f"expected {self.num_terrain_tokens}."
            )

        query = self.proprio_embedding(proprio).unsqueeze(1)
        terrain_feature, _ = self.cnns["mha"](
            query=query,
            key=local_features,
            value=local_features,
            need_weights=False,
        )
        return torch.cat((proprio, terrain_feature.squeeze(1)), dim=-1)
