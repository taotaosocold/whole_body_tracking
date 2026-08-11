"""Static mesh terrain used by the parkour tracking task."""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils import configclass


def static_obstacle_terrain(
    difficulty: float, cfg: StaticObstacleTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Load an obstacle mesh and add a flat ground plane without changing its local pose."""
    del difficulty

    obstacle_mesh = trimesh.load(cfg.obstacle_file, force="mesh", process=False)
    if not isinstance(obstacle_mesh, trimesh.Trimesh):
        raise ValueError(f"Unable to load obstacle mesh as Trimesh: {cfg.obstacle_file}")

    # TerrainGenerator expects generated vertices in [0, size] and centers them afterwards.
    # The matching translation therefore cancels out and preserves the STL's authored coordinates.
    obstacle_mesh.apply_translation((cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.0))
    ground_mesh = make_plane(cfg.size, cfg.ground_height, center_zero=False)
    origin = np.array((cfg.size[0] / 2.0, cfg.size[1] / 2.0, cfg.ground_height))
    return [ground_mesh, obstacle_mesh], origin


@configclass
class StaticObstacleTerrainCfg(SubTerrainBaseCfg):
    """Configuration for one STL obstacle placed on a generated flat ground."""

    function = static_obstacle_terrain

    obstacle_file: str = MISSING
    ground_height: float = 0.0

