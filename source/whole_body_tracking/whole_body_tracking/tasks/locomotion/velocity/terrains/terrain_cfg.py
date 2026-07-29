"""Custom terrain configuration types for velocity locomotion."""

from dataclasses import MISSING

from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.utils import configclass

from .terrain_generator import concentric_gap_terrain


@configclass
class HfConcentricGapTerrainCfg(HfTerrainBaseCfg):
    """Height field made of alternating concentric ground and gap rings."""

    function = concentric_gap_terrain

    gap_width_range: tuple[float, float] = MISSING
    ground_width_range: tuple[float, float] = MISSING
    ground_height_max: float = MISSING
    gap_depth: float = -2.0
    platform_width: float = 1.0
