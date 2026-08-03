"""Terrain configurations used by velocity-locomotion tasks."""

from .terrain_cfg import (
    HfAlternateColumnStakesTerrainCfg,
    HfConcentricGapTerrainCfg,
    HfDoubleColumnStakesTerrainCfg,
    HfStonesBridgeTerrainCfg,
    FINETUNE_ROUGH_TERRAINS_CFG,
    ROUGH_TERRAINS_CFG,
)

__all__ = [
    "FINETUNE_ROUGH_TERRAINS_CFG",
    "ROUGH_TERRAINS_CFG",
    "HfAlternateColumnStakesTerrainCfg",
    "HfConcentricGapTerrainCfg",
    "HfDoubleColumnStakesTerrainCfg",
    "HfStonesBridgeTerrainCfg",
]
