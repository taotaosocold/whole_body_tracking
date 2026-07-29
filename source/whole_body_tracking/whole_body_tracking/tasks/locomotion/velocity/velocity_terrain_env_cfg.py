"""Rough-terrain velocity locomotion with a height-map observation.

This module intentionally uses the standard RSL-RL MLP policy.  The xyz values
from a 17 x 11 scan are concatenated to the ordinary proprioceptive observations.
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.locomotion.velocity.mdp as mdp
from whole_body_tracking.tasks.locomotion.velocity.terrains import ROUGH_TERRAINS_CFG
from whole_body_tracking.tasks.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityFlatEnvCfg,
    MySceneCfg,
    ObservationsCfg,
)


@configclass
class TerrainSceneCfg(MySceneCfg):
    """Generated rough terrain, robot and terrain sensors."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
                "TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = MISSING
    height_scanner = RayCasterCfg(
        # Robot-specific configs replace this with their actual base link.
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class TerrainObservationsCfg(ObservationsCfg):
    """Proprioception plus a flattened 17 x 11 x 3 terrain scan (561 values)."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        height_scan = ObsTerm(
            func=mdp.elevation_map_xyz,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.03, n_max=0.03),
            clip=(-1.2, 1.2),
        )

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        height_scan = ObsTerm(
            func=mdp.elevation_map_xyz,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.2, 1.2),
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class TerrainCurriculumCfg:
    """Promote or demote each environment based on distance travelled."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class LocomotionVelocityTerrainEnvCfg(LocomotionVelocityFlatEnvCfg):
    """Generic rough-terrain locomotion configuration for standard MLP PPO."""

    scene: TerrainSceneCfg = TerrainSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: TerrainObservationsCfg = TerrainObservationsCfg()
    curriculum: TerrainCurriculumCfg = TerrainCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.terrain.terrain_generator.curriculum = True
