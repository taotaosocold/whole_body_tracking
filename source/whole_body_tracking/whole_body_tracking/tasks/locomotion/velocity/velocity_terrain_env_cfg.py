"""Rough-terrain velocity locomotion with an AME-style XYZ height map."""

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

import whole_body_tracking.tasks.locomotion.velocity.mdp as mdp
from whole_body_tracking.tasks.locomotion.velocity.terrains import FINETUNE_ROUGH_TERRAINS_CFG
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
        terrain_generator=FINETUNE_ROUGH_TERRAINS_CFG,
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
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class TerrainObservationsCfg(ObservationsCfg):
    """Proprioception plus a flattened 33 x 21 x 3 terrain scan (2079 values)."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        height_scan = ObsTerm(
            func=mdp.elevation_map_xyz,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "noise": True},
        )

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        height_scan = ObsTerm(
            func=mdp.elevation_map_xyz,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "noise": False},
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class TerrainCurriculumCfg:
    """Promote or demote each environment based on distance travelled."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class LocomotionVelocityTerrainEnvCfg(LocomotionVelocityFlatEnvCfg):
    """AME second-stage rough-terrain configuration for terrain-encoder PPO."""

    scene: TerrainSceneCfg = TerrainSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: TerrainObservationsCfg = TerrainObservationsCfg()
    curriculum: TerrainCurriculumCfg = TerrainCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.terrain.terrain_generator.curriculum = True

        # AME second-stage command/reset distribution.
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)
        }
        self.events.randomize_reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }

        # AME second-stage reward weights.
        self.rewards.dof_torques_limits.weight = -0.05
        self.rewards.action_rate_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.feet_air_time.weight = 0.5
        self.rewards.feet_air_time_variance.weight = -2.0
        self.rewards.feet_slide.weight = -0.3
        self.rewards.feet_stumble.weight = -5.0
        self.rewards.feet_too_near.weight = -5.0
        self.rewards.joint_coordination.weight = -0.5
