from isaaclab.utils import configclass

from whole_body_tracking.robots.marathon import MARATHON_ACTION_SCALE, MARATHON_CYLINDER_CFG
from whole_body_tracking.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityFlatEnvCfg
from whole_body_tracking.tasks.locomotion.velocity.velocity_terrain_env_cfg import (
    TerrainCurriculumCfg,
    TerrainObservationsCfg,
    TerrainSceneCfg,
)


@configclass
class MARATHONLocomotionFlatEnvCfg(LocomotionVelocityFlatEnvCfg):
    base_link_name = "base_link"
    foot_link_name = ".*_ankle_roll_link"

    def __post_init__(self):
        super().__post_init__()

        # ── Scene ──
        self.scene.robot = MARATHON_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ── Actions ──
        self.actions.joint_pos.scale = MARATHON_ACTION_SCALE
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}

        # ── Observations ──
        # remove base_lin_vel from policy (for domain randomization robustness)
        self.observations.policy.base_lin_vel = None

        # ── Events ──
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # MARATHON has no waist joint.
        self.rewards.joint_deviation_waists = None
        self.terminations.base_contact.params["sensor_cfg"].body_names = self.base_link_name

        # ── Commands ──
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

@configclass
class MARATHONLocomotionTerrainEnvCfg(MARATHONLocomotionFlatEnvCfg):
    """MARATHON velocity locomotion on curriculum rough terrain."""

    scene: TerrainSceneCfg = TerrainSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: TerrainObservationsCfg = TerrainObservationsCfg()
    curriculum: TerrainCurriculumCfg = TerrainCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.prim_path = f"{{ENV_REGEX_NS}}/Robot/{self.base_link_name}"
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.terrain.terrain_generator.curriculum = True
