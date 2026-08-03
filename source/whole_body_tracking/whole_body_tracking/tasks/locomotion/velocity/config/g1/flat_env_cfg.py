from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityFlatEnvCfg
from whole_body_tracking.tasks.locomotion.velocity.terrains import ROUGH_TERRAINS_CFG
from whole_body_tracking.tasks.locomotion.velocity.velocity_terrain_env_cfg import LocomotionVelocityTerrainEnvCfg


@configclass
class G1LocomotionFlatEnvCfg(LocomotionVelocityFlatEnvCfg):
    """G1 29-DOF velocity locomotion on flat terrain."""

    base_link_name = "torso_link"
    foot_link_name = ".*_ankle_roll_link"

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.observations.policy.base_lin_vel = None
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.joint_coordination.params["coord_joints"] = [
            ["left_hip_pitch_joint", "right_shoulder_pitch_joint"],
            ["right_hip_pitch_joint", "left_shoulder_pitch_joint"],
        ]
        self.rewards.joint_deviation_hip.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"]
        )
        self.rewards.joint_deviation_waists.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["waist.*"]
        )
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "torso_link",
            ".*_shoulder_.*_link",
            ".*_hip_.*_link",
            ".*_knee_link",
            ".*_elbow_link",
            "waist_.*_link",
            "pelvis",
        ]


@configclass
class G1LocomotionTerrainEnvCfg(LocomotionVelocityTerrainEnvCfg):
    """G1 29-DOF velocity locomotion on curriculum rough terrain."""

    base_link_name = "torso_link"
    foot_link_name = ".*_ankle_roll_link"

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.observations.policy.base_lin_vel = None
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.joint_coordination.params["coord_joints"] = [
            ["left_hip_pitch_joint", "right_shoulder_pitch_joint"],
            ["right_hip_pitch_joint", "left_shoulder_pitch_joint"],
        ]
        self.rewards.joint_deviation_hip.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"]
        )
        self.rewards.joint_deviation_waists.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["waist.*"]
        )
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "torso_link",
            ".*_shoulder_.*_link",
            ".*_hip_.*_link",
            ".*_knee_link",
            ".*_elbow_link",
            "waist_.*_link",
            "pelvis",
        ]
        self.scene.height_scanner.prim_path = f"{{ENV_REGEX_NS}}/Robot/{self.base_link_name}"


@configclass
class G1LocomotionTerrainNoDREnvCfg(G1LocomotionTerrainEnvCfg):
    """AME first-stage terrain training with material DR but without observation noise."""

    def __post_init__(self):
        super().__post_init__()

        # AME first-stage terrain and command/reset distribution.
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene.terrain.terrain_generator.curriculum = True
        self.commands.base_velocity.ranges.heading = (-3.141592653589793, 3.141592653589793)
        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)
        }

        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_rigid_body_mass_others = None
        self.events.randomize_com_positions = None
        self.events.randomize_push_robot = None

        self.events.randomize_reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

        self.observations.policy.enable_corruption = False
        self.observations.policy.height_scan.params["noise"] = False

        # AME first-stage reward weights.
        self.rewards.dof_torques_limits.weight = -0.01
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.feet_air_time_variance.weight = -0.7
        self.rewards.feet_slide.weight = -0.1
        self.rewards.feet_stumble.weight = -2.0
        self.rewards.feet_too_near.weight = -1.0
        self.rewards.joint_coordination.weight = -0.2
